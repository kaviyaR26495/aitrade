"""Multi-Horizon LSTM: predicts 10 future trading-day actions (BUY=1/HOLD=0/SELL=2)
and a trend-durability score in a single forward pass.

Architecture
────────────
Input  : (batch, seq_len=15, n_features)
       → BiLSTM(hidden=256, layers=2) + LayerNorm
       → per-horizon heads: 10 × Linear(hidden, 3)  [action logits]
       → durability head  : Linear(hidden, 1) + Sigmoid [0-1]

Training targets
────────────────
For training-day d, horizon h ∈ {1…10}:
  - y_h   = discretised ROC(d+h): ROC > +1% → BUY(1), ROC < -1% → SELL(2), else HOLD(0)
  - trend = fraction of h=1..5 steps with the same non-HOLD direction as h=1

Loss: mean of 10 CrossEntropy losses + BCE on trend durability
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)
HORIZON = 10


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class MultiHorizonLSTM(nn.Module):
    """Bidirectional LSTM with 10 per-horizon classification heads."""

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        horizon: int = HORIZON,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.num_classes = num_classes

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.layer_norm = nn.LayerNorm(hidden_size * 2)

        # 10 independent action classification heads
        self.action_heads = nn.ModuleList(
            [nn.Linear(hidden_size * 2, num_classes) for _ in range(horizon)]
        )
        # Trend durability head
        self.trend_head = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """
        x: (batch, seq_len, n_features)
        Returns:
            action_logits: list of 10 tensors each (batch, 3)
            trend_score:   (batch,) ∈ [0,1]
        """
        out, _ = self.lstm(x)                         # (batch, seq_len, hidden*2)
        last = self.layer_norm(out[:, -1, :])         # take last time-step
        action_logits = [head(last) for head in self.action_heads]
        trend_score = self.trend_head(last).squeeze(-1)
        return action_logits, trend_score


# ─────────────────────────────────────────────────────────────────────────────
# Label helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_horizon_labels(
    close_prices: np.ndarray,
    seq_len: int,
    horizon: int = HORIZON,
    buy_thresh: float = 0.01,
    sell_thresh: float = -0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Build multi-horizon labels from a 1-D close price array.

    Returns:
        y_horizon: (n_samples, horizon) int8 — 0=HOLD, 1=BUY, 2=SELL
        y_trend:   (n_samples,) float32 — trend durability [0,1]
    """
    n = len(close_prices) - seq_len - horizon
    if n <= 0:
        raise ValueError(
            f"Not enough bars: len={len(close_prices)}, seq_len={seq_len}, horizon={horizon}"
        )

    y_horizon = np.zeros((n, horizon), dtype=np.int8)
    y_trend = np.zeros(n, dtype=np.float32)

    for i in range(n):
        ref_close = close_prices[i + seq_len - 1]
        actions = []
        for h in range(1, horizon + 1):
            future_close = close_prices[i + seq_len - 1 + h]
            roc = (future_close - ref_close) / ref_close
            # Scale threshold by sqrt(h) so short horizons don't require the
            # same absolute ROC as long horizons (avoids HOLD bias at h=1).
            h_scale = h ** 0.5
            h_buy = buy_thresh * h_scale
            h_sell = sell_thresh * h_scale
            if roc > h_buy:
                actions.append(1)  # BUY
            elif roc < h_sell:
                actions.append(2)  # SELL
            else:
                actions.append(0)  # HOLD
        y_horizon[i] = actions

        # Trend durability: fraction of first-5 horizons agreeing with h=1
        h1 = actions[0]
        if h1 != 0:
            agree = sum(1 for a in actions[:5] if a == h1)
            y_trend[i] = agree / 5.0
        else:
            y_trend[i] = 0.0

    return y_horizon, y_trend


def build_training_data(
    close_prices: np.ndarray,
    feature_matrix: np.ndarray,
    seq_len: int = 15,
    horizon: int = HORIZON,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert full feature matrix + close prices into training tensors.

    Args:
        close_prices: (T,) daily close prices
        feature_matrix: (T, n_features) normalised feature matrix
        seq_len: look-back window
        horizon: forecast horizon

    Returns:
        X: (n_samples, seq_len, n_features)
        y_horizon: (n_samples, horizon) int8
        y_trend: (n_samples,) float32
    """
    n = len(feature_matrix) - seq_len - horizon
    if n <= 0:
        raise ValueError("Insufficient data for multi-horizon training")

    X = np.stack([feature_matrix[i : i + seq_len] for i in range(n)])
    y_horizon, y_trend = _build_horizon_labels(close_prices, seq_len=seq_len, horizon=horizon)
    return X, y_horizon, y_trend


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_multi_horizon(
    X: np.ndarray,
    y_horizon: np.ndarray,
    y_trend: np.ndarray,
    hidden_size: int = 256,
    num_layers: int = 2,
    dropout: float = 0.3,
    lr: float = 3e-4,
    epochs: int = 80,
    patience: int = 15,
    batch_size: int = 64,
    train_ratio: float = 0.8,
    log_fn: Callable[[str], None] | None = None,
) -> tuple[MultiHorizonLSTM, dict[str, Any]]:
    """Train a MultiHorizonLSTM.

    Returns:
        model: trained MultiHorizonLSTM
        metrics: dict with train/val loss, per-horizon accuracy, trend_mse
    """

    def _log(msg: str) -> None:
        logger.info(msg)
        if log_fn:
            log_fn(msg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(f"MultiHorizon training on {device}  X={X.shape}")

    n_samples, seq_len, n_features = X.shape
    horizon = y_horizon.shape[1]
    split = int(n_samples * train_ratio)

    X_tr, X_va = X[:split], X[split:]
    yh_tr, yh_va = y_horizon[:split], y_horizon[split:]
    yt_tr, yt_va = y_trend[:split], y_trend[split:]

    def _to_tensors(xa, yha, yta):
        return (
            torch.tensor(xa, dtype=torch.float32),
            torch.tensor(yha, dtype=torch.long),
            torch.tensor(yta, dtype=torch.float32),
        )

    t_X, t_yh, t_yt = _to_tensors(X_tr, yh_tr, yt_tr)
    v_X, v_yh, v_yt = _to_tensors(X_va, yh_va, yt_va)

    train_loader = DataLoader(
        TensorDataset(t_X, t_yh, t_yt), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(v_X, v_yh, v_yt), batch_size=batch_size * 2, shuffle=False
    )

    model = MultiHorizonLSTM(
        n_features=n_features,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        horizon=horizon,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    ce_loss = nn.CrossEntropyLoss()
    bce_loss = nn.BCELoss()

    best_val_loss = float("inf")
    best_state: dict = {}
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for bx, by_h, by_t in train_loader:
            bx, by_h, by_t = bx.to(device), by_h.to(device), by_t.to(device)
            optimizer.zero_grad()
            act_logits, trend_score = model(bx)
            loss = sum(ce_loss(act_logits[h], by_h[:, h]) for h in range(horizon))
            loss += bce_loss(trend_score, by_t)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item()

        model.eval()
        va_loss = 0.0
        horizon_correct = np.zeros(horizon)
        horizon_total = 0
        with torch.no_grad():
            for bx, by_h, by_t in val_loader:
                bx, by_h, by_t = bx.to(device), by_h.to(device), by_t.to(device)
                act_logits, trend_score = model(bx)
                loss = sum(ce_loss(act_logits[h], by_h[:, h]) for h in range(horizon))
                loss += bce_loss(trend_score, by_t)
                va_loss += loss.item()
                for h in range(horizon):
                    preds = act_logits[h].argmax(dim=1)
                    horizon_correct[h] += (preds == by_h[:, h]).sum().item()
                horizon_total += bx.size(0)

        tr_loss /= len(train_loader)
        va_loss /= len(val_loader)

        if epoch % 10 == 0 or epoch == 1:
            h1_acc = horizon_correct[0] / max(horizon_total, 1)
            _log(
                f"Epoch {epoch:3d}/{epochs}  tr={tr_loss:.4f}  va={va_loss:.4f}  "
                f"h1_acc={h1_acc:.4f}"
            )

        if va_loss < best_val_loss - 1e-5:
            best_val_loss = va_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                _log(f"Early stop at epoch {epoch}")
                break

    if best_state:
        model.load_state_dict(best_state)

    per_horizon_acc = (horizon_correct / max(horizon_total, 1)).tolist()
    metrics = {
        "best_val_loss": best_val_loss,
        "h1_accuracy": per_horizon_acc[0],
        "per_horizon_accuracy": per_horizon_acc,
        "n_train": split,
        "n_val": n_samples - split,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "horizon": horizon,
        "n_features": n_features,
        "seq_len": seq_len,
    }
    _log(
        f"Training complete  best_val_loss={best_val_loss:.4f}  "
        f"h1_acc={per_horizon_acc[0]:.4f}  "
        f"h5_acc={per_horizon_acc[4]:.4f}"
    )
    return model, metrics


# ─────────────────────────────────────────────────────────────────────────────
# Save / Load
# ─────────────────────────────────────────────────────────────────────────────

def save_multi_horizon_model(
    model: MultiHorizonLSTM,
    metrics: dict,
    save_dir: str | Path,
    model_name: str = "mh_lstm",
) -> dict[str, str]:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = str(save_dir / f"{model_name}.pt")
    torch.save(
        {
            "model_state": model.state_dict(),
            "n_features": metrics["n_features"],
            "hidden_size": metrics["hidden_size"],
            "num_layers": metrics["num_layers"],
            "horizon": metrics["horizon"],
            "seq_len": metrics["seq_len"],
            "metrics": metrics,
        },
        model_path,
    )
    return {"model_path": model_path}


def load_multi_horizon_model(
    model_path: str,
    device: str | None = None,
) -> MultiHorizonLSTM:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(model_path, map_location=device)
    model = MultiHorizonLSTM(
        n_features=ckpt["n_features"],
        hidden_size=ckpt["hidden_size"],
        num_layers=ckpt["num_layers"],
        horizon=ckpt["horizon"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def predict_multi_horizon(
    model: MultiHorizonLSTM,
    X: np.ndarray,
    device: str | None = None,
) -> dict[str, np.ndarray]:
    """Run inference.

    Args:
        model: loaded MultiHorizonLSTM
        X: (n_samples, seq_len, n_features)

    Returns dict:
        actions: (n_samples, horizon) int — 0=HOLD, 1=BUY, 2=SELL
        confidences: (n_samples, horizon) float32 ∈ [0,1]
        trend_score: (n_samples,) float32 ∈ [0,1]
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    model.to(device)

    x_t = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        act_logits, trend = model(x_t)

    horizon = len(act_logits)
    n = X.shape[0]
    actions = np.zeros((n, horizon), dtype=np.int32)
    confs = np.zeros((n, horizon), dtype=np.float32)

    for h, logits in enumerate(act_logits):
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        actions[:, h] = probs.argmax(axis=1)
        confs[:, h] = probs.max(axis=1)

    return {
        "actions": actions,
        "confidences": confs,
        "trend_score": trend.cpu().numpy(),
    }
