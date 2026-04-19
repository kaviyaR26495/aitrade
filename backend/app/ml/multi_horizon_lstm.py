"""Multi-Horizon LSTM: predicts 10 future trading-day *percentage returns*
(μ, σ) and a trend-durability score in a single forward pass.

Architecture
────────────
Input  : (batch, seq_len=15, n_features)
       → BiLSTM(hidden=256, layers=2) + LayerNorm
       → per-horizon heads: 10 × Linear(hidden, 2)  [μ, log_σ]
       → durability head  : Linear(hidden, 1) + Sigmoid [0-1]

Training targets
────────────────
For training-day d, horizon h ∈ {1…10}:
  - y_h   = percentage return at horizon h: (close[d+h] - close[d]) / close[d]
  - trend = fraction of h=1..5 steps with the same sign as h=1

Loss: sum of 10 GaussianNLLLoss(μ_h, σ_h², y_h) + BCE on trend durability
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
    """Bidirectional LSTM with 10 per-horizon regression heads (μ, log_σ)."""

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        horizon: int = HORIZON,
    ) -> None:
        super().__init__()
        self.horizon = horizon

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.layer_norm = nn.LayerNorm(hidden_size * 2)

        # 10 independent regression heads: each outputs (μ, log_σ)
        self.return_heads = nn.ModuleList(
            [nn.Linear(hidden_size * 2, 2) for _ in range(horizon)]
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
            return_params: list of 10 tensors each (batch, 2) → [μ, log_σ]
            trend_score:   (batch,) ∈ [0,1]
        """
        out, _ = self.lstm(x)                         # (batch, seq_len, hidden*2)
        last = self.layer_norm(out[:, -1, :])         # take last time-step
        return_params = [head(last) for head in self.return_heads]
        trend_score = self.trend_head(last).squeeze(-1)
        return return_params, trend_score


# ─────────────────────────────────────────────────────────────────────────────
# Label helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_horizon_labels(
    close_prices: np.ndarray,
    seq_len: int,
    horizon: int = HORIZON,
) -> tuple[np.ndarray, np.ndarray]:
    """Build multi-horizon regression labels from a 1-D close price array.

    Returns:
        y_returns: (n_samples, horizon) float32 — percentage returns
        y_trend:   (n_samples,) float32 — trend durability [0,1]
    """
    n = len(close_prices) - seq_len - horizon
    if n <= 0:
        raise ValueError(
            f"Not enough bars: len={len(close_prices)}, seq_len={seq_len}, horizon={horizon}"
        )

    y_returns = np.zeros((n, horizon), dtype=np.float32)
    y_trend = np.zeros(n, dtype=np.float32)

    for i in range(n):
        ref_close = close_prices[i + seq_len - 1]
        returns = []
        for h in range(1, horizon + 1):
            future_close = close_prices[i + seq_len - 1 + h]
            pct_return = (future_close - ref_close) / ref_close
            returns.append(pct_return)
        y_returns[i] = returns

        # Trend durability: fraction of first-5 horizons with same sign as h=1
        h1_sign = 1 if returns[0] > 0 else (-1 if returns[0] < 0 else 0)
        if h1_sign != 0:
            agree = sum(
                1 for r in returns[:5]
                if (r > 0 and h1_sign > 0) or (r < 0 and h1_sign < 0)
            )
            y_trend[i] = agree / 5.0
        else:
            y_trend[i] = 0.0

    return y_returns, y_trend


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
        y_returns: (n_samples, horizon) float32 — pct returns
        y_trend: (n_samples,) float32
    """
    n = len(feature_matrix) - seq_len - horizon
    if n <= 0:
        raise ValueError("Insufficient data for multi-horizon training")

    X = np.stack([feature_matrix[i : i + seq_len] for i in range(n)])
    y_returns, y_trend = _build_horizon_labels(close_prices, seq_len=seq_len, horizon=horizon)
    return X, y_returns, y_trend


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_multi_horizon(
    X: np.ndarray,
    y_returns: np.ndarray,
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
    """Train a MultiHorizonLSTM with GaussianNLL regression.

    Returns:
        model: trained MultiHorizonLSTM
        metrics: dict with train/val loss, per-horizon MAE, trend_mse
    """

    def _log(msg: str) -> None:
        logger.info(msg)
        if log_fn:
            log_fn(msg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(f"MultiHorizon training on {device}  X={X.shape}")

    n_samples, seq_len, n_features = X.shape
    horizon = y_returns.shape[1]
    split = int(n_samples * train_ratio)

    X_tr, X_va = X[:split], X[split:]
    yr_tr, yr_va = y_returns[:split], y_returns[split:]
    yt_tr, yt_va = y_trend[:split], y_trend[split:]

    def _to_tensors(xa, yra, yta):
        return (
            torch.tensor(xa, dtype=torch.float32),
            torch.tensor(yra, dtype=torch.float32),
            torch.tensor(yta, dtype=torch.float32),
        )

    t_X, t_yr, t_yt = _to_tensors(X_tr, yr_tr, yt_tr)
    v_X, v_yr, v_yt = _to_tensors(X_va, yr_va, yt_va)

    train_loader = DataLoader(
        TensorDataset(t_X, t_yr, t_yt), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(v_X, v_yr, v_yt), batch_size=batch_size * 2, shuffle=False
    )

    model = MultiHorizonLSTM(
        n_features=n_features,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        horizon=horizon,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    gnll = nn.GaussianNLLLoss()
    bce_loss = nn.BCELoss()

    best_val_loss = float("inf")
    best_state: dict = {}
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for bx, by_r, by_t in train_loader:
            bx, by_r, by_t = bx.to(device), by_r.to(device), by_t.to(device)
            optimizer.zero_grad()
            return_params, trend_score = model(bx)
            loss = torch.tensor(0.0, device=device)
            for h in range(horizon):
                mu = return_params[h][:, 0]
                log_sigma = return_params[h][:, 1]
                var = torch.exp(2 * log_sigma).clamp(min=1e-6)
                loss = loss + gnll(mu, by_r[:, h], var)
            loss = loss + bce_loss(trend_score, by_t)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item()

        model.eval()
        va_loss = 0.0
        horizon_mae = np.zeros(horizon)
        horizon_total = 0
        with torch.no_grad():
            for bx, by_r, by_t in val_loader:
                bx, by_r, by_t = bx.to(device), by_r.to(device), by_t.to(device)
                return_params, trend_score = model(bx)
                loss = torch.tensor(0.0, device=device)
                for h in range(horizon):
                    mu = return_params[h][:, 0]
                    log_sigma = return_params[h][:, 1]
                    var = torch.exp(2 * log_sigma).clamp(min=1e-6)
                    loss = loss + gnll(mu, by_r[:, h], var)
                    horizon_mae[h] += (mu - by_r[:, h]).abs().sum().item()
                loss = loss + bce_loss(trend_score, by_t)
                va_loss += loss.item()
                horizon_total += bx.size(0)

        tr_loss /= len(train_loader)
        va_loss /= len(val_loader)

        if epoch % 10 == 0 or epoch == 1:
            h1_mae = horizon_mae[0] / max(horizon_total, 1)
            _log(
                f"Epoch {epoch:3d}/{epochs}  tr={tr_loss:.4f}  va={va_loss:.4f}  "
                f"h1_mae={h1_mae:.6f}"
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

    per_horizon_mae = (horizon_mae / max(horizon_total, 1)).tolist()
    metrics = {
        "best_val_loss": best_val_loss,
        "h1_mae": per_horizon_mae[0],
        "per_horizon_mae": per_horizon_mae,
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
        f"h1_mae={per_horizon_mae[0]:.6f}  "
        f"h5_mae={per_horizon_mae[4]:.6f}"
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
        mu: (n_samples, horizon) float32 — predicted mean pct return
        sigma: (n_samples, horizon) float32 — predicted std (uncertainty)
        trend_score: (n_samples,) float32 ∈ [0,1]
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    model.to(device)

    x_t = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        return_params, trend = model(x_t)

    horizon = len(return_params)
    n = X.shape[0]
    mu_arr = np.zeros((n, horizon), dtype=np.float32)
    sigma_arr = np.zeros((n, horizon), dtype=np.float32)

    for h, params in enumerate(return_params):
        mu_arr[:, h] = params[:, 0].cpu().numpy()
        sigma_arr[:, h] = torch.exp(params[:, 1]).cpu().numpy()  # exp(log_σ) → σ

    return {
        "mu": mu_arr,
        "sigma": sigma_arr,
        "trend_score": trend.cpu().numpy(),
    }
