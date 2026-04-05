"""LSTM Distillation — train LSTM from golden patterns extracted by RL.

Step 5.3: LSTM preserves temporal ordering (unlike KNN which flattens).

Architecture:
  Input: (batch, seq_len, num_features + regime_features)
  → LSTM(hidden_size=128, num_layers=2, dropout=0.3)
  → Linear(128, 64) → ReLU → Dropout(0.3) → Linear(64, 3) → Softmax
  Output: 3-class probabilities (HOLD, BUY, SELL)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class TradeLSTM(nn.Module):
    """LSTM classifier for BUY/HOLD/SELL prediction."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Use final hidden state from last layer
        final_hidden = h_n[-1]  # (batch, hidden_size)
        logits = self.classifier(final_hidden)
        return logits


def train_lstm(
    X: np.ndarray,
    y: np.ndarray,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.3,
    lr: float = 1e-3,
    batch_size: int = 64,
    max_epochs: int = 100,
    patience: int = 10,
    train_ratio: float = 0.8,
    augment_jitter: bool = False,
    jitter_noise_std: float = 0.001,
    jitter_copies: int = 1,
    device: str | None = None,
    log_fn: Any = None,
) -> tuple[TradeLSTM, dict[str, Any]]:
    """
    Train LSTM on golden patterns.

    X: (n_samples, seq_len, n_features)
    y: (n_samples,) — 0=HOLD, 1=BUY, 2=SELL

    augment_jitter: if True, Gaussian jitter is applied *only* to the training
        split after the chronological split, so the validation set remains
        composed entirely of original (unaugmented) market data.

    Returns (model, metrics_dict).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    def _log(msg: str) -> None:
        logger.info(msg)
        if log_fn:
            log_fn(msg)

    n_samples, seq_len, n_features = X.shape

    # Chronological split
    split_idx = int(n_samples * train_ratio)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    _log(
        f"INFO   LSTM split: train={len(X_train)}, val={len(X_val)}.  Classes: {dict(zip(*np.unique(y_train, return_counts=True)))}"
    )

    # Jitter augmentation — applied to X_train ONLY, after the split,
    # so the validation set stays pure (no synthetic data leaks in)
    if augment_jitter and len(X_train) > 0:
        from app.ml.pattern_extractor import jitter_augment
        n_before = len(X_train)
        X_train, y_train = jitter_augment(X_train, y_train, noise_std=jitter_noise_std, copies=jitter_copies)
        _log(f"INFO   Jitter augmentation applied to train only: {n_before} → {len(X_train)} samples.")

    # Class weights for imbalanced data (always size num_classes=3)
    num_classes = 3
    classes, counts = np.unique(y_train, return_counts=True)
    total = counts.sum()
    weight_array = np.ones(num_classes, dtype=np.float32)
    for cls, cnt in zip(classes, counts):
        weight_array[int(cls)] = total / (len(classes) * cnt)
    class_weights = torch.tensor(weight_array, dtype=torch.float32).to(device)

    # DataLoaders
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Model
    model = TradeLSTM(
        input_size=n_features,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    # Training loop
    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(max_epochs):
        # Train
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(X_batch)
        train_loss /= len(train_ds)

        # Validate
        model.eval()
        val_loss = 0
        correct = 0
        total_val = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item() * len(X_batch)
                preds = logits.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total_val += len(y_batch)
        val_loss /= len(val_ds)
        val_acc = correct / total_val if total_val > 0 else 0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        scheduler.step(val_loss)

        if epoch % 10 == 0:
            _log(
                f"INFO   Epoch {epoch}/{max_epochs}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
            )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                _log(f"INFO   Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    from sklearn.metrics import accuracy_score, classification_report, precision_score
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)

    metrics = {
        "accuracy": round(float(accuracy), 4),
        "precision_buy": round(float(report.get("1", {}).get("precision", 0)), 4),
        "precision_sell": round(float(report.get("2", {}).get("precision", 0)), 4),
        "best_val_loss": round(float(best_val_loss), 6),
        "epochs_trained": len(history["train_loss"]),
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "classification_report": report,
        "history": history,
    }

    logger.info("LSTM accuracy: %.4f, prec_buy: %.4f, prec_sell: %.4f", accuracy, metrics["precision_buy"], metrics["precision_sell"])

    return model.cpu(), metrics


def fine_tune_lstm(
    base_model: TradeLSTM,
    X: np.ndarray,
    y: np.ndarray,
    freeze_lstm: bool = True,
    lr: float = 1e-4,
    max_epochs: int = 30,
    patience: int = 7,
    batch_size: int = 32,
    train_ratio: float = 0.8,
    device: str | None = None,
    log_fn: Any = None,
) -> tuple[TradeLSTM, dict[str, Any]]:
    """Fine-tune a pre-trained TradeLSTM on stock-specific data.

    Two-phase strategy
    ------------------
    Phase 1 (frozen LSTM layers):
        Only the classifier head is trainable.  The model adapts its output
        mapping to the new stock's statistics without forgetting the temporal
        patterns learned on the base dataset.  Runs for half of max_epochs.

    Phase 2 (all layers unfrozen):
        Full fine-tune at a 10× lower LR.  Allows the LSTM hidden
        representations to also shift slightly toward the target stock.

    This is equivalent to "feature extraction → full fine-tune" transfer
    learning used in computer vision (e.g., ResNet pre-trained on ImageNet).

    Parameters
    ----------
    base_model : TradeLSTM
        A model already trained (e.g. on pooled sector or NIFTY 500 data).
        Will be deep-copied so the original is not mutated.
    X, y : np.ndarray
        Stock-specific training data from ``patterns_to_training_data()``.
    freeze_lstm : bool
        Whether to run Phase 1 with frozen LSTM.  Set False to skip directly
        to full fine-tune (useful when X is large enough on its own).
    lr : float
        Learning rate for Phase 2 (full fine-tune).  Phase 1 uses ``lr * 5``.
    """
    import copy

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    def _log(msg: str) -> None:
        logger.info(msg)
        if log_fn:
            log_fn(msg)

    if len(X) == 0:
        raise ValueError("No fine-tuning data provided")

    model = copy.deepcopy(base_model).to(device)

    n_samples = X.shape[0]
    split_idx = int(n_samples * train_ratio)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    _log(f"INFO   Fine-tune split: train={len(X_train)}, val={len(X_val)}")

    # Class weights
    num_classes = 3
    classes, counts = np.unique(y_train, return_counts=True)
    total = counts.sum()
    weight_array = np.ones(num_classes, dtype=np.float32)
    for cls, cnt in zip(classes, counts):
        weight_array[int(cls)] = total / (len(classes) * cnt)
    class_weights = torch.tensor(weight_array).to(device)

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    history: dict[str, list] = {"train_loss": [], "val_loss": [], "val_acc": [], "phase": []}

    def _run_phase(phase_name: str, phase_lr: float, phase_epochs: int) -> None:
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=phase_lr
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=3, factor=0.5
        )
        _log(f"INFO   Fine-tune {phase_name}: lr={phase_lr}, epochs={phase_epochs}")

        best_val = float("inf")
        best_state = None
        no_improve = 0

        for epoch in range(phase_epochs):
            model.train()
            train_loss = 0.0
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(Xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item() * len(Xb)
            train_loss /= len(train_ds)

            model.eval()
            val_loss = 0.0
            correct = 0
            with torch.no_grad():
                for Xb, yb in val_loader:
                    Xb, yb = Xb.to(device), yb.to(device)
                    out = model(Xb)
                    val_loss += criterion(out, yb).item() * len(Xb)
                    correct  += (out.argmax(1) == yb).sum().item()
            val_loss /= max(len(val_ds), 1)
            val_acc   = correct / max(len(val_ds), 1)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["phase"].append(phase_name)

            scheduler.step(val_loss)
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    _log(f"INFO   Early stop ({phase_name}) at epoch {epoch}")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

    # Phase 1: classifier-head only
    if freeze_lstm:
        for param in model.lstm.parameters():
            param.requires_grad = False
        _run_phase("head-only", lr * 5, max(max_epochs // 2, 5))
        # Unfreeze for Phase 2
        for param in model.lstm.parameters():
            param.requires_grad = True

    # Phase 2: full fine-tune
    _run_phase("full", lr, max_epochs)

    # Final metrics on validation set
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for Xb, yb in val_loader:
            logits = model(Xb.to(device))
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_labels.extend(yb.tolist())

    from sklearn.metrics import accuracy_score, classification_report
    all_preds_np  = np.array(all_preds)
    all_labels_np = np.array(all_labels)
    accuracy = accuracy_score(all_labels_np, all_preds_np)
    report   = classification_report(all_labels_np, all_preds_np, output_dict=True, zero_division=0)

    metrics = {
        "accuracy": round(float(accuracy), 4),
        "precision_buy":  round(float(report.get("1", {}).get("precision", 0)), 4),
        "precision_sell": round(float(report.get("2", {}).get("precision", 0)), 4),
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "epochs_trained": len(history["train_loss"]),
        "classification_report": report,
        "history": history,
    }
    _log(
        f"INFO   Fine-tune complete: accuracy={accuracy:.4f}, "
        f"prec_buy={metrics['precision_buy']:.4f}, prec_sell={metrics['precision_sell']:.4f}"
    )
    return model.cpu(), metrics


def save_lstm_model(
    model: TradeLSTM,
    metrics: dict,
    save_dir: str | Path,
    model_name: str,
    feature_cols: list[str] | None = None,
) -> dict[str, str]:
    """Save LSTM model and artifacts."""
    save_path = Path(save_dir) / model_name
    save_path.mkdir(parents=True, exist_ok=True)

    # Save model weights
    model_file = save_path / "lstm_model.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "input_size": model.lstm.input_size,
        "hidden_size": model.lstm.hidden_size,
        "num_layers": model.lstm.num_layers,
        "num_classes": model.classifier[-1].out_features,
    }, model_file)

    # Save metadata
    meta = {
        "model_name": model_name,
        "metrics": metrics,
        "feature_cols": feature_cols,
    }
    meta_file = save_path / "metadata.json"
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    return {
        "model_path": str(model_file),
        "metadata_path": str(meta_file),
    }


def load_lstm_model(model_path: str, device: str = "cpu") -> TradeLSTM:
    """Load a saved LSTM model."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model = TradeLSTM(
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        num_classes=checkpoint["num_classes"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def predict_lstm(
    model: TradeLSTM,
    X: np.ndarray,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict with LSTM. Returns (predictions, probabilities).
    X: (n_samples, seq_len, n_features).
    """
    model.eval()
    model = model.to(device)

    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()

    return preds, probs
