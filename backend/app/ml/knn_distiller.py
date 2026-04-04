"""KNN Distillation — train KNN from golden patterns extracted by RL.

Step 5.2:
1. Load golden patterns from DB
2. Handle class imbalance (SMOTE)
3. Flatten feature windows + append regime features
4. Train KNeighborsClassifier
5. Chronological train/test split (80/20)
6. Save artifacts: model.joblib + norm_params.json
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
)

logger = logging.getLogger(__name__)


def train_knn(
    X: np.ndarray,
    y: np.ndarray,
    k_neighbors: int = 5,
    train_ratio: float = 0.8,
    use_smote: bool = True,
    log_fn: Any = None,
) -> tuple[KNeighborsClassifier, dict[str, Any]]:
    """
    Train KNN on golden patterns.

    X: (n_samples, seq_len, n_features) — will be flattened
    y: (n_samples,) — 0=HOLD, 1=BUY, 2=SELL

    Returns (model, metrics_dict).
    """
    def _log(msg: str) -> None:
        logger.info(msg)
        if log_fn:
            log_fn(msg)

    if len(X) == 0:
        raise ValueError("No training data provided")

    # Flatten: (n, seq_len, features) → (n, seq_len * features)
    n_samples = X.shape[0]
    X_flat = X.reshape(n_samples, -1)

    # Chronological split (no shuffle — time series!)
    split_idx = int(n_samples * train_ratio)
    X_train, X_test = X_flat[:split_idx], X_flat[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    _log(
        f"INFO   KNN split: train={len(X_train)}, test={len(X_test)}.  Classes: {dict(zip(*np.unique(y_train, return_counts=True)))}"
    )

    # SMOTE oversampling for class imbalance
    if use_smote and len(np.unique(y_train)) > 1:
        try:
            from imblearn.over_sampling import SMOTE
            _log("INFO   Running SMOTE oversampling...")
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            _log(f"INFO   After SMOTE: {dict(zip(*np.unique(y_train, return_counts=True)))}")
        except ImportError:
            _log("WARN   imblearn not installed, skipping SMOTE")

    # Train KNN
    _log(f"INFO   Fitting KNN (k={k_neighbors}, n_train={len(X_train)}, features={X_flat.shape[1]})...")
    knn = KNeighborsClassifier(
        n_neighbors=k_neighbors,
        weights="distance",
        n_jobs=-1,
    )
    knn.fit(X_train, y_train)
    _log("INFO   KNN fit complete. Evaluating...")

    # Evaluate
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Per-class precision
    labels_present = sorted(np.unique(y_test))
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    prec_buy = report.get("1", {}).get("precision", 0.0)
    prec_sell = report.get("2", {}).get("precision", 0.0)

    metrics = {
        "accuracy": round(float(accuracy), 4),
        "precision_buy": round(float(prec_buy), 4),
        "precision_sell": round(float(prec_sell), 4),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "k_neighbors": k_neighbors,
        "class_distribution_train": dict(zip(*[a.tolist() for a in np.unique(y_train, return_counts=True)])),
        "class_distribution_test": dict(zip(*[a.tolist() for a in np.unique(y_test, return_counts=True)])),
        "classification_report": report,
    }

    return knn, metrics


def save_knn_model(
    model: KNeighborsClassifier,
    metrics: dict,
    save_dir: str | Path,
    model_name: str,
    feature_cols: list[str] | None = None,
    norm_params: dict | None = None,
) -> dict[str, str]:
    """Save KNN model and associated artifacts."""
    save_path = Path(save_dir) / model_name
    save_path.mkdir(parents=True, exist_ok=True)

    # Save model
    model_file = save_path / "knn_model.joblib"
    joblib.dump(model, model_file)

    # Save metadata
    meta = {
        "model_name": model_name,
        "metrics": metrics,
        "feature_cols": feature_cols,
    }
    meta_file = save_path / "metadata.json"
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    # Save norm params
    if norm_params:
        norm_file = save_path / "norm_params.json"
        with open(norm_file, "w") as f:
            json.dump(norm_params, f, indent=2, default=str)

    return {
        "model_path": str(model_file),
        "metadata_path": str(meta_file),
        "norm_params_path": str(save_path / "norm_params.json") if norm_params else None,
    }


def load_knn_model(model_path: str) -> KNeighborsClassifier:
    """Load a saved KNN model."""
    return joblib.load(model_path)


def predict_knn(
    model: KNeighborsClassifier,
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict with KNN. Returns (predictions, probabilities).
    X: (n_samples, seq_len, n_features) or already flattened.
    """
    if X.ndim == 3:
        X = X.reshape(X.shape[0], -1)

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    return predictions, probabilities
