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
from sklearn.neighbors import KNeighborsClassifier  # kept for loading legacy saved models
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
)
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class FaissKNNClassifier:
    """FAISS-backed approximate nearest-neighbor classifier.

    Drop-in replacement for sklearn KNeighborsClassifier.  Uses
    IndexFlatL2 for small datasets (< 10 000 samples) and IndexIVFFlat
    for larger ones, keeping search latency in the millisecond range.
    Fully compatible with joblib.dump / joblib.load via custom
    __getstate__ / __setstate__ that serialize the FAISS index to bytes.
    """

    def __init__(self, k: int = 5) -> None:
        self.k = k
        self._index = None
        self._train_y: np.ndarray | None = None
        self._classes: np.ndarray | None = None
        self._n_features: int | None = None

    # ------------------------------------------------------------------
    # Joblib / pickle compatibility
    # ------------------------------------------------------------------
    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        if self._index is not None:
            import faiss
            state["_index_bytes"] = faiss.serialize_index(self._index).tobytes()
        del state["_index"]
        return state

    def __setstate__(self, state: dict) -> None:
        index_bytes = state.pop("_index_bytes", None)
        self.__dict__.update(state)
        if index_bytes is not None:
            import faiss
            arr = np.frombuffer(index_bytes, dtype=np.uint8)
            self._index = faiss.deserialize_index(arr)
        else:
            self._index = None

    # ------------------------------------------------------------------
    # sklearn-compatible interface
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "FaissKNNClassifier":
        import faiss
        X = np.ascontiguousarray(X, dtype=np.float32)
        n, d = X.shape
        self._n_features = d
        self._train_y = np.asarray(y)
        self._classes = np.unique(y)
        if n < 10_000:
            index: faiss.Index = faiss.IndexFlatL2(d)
        else:
            nlist = min(int(np.sqrt(n)), 256)
            quantizer = faiss.IndexFlatL2(d)
            index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
            index.train(X)
            index.nprobe = max(10, nlist // 10)
        index.add(X)
        self._index = index
        return self

    def _query(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.ascontiguousarray(X, dtype=np.float32)
        k = min(self.k, self._index.ntotal)
        distances, indices = self._index.search(X, k)
        return distances, indices

    def predict(self, X: np.ndarray) -> np.ndarray:
        distances, indices = self._query(X)
        n = X.shape[0]
        preds = np.empty(n, dtype=self._train_y.dtype)
        for i in range(n):
            neighbor_labels = self._train_y[indices[i]]
            weights = 1.0 / (distances[i] + 1e-8)
            class_scores: dict = {c: 0.0 for c in self._classes}
            for label, w in zip(neighbor_labels, weights):
                class_scores[label] += w
            preds[i] = max(class_scores, key=class_scores.__getitem__)
        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        distances, indices = self._query(X)
        n = X.shape[0]
        n_classes = len(self._classes)
        class_idx = {c: j for j, c in enumerate(self._classes)}
        probs = np.zeros((n, n_classes), dtype=np.float32)
        for i in range(n):
            neighbor_labels = self._train_y[indices[i]]
            weights = 1.0 / (distances[i] + 1e-8)
            for label, w in zip(neighbor_labels, weights):
                probs[i, class_idx[label]] += w
            total = probs[i].sum()
            if total > 0:
                probs[i] /= total
        return probs

    @property
    def classes_(self) -> np.ndarray:
        return self._classes


def train_knn(
    X: np.ndarray,
    y: np.ndarray,
    k_neighbors: int = 5,
    train_ratio: float = 0.8,
    use_smote: bool = True,
    augment_jitter: bool = False,
    jitter_noise_std: float = 0.001,
    jitter_copies: int = 1,
    log_fn: Any = None,
) -> tuple[KNeighborsClassifier, dict[str, Any]]:
    """
    Train KNN on golden patterns.

    X: (n_samples, seq_len, n_features) — will be flattened
    y: (n_samples,) — 0=HOLD, 1=BUY, 2=SELL

    augment_jitter: if True, Gaussian jitter is applied *only* to the training
        split after the chronological split, so the validation set remains
        composed entirely of original (unaugmented) market data.

    Returns (model, metrics_dict).
    """
    def _log(msg: str) -> None:
        logger.info(msg)
        if log_fn:
            log_fn(msg)

    if len(X) < 15:
        raise ValueError(
            f"Insufficient patterns extracted for KNN: found only {len(X)} samples. "
            "The RL model likely learned to only HOLD. Needs more RL training timesteps."
        )

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

    # Z-score normalization — fit ONLY on train to prevent data leakage
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    norm_params: dict = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
    }
    _log(f"INFO   StandardScaler fitted (features={X_flat.shape[1]}).")

    # Jitter augmentation — applied to X_train ONLY, after the split,
    # so the validation set stays pure (no synthetic data leaks in)
    if augment_jitter and len(X_train) > 0:
        from app.ml.pattern_extractor import jitter_augment
        # Unflatten → jitter → re-flatten to preserve seq structure in noise
        n_tr = X_train.shape[0]
        seq_len = X.shape[1] if X.ndim == 3 else 1
        n_feat = X_flat.shape[1] // seq_len if seq_len > 1 else X_flat.shape[1]
        X_3d = X_train.reshape(n_tr, seq_len, n_feat)
        X_3d_aug, y_train = jitter_augment(X_3d, y_train, noise_std=jitter_noise_std, copies=jitter_copies)
        X_train = X_3d_aug.reshape(len(X_3d_aug), -1)
        _log(f"INFO   Jitter augmentation applied to train only: {n_tr} → {len(X_train)} samples.")

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

    # Train FAISS-KNN
    _log(f"INFO   Fitting FaissKNN (k={k_neighbors}, n_train={len(X_train)}, features={X_flat.shape[1]})...")
    knn = FaissKNNClassifier(k=k_neighbors)
    knn.fit(X_train, y_train)
    _log("INFO   FaissKNN fit complete. Evaluating...")

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
        "norm_params": norm_params,  # saved alongside model for inference-time normalization
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
    """Save KNN model and associated artifacts.

    Files are saved with a UTC timestamp suffix so that each training run
    produces a *new* artifact rather than overwriting the previous one.
    The returned ``model_path`` should be persisted to the DB
    ``KNNModel.model_path`` column so the predictor can load the exact
    version that is currently marked active — enabling instant rollback by
    pointing the DB row at an older versioned file.
    """
    from datetime import datetime as _dt

    timestamp = _dt.now().strftime("%Y%m%d_%H%M")
    save_path = Path(save_dir) / model_name
    save_path.mkdir(parents=True, exist_ok=True)

    # Save model — versioned so old artifacts are preserved on disk
    model_file = save_path / f"knn_model_{timestamp}.joblib"
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

    # Pull norm_params out of metrics dict if not explicitly supplied
    if norm_params is None:
        norm_params = metrics.get("norm_params")

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


def load_knn_model(model_path: str) -> "FaissKNNClassifier | KNeighborsClassifier":
    """Load a saved KNN model (FaissKNNClassifier for new models, legacy sklearn KNN for old ones)."""
    return joblib.load(model_path)


def load_knn_norm_params(model_dir: str | Path) -> dict | None:
    """Load norm_params.json from a trained KNN model directory."""
    norm_file = Path(model_dir) / "norm_params.json"
    if norm_file.exists():
        with open(norm_file) as f:
            return json.load(f)
    return None


def predict_knn(
    model,
    X: np.ndarray,
    norm_params: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict with KNN. Returns (predictions, probabilities).

    X: (n_samples, seq_len, n_features) or already flattened.
    norm_params: dict with 'mean' and 'scale' lists produced during training.
                 When supplied the input is Z-score normalized identically to
                 how training data was transformed, preventing feature-scale bias.
    """
    if X.ndim == 3:
        X = X.reshape(X.shape[0], -1).astype(np.float64)
    else:
        X = X.astype(np.float64)

    if norm_params is not None:
        mean = np.asarray(norm_params["mean"], dtype=np.float64)
        scale = np.asarray(norm_params["scale"], dtype=np.float64)
        X = (X - mean) / np.where(scale > 0, scale, 1.0)

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    return predictions, probabilities
