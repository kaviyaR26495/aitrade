"""Meta-Classifier — XGBoost Probability-of-Profit (PoP) gate.

Replaces hardcoded confluence thresholds with a learned binary classifier
that predicts whether a signal candidate will be profitable.

Training
────────
- Trained on a rolling 6-month window of past TradeSignal outcomes
- Features: confluence_score, fqs_score, execution_cost_pct, rr_ratio,
            lstm_mu, lstm_sigma, knn_win_rate, regime_id, sr_features
- Label: 1 if signal hit target, 0 if hit SL or expired

Inference
─────────
- Produces pop_score ∈ [0, 1] for each TradeSignalCandidate
- Only signals with pop_score > threshold proceed to OMS

This module requires ``xgboost`` which is already in requirements.txt.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Feature names in the order expected by the model
META_FEATURES = [
    "confluence_score",
    "fqs_score",
    "execution_cost_pct",
    "initial_rr_ratio",
    "net_expected_return_pct",
    "lstm_mu",
    "lstm_sigma",
    "knn_median_return",
    "knn_win_rate",
    "regime_id",
    "sr_support_dist",
    "sr_resistance_dist",
    "sr_support_strength",
    "sr_resistance_strength",
    "sr_zone_count",
    "sr_rr_ratio",
]


@dataclass
class MetaClassifierResult:
    """Output of the meta-classifier for a single signal."""
    pop_score: float        # probability of profit ∈ [0, 1]
    feature_importances: Optional[dict[str, float]] = None


class MetaClassifier:
    """XGBoost-based PoP gate for trade signal candidates."""

    def __init__(self, threshold: float = 0.55) -> None:
        self.threshold = threshold
        self._model = None
        self._feature_names = META_FEATURES

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    def build_feature_vector(
        self,
        confluence_score: float,
        fqs_score: float,
        execution_cost_pct: float,
        initial_rr_ratio: float,
        net_expected_return_pct: float,
        lstm_mu: float,
        lstm_sigma: float,
        knn_median_return: float,
        knn_win_rate: float,
        regime_id: int,
        sr_features: dict[str, float],
    ) -> np.ndarray:
        """Build a feature vector for a single signal candidate."""
        return np.array([
            confluence_score,
            fqs_score,
            execution_cost_pct,
            initial_rr_ratio,
            net_expected_return_pct,
            lstm_mu,
            lstm_sigma,
            knn_median_return,
            knn_win_rate,
            float(regime_id),
            sr_features.get("sr_support_dist", 0.0),
            sr_features.get("sr_resistance_dist", 0.0),
            sr_features.get("sr_support_strength", 0.0),
            sr_features.get("sr_resistance_strength", 0.0),
            sr_features.get("sr_zone_count", 0.0),
            sr_features.get("sr_rr_ratio", 1.0),
        ], dtype=np.float32)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_ratio: float = 0.2,
        **xgb_params: Any,
    ) -> dict[str, Any]:
        """Train the meta-classifier on historical signal outcomes.

        Parameters
        ----------
        X : ndarray (n_samples, n_features)
            Feature matrix from past signals.
        y : ndarray (n_samples,)
            Binary labels: 1 = profitable, 0 = unprofitable.
        eval_ratio : float
            Fraction of data for early stopping validation.
        **xgb_params
            Override XGBoost hyperparameters.

        Returns
        -------
        dict with training metrics (AUC, accuracy, feature importance).
        """
        import xgboost as xgb
        from sklearn.metrics import roc_auc_score, accuracy_score

        n = len(X)
        # Data MUST be sorted chronologically before calling train().
        # Purged split: leave a 5-sample gap between train and validation
        # to prevent overlapping hold-period target leakage.
        purge_gap = 5
        split = int(n * (1 - eval_ratio))
        X_tr, X_va = X[:split - purge_gap], X[split:]
        y_tr, y_va = y[:split - purge_gap], y[split:]

        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "n_estimators": 300,
            "early_stopping_rounds": 20,
        }
        params.update(xgb_params)

        n_estimators = params.pop("n_estimators", 300)
        early_stop = params.pop("early_stopping_rounds", 20)

        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            early_stopping_rounds=early_stop,
            **params,
        )
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
        )
        self._model = model

        # Evaluate
        y_pred_proba = model.predict_proba(X_va)[:, 1]
        y_pred = (y_pred_proba >= self.threshold).astype(int)

        auc = roc_auc_score(y_va, y_pred_proba) if len(np.unique(y_va)) > 1 else 0.0
        acc = accuracy_score(y_va, y_pred)

        # Feature importance
        importance = dict(zip(
            self._feature_names,
            model.feature_importances_.tolist(),
        ))

        metrics = {
            "auc": round(float(auc), 4),
            "accuracy": round(float(acc), 4),
            "n_train": split,
            "n_val": n - split,
            "best_iteration": model.best_iteration if hasattr(model, "best_iteration") else n_estimators,
            "feature_importances": importance,
            "threshold": self.threshold,
        }
        logger.info(f"MetaClassifier trained: AUC={auc:.4f}, acc={acc:.4f}")
        return metrics

    def predict(self, X: np.ndarray) -> list[MetaClassifierResult]:
        """Predict PoP scores for signal candidates.

        Parameters
        ----------
        X : ndarray (n_samples, n_features)

        Returns
        -------
        List of MetaClassifierResult.
        """
        if self._model is None:
            # No model trained yet — return neutral scores
            return [
                MetaClassifierResult(pop_score=0.5)
                for _ in range(X.shape[0])
            ]

        probas = self._model.predict_proba(X)[:, 1]
        return [
            MetaClassifierResult(pop_score=round(float(p), 4))
            for p in probas
        ]

    def should_trade(self, pop_score: float) -> bool:
        """Check if a signal passes the PoP threshold."""
        return pop_score >= self.threshold

    def save(self, path: str | Path) -> str:
        """Save model to disk."""
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self._model,
            "threshold": self.threshold,
            "feature_names": self._feature_names,
        }, path)
        return str(path)

    def load(self, path: str | Path) -> None:
        """Load model from disk."""
        import joblib
        data = joblib.load(path)
        self._model = data["model"]
        self.threshold = data.get("threshold", 0.55)
        self._feature_names = data.get("feature_names", META_FEATURES)
