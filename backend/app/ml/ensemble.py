"""Ensemble — weighted KNN + LSTM predictions for final trading decisions.

Step 5.4: combine KNN and LSTM predictions with configurable weights
(derived from backtest accuracy).
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def ensemble_predict(
    knn_preds: np.ndarray,
    knn_probs: np.ndarray,
    lstm_preds: np.ndarray,
    lstm_probs: np.ndarray,
    knn_weight: float = 0.5,
    lstm_weight: float = 0.5,
    agreement_required: bool = True,
) -> list[dict[str, Any]]:
    """
    Combine KNN and LSTM predictions.

    knn_preds, lstm_preds: (n,) with class labels 0=HOLD, 1=BUY, 2=SELL
    knn_probs, lstm_probs: (n, 3) with class probabilities

    Returns list of prediction dicts.
    """
    n = len(knn_preds)
    results = []

    # Normalize weights
    total_w = knn_weight + lstm_weight
    knn_w = knn_weight / total_w
    lstm_w = lstm_weight / total_w

    for i in range(n):
        # Weighted probability combination
        combined_probs = knn_w * knn_probs[i] + lstm_w * lstm_probs[i]
        action = int(np.argmax(combined_probs))
        confidence = float(combined_probs[action])

        # Agreement check
        knn_agrees = int(knn_preds[i]) == action
        lstm_agrees = int(lstm_preds[i]) == action
        both_agree = int(knn_preds[i]) == int(lstm_preds[i])

        # If agreement required but models disagree → HOLD
        if agreement_required and not both_agree:
            action = 0
            confidence = float(combined_probs[0])

        results.append({
            "action": action,
            "confidence": round(confidence, 4),
            "knn_action": int(knn_preds[i]),
            "knn_confidence": round(float(knn_probs[i].max()), 4),
            "lstm_action": int(lstm_preds[i]),
            "lstm_confidence": round(float(lstm_probs[i].max()), 4),
            "agreement": both_agree,
            "combined_probs": {
                "hold": round(float(combined_probs[0]), 4),
                "buy": round(float(combined_probs[1]), 4),
                "sell": round(float(combined_probs[2]), 4),
            },
        })

    return results


def batch_predict(
    knn_model,
    lstm_model,
    X: np.ndarray,
    knn_weight: float = 0.5,
    lstm_weight: float = 0.5,
    agreement_required: bool = True,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    """
    Run both models and combine predictions.

    X: (n_samples, seq_len, n_features)
    """
    from app.ml.knn_distiller import predict_knn
    from app.ml.lstm_distiller import predict_lstm

    knn_preds, knn_probs = predict_knn(knn_model, X)
    lstm_preds, lstm_probs = predict_lstm(lstm_model, X, device=device)

    return ensemble_predict(
        knn_preds, knn_probs,
        lstm_preds, lstm_probs,
        knn_weight=knn_weight,
        lstm_weight=lstm_weight,
        agreement_required=agreement_required,
    )
