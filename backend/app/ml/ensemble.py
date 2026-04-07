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

        both_agree = int(knn_preds[i]) == int(lstm_preds[i])

        # Directional agreement check (replaces exact-label agreement).
        #
        # Three cases when agreement_required=True:
        #   1. knn == lstm (exact)          → keep combined action (strictest signal)
        #   2. BUY vs SELL (hard conflict)  → always HOLD (contradictory views)
        #   3. HOLD vs directional          → allow if combined probability already
        #                                     favours that direction over HOLD;
        #                                     otherwise fall back to HOLD.
        #
        # This preserves the precision filter while avoiding the "deadlock" where
        # KNN (geometric) and LSTM (temporal) rarely agree on the exact same candle.
        if agreement_required and not both_agree:
            knn_a = int(knn_preds[i])
            lstm_a = int(lstm_preds[i])
            hard_conflict = (knn_a != 0 and lstm_a != 0 and knn_a != lstm_a)
            if hard_conflict:
                # BUY vs SELL — genuinely contradictory views; force HOLD
                action = 0
                confidence = float(combined_probs[0])
            else:
                # One of them is HOLD; the other has a directional view.
                # Trust the direction only if the *combined* probability for
                # that direction exceeds the HOLD probability.
                direction = knn_a if knn_a != 0 else lstm_a
                if combined_probs[direction] > combined_probs[0]:
                    action = direction
                    confidence = float(combined_probs[direction])
                else:
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


def optimize_ensemble_weights(
    knn_preds: np.ndarray,
    knn_probs: np.ndarray,
    lstm_preds: np.ndarray,
    lstm_probs: np.ndarray,
    y_true: np.ndarray,
    weight_steps: int = 9,
) -> tuple[float, float]:
    """Grid-search the KNN/LSTM weight ratio on historical predictions.

    Tries *weight_steps* ratios from (0.1/0.9) to (0.9/0.1) and picks the
    pair that maximises precision on non-HOLD (BUY + SELL) signals, which is
    the metric that directly drives P&L reliability.

    Args:
        knn_preds/knn_probs: outputs from predict_knn over the history window.
        lstm_preds/lstm_probs: outputs from predict_lstm over the same window.
        y_true: ground-truth labels, same length as the prediction arrays.
        weight_steps: number of evenly-spaced candidates to evaluate.

    Returns:
        (knn_weight, lstm_weight) that sum to 1.0.
    """
    y_true = np.asarray(y_true)
    best_score = -1.0
    best_knn_w = 0.5

    for knn_w in np.linspace(0.1, 0.9, weight_steps):
        lstm_w = 1.0 - knn_w
        preds = ensemble_predict(
            knn_preds, knn_probs, lstm_preds, lstm_probs,
            knn_weight=float(knn_w), lstm_weight=float(lstm_w),
            agreement_required=False,
        )
        actions = np.array([p["action"] for p in preds])
        non_hold = actions != 0
        if non_hold.sum() == 0:
            continue
        # Precision: fraction of non-HOLD calls where model was correct
        precision = (actions[non_hold] == y_true[non_hold]).sum() / non_hold.sum()
        if precision > best_score:
            best_score = precision
            best_knn_w = float(knn_w)

    return round(best_knn_w, 2), round(1.0 - best_knn_w, 2)


def per_stock_optimal_weights(
    knn_model,
    lstm_model,
    X_history: np.ndarray,
    y_history: np.ndarray,
    norm_params: dict | None = None,
    device: str = "cpu",
    weight_steps: int = 9,
) -> tuple[float, float]:
    """Determine per-stock optimal KNN/LSTM weight ratio.

    Runs both models over the supplied history window (typically the last
    12 months of windowed features) and calls optimize_ensemble_weights to
    find the ratio that best matches ground-truth labels.

    The result can be persisted in the DB (knn_weight / lstm_weight columns
    on the EnsemblePrediction model) so that each stock is evaluated with
    its own calibrated weights rather than a single global 0.5/0.5 default.

    Args:
        X_history: (n_samples, seq_len, n_features) history windows.
        y_history: (n_samples,) ground-truth labels for those windows.
        norm_params: scaler params dict from KNN training (norm_params.json).

    Returns:
        (knn_weight, lstm_weight) in [0, 1] summing to 1.
    """
    from app.ml.knn_distiller import predict_knn
    from app.ml.lstm_distiller import predict_lstm

    knn_preds, knn_probs = predict_knn(knn_model, X_history, norm_params=norm_params)
    lstm_preds, lstm_probs = predict_lstm(lstm_model, X_history, device=device)

    return optimize_ensemble_weights(
        knn_preds, knn_probs,
        lstm_preds, lstm_probs,
        y_true=y_history,
        weight_steps=weight_steps,
    )
