"""Daily prediction service — runs KNN+LSTM ensemble on all active stocks."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.data_service import get_model_ready_data
from app.core.normalizer import prepare_model_input
from app.db import crud
from app.ml.knn_distiller import load_knn_model, predict_knn, load_knn_norm_params
from app.ml.lstm_distiller import load_lstm_model, predict_lstm
from app.ml.ensemble import ensemble_predict

logger = logging.getLogger(__name__)

ACTION_MAP = {0: "HOLD", 1: "BUY", 2: "SELL"}


def _latest_artifact(directory: Path, glob_pattern: str, legacy_name: str) -> str:
    """Return the path of the most-recent versioned artifact in *directory*.

    Versioned files follow the ``{base}_{YYYYMMDD_HHMM}.{ext}`` naming
    convention produced by the save functions and are lexicographically
    sortable by date.  Falls back to *legacy_name* for models saved before
    the versioning scheme was introduced.
    """
    candidates = sorted(directory.glob(glob_pattern))
    if candidates:
        return str(candidates[-1])
    legacy = directory / legacy_name
    if legacy.exists():
        return str(legacy)
    raise FileNotFoundError(
        f"No artifact found in {directory} matching '{glob_pattern}' or '{legacy_name}'"
    )


async def run_daily_predictions(
    db: AsyncSession,
    model_dir: str | None = None,
    knn_name: str = "latest",
    lstm_name: str = "latest",
    knn_weight: float = 0.5,
    lstm_weight: float = 0.5,
    agreement_required: bool = True,
    target_date: date | None = None,
    interval: str = "day",
    stock_ids: list[int] | None = None,
    ensemble_config_id: int | None = None,
    knn_model_id: int | None = None,
    lstm_model_id: int | None = None,
) -> dict[str, Any]:
    """Run ensemble predictions for all active stocks and store results.

    When ``ensemble_config_id`` is provided the function queries the DB for
    per-stock calibrated KNN/LSTM weights (written by
    ``ensemble.per_stock_optimal_weights``).  If no per-stock row exists for a
    given stock the global ``knn_weight`` / ``lstm_weight`` defaults are used.

    When ``knn_model_id`` or ``lstm_model_id`` are provided the function loads
    the model from the exact filesystem path stored in the DB
    ``KNNModel.model_path`` / ``LSTMModel.model_path`` columns.  This enables
    instant rollback: point the DB row at an older timestamped artifact via the
    API and the next prediction run automatically uses the older weights.
    """
    model_dir = model_dir or settings.MODEL_DIR
    model_path = Path(model_dir)
    target_date = target_date or date.today()

    # ── Load KNN model ────────────────────────────────────────────────
    if knn_model_id is not None:
        knn_db = await crud.get_knn_model(db, knn_model_id)
        if knn_db is None or not knn_db.model_path:
            raise ValueError(f"KNN model id={knn_model_id} has no saved artifact in DB")
        knn_model = load_knn_model(knn_db.model_path)
        knn_norm_params = load_knn_norm_params(Path(knn_db.model_path).parent)
    else:
        knn_model_dir = model_path / "knn" / knn_name
        knn_model = load_knn_model(
            _latest_artifact(knn_model_dir, "knn_model_*.joblib", "knn_model.joblib")
        )
        knn_norm_params = load_knn_norm_params(knn_model_dir)

    # ── Load LSTM model ───────────────────────────────────────────────
    if lstm_model_id is not None:
        lstm_db = await crud.get_lstm_model(db, lstm_model_id)
        if lstm_db is None or not lstm_db.model_path:
            raise ValueError(f"LSTM model id={lstm_model_id} has no saved artifact in DB")
        lstm_model = load_lstm_model(lstm_db.model_path)
    else:
        lstm_model_dir = model_path / "lstm" / lstm_name
        lstm_model = load_lstm_model(
            _latest_artifact(lstm_model_dir, "lstm_model_*.pt", "lstm_model.pt")
        )

    # Get stocks to predict
    if stock_ids:
        stocks = [await crud.get_stock_by_id(db, sid) for sid in stock_ids]
        stocks = [s for s in stocks if s is not None]
    else:
        stocks = await crud.get_all_active_stocks(db)

    results = []
    errors = []
    seq_len = settings.DEFAULT_SEQ_LEN_DAILY if interval == "day" else settings.DEFAULT_SEQ_LEN_WEEKLY

    for stock in stocks:
        try:
            # Get model-ready data
            df, feature_cols = await get_model_ready_data(
                db, stock.id, interval=interval, seq_len=seq_len
            )

            if df is None or len(df) < seq_len:
                logger.warning(f"Insufficient data for {stock.symbol}, skipping")
                continue

            # Prepare input — take last seq_len rows for "today's" prediction
            X = prepare_model_input(df, feature_cols, seq_len=seq_len)
            if len(X) == 0:
                continue

            # Use the last window only (most recent)
            X_last = X[-1:].copy()

            # Resolve per-stock calibrated weights (fall back to global defaults)
            stock_knn_w, stock_lstm_w = knn_weight, lstm_weight
            if ensemble_config_id is not None:
                stock_weights = await crud.get_stock_ensemble_weight(
                    db, ensemble_config_id=ensemble_config_id, stock_id=stock.id
                )
                if stock_weights is not None:
                    stock_knn_w = stock_weights.knn_weight
                    stock_lstm_w = stock_weights.lstm_weight

            knn_preds, knn_probs = predict_knn(knn_model, X_last, norm_params=knn_norm_params)
            lstm_preds, lstm_probs = predict_lstm(lstm_model, X_last)

            preds = ensemble_predict(
                knn_preds, knn_probs, lstm_preds, lstm_probs,
                knn_weight=stock_knn_w, lstm_weight=stock_lstm_w,
                agreement_required=agreement_required,
            )

            if preds:
                p = preds[0]
                results.append({
                    "stock_id": stock.id,
                    "symbol": stock.symbol,
                    "date": target_date,
                    "interval": interval,
                    "action": ACTION_MAP.get(p["action"], "HOLD"),
                    "confidence": float(p["confidence"]),
                    "knn_action": ACTION_MAP.get(p["knn_action"], "HOLD"),
                    "knn_confidence": float(p["combined_probs"][p["knn_action"]]),
                    "lstm_action": ACTION_MAP.get(p["lstm_action"], "HOLD"),
                    "lstm_confidence": float(p["combined_probs"][p["lstm_action"]]),
                    "agreement": p["agreement"],
                    "regime_id": None,  # filled below if available
                })

        except Exception as e:
            logger.error(f"Prediction failed for {stock.symbol}: {e}")
            errors.append({"stock_id": stock.id, "symbol": stock.symbol, "error": str(e)})

    # Store predictions in DB
    if results:
        from app.db.models import EnsemblePrediction
        rows = []
        for r in results:
            rows.append(EnsemblePrediction(
                stock_id=r["stock_id"],
                date=r["date"],
                interval=r["interval"],
                action=r["action"],
                confidence=r["confidence"],
                knn_action=r["knn_action"],
                knn_confidence=r["knn_confidence"],
                lstm_action=r["lstm_action"],
                lstm_confidence=r["lstm_confidence"],
                agreement=r["agreement"],
                regime_id=r["regime_id"],
            ))
        db.add_all(rows)
        await db.commit()

    return {
        "date": str(target_date),
        "total_stocks": len(stocks),
        "predictions_made": len(results),
        "errors": len(errors),
        "buy_signals": sum(1 for r in results if r["action"] == "BUY"),
        "sell_signals": sum(1 for r in results if r["action"] == "SELL"),
        "hold_signals": sum(1 for r in results if r["action"] == "HOLD"),
        "results": results,
        "error_details": errors,
    }
