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
from app.ml.knn_distiller import load_knn_model, predict_knn
from app.ml.lstm_distiller import load_lstm_model, predict_lstm
from app.ml.ensemble import ensemble_predict

logger = logging.getLogger(__name__)

ACTION_MAP = {0: "HOLD", 1: "BUY", 2: "SELL"}


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
) -> dict[str, Any]:
    """Run ensemble predictions for all active stocks and store results."""
    model_dir = model_dir or settings.MODEL_DIR
    model_path = Path(model_dir)
    target_date = target_date or date.today()

    # Load models
    knn_model = load_knn_model(str(model_path / "knn" / knn_name / "knn_model.joblib"))
    lstm_model = load_lstm_model(str(model_path / "lstm" / lstm_name / "lstm_model.pt"))

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

            knn_preds, knn_probs = predict_knn(knn_model, X_last)
            lstm_preds, lstm_probs = predict_lstm(lstm_model, X_last)

            preds = ensemble_predict(
                knn_preds, knn_probs, lstm_preds, lstm_probs,
                knn_weight=knn_weight, lstm_weight=lstm_weight,
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
