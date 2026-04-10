"""Daily prediction service — runs KNN+LSTM ensemble on all active stocks."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
import uuid
from datetime import datetime
from typing import Any, Sequence

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from sqlalchemy import select, desc
from app.config import settings
from app.core.data_service import get_model_ready_data
from app.core.normalizer import prepare_model_input
from app.db import crud
from app.db.models import KNNModel, LSTMModel, EnsembleConfig, EnsemblePrediction
from app.db.database import async_session_factory
from app.ml.knn_distiller import load_knn_model, predict_knn, load_knn_norm_params
from app.ml.lstm_distiller import load_lstm_model, predict_lstm
from app.ml.ensemble import ensemble_predict
from app.core.regime_classifier import classify_regimes

logger = logging.getLogger(__name__)

ACTION_MAP = {0: "HOLD", 1: "BUY", 2: "SELL"}

# How often to open a secondary DB session to check for job cancellation.
# Opening a new connection on every stock causes excessive overhead at scale.
_CANCEL_CHECK_INTERVAL = 10
# Predictions are flushed to DB in chunks; avoids per-row commit round-trips.
_COMMIT_BATCH_SIZE = 50

# ── Model Cache ─────────────────────────────────────────────────────────
# Avoids re-loading heavy KNN/LSTM files from disk on every prediction run.
# Cache is keyed by file path so it auto-refreshes if a new model is trained.
_knn_cache: dict[str, Any] = {}   # path -> (model, norm_params)
_lstm_cache: dict[str, Any] = {}  # path -> model

def _load_knn_cached(path: str, norm_dir: Path):
    if path not in _knn_cache:
        logger.info(f"Loading KNN model from disk: {path}")
        _knn_cache[path] = (load_knn_model(path), load_knn_norm_params(norm_dir))
    return _knn_cache[path]

def _load_lstm_cached(path: str):
    if path not in _lstm_cache:
        logger.info(f"Loading LSTM model from disk: {path}")
        _lstm_cache[path] = load_lstm_model(path)
    return _lstm_cache[path]

def clear_model_cache():
    """Call after training a new model to force a fresh load on next run."""
    _knn_cache.clear()
    _lstm_cache.clear()
    logger.info("Model cache cleared.")


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
    job_id: str | None = None,
    batch_id: str | None = None,
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
        knn_model, knn_norm_params = _load_knn_cached(knn_db.model_path, Path(knn_db.model_path).parent)
    else:
        # Resolve 'latest' via DB
        q = select(KNNModel).where(KNNModel.status == "completed").order_by(desc(KNNModel.created_at)).limit(1)
        res = await db.execute(q)
        knn_db = res.scalar_one_or_none()
        
        if knn_db and knn_db.model_path:
            knn_model, knn_norm_params = _load_knn_cached(knn_db.model_path, Path(knn_db.model_path).parent)
        else:
            # Fallback legacy path
            knn_model_dir = model_path / "knn" / knn_name
            knn_path = _latest_artifact(knn_model_dir, "knn_model_*.joblib", "knn_model.joblib")
            knn_model, knn_norm_params = _load_knn_cached(knn_path, knn_model_dir)

    # ── Load LSTM model ───────────────────────────────────────────────
    if lstm_model_id is not None:
        lstm_db = await crud.get_lstm_model(db, lstm_model_id)
        if lstm_db is None or not lstm_db.model_path:
            raise ValueError(f"LSTM model id={lstm_model_id} has no saved artifact in DB")
        lstm_model = _load_lstm_cached(lstm_db.model_path)
    else:
        # Resolve 'latest' via DB
        q = select(LSTMModel).where(LSTMModel.status == "completed").order_by(desc(LSTMModel.created_at)).limit(1)
        res = await db.execute(q)
        lstm_db = res.scalar_one_or_none()
        
        if lstm_db and lstm_db.model_path:
            lstm_model = _load_lstm_cached(lstm_db.model_path)
        else:
            # Fallback legacy path
            lstm_model_dir = model_path / "lstm" / lstm_name
            lstm_path = _latest_artifact(lstm_model_dir, "lstm_model_*.pt", "lstm_model.pt")
            lstm_model = _load_lstm_cached(lstm_path)

    # ── Resolve Ensemble Config ──────────────────────────────────────────
    try:
        if ensemble_config_id is None:
            # Try to find latest config
            q_cfg = select(EnsembleConfig).order_by(desc(EnsembleConfig.created_at)).limit(1)
            res_cfg = await db.execute(q_cfg)
            cfg_db = res_cfg.scalar_one_or_none()
            
            if cfg_db:
                ensemble_config_id = cfg_db.id
            else:
                # Create a default config if we have model IDs
                k_id = knn_db.id if (knn_db and hasattr(knn_db, "id")) else None
                l_id = lstm_db.id if (lstm_db and hasattr(lstm_db, "id")) else None
                
                if k_id and l_id:
                    new_cfg = await crud.create_ensemble_config(
                        db,
                        name=f"Auto-generated Config ({target_date})",
                        knn_model_id=k_id,
                        lstm_model_id=l_id,
                        interval=interval,
                    )
                    ensemble_config_id = new_cfg.id
                    logger.info(f"Created default EnsembleConfig id={ensemble_config_id}")
                else:
                    logger.warning("Cannot find/create EnsembleConfig: missing model IDs")
    except Exception as e:
        logger.error(f"Failed to resolve EnsembleConfig: {e}", exc_info=True)
        raise

    # Get stocks to predict — bulk fetch to avoid N+1 queries
    if stock_ids:
        from app.db.models import Stock
        res = await db.execute(select(Stock).where(Stock.id.in_(stock_ids)))
        stocks_map = {s.id: s for s in res.scalars().all()}
        stocks = [stocks_map[sid] for sid in stock_ids if sid in stocks_map]
    else:
        from app.core import data_service
        stocks = await data_service.get_universe_stocks(db)

    # Prepare batch metadata
    if not batch_id:
        batch_id = str(uuid.uuid4())
    run_at = datetime.now()
    
    # Associate batch with job immediately if not already done
    if job_id:
        await crud.update_prediction_job(db, job_id, batch_id=batch_id)

    results = []
    errors = []
    seq_len = settings.DEFAULT_SEQ_LEN_DAILY if interval == "day" else settings.DEFAULT_SEQ_LEN_WEEKLY

    processed_count = 0
    pending_predictions: list[EnsemblePrediction] = []
    for i, stock in enumerate(stocks):
        try:
            # Check for cancellation every N stocks to avoid per-stock session overhead
            if job_id and i % _CANCEL_CHECK_INTERVAL == 0:
                async with async_session_factory() as check_db:
                    job = await crud.get_prediction_job(check_db, job_id)
                    if job and job.status == "cancelled":
                        logger.info(f"Prediction job {job_id} cancelled by user.")
                        break

            # Get model-ready data
            df, feature_cols = await get_model_ready_data(
                db, stock.id, interval=interval, seq_len=seq_len,
                end_date=target_date
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

            # Fail-safe regime classification: compute on-the-fly to ensure it is never null/missing
            try:
                df_regime = classify_regimes(df.iloc[-1:])
                regime_id = int(df_regime["regime_id"].iloc[0])
            except Exception as reg_exc:
                logger.warning(f"On-the-fly regime classification failed for {stock.symbol}: {reg_exc}")
                regime_id = int(df["regime_id"].iloc[-1]) if "regime_id" in df.columns else 0

            preds = ensemble_predict(
                knn_preds, knn_probs, lstm_preds, lstm_probs,
                knn_weight=stock_knn_w, lstm_weight=stock_lstm_w,
                agreement_required=agreement_required,
            )

            if preds:
                p = preds[0]
                pred_row = {
                    "batch_id": batch_id,
                    "run_at": run_at,
                    "ensemble_config_id": ensemble_config_id,
                    "stock_id": stock.id,
                    "date": target_date,
                    "interval": interval,
                    "action": int(p["action"]),
                    "confidence": float(p["confidence"]),
                    "knn_action": int(p["knn_action"]),
                    "knn_confidence": float(p["combined_probs"][ACTION_MAP[p["knn_action"]].lower()]),
                    "lstm_action": int(p["lstm_action"]),
                    "lstm_confidence": float(p["combined_probs"][ACTION_MAP[p["lstm_action"]].lower()]),
                    "agreement": p["agreement"],
                    "regime_id": regime_id,
                }
                
                pending_predictions.append(EnsemblePrediction(**pred_row))
                # Batch-commit to avoid per-row transaction overhead
                if len(pending_predictions) >= _COMMIT_BATCH_SIZE:
                    db.add_all(pending_predictions)
                    await db.commit()
                    pending_predictions.clear()

                results.append(pred_row)

        finally:
            processed_count += 1
            # Progressive throttling: update every stock for first 10, then every 5
            is_last = processed_count == len(stocks)
            if job_id and (processed_count <= 10 or processed_count % 5 == 0 or is_last):
                progress = int((processed_count / len(stocks)) * 100)
                await crud.update_prediction_job(
                    db, job_id, 
                    completed_stocks=processed_count, 
                    progress=progress
                )

        
    # Flush any remaining predictions that didn't fill a full batch
    if pending_predictions:
        db.add_all(pending_predictions)
        await db.commit()
        pending_predictions.clear()

    # Finalize job status
    if job_id:
        is_cancelled = False
        job = await crud.get_prediction_job(db, job_id)
        if job and job.status == "cancelled":
            is_cancelled = True
        
        await crud.update_prediction_job(
            db, job_id, 
            status="cancelled" if is_cancelled else "completed",
            batch_id=batch_id,
            progress=100
        )

    return {
        "date": str(target_date),
        "total_stocks": len(stocks),
        "predictions_made": len(results),
        "errors": len(errors),
        "buy_signals": sum(1 for r in results if r["action"] == 1),
        "sell_signals": sum(1 for r in results if r["action"] == 2),
        "hold_signals": sum(1 for r in results if r["action"] == 0),
        "results": results,
        "error_details": errors,
    }
