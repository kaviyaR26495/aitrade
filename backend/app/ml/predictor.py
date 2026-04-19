"""Daily prediction service — runs KNN+LSTM ensemble on all active stocks."""

from __future__ import annotations

import gc
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
from app.db.models import (
    KNNModel, LSTMModel, EnsembleConfig, EnsemblePrediction, Stock,
    LSTMHorizonModel, TradeSignal, SignalStatus,
)
from app.db.database import async_session_factory
from app.ml.knn_distiller import load_knn_model, predict_knn, load_knn_norm_params, predict_knn_returns
from app.ml.lstm_distiller import load_lstm_model, predict_lstm
from app.ml.multi_horizon_lstm import load_multi_horizon_model, predict_multi_horizon
from app.ml.ensemble import ensemble_predict
from app.ml.signal_synthesizer import synthesize_signal, SynthesizerConfig
from app.ml.meta_classifier import MetaClassifier
from app.ml.position_sizer import sector_concentration_multiplier
from app.core.regime_classifier import classify_regimes
from app.core.support_resistance import compute_sr_zones, sr_features
from app.core.fundamental_scorer import compute_fundamental_score, fundamental_features

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
    sector_guard: bool = True,
    weekly_confluence_filter: bool = True,
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

    # Reconcile any open orders against Zerodha before generating new signals.
    # Wrapped in try/except so a broker outage never blocks the prediction run.
    try:
        from app.core.oms import reconcile_open_orders
        reconciled = await reconcile_open_orders(db)
        logger.info("OMS reconciliation: %d orders updated before prediction run", reconciled)
    except Exception as oms_exc:
        logger.warning("OMS reconciliation failed (non-fatal): %s", oms_exc)

    # Snapshot open BUY positions for sector concentration guard (best-effort)
    open_positions_for_sector: dict[str, dict] = {}
    total_portfolio_equity: float = 1.0
    if sector_guard:
        try:
            # Build sector map from ALL stocks (not just prediction universe)
            # so open orders for stocks outside the universe are still counted.
            all_stocks_res = await db.execute(select(Stock))
            stock_sector_map: dict[int, str | None] = {
                s.id: s.sector for s in all_stocks_res.scalars().all()
            }

            raw_orders = await crud.get_open_trade_orders(db)
            for o in raw_orders:
                if o.transaction_type == "BUY":
                    price = o.avg_fill_price or o.price or 0.0
                    open_positions_for_sector[str(o.id)] = {
                        "quantity": o.filled_quantity or o.quantity,
                        "price": price,
                        "sector": stock_sector_map.get(o.stock_id),
                    }

            # Use PortfolioSnapshot for total equity (cash + holdings) when available
            snap = await crud.get_latest_portfolio_snapshot(db)
            if snap and (snap.cash_available or 0) + (snap.holdings_value or 0) > 0:
                total_portfolio_equity = (snap.cash_available or 0) + (snap.holdings_value or 0)
            else:
                # Fallback: sum open position values (underestimates equity)
                total_portfolio_equity = max(
                    sum(p["quantity"] * p["price"] for p in open_positions_for_sector.values()),
                    1.0,
                )
        except Exception as sp_exc:
            logger.warning("Failed to build open positions for sector guard (non-fatal): %s", sp_exc)

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
            latest_row = df.iloc[-1:].copy()
            X_last = X[-1:].copy()
            del df  # release per-stock DataFrame — reduces peak RSS at scale

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
                df_regime = classify_regimes(latest_row)
                regime_id = int(df_regime["regime_id"].iloc[0])
            except Exception as reg_exc:
                logger.warning(f"On-the-fly regime classification failed for {stock.symbol}: {reg_exc}")
                regime_id = int(latest_row["regime_id"].iloc[-1]) if "regime_id" in latest_row.columns else 0

            preds = ensemble_predict(
                knn_preds, knn_probs, lstm_preds, lstm_probs,
                knn_weight=stock_knn_w, lstm_weight=stock_lstm_w,
                agreement_required=agreement_required,
            )

            # ── Sector concentration guard ──────────────────────────────────
            if preds and sector_guard and stock.sector and preds[0]["action"] == 1:
                sect_mult = sector_concentration_multiplier(
                    stock.sector, open_positions_for_sector, total_portfolio_equity
                )
                if sect_mult == 0.0:
                    preds[0]["action"] = 0
                    preds[0]["sector_cap_flag"] = True

            # ── Weekly confluence filter ─────────────────────────────────────
            if preds and weekly_confluence_filter and preds[0]["action"] == 1:
                try:
                    w_rsi = float(latest_row["weekly_rsi"].iloc[0]) if "weekly_rsi" in latest_row.columns else 1.0
                    w_roc = float(latest_row["weekly_roc_1"].iloc[0]) if "weekly_roc_1" in latest_row.columns else 0.0
                    if w_rsi < 0.4 and w_roc < 0:
                        preds[0]["action"] = 0
                        preds[0]["weekly_confluence_blocked"] = True
                except Exception:
                    pass

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
                
                pending_predictions.append(pred_row)
                # Batch-commit to avoid per-row transaction overhead
                if len(pending_predictions) >= _COMMIT_BATCH_SIZE:
                    await crud.bulk_upsert_ensemble_predictions(db, pending_predictions)
                    pending_predictions.clear()
                    gc.collect()  # free cyclic garbage at existing periodic boundary

                results.append(pred_row)

        except Exception as stock_exc:
            logger.error("Prediction failed for %s: %s", stock.symbol, stock_exc, exc_info=True)
            errors.append({"stock_id": stock.id, "symbol": stock.symbol, "error": str(stock_exc)})
        finally:
            processed_count += 1
            # Progressive throttling: update every stock for first 10, then every 5
            is_last = processed_count == len(stocks)
            if job_id and (processed_count <= 10 or processed_count % 5 == 0 or is_last):
                progress = int((processed_count / len(stocks)) * 100)
                try:
                    await crud.update_prediction_job(
                        db, job_id,
                        completed_stocks=processed_count,
                        progress=progress
                    )
                except Exception as prog_exc:
                    logger.warning("Failed to update prediction job progress (non-fatal): %s", prog_exc)

        
    # Flush any remaining predictions that didn't fill a full batch
    if pending_predictions:
        await crud.bulk_upsert_ensemble_predictions(db, pending_predictions)
        pending_predictions.clear()

    # Finalize job status — use a fresh session to avoid "transaction aborted" errors
    # that can occur when the long-running db session hit an earlier error/rollback.
    if job_id:
        try:
            async with async_session_factory() as final_db:
                job_check = await crud.get_prediction_job(final_db, job_id)
                is_cancelled = bool(job_check and job_check.status == "cancelled")
                await crud.update_prediction_job(
                    final_db, job_id,
                    status="cancelled" if is_cancelled else "completed",
                    batch_id=batch_id,
                    progress=100
                )
        except Exception as finalize_exc:
            logger.error("Failed to finalize prediction job %s status: %s", job_id, finalize_exc)

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


# ─────────────────────────────────────────────────────────────────────────────
# Target-Price Prediction Pipeline (TPML)
# ─────────────────────────────────────────────────────────────────────────────
# Replaces classification-based predictions with target-price signals:
#   1. LSTM regression → (μ, σ) per horizon
#   2. KNN neighbor return stats → (median, std, win_rate)
#   3. S/R zones → nearest support/resistance
#   4. Fundamental quality score
#   5. Signal synthesiser → entry/target/SL
#   6. Meta-classifier gate → PoP score

_mh_lstm_cache: dict[str, Any] = {}  # path -> model
_meta_clf: MetaClassifier | None = None


def _load_mh_lstm_cached(path: str):
    if path not in _mh_lstm_cache:
        logger.info(f"Loading MultiHorizon LSTM from disk: {path}")
        _mh_lstm_cache[path] = load_multi_horizon_model(path)
    return _mh_lstm_cache[path]


async def run_target_price_predictions(
    db: AsyncSession,
    model_dir: str | None = None,
    target_date: date | None = None,
    interval: str = "day",
    stock_ids: list[int] | None = None,
    job_id: str | None = None,
    pop_threshold: float = 0.55,
    config: SynthesizerConfig | None = None,
) -> dict[str, Any]:
    """Run the target-price prediction pipeline for all active stocks.

    This is the TPML replacement for ``run_daily_predictions()``.
    Instead of BUY/HOLD/SELL classification, it produces trade signals
    with entry_price, target_price, stoploss_price, and PoP score.

    Returns
    -------
    dict with summary stats and list of generated TradeSignal IDs.
    """
    global _meta_clf

    model_dir = model_dir or settings.MODEL_DIR
    model_path = Path(model_dir)
    target_date = target_date or date.today()
    config = config or SynthesizerConfig()

    # ── Load Multi-Horizon LSTM ───────────────────────────────────────
    q = (select(LSTMHorizonModel)
         .where(LSTMHorizonModel.status == "completed")
         .order_by(desc(LSTMHorizonModel.created_at))
         .limit(1))
    res = await db.execute(q)
    mh_lstm_db = res.scalar_one_or_none()

    if mh_lstm_db and mh_lstm_db.model_path:
        mh_lstm = _load_mh_lstm_cached(mh_lstm_db.model_path)
    else:
        # Fallback: try filesystem
        mh_path = model_path / "lstm" / "mh_lstm.pt"
        if not mh_path.exists():
            raise FileNotFoundError("No trained MultiHorizon LSTM model found")
        mh_lstm = _load_mh_lstm_cached(str(mh_path))

    # ── Load KNN model (for return stats) ─────────────────────────────
    q = (select(KNNModel)
         .where(KNNModel.status == "completed")
         .order_by(desc(KNNModel.created_at))
         .limit(1))
    res = await db.execute(q)
    knn_db = res.scalar_one_or_none()
    knn_model = None
    knn_norm_params = None
    knn_neighbor_returns = None

    if knn_db and knn_db.model_path:
        knn_model, knn_norm_params = _load_knn_cached(
            knn_db.model_path, Path(knn_db.model_path).parent
        )
        # Load neighbor returns (pnl_percent from golden patterns)
        try:
            from app.db.models import GoldenPattern
            gp_q = (select(GoldenPattern.pnl_percent)
                     .where(GoldenPattern.rl_model_id == knn_db.source_rl_model_id)
                     .order_by(GoldenPattern.id))
            gp_res = await db.execute(gp_q)
            knn_neighbor_returns = np.array(
                [r[0] if r[0] is not None else 0.0 for r in gp_res.all()],
                dtype=np.float32,
            )
        except Exception as e:
            logger.warning("Failed to load KNN neighbor returns: %s", e)

    # ── Load or initialise Meta-Classifier ────────────────────────────
    meta_path = model_path / "meta_classifier" / "meta_clf.joblib"
    if _meta_clf is None:
        _meta_clf = MetaClassifier(threshold=pop_threshold)
        if meta_path.exists():
            try:
                _meta_clf.load(meta_path)
                logger.info("MetaClassifier loaded from %s", meta_path)
            except Exception as e:
                logger.warning("Failed to load MetaClassifier: %s", e)

    # ── Get stock universe ────────────────────────────────────────────
    if stock_ids:
        res = await db.execute(select(Stock).where(Stock.id.in_(stock_ids)))
        stocks = list(res.scalars().all())
    else:
        from app.core import data_service
        stocks = await data_service.get_universe_stocks(db)

    seq_len = settings.DEFAULT_SEQ_LEN_DAILY if interval == "day" else settings.DEFAULT_SEQ_LEN_WEEKLY

    signals_created: list[dict] = []
    signals_rejected: list[dict] = []
    errors: list[dict] = []
    processed_count = 0

    for i, stock in enumerate(stocks):
        try:
            # ── Get OHLCV + indicators ────────────────────────────────
            df, feature_cols = await get_model_ready_data(
                db, stock.id, interval=interval, seq_len=seq_len,
                end_date=target_date,
            )
            if df is None or len(df) < seq_len:
                continue

            X = prepare_model_input(df, feature_cols, seq_len=seq_len)
            if len(X) == 0:
                continue

            X_last = X[-1:].copy()
            latest = df.iloc[-1]
            current_price = float(latest["close"])
            atr = float(latest.get("atr", current_price * 0.02))

            # ── 1. LSTM regression ────────────────────────────────────
            lstm_out = predict_multi_horizon(mh_lstm, X_last)
            # Use h=5 (5-day horizon) as the primary signal
            h_idx = 4  # 0-indexed
            lstm_mu = float(lstm_out["mu"][0, h_idx])
            lstm_sigma = float(lstm_out["sigma"][0, h_idx])

            # ── 2. KNN return stats ───────────────────────────────────
            knn_median = 0.0
            knn_win = 0.5
            knn_std = 0.10  # default moderate uncertainty
            if knn_model is not None and knn_neighbor_returns is not None:
                try:
                    knn_stats = predict_knn_returns(
                        knn_model, X_last, knn_neighbor_returns,
                        norm_params=knn_norm_params,
                    )
                    if knn_stats:
                        knn_median = knn_stats[0].median_return
                        knn_win = knn_stats[0].win_rate
                        knn_std = knn_stats[0].return_std
                except Exception as e:
                    logger.warning("KNN returns failed for %s: %s", stock.symbol, e)

            # ── 3. S/R zones ──────────────────────────────────────────
            sr_result = compute_sr_zones(df, current_price, atr)
            sr_feat = sr_features(sr_result, current_price, atr)

            # ── 4. Fundamental quality score ──────────────────────────
            fqs = 0.5  # neutral default
            try:
                from app.db.models import StockFundamentalZScore, StockFundamentalPIT, StockSentiment
                zs_q = (select(StockFundamentalZScore)
                        .where(StockFundamentalZScore.stock_id == stock.id)
                        .order_by(desc(StockFundamentalZScore.date))
                        .limit(1))
                zs_row = (await db.execute(zs_q)).scalar_one_or_none()

                pit_q = (select(StockFundamentalPIT)
                         .where(StockFundamentalPIT.stock_id == stock.id)
                         .order_by(desc(StockFundamentalPIT.date))
                         .limit(1))
                pit_row = (await db.execute(pit_q)).scalar_one_or_none()

                sent_q = (select(StockSentiment)
                          .where(StockSentiment.stock_id == stock.id)
                          .order_by(desc(StockSentiment.date))
                          .limit(1))
                sent_row = (await db.execute(sent_q)).scalar_one_or_none()

                fqs_result = compute_fundamental_score(
                    pe_zscore_3y=getattr(zs_row, "pe_zscore_3y", None),
                    pe_zscore_sector=getattr(zs_row, "pe_zscore_sector", None),
                    roe_norm=getattr(zs_row, "roe_norm", None),
                    debt_equity_norm=getattr(zs_row, "debt_equity_norm", None),
                    pe_ratio=getattr(pit_row, "pe_ratio", None),
                    forward_pe=getattr(pit_row, "forward_pe", None),
                    dividend_yield=getattr(pit_row, "dividend_yield", None),
                    avg_finbert_score=getattr(sent_row, "avg_finbert_score", None),
                )
                fqs = fqs_result.fqs
            except Exception as e:
                logger.debug("Fundamental score unavailable for %s: %s", stock.symbol, e)

            # ── 5. Regime ─────────────────────────────────────────────
            try:
                df_regime = classify_regimes(df.iloc[-1:])
                regime_id = int(df_regime["regime_id"].iloc[0])
            except Exception:
                regime_id = 0

            # ── 6. Signal synthesis ───────────────────────────────────
            # Get bid/ask if available
            bid, ask = None, None
            try:
                from app.core.zerodha import _get_bid_ask
                bid, ask = _get_bid_ask(stock.symbol, stock.exchange)
            except Exception:
                pass

            realized_vol = float(latest.get("realized_vol", 0.0))

            candidate = synthesize_signal(
                stock_id=stock.id,
                current_price=current_price,
                atr=atr,
                lstm_mu=lstm_mu,
                lstm_sigma=lstm_sigma,
                knn_median_return=knn_median,
                knn_win_rate=knn_win,
                knn_return_std=knn_std,
                sr_result=sr_result,
                fqs_score=fqs,
                regime_id=regime_id,
                bid=bid,
                ask=ask,
                realized_vol=realized_vol,
                config=config,
            )

            # ── 7. Meta-classifier gate ──────────────────────────────
            pop_score = 0.5
            if _meta_clf is not None:
                feat_vec = _meta_clf.build_feature_vector(
                    confluence_score=candidate.confluence_score,
                    fqs_score=candidate.fqs_score,
                    execution_cost_pct=candidate.execution_cost_pct,
                    initial_rr_ratio=candidate.initial_rr_ratio,
                    net_expected_return_pct=candidate.net_expected_return_pct,
                    lstm_mu=candidate.lstm_mu,
                    lstm_sigma=candidate.lstm_sigma,
                    knn_median_return=candidate.knn_median_return,
                    knn_win_rate=candidate.knn_win_rate,
                    regime_id=regime_id,
                    sr_features=sr_feat,
                )
                meta_results = _meta_clf.predict(feat_vec.reshape(1, -1))
                pop_score = meta_results[0].pop_score

            # ── 8. Final decision ─────────────────────────────────────
            if candidate.is_buy and _meta_clf.should_trade(pop_score):
                signal = TradeSignal(
                    stock_id=stock.id,
                    signal_date=target_date,
                    entry_price=candidate.entry_price,
                    target_price=candidate.target_price,
                    stoploss_price=candidate.stoploss_price,
                    current_stoploss=candidate.stoploss_price,
                    pop_score=pop_score,
                    fqs_score=fqs,
                    confluence_score=candidate.confluence_score,
                    execution_cost_pct=candidate.execution_cost_pct,
                    initial_rr_ratio=candidate.initial_rr_ratio,
                    current_rr_ratio=candidate.initial_rr_ratio,
                    lstm_mu=candidate.lstm_mu,
                    lstm_sigma=candidate.lstm_sigma,
                    knn_median_return=candidate.knn_median_return,
                    knn_win_rate=candidate.knn_win_rate,
                    regime_id=regime_id,
                    status=SignalStatus.pending,
                )
                db.add(signal)
                await db.flush()
                signals_created.append({
                    "signal_id": signal.id,
                    "stock_id": stock.id,
                    "symbol": stock.symbol,
                    "entry": candidate.entry_price,
                    "target": candidate.target_price,
                    "stoploss": candidate.stoploss_price,
                    "rr_ratio": candidate.initial_rr_ratio,
                    "pop_score": pop_score,
                    "fqs": fqs,
                })
            else:
                reason = candidate.reject_reason or f"PoP {pop_score:.3f} < {pop_threshold}"
                signals_rejected.append({
                    "stock_id": stock.id,
                    "symbol": stock.symbol,
                    "reason": reason,
                })

            del df
            gc.collect()

        except Exception as e:
            logger.error("TPML failed for %s: %s", stock.symbol, e, exc_info=True)
            errors.append({"stock_id": stock.id, "symbol": stock.symbol, "error": str(e)})
        finally:
            processed_count += 1
            if job_id and (processed_count <= 10 or processed_count % 5 == 0
                           or processed_count == len(stocks)):
                try:
                    await crud.update_prediction_job(
                        db, job_id,
                        completed_stocks=processed_count,
                        progress=int((processed_count / len(stocks)) * 100),
                    )
                except Exception:
                    pass

    await db.commit()

    if job_id:
        try:
            async with async_session_factory() as final_db:
                await crud.update_prediction_job(
                    final_db, job_id, status="completed", progress=100,
                )
        except Exception as e:
            logger.error("Failed to finalize TPML job %s: %s", job_id, e)

    return {
        "date": str(target_date),
        "total_stocks": len(stocks),
        "signals_created": len(signals_created),
        "signals_rejected": len(signals_rejected),
        "errors": len(errors),
        "signals": signals_created,
        "rejected": signals_rejected,
        "error_details": errors,
    }
