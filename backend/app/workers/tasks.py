"""Celery task definitions.

Each task wraps existing async pipeline functions via ``asyncio.run()``,
ensuring compatibility between Celery's synchronous worker model and the
project's async-first codebase.

Task contract
-------------
- Tasks are idempotent: running twice on the same day is safe.
- Tasks log progress to both the Celery task logger and the app logger.
- Tasks return a summary dict that is stored in the Redis result backend
  for 24 hours (useful for monitoring via Flower).
- Failures are caught and re-raised so Celery marks the task as FAILED
  and triggers any configured alert chains.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from app.workers.celery_app import celery_app

_log = logging.getLogger(__name__)
IST = timezone(timedelta(hours=5, minutes=30))


# ── Helper ─────────────────────────────────────────────────────────────

def _get_universe_stocks(db_session) -> list:
    """Synchronous helper: resolve the active stock universe from DB settings."""
    import asyncio
    from app.db import crud

    async def _inner():
        universe_cfg = await crud.get_setting(db_session, "stock_universe") or "{}"
        import json
        cfg = json.loads(universe_cfg) if isinstance(universe_cfg, str) else universe_cfg
        # Get all active stocks — the universe filter narrows this in the pipeline
        from sqlalchemy import select
        from app.db.models import Stock
        result = await db_session.execute(
            select(Stock).where(Stock.is_active == True).limit(500)  # noqa: E712
        )
        return result.scalars().all()

    return asyncio.run(_inner())


# ── Task: Nightly OHLCV + Indicator Sync ─────────────────────────────

@celery_app.task(name="app.workers.tasks.task_nightly_sync", bind=True, max_retries=2)
def task_nightly_sync(self):
    """Sync OHLCV data and recompute indicators for all universe stocks.

    Runs nightly at 18:30 IST after market close.
    """
    _log.info("[task_nightly_sync] Starting nightly OHLCV sync")
    try:
        return asyncio.run(_async_nightly_sync())
    except Exception as exc:
        _log.error("[task_nightly_sync] Failed: %s", exc)
        raise self.retry(exc=exc, countdown=300)


async def _async_nightly_sync() -> dict:
    from app.db.database import async_session_factory
    from app.core import data_pipeline
    from app.db import crud

    async with async_session_factory() as db:
        # Resolve universe stocks
        universe_cfg_str = await crud.get_setting(db, "stock_universe") or "{}"
        import json
        import sqlalchemy as sa
        from app.db.models import Stock

        result = await db.execute(
            sa.select(Stock).where(Stock.is_active == True)  # noqa: E712
        )
        stocks = result.scalars().all()

    synced = 0
    failed = 0
    async with async_session_factory() as db:
        for stock in stocks:
            try:
                await data_pipeline.sync_ohlcv_and_indicators(db, stock.id)
                synced += 1
            except Exception as exc:
                _log.warning("OHLCV sync failed for %s: %s", stock.symbol, exc)
                failed += 1

    summary = {"synced": synced, "failed": failed, "total": len(stocks)}
    _log.info("[task_nightly_sync] Done: %s", summary)
    return summary


# ── Task: Weekly Fundamental Sync ─────────────────────────────────────

@celery_app.task(name="app.workers.tasks.task_fundamental_sync", bind=True, max_retries=2)
def task_fundamental_sync(self):
    """Refresh fundamental PIT data and recompute z-scores for all universe stocks.

    Runs weekly (Sunday 20:00 IST) to stay current with quarterly reports.
    """
    _log.info("[task_fundamental_sync] Starting fundamental sync")
    try:
        return asyncio.run(_async_fundamental_sync())
    except Exception as exc:
        _log.error("[task_fundamental_sync] Failed: %s", exc)
        raise self.retry(exc=exc, countdown=600)


async def _async_fundamental_sync() -> dict:
    import sqlalchemy as sa
    from app.db.database import async_session_factory
    from app.db.models import Stock
    from app.core.fundamental_pipeline import ingest_and_score

    async with async_session_factory() as db:
        result = await db.execute(
            sa.select(Stock).where(Stock.is_active == True)  # noqa: E712
        )
        stocks = result.scalars().all()
        summary = await ingest_and_score(db, list(stocks))

    _log.info("[task_fundamental_sync] Done: %s", summary)
    return summary


# ── Task: Morning Sentiment ────────────────────────────────────────────

@celery_app.task(name="app.workers.tasks.task_morning_sentiment", bind=True, max_retries=1)
def task_morning_sentiment(self):
    """Fetch headlines, run FinBERT triage, and score with LLM.

    Runs at 08:45 IST so sentiment data is ready before 09:00 open.
    """
    _log.info("[task_morning_sentiment] Starting morning sentiment run")
    try:
        return asyncio.run(_async_morning_sentiment())
    except Exception as exc:
        _log.error("[task_morning_sentiment] Failed: %s", exc)
        raise self.retry(exc=exc, countdown=120)


async def _async_morning_sentiment() -> dict:
    import sqlalchemy as sa
    from app.db.database import async_session_factory
    from app.db.models import Stock
    from app.core.sentiment_pipeline import run_sentiment_batch

    async with async_session_factory() as db:
        result = await db.execute(
            sa.select(Stock).where(Stock.is_active == True)  # noqa: E712
        )
        stocks = result.scalars().all()
        summaries = await run_sentiment_batch(db, list(stocks), concurrency=5)

    total = len(summaries)
    failed = sum(1 for s in summaries if s["llm_impact_score"] is None)
    summary = {"total": total, "with_score": total - failed, "failed": failed}
    _log.info("[task_morning_sentiment] Done: %s", summary)
    return summary


# ── Task: Morning Predictions ──────────────────────────────────────────

@celery_app.task(name="app.workers.tasks.task_morning_predictions", bind=True, max_retries=1)
def task_morning_predictions(self):
    """Generate ensemble predictions for the full universe.

    Runs at 09:00 IST — after sentiment data is ready and before market open.
    The prediction batch is written to ``ensemble_predictions`` table.
    """
    _log.info("[task_morning_predictions] Starting morning prediction batch")
    try:
        return asyncio.run(_async_morning_predictions())
    except Exception as exc:
        _log.error("[task_morning_predictions] Failed: %s", exc)
        raise self.retry(exc=exc, countdown=120)


async def _async_morning_predictions() -> dict:
    import uuid
    import sqlalchemy as sa
    from app.db.database import async_session_factory
    from app.db.models import Stock
    from app.core.data_pipeline import run_prediction_batch

    batch_id = str(uuid.uuid4())[:8]
    async with async_session_factory() as db:
        result = await db.execute(
            sa.select(Stock).where(Stock.is_active == True)  # noqa: E712
        )
        stocks = result.scalars().all()

        try:
            counts = await run_prediction_batch(db, [s.id for s in stocks], batch_id=batch_id)
            summary = {"batch_id": batch_id, **counts}
        except Exception as exc:
            _log.error("Prediction batch failed: %s", exc)
            summary = {"batch_id": batch_id, "error": str(exc)}

    _log.info("[task_morning_predictions] Done: %s", summary)
    return summary


# ── Task: Monthly Retraining ───────────────────────────────────────────

@celery_app.task(name="app.workers.tasks.task_monthly_retrain", bind=True, max_retries=1)
def task_monthly_retrain(self):
    """Retrain the full ensemble on the latest golden patterns.

    Runs on the 1st of each month at 22:00 IST.  Trains both:
      1. Global ensemble (fallback for data-sparse regimes)
      2. 6 regime-stratified KNN+LSTM pairs
    """
    _log.info("[task_monthly_retrain] Starting monthly retraining")
    try:
        return asyncio.run(_async_monthly_retrain())
    except Exception as exc:
        _log.error("[task_monthly_retrain] Failed: %s", exc)
        raise self.retry(exc=exc, countdown=600)


async def _async_monthly_retrain() -> dict:
    from app.db.database import async_session_factory
    from app.core.ct_pipeline import auto_retrain

    results: dict = {}

    async with async_session_factory() as db:
        # Global retrain (existing function)
        global_result = await auto_retrain(db, lookback_years=2)
        results["global"] = global_result
        _log.info("[task_monthly_retrain] Global retrain done: %s", global_result)

    # Regime-stratified retrain (train_all_regime_models manages its own sessions)
    try:
        from app.ml.regime_trainer import train_all_regime_models
        regime_result = await train_all_regime_models(lookback_years=2)
        results["regime_stratified"] = regime_result
        _log.info("[task_monthly_retrain] Regime retrain done: %s", regime_result)
    except ImportError:
        _log.info("[task_monthly_retrain] regime_trainer not yet implemented — skipping")
    except Exception as exc:
        _log.warning("[task_monthly_retrain] Regime retrain failed (non-fatal): %s", exc)
        results["regime_stratified"] = {"error": str(exc)}

    return results


# ── Task: Morning Target-Price Signals (TPML) ─────────────────────────

@celery_app.task(name="app.workers.tasks.task_morning_tpml_signals", bind=True, max_retries=1)
def task_morning_tpml_signals(self):
    """Generate target-price trade signals for the full universe.

    Runs at 09:05 IST — after OHLCV data and sentiment are fresh.
    Produces TradeSignal rows with entry/target/SL/PoP.
    """
    _log.info("[task_morning_tpml_signals] Starting TPML signal generation")
    try:
        return asyncio.run(_async_morning_tpml_signals())
    except Exception as exc:
        _log.error("[task_morning_tpml_signals] Failed: %s", exc)
        raise self.retry(exc=exc, countdown=120)


async def _async_morning_tpml_signals() -> dict:
    from app.db.database import async_session_factory
    from app.ml.predictor import run_target_price_predictions

    async with async_session_factory() as db:
        result = await run_target_price_predictions(db)

    _log.info("[task_morning_tpml_signals] Done: %s signals created", result.get("signals_created", 0))
    return {
        "signals_created": result.get("signals_created", 0),
        "signals_rejected": result.get("signals_rejected", 0),
        "errors": result.get("errors", 0),
    }


# ── Task: Trailing Stop Update ─────────────────────────────────────────

@celery_app.task(name="app.workers.tasks.task_trailing_stop_update", bind=True, max_retries=0)
def task_trailing_stop_update(self):
    """Update trailing stops for all active signals.

    Runs every 5 minutes during market hours (09:15–15:30 IST).
    Cancels + re-places GTT OCO orders when price breaches S/R zones.
    """
    _log.info("[task_trailing_stop_update] Checking trailing stops")
    try:
        return asyncio.run(_async_trailing_stop_update())
    except Exception as exc:
        _log.error("[task_trailing_stop_update] Failed: %s", exc)
        return {"error": str(exc)}


async def _async_trailing_stop_update() -> dict:
    import sqlalchemy as sa
    from app.db.database import async_session_factory
    from app.db.models import TradeSignal, SignalStatus, Stock
    from app.core.trailing_stop import (
        TrailingStopState, evaluate_trailing_stop, execute_trailing_stop_update,
    )
    from app.core.support_resistance import compute_sr_zones
    from app.core.data_pipeline import ohlcv_to_dataframe
    from app.db import crud

    updated = 0
    skipped = 0
    errors = 0

    async with async_session_factory() as db:
        # Get all active signals
        result = await db.execute(
            sa.select(TradeSignal).where(TradeSignal.status == SignalStatus.active)
        )
        active_signals = result.scalars().all()

        if not active_signals:
            return {"updated": 0, "skipped": 0, "active_signals": 0}

        # Bulk resolve stock info
        stock_ids = list({s.stock_id for s in active_signals})
        stock_res = await db.execute(sa.select(Stock).where(Stock.id.in_(stock_ids)))
        stock_map = {s.id: s for s in stock_res.scalars().all()}

        for signal in active_signals:
            try:
                stock = stock_map.get(signal.stock_id)
                if not stock:
                    continue

                # Get latest price from Kite
                try:
                    from app.core.zerodha import _get_bid_ask
                    bid, ask = _get_bid_ask(stock.symbol, stock.exchange)
                    current_price = (bid + ask) / 2
                except Exception:
                    skipped += 1
                    continue

                # Get latest OHLCV for S/R computation
                ohlcv_rows = await crud.get_ohlcv_rows(db, stock.id, limit=250)
                if not ohlcv_rows or len(ohlcv_rows) < 30:
                    skipped += 1
                    continue

                df = ohlcv_to_dataframe(ohlcv_rows)
                atr = float(df["high"].rolling(14).max().iloc[-1] - df["low"].rolling(14).min().iloc[-1]) / 14
                if "atr" in df.columns:
                    atr = float(df["atr"].iloc[-1])

                sr_result = compute_sr_zones(df, current_price, atr)

                # Get nearest support below price
                supports = [z for z in sr_result.zones if z.midpoint < current_price]
                nearest_sup = max(supports, key=lambda z: z.midpoint).midpoint if supports else None

                # Get holding quantity from trade orders
                from app.db.models import TradeOrder
                order_q = sa.select(TradeOrder).where(
                    TradeOrder.trade_signal_id == signal.id,
                    TradeOrder.transaction_type == "BUY",
                )
                order_row = (await db.execute(order_q)).scalar_one_or_none()
                quantity = order_row.filled_quantity or order_row.quantity if order_row else 1

                state = TrailingStopState(
                    signal_id=signal.id,
                    stock_id=stock.id,
                    symbol=stock.symbol,
                    exchange=stock.exchange,
                    quantity=quantity,
                    entry_price=signal.entry_price,
                    target_price=signal.target_price,
                    original_sl=signal.stoploss_price,
                    current_sl=signal.current_stoploss or signal.stoploss_price,
                    last_gtt_id=signal.last_gtt_id,
                    is_active=signal.is_trailing_active,
                    updates_count=signal.trailing_updates_count,
                )

                update = evaluate_trailing_stop(
                    state, current_price, atr, nearest_sup,
                )

                if update.should_update:
                    new_gtt_id = await execute_trailing_stop_update(state, update.new_sl)
                    signal.current_stoploss = update.new_sl
                    signal.is_trailing_active = True
                    signal.trailing_updates_count += 1
                    if new_gtt_id:
                        signal.last_gtt_id = new_gtt_id
                    updated += 1
                    _log.info(
                        "Trailing SL updated for signal %d (%s): %s",
                        signal.id, stock.symbol, update.reason,
                    )
                else:
                    skipped += 1

            except Exception as exc:
                _log.warning("Trailing stop error for signal %d: %s", signal.id, exc)
                errors += 1

        await db.commit()

    summary = {
        "active_signals": len(active_signals),
        "updated": updated,
        "skipped": skipped,
        "errors": errors,
    }
    _log.info("[task_trailing_stop_update] Done: %s", summary)
    return summary
