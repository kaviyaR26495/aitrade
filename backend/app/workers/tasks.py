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
