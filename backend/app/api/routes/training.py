"""Continuous Training (CT) routes.

POST /api/training/auto-retrain  — trigger rolling window retraining
GET  /api/training/retrain-status — check staleness and last retrain date
"""
from __future__ import annotations

import logging
from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, Depends
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.db import crud
from app.db.database import async_session_factory
from app.db.models import EnsembleConfig

router = APIRouter()
logger = logging.getLogger(__name__)

_STALE_AFTER_DAYS = 30  # recommend retrain if last run > 30 days ago


class AutoRetrainRequest(BaseModel):
    lookback_years: int = 2


@router.post("/auto-retrain")
async def trigger_auto_retrain(
    req: AutoRetrainRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """Launch KNN + LSTM auto-retrain as a background task.

    New models are staged (not activated) — activate via the EnsembleConfig
    UI after reviewing accuracy metrics.
    """
    from app.core.ct_pipeline import auto_retrain

    async def _run() -> None:
        async with async_session_factory() as bdb:
            try:
                result = await auto_retrain(bdb, lookback_years=req.lookback_years)
                logger.info("Auto-retrain finished: %s", result)
            except Exception as exc:
                logger.error("Auto-retrain failed: %s", exc, exc_info=True)

    background_tasks.add_task(_run)
    return {
        "message": "Auto-retrain started in background",
        "lookback_years": req.lookback_years,
    }


@router.get("/retrain-status")
async def get_retrain_status(db: AsyncSession = Depends(get_db)):
    """Return when the ensemble was last refreshed and whether a new retrain is recommended.

    Response:
      last_retrain_at   ISO-8601 string or null
      days_since_retrain  integer or null
      needs_retrain       bool (true if > 30 days or never run)
    """
    last_str = await crud.get_setting(db, "last_auto_retrain_at")
    last_dt: datetime | None = None

    if last_str:
        try:
            last_dt = datetime.fromisoformat(last_str)
        except ValueError:
            last_dt = None

    latest_ensemble_result = await db.execute(
        select(EnsembleConfig.created_at).order_by(EnsembleConfig.created_at.desc()).limit(1)
    )
    latest_ensemble_dt = latest_ensemble_result.scalar_one_or_none()

    if latest_ensemble_dt and (last_dt is None or latest_ensemble_dt > last_dt):
        last_dt = latest_ensemble_dt
        last_str = latest_ensemble_dt.isoformat()

    if last_dt is not None:
        days_since = max(0, (datetime.now().date() - last_dt.date()).days)
        return {
            "last_retrain_at": last_str,
            "days_since_retrain": days_since,
            "needs_retrain": days_since >= _STALE_AFTER_DAYS,
        }

    return {
        "last_retrain_at": None,
        "days_since_retrain": None,
        "needs_retrain": True,
    }
