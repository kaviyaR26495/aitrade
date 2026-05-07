"""Backfill predictions API.

POST /api/backfill/start       — start backfill job in background
GET  /api/backfill/status      — get current job status + progress
POST /api/backfill/stop        — stop running backfill
GET  /api/backfill/coverage    — return (min_date, max_date, total_days) per ensemble config
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import func, select, distinct
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.db.database import async_session_factory
from app.db.models import EnsembleConfig, EnsemblePrediction, StockOHLCV

router = APIRouter()
logger = logging.getLogger(__name__)


# ── In-process job state ─────────────────────────────────────────────────────

@dataclass
class BackfillJobState:
    status: str = "idle"           # idle | running | stopping | completed | failed
    start_date: str = ""
    end_date: str = ""
    ensemble_config_id: int | None = None
    override_existing: bool = False
    total: int = 0
    done: int = 0
    skipped: int = 0
    errors: int = 0
    current_date: str = ""
    log: list[str] = field(default_factory=list)
    started_at: float = 0.0
    finished_at: float = 0.0
    stop_flag: bool = False

    def to_dict(self) -> dict[str, Any]:
        elapsed = (self.finished_at or time.monotonic()) - self.started_at if self.started_at else 0
        per_date = elapsed / max(self.done, 1)
        remaining = (self.total - self.done) * per_date
        return {
            "status": self.status,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "ensemble_config_id": self.ensemble_config_id,
            "override_existing": self.override_existing,
            "total": self.total,
            "done": self.done,
            "skipped": self.skipped,
            "errors": self.errors,
            "current_date": self.current_date,
            "progress_pct": round(100 * self.done / max(self.total, 1), 1),
            "elapsed_seconds": round(elapsed),
            "eta_seconds": round(remaining) if self.status == "running" and self.done > 0 else None,
            "log": self.log[-100:],  # last 100 lines
        }


_job = BackfillJobState()
_job_lock = asyncio.Lock()


# ── Background worker ────────────────────────────────────────────────────────

async def _run_backfill(
    start_d: date,
    end_d: date,
    ensemble_config_id: int,
    override_existing: bool,
) -> None:
    global _job
    from app.ml.predictor import run_daily_predictions

    # Fetch trading dates
    async with async_session_factory() as db:
        res = await db.execute(
            select(distinct(StockOHLCV.date))
            .where(StockOHLCV.date >= start_d, StockOHLCV.date <= end_d)
            .order_by(StockOHLCV.date)
        )
        all_dates: list[date] = [row[0] for row in res.all()]

    if not all_dates:
        async with _job_lock:
            _job.status = "completed"
            _job.finished_at = time.monotonic()
            _job.log.append("No trading dates found in range — nothing to do.")
        return

    # Optionally skip dates already covered
    if not override_existing:
        async with async_session_factory() as db:
            res2 = await db.execute(
                select(distinct(EnsemblePrediction.date))
                .where(
                    EnsemblePrediction.ensemble_config_id == ensemble_config_id,
                    EnsemblePrediction.date.in_(all_dates),
                )
            )
            already_done: set[date] = {row[0] for row in res2.all()}
        dates_to_process = [d for d in all_dates if d not in already_done]
        skipped_count = len(already_done)
    else:
        dates_to_process = all_dates
        skipped_count = 0

    async with _job_lock:
        _job.total = len(dates_to_process)
        _job.skipped = skipped_count
        _job.log.append(
            f"Found {len(all_dates)} trading days | "
            f"Skipping {skipped_count} already done | "
            f"Processing {len(dates_to_process)}"
        )

    for idx, target_date in enumerate(dates_to_process, 1):
        async with _job_lock:
            if _job.stop_flag:
                _job.status = "stopped"
                _job.finished_at = time.monotonic()
                _job.log.append(f"Stopped by user after {_job.done} dates.")
                return
            _job.current_date = str(target_date)

        t0 = time.monotonic()
        try:
            async with async_session_factory() as db:
                result = await run_daily_predictions(
                    db=db,
                    target_date=target_date,
                    interval="day",
                    ensemble_config_id=ensemble_config_id,
                    sector_guard=False,
                    weekly_confluence_filter=False,
                )
            saved = result.get("predictions_made", 0)
            duration = time.monotonic() - t0
            msg = f"[{idx}/{len(dates_to_process)}] {target_date}  saved={saved}  {duration:.0f}s"
            async with _job_lock:
                _job.done += 1
                _job.log.append(f"✓ {msg}")
        except Exception as exc:
            duration = time.monotonic() - t0
            msg = f"[{idx}/{len(dates_to_process)}] {target_date}  FAILED: {exc}  {duration:.0f}s"
            logger.error("Backfill date %s failed: %s", target_date, exc, exc_info=True)
            async with _job_lock:
                _job.errors += 1
                _job.done += 1
                _job.log.append(f"✗ {msg}")

    async with _job_lock:
        _job.status = "completed"
        _job.finished_at = time.monotonic()
        _job.log.append(
            f"Backfill complete — {len(dates_to_process)} processed, "
            f"{_job.errors} errors, {skipped_count} skipped."
        )


# ── API models ────────────────────────────────────────────────────────────────

class BackfillStartRequest(BaseModel):
    start_date: str  # YYYY-MM-DD
    end_date: str    # YYYY-MM-DD
    ensemble_config_id: int | None = None
    override_existing: bool = False


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/start")
async def start_backfill(
    req: BackfillStartRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    global _job
    async with _job_lock:
        if _job.status == "running":
            raise HTTPException(status_code=409, detail="A backfill job is already running.")

    # Parse dates
    try:
        start_d = date.fromisoformat(req.start_date)
        end_d = date.fromisoformat(req.end_date)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Invalid date: {e}")

    if start_d > end_d:
        raise HTTPException(status_code=422, detail="start_date must be <= end_date")

    # Resolve ensemble config
    cfg_id = req.ensemble_config_id
    if cfg_id is None:
        res = await db.execute(
            select(EnsembleConfig.id).order_by(EnsembleConfig.created_at.desc()).limit(1)
        )
        cfg_id = res.scalar_one_or_none()
        if cfg_id is None:
            raise HTTPException(status_code=400, detail="No ensemble config found. Train models first.")

    async with _job_lock:
        _job.__init__()  # reset to defaults
        _job.status = "running"
        _job.start_date = str(start_d)
        _job.end_date = str(end_d)
        _job.ensemble_config_id = cfg_id
        _job.override_existing = req.override_existing
        _job.started_at = time.monotonic()
        _job.log.append(
            f"Starting backfill {start_d} → {end_d}  config={cfg_id}  "
            f"override={req.override_existing}"
        )

    background_tasks.add_task(
        _run_backfill, start_d, end_d, cfg_id, req.override_existing
    )

    return {"message": "Backfill started", "ensemble_config_id": cfg_id}


@router.get("/status")
async def get_status():
    async with _job_lock:
        return _job.to_dict()


@router.post("/stop")
async def stop_backfill():
    global _job
    async with _job_lock:
        if _job.status != "running":
            raise HTTPException(status_code=409, detail="No backfill is currently running.")
        _job.stop_flag = True
        _job.status = "stopping"
    return {"message": "Stop signal sent — current date will finish then backfill will halt."}


@router.get("/coverage")
async def get_coverage(db: AsyncSession = Depends(get_db)):
    """Return prediction coverage per ensemble config."""
    res = await db.execute(
        select(
            EnsemblePrediction.ensemble_config_id,
            func.min(EnsemblePrediction.date),
            func.max(EnsemblePrediction.date),
            func.count(distinct(EnsemblePrediction.date)),
            func.count(EnsemblePrediction.id),
        ).group_by(EnsemblePrediction.ensemble_config_id)
        .order_by(EnsemblePrediction.ensemble_config_id)
    )
    rows = []
    for cfg_id, min_d, max_d, days, total in res.all():
        rows.append({
            "ensemble_config_id": cfg_id,
            "min_date": str(min_d) if min_d else None,
            "max_date": str(max_d) if max_d else None,
            "distinct_days": days,
            "total_rows": total,
        })

    # Also get list of ensemble configs
    cfg_res = await db.execute(
        select(EnsembleConfig.id, EnsembleConfig.name, EnsembleConfig.created_at)
        .order_by(EnsembleConfig.created_at.desc())
    )
    configs = [
        {"id": r.id, "name": r.name, "created_at": str(r.created_at)}
        for r in cfg_res.all()
    ]

    return {"coverage": rows, "configs": configs}
