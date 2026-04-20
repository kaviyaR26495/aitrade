"""One-Click Universal Training Pipeline route.

POST /pipeline/start              — kicks off a multi-stage background pipeline
GET  /pipeline/status/{job_id}    — poll for progress
DELETE /pipeline/{job_id}         — terminate + optionally purge all created data/files

Pipeline stages:
  0  data_sync        — Sync OHLCV + compute indicators for every symbol
  1  cql_pretrain     — Offline CQL pre-training (skipped if d3rlpy not installed)
  2  bc_warmup        — Behavioral Cloning warmup (skipped if d3rlpy not installed)
  3  ppo_finetune     — Online AttentionPPO fine-tuning
  4  ensemble_distill — KNN + LSTM distillation from RL patterns
  5  backtest         — Backtest models
  6  ready            — All models ready for live trading
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from app.config import settings
from app.db import crud
from app.db.database import async_session_factory

logger = logging.getLogger(__name__)
router = APIRouter()

# ── Cancellation check (DB-backed) ───────────────────────────────────────────
# Previously used an in-process set which broke under multi-worker deployments.
# Now queries the DB so any worker can reliably detect a cancel/purge.
async def _is_cancelled(job_id: str) -> bool:
    async with async_session_factory() as db:
        job = await crud.get_pipeline_job(db, job_id)
        return job is not None and job.status in ("cancelled", "purged")


# ── Thread pool shared with models router (limit concurrent heavy jobs)
_PIPELINE_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pipeline")

# ── Stage definitions (matches AutoPilotPipeline frontend) ────────────────────
_STAGE_NAMES = [
    "data_sync",
    "cql_pretrain",
    "bc_warmup",
    "ppo_finetune",
    "ensemble_distill",
    "backtest",
    "ready",
]


def _make_stages() -> list[dict]:
    return [
        {"stage": i, "name": name, "status": "pending", "progress": 0, "message": ""}
        for i, name in enumerate(_STAGE_NAMES)
    ]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _update_stage(job_id: str, stage_idx: int, *, status: str, progress: int = 0, message: str = "") -> bool:
    """Update a specific stage and the job's overall status/stage in the DB.

    Returns True when the pipeline has been cancelled/purged/failed (circuit
    breaker). Callers should abort when True is returned.
    """
    async with async_session_factory() as db:
        job_record = await crud.get_pipeline_job(db, job_id)
        if not job_record:
            return True  # job gone — abort

        # CIRCUIT BREAKER: never overwrite a terminal state; signalling
        # is_cancelled=True lets the caller stop the running trainer thread.
        if job_record.status in ("cancelled", "purged", "failed"):
            return True

        stages = list(job_record.stages or [])
        if stage_idx < len(stages):
            stages[stage_idx]["status"] = status
            stages[stage_idx]["progress"] = progress
            stages[stage_idx]["message"] = message

        await crud.update_pipeline_job(
            db,
            job_id,
            status="running" if status in ["running", "pending"] else None,
            current_stage=stages[stage_idx]["name"] if stage_idx < len(stages) else str(stage_idx),
        )

        from sqlalchemy import update as sqla_update
        from app.db.models import PipelineJob
        await db.execute(
            sqla_update(PipelineJob)
            .where(PipelineJob.id == job_id)
            .values(stages=stages, updated_at=datetime.now(timezone.utc))
        )
        await db.commit()
        return False


# ── Schemas ───────────────────────────────────────────────────────────────────

class PipelineStartRequest(BaseModel):
    symbols: list[str]
    skip_sync: bool = False
    force_sync: bool = False
    use_regime_pooling: bool = True
    resume_job_id: str | None = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/start")
async def start_pipeline(req: PipelineStartRequest, background_tasks: BackgroundTasks):
    if not req.symbols:
        raise HTTPException(status_code=422, detail="symbols must not be empty")

    # Check for existing pending/failed job for the same symbols
    async with async_session_factory() as db:
        from sqlalchemy import select as sqla_select, and_
        from app.db.models import PipelineJob
        
        # Look for a failed or manual-retry job with the same symbols
        q = sqla_select(PipelineJob).where(
            and_(
                PipelineJob.status.in_(["failed", "cancelled"]),
                # Symbols match exactly
            )
        ).order_by(PipelineJob.created_at.desc()).limit(1)
        
        # For simplicity in this fix, we'll check if the user is asking to resume
        # by looking for the last job if it failed.
        result = await db.execute(q)
        existing_job = result.scalar_one_or_none()
        
        # If the last job failed and has matching symbols, we can theoretically resume.
        # However, to avoid complexity with UUIDs and frontend expectations, 
        # we will create a NEW job but skip stages that were already completed.
    
    job_id = str(uuid.uuid4())
    stages = _make_stages()
        
    async with async_session_factory() as db:
        if req.resume_job_id:
            from sqlalchemy import select as sqla_select
            from app.db.models import PipelineJob
            
            job = await crud.get_pipeline_job(db, req.resume_job_id)
            if job and job.symbols == req.symbols:
                old_stages = job.stages or []
                for stage in stages:
                    # Look up matching stage in old job
                    old_stage = next((s for s in old_stages if s.get("name") == stage["name"]), None)
                    if old_stage and old_stage.get("status") == "completed":
                        stage["status"] = "completed"
                # Set previous IDs to skip the work later
        
        await crud.create_pipeline_job(
            db,
            job_id=job_id,
            symbols=req.symbols,
            stages=stages,
            status="queued"
        )

    background_tasks.add_task(_run_pipeline, job_id, req.symbols, req.skip_sync, req.use_regime_pooling, req.force_sync, req.resume_job_id)
    return {"job_id": job_id}


@router.get("/status/{job_id}")
async def get_pipeline_status(job_id: str):
    async with async_session_factory() as db:
        job = await crud.get_pipeline_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Pipeline job not found")
    
    # current_stage may be stored as a numeric string ("3") or a stage name
    # ("ppo_finetune") depending on which path wrote it — handle both.
    def _resolve_stage_index(val: str | None) -> int:
        if val is None:
            return 0
        try:
            return int(val)
        except ValueError:
            try:
                return _STAGE_NAMES.index(val)
            except ValueError:
                return 0

    return {
        "job_id": job.id,
        "symbols": job.symbols,
        "status": job.status,
        "current_stage": _resolve_stage_index(job.current_stage),
        "stages": job.stages,
        "created_at": job.created_at.isoformat(),
        "updated_at": job.updated_at.isoformat(),
        "error": job.error,
    }


@router.get("/latest")
async def get_latest_pipeline():
    """Return the most recent *meaningful* pipeline job.

    Priority:
      1. Any job currently running or queued (most recent first).
      2. The most recent failed/completed job that has at least one completed stage
         (i.e., something actually ran — not an immediately-failed validation error).

    Returns 404 when no meaningful jobs exist.
    """
    from sqlalchemy import select as sqla_select, desc
    from app.db.models import PipelineJob

    def _resolve_stage_index(val: str | None) -> int:
        if val is None:
            return 0
        try:
            return int(val)
        except ValueError:
            try:
                return _STAGE_NAMES.index(val)
            except ValueError:
                return 0

    def _has_progress(job) -> bool:
        """True if at least one stage completed (not an immediately-failed job)."""
        return any(
            s.get("status") == "completed"
            for s in (job.stages or [])
        )

    async with async_session_factory() as db:
        # First priority: running / queued jobs
        res = await db.execute(
            sqla_select(PipelineJob)
            .where(PipelineJob.status.in_(["running", "queued"]))
            .order_by(desc(PipelineJob.created_at))
            .limit(1)
        )
        job = res.scalar_one_or_none()

        if job is None:
            # Second priority: most recent job with actual progress
            res = await db.execute(
                sqla_select(PipelineJob)
                .where(PipelineJob.status.in_(["failed", "completed", "cancelled"]))
                .order_by(desc(PipelineJob.created_at))
                .limit(20)  # scan recent jobs to find one with progress
            )
            candidates = res.scalars().all()
            for candidate in candidates:
                if _has_progress(candidate):
                    job = candidate
                    break

    if not job:
        raise HTTPException(status_code=404, detail="No meaningful pipeline jobs found")

    return {
        "job_id": job.id,
        "symbols": job.symbols,
        "status": job.status,
        "current_stage": _resolve_stage_index(job.current_stage),
        "stages": job.stages,
        "created_at": job.created_at.isoformat(),
        "updated_at": job.updated_at.isoformat(),
        "error": job.error,
    }


@router.get("/sync-status")
async def get_sync_status() -> dict:
    """Return the latest OHLCV and regime dates for every active stock.

    Used by the frontend to show "Data last synced: April 18, 2026" and to
    highlight stale stocks before the user starts a pipeline run.

    Response shape::

        {
            "RELIANCE": {
                "ohlcv_latest": "2026-04-18",
                "regime_latest": "2026-04-18",
                "is_stale": false
            },
            ...
        }

    ``is_stale`` is True when either ohlcv_latest or regime_latest is more than
    1 calendar day behind today (weekends/holidays are not accounted for here —
    that is handled server-side by the data sync logic).
    """
    from datetime import date as date_cls
    from sqlalchemy import select as sqla_select, func, text
    from app.db.models import Stock, StockOHLCV, StockRegime

    today = date_cls.today()

    async with async_session_factory() as db:
        # Latest OHLCV date per stock
        ohlcv_q = (
            sqla_select(StockOHLCV.stock_id, func.max(StockOHLCV.date).label("latest"))
            .where(StockOHLCV.interval == "day")
            .group_by(StockOHLCV.stock_id)
        )
        ohlcv_rows = (await db.execute(ohlcv_q)).all()
        ohlcv_map = {r.stock_id: r.latest for r in ohlcv_rows}

        # Latest regime date per stock
        regime_q = (
            sqla_select(StockRegime.stock_id, func.max(StockRegime.date).label("latest"))
            .where(StockRegime.interval == "day")
            .group_by(StockRegime.stock_id)
        )
        regime_rows = (await db.execute(regime_q)).all()
        regime_map = {r.stock_id: r.latest for r in regime_rows}

        # All active stocks
        stocks_res = await db.execute(sqla_select(Stock).where(Stock.is_active == True))
        stocks = stocks_res.scalars().all()

    result: dict = {}
    for stock in stocks:
        ohlcv_latest = ohlcv_map.get(stock.id)
        regime_latest = regime_map.get(stock.id)
        is_stale = (
            ohlcv_latest is None
            or regime_latest is None
            or (today - ohlcv_latest).days > 1
            or (today - regime_latest).days > 1
        )
        result[stock.symbol] = {
            "ohlcv_latest": ohlcv_latest.isoformat() if ohlcv_latest else None,
            "regime_latest": regime_latest.isoformat() if regime_latest else None,
            "is_stale": is_stale,
        }

    return result


@router.delete("/{job_id}")
async def terminate_pipeline(job_id: str, purge: bool = False):
    """Terminate a running pipeline and optionally purge all data/files it created.

    - ``purge=false`` (default): cancel the running pipeline, keep DB records + files.
    - ``purge=true``: cancel (if running) **and** delete all DB records and model files
      that were created by this pipeline session.
    """
    async with async_session_factory() as db:
        job = await crud.get_pipeline_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Pipeline job not found")

    # Signal cancellation by writing the status to DB (works across workers)
    if job.status in ("running", "queued"):
        async with async_session_factory() as db:
            await crud.update_pipeline_job(db, job_id, status="cancelled")

    if purge:
        deleted = await _purge_pipeline_data(job_id)
        return {"job_id": job_id, "status": "purged", **deleted}

    return {"job_id": job_id, "status": "terminating"}


async def _purge_pipeline_data(job_id: str) -> dict:
    """Delete all DB records and disk files created by a pipeline session.

    Cleans up (in FK-safe order):
      EnsembleConfigs & Predictions → backtest_results → knn/lstm models (+ directory & logs) → 
      golden_patterns → rl_training_runs → rl_models (+ directory, parquets & logs) → 
      cql/bc files → pipeline_job
    """
    import shutil
    from sqlalchemy import delete as sqla_delete, select as sqla_select, or_ as sqla_or
    from app.db.models import (
        RLModel, RLTrainingRun, GoldenPattern,
        KNNModel, LSTMModel, BacktestResult, PipelineJob,
        EnsembleConfig, StockEnsembleWeights, EnsemblePrediction,
        KNNPrediction, LSTMPrediction
    )

    files_deleted: list[str] = []
    records_deleted: dict[str, int] = {}

    def _remove_file(path: str | None) -> None:
        if not path:
            return
        p = Path(path)
        if p.exists():
            p.unlink(missing_ok=True)
            files_deleted.append(str(p.name))

    async with async_session_factory() as db:
        # ── Find RL models created by this pipeline ───────────────────────
        rl_result = await db.execute(sqla_select(RLModel))
        all_rl = rl_result.scalars().all()
        pipeline_rl_ids = [
            m.id for m in all_rl
            if (m.training_config or {}).get("pipeline_job_id") == job_id
        ]

        # ── Find KNN / LSTM models built from those RL models ─────────────
        knn_models: list[KNNModel] = []
        lstm_models: list[LSTMModel] = []
        if pipeline_rl_ids:
            knn_res = await db.execute(
                sqla_select(KNNModel).where(KNNModel.source_rl_model_id.in_(pipeline_rl_ids))
            )
            knn_models = list(knn_res.scalars().all())
            lstm_res = await db.execute(
                sqla_select(LSTMModel).where(LSTMModel.source_rl_model_id.in_(pipeline_rl_ids))
            )
            lstm_models = list(lstm_res.scalars().all())

        knn_ids = [m.id for m in knn_models]
        lstm_ids = [m.id for m in lstm_models]

        # ── Find EnsembleConfig tied to these KNN/LSTM models ─────────────
        ensemble_configs: list[EnsembleConfig] = []
        if knn_ids or lstm_ids:
            conds = []
            if knn_ids:
                conds.append(EnsembleConfig.knn_model_id.in_(knn_ids))
            if lstm_ids:
                conds.append(EnsembleConfig.lstm_model_id.in_(lstm_ids))
            enc_res = await db.execute(
                sqla_select(EnsembleConfig).where(sqla_or(*conds))
            )
            ensemble_configs = list(enc_res.scalars().all())
        ensemble_ids = [e.id for e in ensemble_configs]

        # ── Delete Ensemble Weights & Predictions ─────────────────────────
        if ensemble_ids:
            await db.execute(sqla_delete(StockEnsembleWeights).where(StockEnsembleWeights.ensemble_config_id.in_(ensemble_ids)))
            await db.execute(sqla_delete(EnsemblePrediction).where(EnsemblePrediction.ensemble_config_id.in_(ensemble_ids)))
        
        # ── Delete KNN & LSTM Predictions ─────────────────────────────────
        if knn_ids:
            await db.execute(sqla_delete(KNNPrediction).where(KNNPrediction.knn_model_id.in_(knn_ids)))
        if lstm_ids:
            await db.execute(sqla_delete(LSTMPrediction).where(LSTMPrediction.lstm_model_id.in_(lstm_ids)))

        # ── Delete backtest results tied to these models ───────────────────
        bt_count = 0
        if ensemble_ids:
            r = await db.execute(
                sqla_delete(BacktestResult).where(
                    BacktestResult.model_type == "ensemble",
                    sqla_or(BacktestResult.model_id.in_(ensemble_ids), BacktestResult.model_id == 0)
                )
            )
            bt_count += r.rowcount
        if knn_ids:
            r = await db.execute(
                sqla_delete(BacktestResult).where(
                    BacktestResult.model_type == "knn",
                    sqla_or(BacktestResult.model_id.in_(knn_ids), BacktestResult.model_id == 0)
                )
            )
            bt_count += r.rowcount
        if lstm_ids:
            r = await db.execute(
                sqla_delete(BacktestResult).where(
                    BacktestResult.model_type == "lstm",
                    sqla_or(BacktestResult.model_id.in_(lstm_ids), BacktestResult.model_id == 0)
                )
            )
            bt_count += r.rowcount
        if pipeline_rl_ids:
            r = await db.execute(
                sqla_delete(BacktestResult).where(
                    BacktestResult.model_type == "rl",
                    sqla_or(BacktestResult.model_id.in_(pipeline_rl_ids), BacktestResult.model_id == 0)
                )
            )
            bt_count += r.rowcount
        records_deleted["backtest_results"] = bt_count

        # ── Delete EnsembleConfig DB rows ─────────────────────────────────
        if ensemble_ids:
            await db.execute(sqla_delete(EnsembleConfig).where(EnsembleConfig.id.in_(ensemble_ids)))
            records_deleted["ensemble_configs"] = len(ensemble_ids)

        # ── Delete KNN / LSTM model dirs + norm params + DB rows ──────────
        for m in knn_models:
            _remove_file(m.norm_params_path)
            if m.model_path:
                art_dir = Path(m.model_path).parent
                if art_dir.exists():
                    shutil.rmtree(art_dir, ignore_errors=True)
                    files_deleted.append(str(art_dir.name))
        for m in lstm_models:
            if m.model_path:
                art_dir = Path(m.model_path).parent
                if art_dir.exists():
                    shutil.rmtree(art_dir, ignore_errors=True)
                    files_deleted.append(str(art_dir.name))

        if knn_ids:
            await db.execute(sqla_delete(KNNModel).where(KNNModel.id.in_(knn_ids)))
        if lstm_ids:
            await db.execute(sqla_delete(LSTMModel).where(LSTMModel.id.in_(lstm_ids)))
        records_deleted["knn_models"] = len(knn_ids)
        records_deleted["lstm_models"] = len(lstm_ids)

        # ── Delete RL models and their dependents ─────────────────────────
        if pipeline_rl_ids:
            await db.execute(sqla_delete(GoldenPattern).where(GoldenPattern.rl_model_id.in_(pipeline_rl_ids)))
            await db.execute(sqla_delete(RLTrainingRun).where(RLTrainingRun.rl_model_id.in_(pipeline_rl_ids)))
            for m in all_rl:
                if m.id in pipeline_rl_ids:
                    if m.model_path:
                        art_dir = Path(m.model_path).parent
                        if art_dir.exists():
                            shutil.rmtree(art_dir, ignore_errors=True)
                            files_deleted.append(str(art_dir.name))
            await db.execute(sqla_delete(RLModel).where(RLModel.id.in_(pipeline_rl_ids)))
        records_deleted["rl_models"] = len(pipeline_rl_ids)

        # ── Delete pattern datasets and log files ─────────────────────────
        parquet_dir = Path(settings.MODEL_DIR) / "patterns"
        if parquet_dir.exists() and pipeline_rl_ids:
            for pid in pipeline_rl_ids:
                for p in parquet_dir.glob(f"patterns_{pid}_*.parquet"):
                    p.unlink(missing_ok=True)
                    files_deleted.append(p.name)
        
        log_dir = Path(__file__).resolve().parents[3] / "logs" / "training"
        if log_dir.exists():
            for pid in pipeline_rl_ids:
                p = log_dir / f"model_{pid}.log"
                if p.exists():
                    p.unlink(missing_ok=True)
                    files_deleted.append(p.name)
            for kid in knn_ids:
                p = log_dir / f"distill_{kid}.log"
                if p.exists():
                    p.unlink(missing_ok=True)
                    files_deleted.append(p.name)

        # ── Delete CQL / BC warmup files (named after job_id prefix) ──────
        short_id = job_id[:8]
        for candidate_dir in [settings.MODEL_DIR, Path(settings.MODEL_DIR) / "rl"]:
            candidate_dir = Path(candidate_dir)
            if candidate_dir.exists():
                for f in candidate_dir.glob(f"*pipeline_cql_{short_id}*"):
                    f.unlink(missing_ok=True)
                    files_deleted.append(f.name)
                for f in candidate_dir.glob(f"*bc_warmup*{short_id}*"):
                    f.unlink(missing_ok=True)
                    files_deleted.append(f.name)

        # ── Finally delete the pipeline_job record itself ─────────────────
        await db.execute(sqla_delete(PipelineJob).where(PipelineJob.id == job_id))
        await db.commit()

    logger.info(
        "Pipeline %s purged — records: %s, files: %s",
        job_id, records_deleted, files_deleted,
    )
    return {"records_deleted": records_deleted, "files_deleted": files_deleted}


# ── Background pipeline runner ─────────────────────────────────────────────────

async def _run_pipeline(job_id: str, symbols: list[str], skip_sync: bool = False, use_regime_pooling: bool = True, force_sync: bool = False, resume_job_id: str | None = None) -> None:
    async with async_session_factory() as db:
        await crud.update_pipeline_job(db, job_id, status="running")

    # Resume Logic — validate previous job artifacts before deciding what to skip.
    # When resume_job_id is provided the symbol set must match exactly; any disk
    # artifact that has been deleted since the previous run causes that stage (and
    # all later stages) to be demoted back to pending so the pipeline re-runs them.

    validated_stages: list[str] = []
    existing_rl_id: int | None = None
    existing_knn_id: int | None = None
    existing_lstm_id: int | None = None
    prev_sync_ok = False

    if resume_job_id:
        async with async_session_factory() as db:
            resume_from_prev = await crud.get_pipeline_job(db, resume_job_id)

        if resume_from_prev is None:
            raise RuntimeError(f"Resume job {resume_job_id} not found in database.")

        try:
            validated_stages, existing_rl_id, existing_knn_id, existing_lstm_id = \
                await _validate_resume_artifacts(resume_from_prev, symbols)
        except ValueError as sym_err:
            # Symbol-set mismatch — fail fast with a clear message
            async with async_session_factory() as db:
                await crud.update_pipeline_job(db, job_id, status="failed", error=str(sym_err))
            raise RuntimeError(str(sym_err)) from sym_err

        prev_sync_ok = "data_sync" in validated_stages
        logger.info(
            "Pipeline %s resume from %s: validated stages=%s, "
            "rl=%s knn=%s lstm=%s",
            job_id, resume_job_id, validated_stages,
            existing_rl_id, existing_knn_id, existing_lstm_id,
        )

    try:
        # Resolve stock IDs for the given symbols
        stock_ids = await _resolve_stock_ids(symbols)
        if not stock_ids:
            raise RuntimeError(
                "None of the requested symbols were found in the database. "
                "Run 'Sync Stock List' in Data Manager first."
            )

        # Stage 0 — Data Sync
        # Skip when: user opted out (skip_sync) OR resuming with valid prev sync,
        # UNLESS force_sync=True which overrides both.
        if (skip_sync or prev_sync_ok) and not force_sync:
            await _update_stage(
                job_id, 0, status="completed", progress=100,
                message="Data sync skipped (using previous sync).",
            )
        else:
            if await _is_cancelled(job_id):
                raise asyncio.CancelledError()
            await _stage_data_sync(job_id, stock_ids, symbols, force_sync)

        # Stages 1–3 — CQL / BC / PPO
        # Re-use previous RL model only when resume validation confirmed the artifact.
        if existing_rl_id is not None:
            rl_model_id = existing_rl_id
            await _update_stage(job_id, 1, status="completed", progress=100, message="Resumed: CQL skipped.")
            await _update_stage(job_id, 2, status="completed", progress=100, message="Resumed: BC warmup skipped.")
            await _update_stage(
                job_id, 3, status="completed", progress=100,
                message=f"Resumed: using existing RL model #{rl_model_id}",
            )
        else:
            # Stage 1 — CQL Pre-training
            if await _is_cancelled(job_id):
                raise asyncio.CancelledError()
            cql_path = await _stage_cql_pretrain(job_id, stock_ids, use_regime_pooling)

            # Stage 2 — BC Warmup
            if await _is_cancelled(job_id):
                raise asyncio.CancelledError()
            warmup_path = await _stage_bc_warmup(job_id, cql_path, stock_ids, use_regime_pooling)

            # Stage 3 — PPO Fine-tuning
            if await _is_cancelled(job_id):
                raise asyncio.CancelledError()
            rl_model_id = await _stage_ppo_finetune(job_id, stock_ids, warmup_path)

        # Stage 4 — Ensemble Distillation
        # Re-use previous KNN + LSTM only when resume validation confirmed both artifacts.
        if existing_knn_id is not None and existing_lstm_id is not None:
            knn_model_id = existing_knn_id
            lstm_model_id = existing_lstm_id
            await _update_stage(
                job_id, 4, status="completed", progress=100,
                message=f"Resumed: using existing KNN #{knn_model_id} + LSTM #{lstm_model_id}",
            )
        else:
            if await _is_cancelled(job_id):
                raise asyncio.CancelledError()
            knn_model_id, lstm_model_id = await _stage_ensemble_distill(job_id, rl_model_id, stock_ids)

        # Stage 5 — Backtest
        if await _is_cancelled(job_id):
            raise asyncio.CancelledError()
        await _stage_backtest(job_id, rl_model_id, knn_model_id, lstm_model_id, stock_ids)

        # Stage 6 — Deploy: run predictions + train meta-classifier
        if await _is_cancelled(job_id):
            raise asyncio.CancelledError()
        await _stage_deploy(job_id, rl_model_id, knn_model_id, lstm_model_id, stock_ids)

        async with async_session_factory() as db:
            await crud.update_pipeline_job(db, job_id, status="completed")

    except asyncio.CancelledError:
        logger.info("Pipeline %s was cancelled", job_id)
        async with async_session_factory() as db:
            job_rec = await crud.get_pipeline_job(db, job_id)
            if job_rec:
                stages = list(job_rec.stages or [])
                for stage in stages:
                    if stage["status"] in ("running", "pending"):
                        stage["status"] = "cancelled"
                        stage["message"] = "Terminated by user."
                await crud.update_pipeline_job(db, job_id, status="cancelled", error="Terminated by user.")
                from sqlalchemy import update as sqla_update
                from app.db.models import PipelineJob
                await db.execute(sqla_update(PipelineJob).where(PipelineJob.id == job_id).values(stages=stages))
                await db.commit()

    except Exception as exc:
        logger.exception("Pipeline %s failed", job_id)
        async with async_session_factory() as db:
            job_rec = await crud.get_pipeline_job(db, job_id)
            if job_rec:
                stages = list(job_rec.stages or [])
                for stage in stages:
                    if stage["status"] == "running":
                        stage["status"] = "failed"
                        stage["message"] = str(exc)
                await crud.update_pipeline_job(db, job_id, status="failed", error=str(exc))
                
                # Manual JSON update for stages
                from sqlalchemy import update as sqla_update
                from app.db.models import PipelineJob
                await db.execute(sqla_update(PipelineJob).where(PipelineJob.id == job_id).values(stages=stages))
                await db.commit()


# ── Helper: resolve symbols → stock IDs ───────────────────────────────────────

async def _resolve_stock_ids(symbols: list[str]) -> list[int]:
    from app.db.database import async_session_factory

    ids: list[int] = []
    async with async_session_factory() as db:
        for sym in symbols:
            stock = await crud.get_stock_by_symbol(db, sym)
            if stock:
                ids.append(stock.id)
            else:
                logger.warning("Pipeline: symbol %s not found in DB — skipping", sym)
    return ids


async def _validate_resume_artifacts(
    prev_job,
    symbols: list[str],
) -> tuple[list[str], int | None, int | None, int | None]:
    """Validate that a previous job's artifacts still exist on disk.

    Returns
    -------
    (validated_stage_names, rl_model_id, knn_model_id, lstm_model_id)

    ``validated_stage_names`` is the subset of the previous job's completed
    stages whose disk artifacts still exist.  Missing artifacts cause that
    stage (and all later stages) to be demoted back to pending.

    Raises ``ValueError`` when the symbol sets differ — resume is not safe.
    """
    from sqlalchemy import select as sqla_select
    from app.db.models import RLModel, KNNModel, LSTMModel

    # ── 1. Symbol-set guard ───────────────────────────────────────────
    prev_symbols = set(prev_job.symbols or [])
    new_symbols = set(symbols)
    if prev_symbols != new_symbols:
        added = new_symbols - prev_symbols
        removed = prev_symbols - new_symbols
        parts = []
        if added:
            parts.append(f"added: {sorted(added)}")
        if removed:
            parts.append(f"removed: {sorted(removed)}")
        raise ValueError(
            f"Cannot resume: symbol set changed ({', '.join(parts)}). "
            "Use 'Discard & Restart' to start a fresh pipeline with the new symbols."
        )

    # ── 2. Determine which stages completed in prev job ───────────────
    prev_stages = {s["name"]: s for s in (prev_job.stages or [])}

    rl_model_id: int | None = None
    knn_model_id: int | None = None
    lstm_model_id: int | None = None
    validated: list[str] = []  # stage names whose artifacts are confirmed on disk

    async with async_session_factory() as db:
        # ── Stage 3: PPO / RL model ──────────────────────────────────
        ppo_stage = prev_stages.get("ppo_finetune", {})
        if ppo_stage.get("status") == "completed":
            rl_res = await db.execute(
                sqla_select(RLModel).where(
                    RLModel.training_config["pipeline_job_id"].as_string() == prev_job.id,
                    RLModel.status == "completed",
                ).order_by(RLModel.created_at.desc())
            )
            rl_model = rl_res.scalars().first()
            if rl_model and rl_model.model_path and Path(rl_model.model_path).exists():
                rl_model_id = rl_model.id
                validated += ["cql_pretrain", "bc_warmup", "ppo_finetune"]
                logger.info(
                    "Resume validation: RL model #%s artifact OK at %s",
                    rl_model_id, rl_model.model_path,
                )
            else:
                logger.warning(
                    "Resume validation: RL model artifact missing for job %s — "
                    "will retrain from stage 1",
                    prev_job.id,
                )

        # ── Stage 4: KNN + LSTM ensemble distillation ────────────────
        distill_stage = prev_stages.get("ensemble_distill", {})
        if distill_stage.get("status") == "completed" and rl_model_id is not None:
            knn_res = await db.execute(
                sqla_select(KNNModel).where(
                    KNNModel.source_rl_model_id == rl_model_id,
                    KNNModel.status == "completed",
                ).order_by(KNNModel.created_at.desc())
            )
            knn_model = knn_res.scalars().first()

            lstm_res = await db.execute(
                sqla_select(LSTMModel).where(
                    LSTMModel.source_rl_model_id == rl_model_id,
                    LSTMModel.status == "completed",
                ).order_by(LSTMModel.created_at.desc())
            )
            lstm_model = lstm_res.scalars().first()

            knn_ok = (
                knn_model is not None
                and knn_model.model_path
                and Path(knn_model.model_path).exists()
            )
            lstm_ok = (
                lstm_model is not None
                and lstm_model.model_path
                and Path(lstm_model.model_path).exists()
            )
            if knn_ok and lstm_ok:
                knn_model_id = knn_model.id
                lstm_model_id = lstm_model.id
                validated.append("ensemble_distill")
                logger.info(
                    "Resume validation: KNN #%s + LSTM #%s artifacts OK",
                    knn_model_id, lstm_model_id,
                )
            else:
                missing = []
                if not knn_ok:
                    missing.append("KNN")
                if not lstm_ok:
                    missing.append("LSTM")
                logger.warning(
                    "Resume validation: %s artifact(s) missing for job %s — "
                    "will re-distill from RL model",
                    "/".join(missing), prev_job.id,
                )

        # ── Stage 5: Backtest ─────────────────────────────────────────
        # Only mark backtest as completed if distillation artifacts are also valid.
        bt_stage = prev_stages.get("backtest", {})
        if bt_stage.get("status") == "completed" and knn_model_id and lstm_model_id:
            validated.append("backtest")

        # ── data_sync ────────────────────────────────────────────────
        sync_stage = prev_stages.get("data_sync", {})
        if sync_stage.get("status") == "completed":
            validated.append("data_sync")

    return validated, rl_model_id, knn_model_id, lstm_model_id


# ── Stage implementations ─────────────────────────────────────────────────────

async def _stage_data_sync(job_id: str, stock_ids: list[int], symbols: list[str], force_sync: bool = False) -> None:
    from app.db.database import async_session_factory
    from app.core import zerodha, data_service

    await _update_stage(job_id, 0, status="running", progress=0, message="Starting OHLCV + Regime sync…")

    if not zerodha.is_authenticated():
        # Data sync requires Zerodha auth; mark as failed with actionable message
        await _update_stage(
            job_id, 0, status="failed", progress=0,
            message="Zerodha not authenticated. Open Settings → Zerodha and login first.",
        )
        raise RuntimeError("Zerodha not authenticated — cannot sync OHLCV data")

    total = len(stock_ids)
    synced = 0

    for i, stock_id in enumerate(stock_ids):
        # Check cancellation before each stock so termination takes effect promptly
        if await _is_cancelled(job_id):
            raise asyncio.CancelledError()

        sym = symbols[i] if i < len(symbols) else f"#{stock_id}"

        for interval in ["day", "week"]:
            if await _is_cancelled(job_id):
                raise asyncio.CancelledError()

            pct = int(i / total * 90) + (5 if interval == "week" else 0)
            await _update_stage(
                job_id, 0, status="running", progress=min(95, pct),
                message=f"Syncing & Classifying {sym} ({interval.capitalize()})…"
            )

            async with async_session_factory() as db:
                try:
                    res = await data_service.sync_and_compute(db, stock_id, interval=interval, force_full=force_sync)
                    if res.get("ohlcv_synced", 0) >= 0 and interval == "day":
                        synced += 1
                except Exception as exc:
                    logger.warning("Pipeline data sync: %s (%s) failed: %s", sym, interval, exc)

    await _update_stage(
        job_id, 0, status="completed", progress=100,
        message=f"Sync + Indicators + Regimes complete. {synced}/{total} stocks ready.",
    )


async def _stage_cql_pretrain(job_id: str, stock_ids: list[int], use_regime_pooling: bool = True) -> str | None:
    """Run offline CQL pre-training. Returns cql_path if successful."""
    await _update_stage(job_id, 1, status="running", progress=0, message="Checking for d3rlpy…")

    try:
        import d3rlpy  # noqa: F401
        has_d3rlpy = True
    except ImportError:
        has_d3rlpy = False

    if not has_d3rlpy:
        await _update_stage(
            job_id, 1, status="completed", progress=100,
            message="d3rlpy not installed — CQL pre-training skipped. PPO will train from scratch.",
        )
        return None

    # d3rlpy is available — run CQL pre-training
    await _update_stage(job_id, 1, status="running", progress=10, message="Building offline dataset from OHLCV…")

    try:
        from app.db.database import async_session_factory
        from app.ml.cql_trainer import collect_offline_transitions, build_offline_dataset, train_cql

        loop = asyncio.get_running_loop()

        from app.core import data_service
        import pandas as _pd
        df = _pd.DataFrame()
        
        if use_regime_pooling:
            await _update_stage(job_id, 1, status="running", progress=20, message=f"Regime pooling: fetching data from {len(stock_ids)} stocks…")
            all_dfs = []
            for _sid in stock_ids:
                async with async_session_factory() as db:
                    _df = await data_service.get_stock_features(db, _sid, "day", normalize=False)
                if not _df.empty and len(_df) > 100:
                    all_dfs.append(_df)
            if not all_dfs:
                raise RuntimeError("No stocks have sufficient data for regime pooling.")
            df = _pd.concat(all_dfs, ignore_index=True)
            await _update_stage(job_id, 1, status="running", progress=25, message=f"Regime pooling: {len(df):,} rows from {len(all_dfs)} stocks.")
        else:
            # Find the first stock with sufficient data
            await _update_stage(job_id, 1, status="running", progress=20, message="Finding first valid stock for offline pre-training…")
            primary_stock_id = stock_ids[0]  # fallback
            for _sid in stock_ids:
                async with async_session_factory() as db:
                    df = await data_service.get_stock_features(db, _sid, "day", normalize=False)
                if not df.empty and len(df) > 100:
                    primary_stock_id = _sid
                    break
        
        if df.empty:
            raise RuntimeError("No stocks have sufficient data for offline pre-training.")

        await _update_stage(job_id, 1, status="running", progress=30, message="Collecting offline RL transitions…")

        def _run_cql():
            transitions = collect_offline_transitions(df)
            dataset = build_offline_dataset(transitions)
            model_path = train_cql(
                dataset,
                n_steps=50_000,
                model_name=f"pipeline_cql_{job_id[:8]}",
            )
            return model_path

        cql_path = await loop.run_in_executor(_PIPELINE_EXECUTOR, _run_cql)
        await _update_stage(job_id, 1, status="completed", progress=100, message=f"CQL trained → {Path(cql_path).name}")
        return cql_path

    except Exception as exc:
        logger.warning("CQL pre-training failed (non-fatal): %s", exc)
        await _update_stage(
            job_id, 1, status="completed", progress=100,
            message=f"CQL pre-training failed ({exc!s:.80}) — continuing.",
        )
        return None


async def _stage_bc_warmup(job_id: str, cql_path: str | None, stock_ids: list[int], use_regime_pooling: bool = True) -> str | None:
    await _update_stage(job_id, 2, status="running", progress=0, message="Checking for BC warmup dependencies…")

    try:
        import d3rlpy  # noqa: F401
        has_d3rlpy = True
    except ImportError:
        has_d3rlpy = False

    if not has_d3rlpy or cql_path is None:
        await _update_stage(
            job_id, 2, status="completed", progress=100,
            message="BC warmup skipped (d3rlpy not installed or no CQL model available).",
        )
        return None

    # If d3rlpy and CQL path exist, run BC to align PPO actor
    await _update_stage(job_id, 2, status="running", progress=10, message="Collecting data for BC warmup…")
    try:
        from app.db.database import async_session_factory
        from app.db import crud as _crud
        from app.ml.rl_trainer import run_bc_warmup
        import pandas as pd

        from app.core import data_service
        import pandas as _pd
        df = _pd.DataFrame()

        if use_regime_pooling:
            await _update_stage(job_id, 2, status="running", progress=15, message=f"Collective BC: pooling data from {len(stock_ids)} stocks…")
            all_dfs = []
            for _sid in stock_ids:
                async with async_session_factory() as db:
                    _df = await data_service.get_stock_features(db, _sid, "day", normalize=False)
                if not _df.empty and len(_df) > 100:
                    all_dfs.append(_df)
            if not all_dfs:
                raise RuntimeError("No stocks have sufficient data for collective BC.")
            df = _pd.concat(all_dfs, ignore_index=True)
            await _update_stage(job_id, 2, status="running", progress=25, message=f"Collective BC: {len(df):,} rows ready.")
        else:
            # Find the first stock with sufficient data (same logic as CQL stage)
            primary_stock_id = stock_ids[0]
            for _sid in stock_ids:
                async with async_session_factory() as db:
                    df = await data_service.get_stock_features(db, _sid, "day", normalize=False)
                if not df.empty and len(df) > 100:
                    primary_stock_id = _sid
                    break

        if df.empty:
            raise RuntimeError("No stocks have sufficient data for BC warmup.")
            
        loop = asyncio.get_running_loop()

        await _update_stage(job_id, 2, status="running", progress=30, message="Running behavioral cloning pass…")
        
        warmup_path = await loop.run_in_executor(
            _PIPELINE_EXECUTOR,
            lambda: run_bc_warmup(
                cql_path=cql_path,
                ohlcv_df=df,
                algorithm="AttentionPPO",
                seq_len=15,
                bc_steps=2000,
            )
        )
        await _update_stage(job_id, 2, status="completed", progress=100, message="BC warmup completed. PPO model aligned.")
        return warmup_path
    except Exception as exc:
        logger.warning("BC warmup failed: %s", exc)
        await _update_stage(
            job_id, 2, status="completed", progress=100,
            message=f"BC warmup skipped: {exc!s:.80}",
        )
        return None


async def _stage_ppo_finetune(job_id: str, stock_ids: list[int], pretrained_path: str | None) -> int:
    """Create and train an AttentionPPO model over the selected stock universe.
    Returns the new rl_model_id.
    """
    from app.db.database import async_session_factory
    from app.db import crud as _crud
    from app.ml.algorithms import ALGORITHM_CONFIGS, get_default_hyperparams

    await _update_stage(job_id, 3, status="running", progress=0, message="Creating RL model record…")

    algorithm = "AttentionPPO"
    if algorithm not in ALGORITHM_CONFIGS:
        algorithm = "PPO"  # fallback if AttentionPPO not registered

    hyperparams = get_default_hyperparams(algorithm)
    # Scale timesteps by universe size — 150k minimum, +15k per stock.
    # 100k was fine for 1 stock but starves the agent when 50+ stocks are used.
    total_timesteps = max(150_000, len(stock_ids) * 15_000)
    # Prevent policy collapse (all-HOLD): inject entropy coef so PPO is penalised
    # for not exploring BUY/SELL.  SB3 default is ent_coef=0.0 which gives the
    # agent zero mathematical incentive to try non-HOLD actions.
    hyperparams["ent_coef"] = 0.05
    hyperparams["learning_rate"] = 0.0003  # nudge up from SB3 default 0.0001
    interval = "day"

    async with async_session_factory() as db:
        rl_model = await _crud.create_rl_model(
            db,
            name=f"Pipeline_{algorithm}_{job_id[:8]}",
            algorithm=algorithm,
            hyperparams=hyperparams,
            training_config={
                "total_timesteps": total_timesteps,
                "stock_ids": stock_ids,
                "min_quality": 0.0,
                "regime_ids": None,
                "reward_function": "risk_adjusted_pnl",
                "seq_len": 15,
                "pipeline_job_id": job_id,
            },
            features=None,
            regime_filter=None,
            interval=interval,
            status="pending",
        )
        rl_model_id = rl_model.id

    async with async_session_factory() as db:
        job_rec = await crud.get_pipeline_job(db, job_id)
        symbols = job_rec.symbols if job_rec else []

    await _update_stage(job_id, 3, status="running", progress=5, message=f"RL model #{rl_model_id} created. Fetching data for {len(stock_ids)} stocks…")

    # Fetch raw OHLCV for all stocks — same pipeline as distillation so
    # Fetch Full Features (OHLCV + Regimes + Indicators) for all stocks
    from app.core import data_service
    import pandas as pd
    multi_dfs: list[pd.DataFrame] = []
    total_stocks = len(stock_ids)

    for idx, sid in enumerate(stock_ids):
        sym = symbols[idx] if idx < len(symbols) else f"#{sid}"
        await _update_stage(job_id, 3, status="running", progress=int(5 + idx / total_stocks * 5),
                      message=f"Loading data for {sym} ({idx+1}/{total_stocks})…")

        async with async_session_factory() as db:
            # FIX: Fetch the exact same DB features used in CQL and BC
            df = await data_service.get_stock_features(db, sid, interval, normalize=False)

        if not df.empty:
            multi_dfs.append(df)
        else:
            logger.warning("Pipeline PPO: no feature data for stock #%d — skipping", sid)

    if not multi_dfs:
        await _update_stage(job_id, 3, status="failed", progress=5, message="No feature data found for any stock. Sync data first.")
        raise RuntimeError("No feature data for any stock in the universe")

    await _update_stage(job_id, 3, status="running", progress=10,
                  message=f"Training {algorithm} across {len(multi_dfs)}/{total_stocks} stocks for {total_timesteps:,} steps…")

    progress_cursor = [10]
    loop = asyncio.get_running_loop()

    def _on_progress(info: dict) -> None:
        step = info.get("timestep", 0)
        pct = min(95, 10 + int(step / total_timesteps * 85))
        if pct > progress_cursor[0]:
            progress_cursor[0] = pct
            msg = ""
            if info.get("ep_rew_mean") is not None:
                msg = f"Step {step:,} | Reward {info['ep_rew_mean']:.4f}"
            # Block briefly so we can act on the circuit-breaker return value.
            future = asyncio.run_coroutine_threadsafe(
                _update_stage(job_id, 3, status="running", progress=pct, message=msg), loop
            )
            try:
                is_cancelled = future.result(timeout=5.0)
                if is_cancelled:
                    # Raise so Stable-Baselines3 callback machinery propagates
                    # the exception and terminates the training thread immediately.
                    raise InterruptedError("Pipeline cancelled by user.")
            except InterruptedError:
                raise
            except Exception as exc:
                logger.error("Progress update failed: %s", exc)

    from app.ml.rl_trainer import train_rl_model

    try:
        result: dict = await loop.run_in_executor(
            _PIPELINE_EXECUTOR,
            lambda: train_rl_model(
                multi_ohlcv_dfs=multi_dfs,
                algorithm=algorithm,
                hyperparams=hyperparams,
                total_timesteps=total_timesteps,
                min_quality=0.0,
                regime_ids=None,
                reward_function="risk_adjusted_pnl",
                seq_len=15,
                model_name=f"Pipeline_{algorithm}_{job_id[:8]}",
                on_progress=_on_progress,
                device="auto",
                pretrained_model_path=pretrained_path,
            ),
        )
    except (InterruptedError, asyncio.CancelledError):
        # Circuit breaker fired from _on_progress — the job was cancelled while
        # PPO was training. Propagate as CancelledError so _run_pipeline's cancel
        # handler cleans up properly instead of marking the job as "failed".
        raise asyncio.CancelledError()

    # Persist training result to DB
    async with async_session_factory() as db:
        await _crud.update_rl_model_completed(
            db,
            rl_model_id,
            total_reward=result.get("final_reward"),
            sharpe_ratio=result.get("sharpe_ratio"),
            model_path=result.get("model_path"),
        )

    # Refresh model from DB to check it saved correctly
    async with async_session_factory() as db:
        rl_model = await _crud.get_rl_model(db, rl_model_id)

    if not rl_model or rl_model.status == "failed":
        raise RuntimeError(f"PPO fine-tuning failed for model #{rl_model_id}")

    await _update_stage(job_id, 3, status="completed", progress=100, message=f"AttentionPPO trained. Model #{rl_model_id} ready.")
    return rl_model_id


async def _stage_ensemble_distill(job_id: str, rl_model_id: int, stock_ids: list[int]) -> tuple[int | None, int | None]:
    from app.db.database import async_session_factory
    from app.db import crud as _crud

    await _update_stage(job_id, 4, status="running", progress=0, message="Creating KNN + LSTM model records…")

    interval = "day"

    # Load RL model to derive config
    async with async_session_factory() as db:
        rl_model = await _crud.get_rl_model(db, rl_model_id)
    if not rl_model:
        raise RuntimeError(f"RL model #{rl_model_id} not found for distillation")
    if not rl_model.model_path:
        raise RuntimeError(f"RL model #{rl_model_id} has no saved model file — training may have failed")

    training_config = rl_model.training_config or {}
    seq_len = training_config.get("seq_len", 15)

    async with async_session_factory() as db:
        knn_model = await _crud.create_knn_model(
            db,
            name=f"Pipeline_KNN_{job_id[:8]}",
            source_rl_model_id=rl_model_id,
            k_neighbors=9,
            seq_len=seq_len,
            interval=interval,
            regime_filter=rl_model.regime_filter,
            status="pending",
        )
        lstm_model = await _crud.create_lstm_model(
            db,
            name=f"Pipeline_LSTM_{job_id[:8]}",
            source_rl_model_id=rl_model_id,
            seq_len=seq_len,
            interval=interval,
            regime_filter=rl_model.regime_filter,
            hidden_size=256,
            num_layers=2,
            dropout=0.3,
            status="pending",
        )
        
        # Create ensemble config to bind them so they appear in Model Studio
        await _crud.create_ensemble_config(
            db,
            name=f"Ensemble_{job_id[:8]}",
            knn_model_id=knn_model.id,
            lstm_model_id=lstm_model.id,
            knn_weight=0.5,
            lstm_weight=0.5,
            agreement_required=True,
            interval=interval,
        )

    knn_model_id = knn_model.id
    lstm_model_id = lstm_model.id

    await _update_stage(job_id, 4, status="running", progress=5, message=f"KNN #{knn_model_id} + LSTM #{lstm_model_id} created. Extracting patterns…")

    # Run distillation using the existing background helper (re-used from models.py)
    from app.api.routes.models import _run_distillation_background  # noqa: PLC0415

    async def _watch_distillation() -> None:
        """Poll the DB status and update pipeline stage progress."""
        from app.db.database import async_session_factory
        from app.db import crud as _crud

        ticks = 0
        while True:
            await asyncio.sleep(5)
            async with async_session_factory() as db:
                knn_m = await _crud.get_knn_model(db, knn_model_id)
            
            if not knn_m or knn_m.status not in ("pending", "training"):
                break
                
            ticks += 1
            pct = min(95, 5 + ticks * 2)
            await _update_stage(job_id, 4, status="running", progress=pct, message="Distillation in progress…")

    distill_task = asyncio.create_task(
        _run_distillation_background(
            knn_model_id=knn_model_id,
            lstm_model_id=lstm_model_id,
            rl_model_id=rl_model_id,
            stock_ids=stock_ids,
            interval=interval,
            k_neighbors=9,
            lstm_hidden_size=256,
            lstm_num_layers=2,
            lstm_dropout=0.3,
            lstm_max_epochs=30,
            min_profit_threshold=0.02,
            profit_horizon=5,
            model_dir=settings.MODEL_DIR,
        )
    )
    watch_task = asyncio.create_task(_watch_distillation())
    await distill_task
    watch_task.cancel()

    # Check final status of the KNN model
    async with async_session_factory() as db:
        knn_final = await _crud.get_knn_model(db, knn_model_id)
        lstm_final = await _crud.get_lstm_model(db, lstm_model_id)

    knn_ok = knn_final and knn_final.status == "completed"
    lstm_ok = lstm_final and lstm_final.status == "completed"

    if knn_ok and lstm_ok:
        acc_knn = round(float(knn_final.accuracy or 0) * 100, 1)
        acc_lstm = round(float(lstm_final.accuracy or 0) * 100, 1)
        await _update_stage(
            job_id, 4, status="completed", progress=100,
            message=f"KNN accuracy {acc_knn}% | LSTM accuracy {acc_lstm}%",
        )
    elif knn_ok or lstm_ok:
        await _update_stage(
            job_id, 4, status="completed", progress=100,
            message="Partial distillation — one model succeeded. Check Model Studio for details.",
        )
    else:
        raise RuntimeError(
            "Ensemble distillation failed — both KNN and LSTM models failed. "
            "Check distillation logs for per-stock details."
        )

    return knn_model_id, lstm_model_id


async def _stage_backtest(job_id: str, rl_model_id: int, knn_model_id: int | None, lstm_model_id: int | None, stock_ids: list[int]) -> None:
    """Run ensemble backtest on a sample of universe stocks and report aggregate metrics."""
    from app.db.database import async_session_factory
    from app.db import crud as _crud
    from datetime import date as _date

    await _update_stage(job_id, 5, status="running", progress=0, message="Running backtest on trained models…")

    # Resolve model paths from DB
    knn_path: str | None = None
    lstm_path: str | None = None
    try:
        if knn_model_id:
            async with async_session_factory() as db:
                knn_rec = await _crud.get_knn_model(db, knn_model_id)
            knn_path = knn_rec.model_path if knn_rec else None
        if lstm_model_id:
            async with async_session_factory() as db:
                lstm_rec = await _crud.get_lstm_model(db, lstm_model_id)
            lstm_path = lstm_rec.model_path if lstm_rec else None
    except Exception as exc:
        logger.warning("Pipeline backtest: could not resolve model paths: %s", exc)

    if not knn_path or not lstm_path:
        await _update_stage(job_id, 5, status="completed", progress=100,
                      message="Backtest skipped — distilled models not available on disk.")
        return

    from app.ml.knn_distiller import load_knn_model, predict_knn, load_knn_norm_params
    from app.ml.lstm_distiller import load_lstm_model, predict_lstm
    from app.ml.ensemble import ensemble_predict
    from app.ml.backtester import BacktestConfig, run_backtest as ml_run_backtest
    from app.core.normalizer import prepare_model_input
    from app.core.data_service import get_model_ready_data
    from app.config import settings
    from app.db.models import BacktestResult as BacktestResultModel
    from datetime import timedelta
    import pandas as pd
    import numpy as np

    try:
        knn_m = load_knn_model(str(knn_path))
        knn_norm_params = load_knn_norm_params(Path(knn_path).parent)
        lstm_m = load_lstm_model(str(lstm_path))
    except Exception as exc:
        await _update_stage(job_id, 5, status="completed", progress=100,
                      message=f"Backtest skipped — model load failed: {exc!s:.80}")
        return

    # Test on up to 5 stocks from the universe
    test_ids = stock_ids[:5]
    seq_len = settings.DEFAULT_SEQ_LEN_DAILY
    end_d = _date.today()
    start_d = _date(2022, 1, 1)
    warmup_start = start_d - timedelta(days=400)

    returns: list[float] = []
    sharpes: list[float] = []
    total_trades_all = 0

    for i, sid in enumerate(test_ids):
        pct = int(10 + i / len(test_ids) * 80)
        await _update_stage(job_id, 5, status="running", progress=pct,
                      message=f"Backtesting stock {i+1}/{len(test_ids)}…")
        try:
            async with async_session_factory() as db:
                df, feat_cols = await get_model_ready_data(
                    db, sid, "day", seq_len=seq_len, start_date=warmup_start, end_date=end_d,
                )

            if df.empty or len(df) < seq_len + 1:
                continue

            X = prepare_model_input(df, feat_cols, seq_len=seq_len)
            if len(X) == 0:
                continue

            knn_preds_raw, knn_probs_raw = predict_knn(knn_m, X, norm_params=knn_norm_params)
            lstm_preds_raw, lstm_probs_raw = predict_lstm(lstm_m, X)

            knn_classes = list(knn_m.classes_)
            if knn_probs_raw.shape[1] != 3:
                expanded = np.zeros((len(knn_probs_raw), 3), dtype=np.float32)
                for col_idx, cls in enumerate(knn_classes):
                    if 0 <= cls <= 2:
                        expanded[:, cls] = knn_probs_raw[:, col_idx]
                knn_probs_raw = expanded
                
            preds_list = ensemble_predict(
                knn_preds_raw, knn_probs_raw,
                lstm_preds_raw, lstm_probs_raw,
                knn_weight=0.5, lstm_weight=0.5,
                agreement_required=True,
            )

            date_col = "date" if "date" in df.columns else None
            all_dates = list(df[date_col]) if date_col else list(df.index)
            # Force strict date objects — DB may return strings or Timestamps
            target_date_set = set(
                pd.to_datetime(d).date() for d in all_dates
                if start_d <= pd.to_datetime(d).date() <= end_d
            )

            predictions: list[dict] = []
            close_arr: list[float] = []
            backtest_dates: list = []
            async with async_session_factory() as db:
                ohlcv_rows = await _crud.get_ohlcv(db, sid, "day", start_d, end_d)

            # Force strict date objects on the map keys as well
            ohlcv_map = {pd.to_datetime(r.date).date(): float(r.close) for r in ohlcv_rows}

            for p_idx, p in enumerate(preds_list):
                date_idx = p_idx + seq_len - 1
                if date_idx >= len(all_dates):
                    break
                d = pd.to_datetime(all_dates[date_idx]).date()
                if d not in target_date_set:
                    continue
                raw_action = p["action"]
                price = ohlcv_map.get(d)
                if price is None or price <= 0:
                    continue  # Skip: missing/zero price would corrupt backtest math
                predictions.append({"action": -1 if raw_action == 2 else raw_action,
                                     "confidence": p["confidence"], "regime_id": None})
                close_arr.append(price)
                backtest_dates.append(d)

            if len(predictions) < 10:
                continue

            bt_cfg = BacktestConfig(initial_capital=100_000.0, stoploss_pct=5.0, min_confidence=0.6)
            bt_res = ml_run_backtest(predictions, np.array(close_arr), backtest_dates, bt_cfg)

            async with async_session_factory() as db:
                rec = BacktestResultModel(
                    model_type="ensemble",
                    model_id=knn_model_id or 0,
                    stock_id=sid,
                    interval="day",
                    start_date=start_d,
                    end_date=end_d,
                    total_return=bt_res.total_return_pct / 100.0,
                    win_rate=bt_res.win_rate / 100.0 if bt_res.win_rate else None,
                    max_drawdown=bt_res.max_drawdown_pct / 100.0 if bt_res.max_drawdown_pct else None,
                    sharpe=bt_res.sharpe_ratio,
                    profit_factor=bt_res.profit_factor if bt_res.profit_factor != float("inf") else 999.0,
                    trades_count=bt_res.total_trades,
                    trade_log=bt_res.trade_log,
                )
                db.add(rec)
                await db.commit()

            returns.append(bt_res.total_return_pct)
            if bt_res.sharpe_ratio is not None:
                sharpes.append(bt_res.sharpe_ratio)
            total_trades_all += bt_res.total_trades

        except Exception as exc:
            logger.warning("Pipeline backtest: stock #%d failed: %s", sid, exc)

    if returns:
        avg_ret = sum(returns) / len(returns)
        avg_sharpe = sum(sharpes) / len(sharpes) if sharpes else 0.0
        await _update_stage(job_id, 5, status="completed", progress=100,
                      message=f"{len(returns)} stocks backtested | Avg return {avg_ret:+.1f}% | Avg Sharpe {avg_sharpe:.2f} | {total_trades_all} trades")
    else:
        await _update_stage(job_id, 5, status="completed", progress=100,
                      message="Backtest could not generate results — check model artifacts.")

async def _stage_deploy(job_id: str, rl_model_id: int, knn_model_id: int, lstm_model_id: int, stock_ids: list[int]) -> None:
    """Stage 6 — Deploy: run predictions + attempt meta-classifier training.

    This stage:
    1. Runs run_daily_predictions so the Live Trading dashboard is populated
       immediately after the pipeline completes (zero manual steps required).
    2. Attempts to train the meta-classifier (XGBoost PoP gate) from historical
       closed TradeSignal outcomes (target_hit / sl_hit).  Requires >= 50 samples;
       skipped if insufficient history exists (first run on a fresh deployment).
    3. Marks the pipeline job stage 6 as "completed".
    """
    from app.db.database import async_session_factory
    from app.ml.predictor import run_daily_predictions

    await _update_stage(job_id, 6, status="running", progress=0, message="Running daily predictions…")

    # ── 1. Trigger predictions ──────────────────────────────────────────────
    try:
        async with async_session_factory() as db:
            pred_result = await run_daily_predictions(
                db,
                stock_ids=stock_ids,
                knn_model_id=knn_model_id,
                lstm_model_id=lstm_model_id,
                interval="day",
                job_id=job_id,
            )
        signal_count = pred_result.get("signals", 0) if isinstance(pred_result, dict) else 0
        logger.info(
            "Pipeline %s: predictions complete — %s signal(s) generated",
            job_id, signal_count,
        )
    except Exception as pred_exc:
        logger.error("Pipeline %s: prediction run failed: %s", job_id, pred_exc, exc_info=True)
        await _update_stage(
            job_id, 6, status="failed", progress=30,
            message=f"Prediction run failed: {pred_exc}",
        )
        raise

    await _update_stage(
        job_id, 6, status="running", progress=40,
        message=f"Predictions done ({signal_count} signal(s)). Training MultiHorizon LSTM…",
    )

    # ── 2. MultiHorizon LSTM training ─────────────────────────────────────
    # The "Train Model" button on the Live Trading page trains this model.
    # By training it here, the user can click "Generate Signal" immediately
    # after the pipeline completes with no manual steps.
    #
    # train_multi_horizon() is a synchronous CPU-bound PyTorch loop — running
    # it inline in an async def would block the event loop for many minutes.
    # Strategy:
    #   a) Async: load stock data and assemble the numpy training arrays (DB I/O)
    #   b) Executor: run the CPU-bound train_multi_horizon() off the event loop
    #   c) Async: save the model file and register it in the DB
    mh_status = "skipped"
    try:
        import numpy as np
        import sqlalchemy as _sa
        from app.db.models import Stock as _Stock, LSTMHorizonModel, ModelStatus as _MS
        from app.core.data_service import get_model_ready_data
        from app.core.normalizer import prepare_model_input
        from app.ml.multi_horizon_lstm import (
            build_training_data as _build_mh_data,
            train_multi_horizon as _train_mh,
            save_multi_horizon_model as _save_mh,
        )

        _seq_len = getattr(settings, "DEFAULT_SEQ_LEN_DAILY", 15)
        _mh_dir = Path(settings.MODEL_DIR) / "lstm"

        # a) Async data assembly ──────────────────────────────────────
        _X_all, _yr_all, _yt_all = [], [], []
        async with async_session_factory() as _db:
            _stocks_res = await _db.execute(
                _sa.select(_Stock).where(_Stock.id.in_(stock_ids), _Stock.is_active == True)  # noqa: E712
            )
            _stocks = _stocks_res.scalars().all()
            logger.info("Pipeline %s MH-LSTM: building data from %d stocks", job_id, len(_stocks))
            
            for _i, _stock in enumerate(_stocks):
                if _i % max(1, len(_stocks) // 10) == 0:
                    await _update_stage(
                        job_id, 6, status="running", progress=40 + int(20 * (_i / len(_stocks))),
                        message=f"Predictions done ({signal_count} signal(s)). Gathering MH-LSTM data ({_i}/{len(_stocks)})...",
                    )
                try:
                    _df, _fcols = await get_model_ready_data(_db, _stock.id, seq_len=_seq_len)
                    if _df is None or len(_df) < _seq_len + 10:
                        continue
                    _close = _df["close"].values.astype("float32")
                    _fmat = _df[_fcols].values.astype("float32")
                    try:
                        _Xs, _yr, _yt = _build_mh_data(_close, _fmat, seq_len=_seq_len)
                        _X_all.append(_Xs)
                        _yr_all.append(_yr)
                        _yt_all.append(_yt)
                    except ValueError:
                        continue
                except Exception as _de:
                    logger.warning("Pipeline %s MH-LSTM skipping %s: %s", job_id, _stock.symbol, _de)

        if not _X_all:
            mh_status = "skipped (no training data available)"
            logger.warning("Pipeline %s MH-LSTM: no training data assembled", job_id)
        else:
            _X_mh = np.concatenate(_X_all, axis=0)
            _y_ret = np.concatenate(_yr_all, axis=0)
            _y_trd = np.concatenate(_yt_all, axis=0)
            logger.info("Pipeline %s MH-LSTM dataset: X=%s", job_id, _X_mh.shape)

            await _update_stage(
                job_id, 6, status="running", progress=60,
                message=f"Training MH-LSTM meta-classifier ({len(_X_mh)} examples)...",
            )

            # b) CPU-bound training in executor ───────────────────────
            _loop = asyncio.get_event_loop()

            def _run_mh_train():
                return _train_mh(
                    _X_mh, _y_ret, _y_trd,
                    hidden_size=256, num_layers=2, dropout=0.3,
                    epochs=60, patience=10,
                    log_fn=lambda m: logger.info("[MH-LSTM] %s", m),
                )

            _mh_model, _mh_metrics = await _loop.run_in_executor(_PIPELINE_EXECUTOR, _run_mh_train)

            # c) Async save + DB registration ─────────────────────────
            _mh_dir.mkdir(parents=True, exist_ok=True)
            _save_res = _save_mh(_mh_model, _mh_metrics, _mh_dir, model_name="mh_lstm")
            _mh_path = _save_res["model_path"]

            async with async_session_factory() as _db2:
                _mh_rec = LSTMHorizonModel(
                    name=f"mh_lstm_{len(_X_mh)}s",
                    hidden_size=256,
                    num_layers=2,
                    seq_len=_seq_len,
                    horizon=_mh_model.horizon,
                    model_path=_mh_path,
                    accuracy=float(1.0 - _mh_metrics.get("best_val_loss", 1.0)),
                    status=_MS.completed,
                )
                _db2.add(_mh_rec)
                await _db2.commit()
                logger.info("Pipeline %s MH-LSTM registered id=%d", job_id, _mh_rec.id)

            mh_status = (
                f"trained (val_loss={_mh_metrics.get('best_val_loss', 0):.4f}, "
                f"n={len(_X_mh)})"
            )

    except Exception as mh_exc:
        mh_status = f"failed ({mh_exc})"
        logger.warning(
            "Pipeline %s: MultiHorizon LSTM training failed (non-fatal): %s",
            job_id, mh_exc, exc_info=True,
        )

    await _update_stage(
        job_id, 6, status="running", progress=75,
        message=f"MH-LSTM {mh_status}. Training meta-classifier…",
    )

    # ── 3. Meta-classifier training ────────────────────────────────────────
    # Build training data from closed TradeSignal history.  sr_features are
    # not stored in the DB so we fill them with 0.0 as a safe default.
    meta_status = "skipped"
    try:
        import numpy as np
        from sqlalchemy import select as sqla_select
        from app.db.models import TradeSignal, SignalStatus
        from app.ml.meta_classifier import MetaClassifier

        async with async_session_factory() as db:
            result = await db.execute(
                sqla_select(TradeSignal).where(
                    TradeSignal.status.in_([SignalStatus.target_hit, SignalStatus.sl_hit]),
                    TradeSignal.confluence_score.isnot(None),
                ).order_by(TradeSignal.signal_date.asc())
            )
            closed_signals = result.scalars().all()

        if len(closed_signals) >= 50:
            rows_X = []
            rows_y = []
            for sig in closed_signals:
                net_return = 0.0
                if sig.initial_rr_ratio and sig.execution_cost_pct is not None:
                    net_return = (
                        (sig.target_price - sig.entry_price) / sig.entry_price
                        - sig.execution_cost_pct
                    ) if sig.entry_price else 0.0
                row = [
                    sig.confluence_score or 0.0,
                    sig.fqs_score or 0.0,
                    sig.execution_cost_pct or 0.0,
                    sig.initial_rr_ratio or 1.0,
                    net_return,
                    sig.lstm_mu or 0.0,
                    sig.lstm_sigma or 0.0,
                    sig.knn_median_return or 0.0,
                    sig.knn_win_rate or 0.0,
                    float(sig.regime_id or 0),
                    # sr_features not persisted — use 0 defaults
                    0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                ]
                rows_X.append(row)
                rows_y.append(1 if sig.status == SignalStatus.target_hit else 0)

            X = np.array(rows_X, dtype=np.float32)
            y = np.array(rows_y, dtype=np.int32)

            mc = MetaClassifier()
            metrics = mc.train(X, y)

            meta_path = Path(settings.MODEL_DIR) / "meta_classifier.joblib"
            mc.save(meta_path)

            meta_status = (
                f"trained (AUC={metrics['auc']:.3f}, "
                f"acc={metrics['accuracy']:.3f}, "
                f"n={metrics['n_train']+metrics['n_val']})"
            )
            logger.info("Pipeline %s: meta-classifier %s", job_id, meta_status)
        else:
            meta_status = (
                f"skipped (only {len(closed_signals)} closed signal(s); "
                "need >= 50 — will auto-train after live trading generates history)"
            )
            logger.info("Pipeline %s: meta-classifier %s", job_id, meta_status)

    except Exception as mc_exc:
        meta_status = f"failed ({mc_exc})"
        logger.warning(
            "Pipeline %s: meta-classifier training failed (non-fatal): %s",
            job_id, mc_exc, exc_info=True,
        )

    await _update_stage(
        job_id, 6, status="completed", progress=100,
        message=(
            f"All models ready for live trading. "
            f"Signals: {signal_count}. MH-LSTM: {mh_status}. Meta-classifier: {meta_status}."
        ),
    )

