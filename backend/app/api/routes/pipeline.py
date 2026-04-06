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
    use_regime_pooling: bool = True


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/start")
async def start_pipeline(req: PipelineStartRequest, background_tasks: BackgroundTasks):
    if not req.symbols:
        raise HTTPException(status_code=422, detail="symbols must not be empty")

    job_id = str(uuid.uuid4())
    async with async_session_factory() as db:
        await crud.create_pipeline_job(
            db,
            job_id=job_id,
            symbols=req.symbols,
            stages=_make_stages(),
            status="queued"
        )

    background_tasks.add_task(_run_pipeline, job_id, req.symbols, req.skip_sync, req.use_regime_pooling)
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
      backtest_results → knn/lstm models (+ files) → golden_patterns →
      rl_training_runs → rl_models (+ files) → cql/bc files → pipeline_job
    """
    from sqlalchemy import delete as sqla_delete, select as sqla_select
    from app.db.models import (
        RLModel, RLTrainingRun, GoldenPattern,
        KNNModel, LSTMModel, BacktestResult, PipelineJob,
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
            # Remove sibling norm-params file if present (KNN pattern)
            norm = p.parent / (p.stem + "_norm_params.pkl")
            if norm.exists():
                norm.unlink(missing_ok=True)
                files_deleted.append(str(norm.name))

    async with async_session_factory() as db:
        # ── Find RL models created by this pipeline ───────────────────────
        # Pipeline tags models via training_config.pipeline_job_id
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

        # ── Delete backtest results tied to these models ───────────────────
        bt_count = 0
        if knn_ids:
            r = await db.execute(
                sqla_delete(BacktestResult).where(
                    BacktestResult.model_id.in_(knn_ids),
                    BacktestResult.model_type == "ensemble",
                )
            )
            bt_count += r.rowcount  # type: ignore[attr-defined]
        if lstm_ids:
            r = await db.execute(
                sqla_delete(BacktestResult).where(
                    BacktestResult.model_id.in_(lstm_ids),
                )
            )
            bt_count += r.rowcount  # type: ignore[attr-defined]
        if pipeline_rl_ids:
            r = await db.execute(
                sqla_delete(BacktestResult).where(
                    BacktestResult.model_id.in_(pipeline_rl_ids),
                    BacktestResult.model_type == "rl",
                )
            )
            bt_count += r.rowcount  # type: ignore[attr-defined]
        records_deleted["backtest_results"] = bt_count

        # ── Delete KNN / LSTM model files + DB rows ───────────────────────
        for m in knn_models:
            _remove_file(m.model_path)
            _remove_file(m.norm_params_path)
        for m in lstm_models:
            _remove_file(m.model_path)

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
                    _remove_file(m.model_path)
            await db.execute(sqla_delete(RLModel).where(RLModel.id.in_(pipeline_rl_ids)))
        records_deleted["rl_models"] = len(pipeline_rl_ids)

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

async def _run_pipeline(job_id: str, symbols: list[str], skip_sync: bool = False, use_regime_pooling: bool = True) -> None:
    async with async_session_factory() as db:
        await crud.update_pipeline_job(db, job_id, status="running")

    try:
        # Resolve stock IDs for the given symbols
        stock_ids = await _resolve_stock_ids(symbols)
        if not stock_ids:
            raise RuntimeError(
                "None of the requested symbols were found in the database. "
                "Run 'Sync Stock List' in Data Manager first."
            )

        # Stage 0 — Data Sync
        if skip_sync:
            await _update_stage(job_id, 0, status="completed", progress=100, message="Data sync skipped per user request.")
        else:
            if await _is_cancelled(job_id):
                raise asyncio.CancelledError()
            await _stage_data_sync(job_id, stock_ids, symbols)

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
        if await _is_cancelled(job_id):
            raise asyncio.CancelledError()
        knn_model_id, lstm_model_id = await _stage_ensemble_distill(job_id, rl_model_id, stock_ids)

        # Stage 5 — Backtest
        if await _is_cancelled(job_id):
            raise asyncio.CancelledError()
        await _stage_backtest(job_id, rl_model_id, knn_model_id, lstm_model_id, stock_ids)

        # Stage 6 — Ready
        await _update_stage(job_id, 6, status="completed", progress=100, message="All models ready for live trading.")
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


# ── Stage implementations ─────────────────────────────────────────────────────

async def _stage_data_sync(job_id: str, stock_ids: list[int], symbols: list[str]) -> None:
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

    # FIX: Moved db session INSIDE the loop to isolate transactions
    for i, stock_id in enumerate(stock_ids):
        sym = symbols[i] if i < len(symbols) else f"#{stock_id}"
        pct = int(i / total * 90)
        await _update_stage(job_id, 0, status="running", progress=pct, message=f"Syncing & Classifying {sym}…")
        
        async with async_session_factory() as db:
            try:
                # Compute and commit immediately per stock
                res = await data_service.sync_and_compute(db, stock_id, interval="day")
                if res.get("ohlcv_synced", 0) >= 0:
                    synced += 1
            except Exception as exc:
                logger.warning("Pipeline data sync: %s failed: %s", sym, exc)

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
    from app.ml.algorithms import ALGORITHM_CONFIGS

    await _update_stage(job_id, 3, status="running", progress=0, message="Creating RL model record…")

    algorithm = "AttentionPPO"
    if algorithm not in ALGORITHM_CONFIGS:
        algorithm = "PPO"  # fallback if AttentionPPO not registered

    hyperparams = dict(ALGORITHM_CONFIGS[algorithm]["defaults"])
    total_timesteps = 100_000  # conservative default for pipeline runs
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
            "Ensemble distillation failed — no golden patterns could be extracted from the RL model."
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

    from app.ml.knn_distiller import load_knn_model, predict_knn
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

            knn_preds_raw, knn_probs_raw = predict_knn(knn_m, X)
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

            if len(predictions) < 10:
                continue

            bt_cfg = BacktestConfig(initial_capital=100_000.0, stoploss_pct=5.0, min_confidence=0.6)
            bt_res = ml_run_backtest(predictions, np.array(close_arr), list(range(len(predictions))), bt_cfg)

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
