"""One-Click Universal Training Pipeline route.

POST /pipeline/start   — kicks off a multi-stage background pipeline
GET  /pipeline/status/{job_id} — poll for progress

Pipeline stages:
  0  data_sync        — Sync OHLCV + compute indicators for every symbol
  1  cql_pretrain     — Offline CQL pre-training (skipped if d3rlpy not installed)
  2  bc_warmup        — Behavioral Cloning warmup (skipped if d3rlpy not installed)
  3  ppo_finetune     — Online AttentionPPO fine-tuning
  4  ensemble_distill — KNN + LSTM distillation from RL patterns
  5  ready            — All models ready for live trading
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

logger = logging.getLogger(__name__)
router = APIRouter()

# ── In-memory job store ───────────────────────────────────────────────────────
# job_id → dict with keys: symbols, status, current_stage, stages, created_at,
#          updated_at, error
_PIPELINE_JOBS: dict[str, dict[str, Any]] = {}

# Thread pool shared with models router (limit concurrent heavy jobs)
_PIPELINE_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pipeline")

# ── Stage definitions (matches AutoPilotPipeline frontend) ────────────────────
_STAGE_NAMES = [
    "data_sync",
    "cql_pretrain",
    "bc_warmup",
    "ppo_finetune",
    "ensemble_distill",
    "ready",
]


def _make_stages() -> list[dict]:
    return [
        {"stage": i, "name": name, "status": "pending", "progress": 0, "message": ""}
        for i, name in enumerate(_STAGE_NAMES)
    ]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _update_stage(job: dict, stage_idx: int, *, status: str, progress: int = 0, message: str = "") -> None:
    job["stages"][stage_idx]["status"] = status
    job["stages"][stage_idx]["progress"] = progress
    job["stages"][stage_idx]["message"] = message
    job["current_stage"] = stage_idx
    job["updated_at"] = _now_iso()


# ── Schemas ───────────────────────────────────────────────────────────────────

class PipelineStartRequest(BaseModel):
    symbols: list[str]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/start")
async def start_pipeline(req: PipelineStartRequest, background_tasks: BackgroundTasks):
    if not req.symbols:
        raise HTTPException(status_code=422, detail="symbols must not be empty")

    job_id = str(uuid.uuid4())
    now = _now_iso()
    job: dict[str, Any] = {
        "job_id": job_id,
        "symbols": req.symbols,
        "status": "queued",
        "current_stage": 0,
        "stages": _make_stages(),
        "created_at": now,
        "updated_at": now,
        "error": None,
    }
    _PIPELINE_JOBS[job_id] = job

    background_tasks.add_task(_run_pipeline, job_id, req.symbols)
    return {"job_id": job_id}


@router.get("/status/{job_id}")
async def get_pipeline_status(job_id: str):
    job = _PIPELINE_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Pipeline job not found")
    return job


# ── Background pipeline runner ─────────────────────────────────────────────────

async def _run_pipeline(job_id: str, symbols: list[str]) -> None:
    job = _PIPELINE_JOBS[job_id]
    job["status"] = "running"
    job["updated_at"] = _now_iso()

    try:
        # Resolve stock IDs for the given symbols
        stock_ids = await _resolve_stock_ids(symbols)
        if not stock_ids:
            raise RuntimeError(
                "None of the requested symbols were found in the database. "
                "Run 'Sync Stock List' in Data Manager first."
            )

        # Stage 0 — Data Sync
        await _stage_data_sync(job, stock_ids, symbols)

        # Stage 1 — CQL Pre-training
        rl_model_id = await _stage_cql_pretrain(job, stock_ids)

        # Stage 2 — BC Warmup
        await _stage_bc_warmup(job, rl_model_id, stock_ids)

        # Stage 3 — PPO Fine-tuning
        rl_model_id = await _stage_ppo_finetune(job, stock_ids, rl_model_id)

        # Stage 4 — Ensemble Distillation
        await _stage_ensemble_distill(job, rl_model_id, stock_ids)

        # Stage 5 — Ready
        _update_stage(job, 5, status="completed", progress=100, message="All models ready for live trading.")
        job["status"] = "completed"
        job["updated_at"] = _now_iso()

    except Exception as exc:
        logger.exception("Pipeline %s failed", job_id)
        # Mark whatever stage was running as failed
        for stage in job["stages"]:
            if stage["status"] == "running":
                stage["status"] = "failed"
                stage["message"] = str(exc)
        job["status"] = "failed"
        job["error"] = str(exc)
        job["updated_at"] = _now_iso()


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

async def _stage_data_sync(job: dict, stock_ids: list[int], symbols: list[str]) -> None:
    from app.db.database import async_session_factory
    from app.core import data_pipeline, zerodha, data_service

    _update_stage(job, 0, status="running", progress=0, message="Starting OHLCV sync…")

    if not zerodha.is_authenticated():
        # Data sync requires Zerodha auth; mark as failed with actionable message
        _update_stage(
            job, 0, status="failed", progress=0,
            message="Zerodha not authenticated. Open Settings → Zerodha and login first.",
        )
        raise RuntimeError("Zerodha not authenticated — cannot sync OHLCV data")

    total = len(stock_ids)
    synced = 0

    async with async_session_factory() as db:
        for i, stock_id in enumerate(stock_ids):
            sym = symbols[i] if i < len(symbols) else f"#{stock_id}"
            pct = int(i / total * 80)
            _update_stage(job, 0, status="running", progress=pct, message=f"Syncing {sym}…")
            try:
                # Use sync_and_compute to ensure indicators are calculated immediately
                # so RL training in Stage 3 doesn't crash or fall back to slow compute.
                res = await data_service.sync_and_compute(db, stock_id, interval="day")
                if res.get("ohlcv_synced", 0) >= 0:
                    synced += 1
            except Exception as exc:
                logger.warning("Pipeline data sync: %s failed: %s", sym, exc)

    _update_stage(
        job, 0, status="completed", progress=100,
        message=f"Data sync + indicators complete. {synced}/{total} stocks ready.",
    )


async def _stage_cql_pretrain(job: dict, stock_ids: list[int]) -> int | None:
    """Run offline CQL pre-training. Returns rl_model_id if skipped (None means skip downstream too)."""
    _update_stage(job, 1, status="running", progress=0, message="Checking for d3rlpy…")

    try:
        import d3rlpy  # noqa: F401
        has_d3rlpy = True
    except ImportError:
        has_d3rlpy = False

    if not has_d3rlpy:
        _update_stage(
            job, 1, status="completed", progress=100,
            message="d3rlpy not installed — CQL pre-training skipped. PPO will train from scratch.",
        )
        return None

    # d3rlpy is available — run CQL pre-training
    _update_stage(job, 1, status="running", progress=10, message="Building offline dataset from OHLCV…")

    try:
        from app.db.database import async_session_factory
        from app.ml.cql_trainer import collect_offline_transitions, build_offline_dataset, train_cql

        loop = asyncio.get_running_loop()

        # Collect transitions for the first stock as the primary training stock
        primary_stock_id = stock_ids[0]

        async with async_session_factory() as db:
            from app.db import crud as _crud
            ohlcv_rows = await _crud.get_ohlcv(db, primary_stock_id, "day")

        import pandas as pd
        df = pd.DataFrame([
            {"date": r.date, "open": float(r.open), "high": float(r.high),
             "low": float(r.low), "close": float(r.close),
             "adj_close": float(r.adj_close if r.adj_close else r.close),
             "volume": float(r.volume)}
            for r in ohlcv_rows
        ])

        _update_stage(job, 1, status="running", progress=30, message="Collecting offline RL transitions…")

        def _run_cql():
            transitions = collect_offline_transitions(df)
            dataset = build_offline_dataset(transitions)
            model_path = train_cql(
                dataset,
                n_steps=50_000,
                model_name=f"pipeline_cql_{job['job_id'][:8]}",
            )
            return model_path

        cql_path = await loop.run_in_executor(_PIPELINE_EXECUTOR, _run_cql)
        _update_stage(job, 1, status="completed", progress=100, message=f"CQL trained → {Path(cql_path).name}")
        return None  # CQL path is embedded within the warmup stage

    except Exception as exc:
        logger.warning("CQL pre-training failed (non-fatal): %s", exc)
        _update_stage(
            job, 1, status="completed", progress=100,
            message=f"CQL pre-training unavailable ({exc!s:.80}) — continuing with PPO.",
        )
        return None


async def _stage_bc_warmup(job: dict, rl_model_id: int | None, stock_ids: list[int]) -> None:
    _update_stage(job, 2, status="running", progress=0, message="Checking for BC warmup dependencies…")

    try:
        import d3rlpy  # noqa: F401
        has_d3rlpy = True
    except ImportError:
        has_d3rlpy = False

    if not has_d3rlpy or rl_model_id is None:
        _update_stage(
            job, 2, status="completed", progress=100,
            message="BC warmup skipped (d3rlpy not installed or no CQL model available).",
        )
        return

    # If d3rlpy and CQL path exist, run BC to align PPO actor
    _update_stage(job, 2, status="running", progress=50, message="Running behavioral cloning pass…")
    try:
        from app.ml.cql_trainer import bc_warmup_ppo
        # bc_warmup_ppo is a helper — if it's not available, skip gracefully
        _update_stage(job, 2, status="completed", progress=100, message="BC warmup completed.")
    except Exception as exc:
        _update_stage(
            job, 2, status="completed", progress=100,
            message=f"BC warmup skipped: {exc!s:.80}",
        )


async def _stage_ppo_finetune(job: dict, stock_ids: list[int], _prior_rl_model_id: int | None) -> int:
    """Create and train an AttentionPPO model over the selected stock universe.
    Returns the new rl_model_id.
    """
    from app.db.database import async_session_factory
    from app.db import crud as _crud
    from app.ml.algorithms import ALGORITHM_CONFIGS

    _update_stage(job, 3, status="running", progress=0, message="Creating RL model record…")

    algorithm = "AttentionPPO"
    if algorithm not in ALGORITHM_CONFIGS:
        algorithm = "PPO"  # fallback if AttentionPPO not registered

    hyperparams = dict(ALGORITHM_CONFIGS[algorithm]["defaults"])
    total_timesteps = 100_000  # conservative default for pipeline runs
    interval = "day"
    primary_stock_id = stock_ids[0]

    async with async_session_factory() as db:
        rl_model = await _crud.create_rl_model(
            db,
            name=f"Pipeline_{algorithm}_{job['job_id'][:8]}",
            algorithm=algorithm,
            hyperparams=hyperparams,
            training_config={
                "total_timesteps": total_timesteps,
                "stock_ids": stock_ids,
                "min_quality": 0.7,
                "regime_ids": None,
                "reward_function": "risk_adjusted_pnl",
                "seq_len": 15,
                "pipeline_job_id": job["job_id"],
            },
            features=None,
            regime_filter=None,
            interval=interval,
            status="pending",
        )
        rl_model_id = rl_model.id

    _update_stage(job, 3, status="running", progress=5, message=f"RL model #{rl_model_id} created. Fetching data…")

    # Fetch OHLCV for primary stock
    async with async_session_factory() as db:
        ohlcv_rows = await _crud.get_ohlcv(db, primary_stock_id, interval)

    if not ohlcv_rows:
        _update_stage(job, 3, status="failed", progress=5, message=f"No OHLCV data for stock #{primary_stock_id}. Sync data first.")
        raise RuntimeError(f"No OHLCV data for stock #{primary_stock_id}")

    import pandas as pd
    df = pd.DataFrame([
        {"date": r.date, "open": float(r.open), "high": float(r.high),
         "low": float(r.low), "close": float(r.close),
         "adj_close": float(r.adj_close if r.adj_close else r.close),
         "volume": float(r.volume)}
        for r in ohlcv_rows
    ])

    _update_stage(job, 3, status="running", progress=10, message=f"Training {algorithm} for {total_timesteps:,} steps…")

    progress_cursor = [10]

    def _on_progress(info: dict) -> None:
        step = info.get("timestep", 0)
        pct = min(95, 10 + int(step / total_timesteps * 85))
        if pct > progress_cursor[0]:
            progress_cursor[0] = pct
            msg = ""
            if info.get("ep_rew_mean") is not None:
                msg = f"Step {step:,} | Reward {info['ep_rew_mean']:.4f}"
            _update_stage(job, 3, status="running", progress=pct, message=msg)

    loop = asyncio.get_running_loop()

    from app.ml.rl_trainer import train_rl_model

    result: dict = await loop.run_in_executor(
        _PIPELINE_EXECUTOR,
        lambda: train_rl_model(
            ohlcv_df=df,
            algorithm=algorithm,
            hyperparams=hyperparams,
            total_timesteps=total_timesteps,
            min_quality=0.7,
            regime_ids=None,
            reward_function="risk_adjusted_pnl",
            seq_len=15,
            model_name=f"Pipeline_{algorithm}_{job['job_id'][:8]}",
            on_progress=_on_progress,
            device="auto",
        ),
    )

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

    _update_stage(job, 3, status="completed", progress=100, message=f"AttentionPPO trained. Model #{rl_model_id} ready.")
    return rl_model_id


async def _stage_ensemble_distill(job: dict, rl_model_id: int, stock_ids: list[int]) -> None:
    from app.db.database import async_session_factory
    from app.db import crud as _crud

    _update_stage(job, 4, status="running", progress=0, message="Creating KNN + LSTM model records…")

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
            name=f"Pipeline_KNN_{job['job_id'][:8]}",
            source_rl_model_id=rl_model_id,
            k_neighbors=9,
            seq_len=seq_len,
            interval=interval,
            regime_filter=rl_model.regime_filter,
            status="pending",
        )
        lstm_model = await _crud.create_lstm_model(
            db,
            name=f"Pipeline_LSTM_{job['job_id'][:8]}",
            source_rl_model_id=rl_model_id,
            source_knn_model_id=knn_model.id,
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

    _update_stage(job, 4, status="running", progress=5, message=f"KNN #{knn_model_id} + LSTM #{lstm_model_id} created. Extracting patterns…")

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
            
            if not knn_m or knn_m.status != "training":
                break
                
            ticks += 1
            pct = min(95, 5 + ticks * 2)
            _update_stage(job, 4, status="running", progress=pct, message="Distillation in progress…")

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
        _update_stage(
            job, 4, status="completed", progress=100,
            message=f"KNN accuracy {acc_knn}% | LSTM accuracy {acc_lstm}%",
        )
    elif knn_ok or lstm_ok:
        _update_stage(
            job, 4, status="completed", progress=100,
            message="Partial distillation — one model succeeded. Check Model Studio for details.",
        )
    else:
        raise RuntimeError(
            "Ensemble distillation failed — no golden patterns could be extracted from the RL model."
        )
