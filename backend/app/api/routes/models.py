from __future__ import annotations

import asyncio
import logging
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, date
from pathlib import Path
from typing import Any
import math

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.config import settings
from app.db import crud
from app.ml.algorithms import list_algorithms, ALGORITHM_CONFIGS

# ── GPU memory helper ─────────────────────────────────────────────────

def _flush_gpu_memory() -> None:
    """Release CUDA cache — safe to call even when CUDA is unavailable."""
    try:
        import gc, torch
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass

# ── Log file directory ─────────────────────────────────────────────────
TRAINING_LOG_DIR = Path(__file__).resolve().parents[3] / "logs" / "training"
TRAINING_LOG_DIR.mkdir(parents=True, exist_ok=True)


def _log_path(model_id: int) -> Path:
    return TRAINING_LOG_DIR / f"model_{model_id}.log"


def _write_log_line(model_id: int, line: str) -> None:
    """Append a single pre-formatted line to the model's log file."""
    try:
        with open(_log_path(model_id), "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def _fmt_log_entry(entry: dict, total_timesteps: int = 0) -> str:
    """Convert a progress dict into a human-readable console line."""
    ts = datetime.now().strftime("%H:%M:%S")
    if "error" in entry:
        return f"[{ts}] ERROR  {entry['error']}"
    if "info" in entry:
        msg = entry["info"]
        if entry.get("stopped"):
            return f"[{ts}] STOPPED  {msg}"
        if entry.get("paused"):
            return f"[{ts}] PAUSED   {msg}"
        if entry.get("completed"):
            return f"[{ts}] DONE   {msg}"
        return f"[{ts}] INFO   {msg}"
    # Progress tick
    step = entry.get("timestep", 0)
    pct = f"{step / total_timesteps * 100:.1f}%" if total_timesteps else ""
    reward = entry.get("ep_rew_mean")
    loss = entry.get("loss")
    fps = entry.get("fps")
    parts = [f"Step {step:,}"]
    if total_timesteps:
        parts[0] += f"/{total_timesteps:,} ({pct})"
    if reward is not None:
        parts.append(f"Reward {reward:.4f}")
    if loss is not None:
        parts.append(f"Loss {loss:.6f}")
    if fps is not None:
        parts.append(f"FPS {fps:.0f}")
    return f"[{ts}] TRAIN  " + " | ".join(parts)


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _safe_float(v) -> float | None:
    if v is None:
        return None
    try:
        val = float(v)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    except (ValueError, TypeError):
        return None

def _get_device_info() -> dict:
    """Detect available compute device. Safe to call at import time."""
    try:
        import torch
        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            return {
                "device": "cuda",
                "gpu_name": torch.cuda.get_device_name(idx),
                "gpu_memory_gb": round(torch.cuda.get_device_properties(idx).total_memory / 1e9, 1),
                "cuda_version": torch.version.cuda,
            }
    except Exception:
        pass
    return {"device": "cpu", "gpu_name": None, "gpu_memory_gb": None, "cuda_version": None}


def _resolve_device(requested: str | None) -> str:
    """Resolve the training device: honour the user's explicit choice, falling back to auto-detect."""
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        # Only use CUDA if actually available
        info = _get_device_info()
        return "cuda" if info["device"] == "cuda" else "cpu"
    # None / "auto" → use whatever is available
    return _get_device_info()["device"]


logger = logging.getLogger(__name__)
router = APIRouter()

# Thread pool for CPU-bound training (max 2 concurrent training jobs)
_training_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="rl_train")

# In-memory training progress store: model_id -> list of progress dicts
_training_progress: dict[int, list[dict]] = {}

# Per-model control events  { model_id: {"stop": Event, "pause": Event} }
_training_controls: dict[int, dict[str, threading.Event]] = {}

# Active distillation runs: knn_model_id -> True while running
_distillation_active: dict[int, bool] = {}

DISTILL_LOG_DIR = TRAINING_LOG_DIR  # same folder, different filename prefix


def _distill_log_path(knn_model_id: int) -> Path:
    return DISTILL_LOG_DIR / f"distill_{knn_model_id}.log"


def _dlog(knn_model_id: int, line: str) -> None:
    """Append a timestamped line to the distillation log file."""
    ts = datetime.now().strftime("%H:%M:%S")
    try:
        with open(_distill_log_path(knn_model_id), "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {line}\n")
    except OSError:
        pass


async def _run_distillation_background(
    knn_model_id: int,
    lstm_model_id: int,
    rl_model_id: int,
    stock_ids: list[int],
    interval: str,
    k_neighbors: int,
    lstm_hidden_size: int,
    lstm_num_layers: int,
    lstm_dropout: float,
    lstm_max_epochs: int,
    min_profit_threshold: float,
    profit_horizon: int,
    model_dir: Path | None = None,
) -> None:
    """Background task: extract RL patterns, train KNN + LSTM, update DB statuses."""
    from app.db.database import async_session_factory

    _distillation_active[knn_model_id] = True

    _dlog(knn_model_id, "=" * 60)
    _dlog(knn_model_id, f"  Distillation  RL#{rl_model_id} → KNN#{knn_model_id} + LSTM#{lstm_model_id}")
    _dlog(knn_model_id, f"  Stocks: {stock_ids}  Interval: {interval}")
    _dlog(knn_model_id, f"  KNN k={k_neighbors}  LSTM hidden={lstm_hidden_size} layers={lstm_num_layers}")
    _dlog(knn_model_id, "=" * 60)

    # ── Mark both models as training ──────────────────────────────────
    try:
        async with async_session_factory() as db:
            await crud.update_knn_model_status(db, knn_model_id, "training")
            await crud.update_lstm_model_status(db, lstm_model_id, "training")
    except Exception as exc:
        _dlog(knn_model_id, f"ERROR  Failed to update status to training: {exc}")
        _distillation_active.pop(knn_model_id, None)
        return

    # ── Load RL model record ──────────────────────────────────────────
    try:
        async with async_session_factory() as db:
            rl_model = await crud.get_rl_model(db, rl_model_id)
        if not rl_model:
            raise ValueError(f"RL model #{rl_model_id} not found")
        if not rl_model.model_path:
            raise ValueError(f"RL model #{rl_model_id} has no saved model file")
    except Exception as exc:
        _dlog(knn_model_id, f"ERROR  {exc}")
        async with async_session_factory() as db:
            await crud.update_knn_model_status(db, knn_model_id, "failed")
            await crud.update_lstm_model_status(db, lstm_model_id, "failed")
        _distillation_active.pop(knn_model_id, None)
        return

    training_config = rl_model.training_config or {}
    seq_len = training_config.get("seq_len", 15)
    min_quality = training_config.get("min_quality", 0.8)
    regime_ids = training_config.get("regime_ids")
    reward_function = training_config.get("reward_function", "risk_adjusted_pnl")
    algorithm = rl_model.algorithm

    # Load obs_mode from metadata.json if available
    obs_mode = "flat"
    try:
        meta_path = Path(rl_model.model_path).parent / "metadata.json"
        if meta_path.exists():
            import json as _json
            with open(meta_path) as f:
                meta = _json.load(f)
            obs_mode = meta.get("obs_mode", "flat")
    except Exception:
        pass

    _dlog(knn_model_id, f"INFO   RL algorithm={algorithm}  seq_len={seq_len}  obs_mode={obs_mode}")

    # ── Fetch OHLCV + extract patterns for each stock ─────────────────
    loop = asyncio.get_event_loop()
    all_patterns: list[dict] = []

    for stock_id in stock_ids:
        _dlog(knn_model_id, f"INFO   Stock #{stock_id}: fetching OHLCV data...")
        try:
            async with async_session_factory() as db:
                ohlcv_rows = await crud.get_ohlcv(db, stock_id, interval)
        except Exception as exc:
            _dlog(knn_model_id, f"ERROR  Stock #{stock_id}: failed to fetch OHLCV: {exc}")
            continue

        if not ohlcv_rows:
            _dlog(knn_model_id, f"ERROR  Stock #{stock_id}: no OHLCV data for interval={interval}")
            continue

        _dlog(knn_model_id, f"INFO   Stock #{stock_id}: {len(ohlcv_rows)} candles loaded. Preparing features...")

        df = pd.DataFrame([
            {
                "date": r.date, "open": float(r.open), "high": float(r.high),
                "low": float(r.low), "close": float(r.close), "volume": float(r.volume),
            }
            for r in ohlcv_rows
        ])

        # Prepare training data (indicators + quality filter + normalization)
        try:
            from app.ml.rl_trainer import prepare_training_data
            df_prep, feature_cols = await loop.run_in_executor(
                _training_executor,
                lambda: prepare_training_data(df, min_quality=min_quality, regime_ids=regime_ids),
            )
        except Exception as exc:
            _dlog(knn_model_id, f"ERROR  Stock #{stock_id}: data preparation failed: {exc}")
            continue

        _dlog(knn_model_id, f"INFO   Stock #{stock_id}: {len(df_prep)} quality rows, {len(feature_cols)} features. Loading RL model...")

        # Load RL model from disk
        try:
            from app.ml.rl_trainer import load_rl_model
            rl_sb3 = await loop.run_in_executor(
                _training_executor,
                lambda: load_rl_model(rl_model.model_path, algorithm),
            )
        except Exception as exc:
            _dlog(knn_model_id, f"ERROR  Stock #{stock_id}: could not load RL model: {exc}")
            continue

        _dlog(knn_model_id, f"INFO   Stock #{stock_id}: replaying RL model to extract golden patterns...")

        # Extract patterns
        try:
            from app.ml.pattern_extractor import extract_patterns
            feature_data = df_prep[feature_cols].values.astype("float32")
            close_prices = df_prep["close"].values.astype("float32")
            dates = df_prep["date"].values if "date" in df_prep.columns else list(range(len(df_prep)))
            regime_ids_arr = df_prep["regime_id"].values.astype(int) if "regime_id" in df_prep.columns else None

            patterns = await loop.run_in_executor(
                _training_executor,
                lambda: extract_patterns(
                    model=rl_sb3,
                    feature_data=feature_data,
                    close_prices=close_prices,
                    dates=dates,
                    regime_ids=regime_ids_arr,
                    seq_len=seq_len,
                    obs_mode=obs_mode,
                    reward_function=reward_function,
                    min_profit_threshold=min_profit_threshold,
                    profit_horizon=profit_horizon,
                    parquet_key=f"{rl_model_id}_{stock_id}_{interval}",
                ),
            )
        except Exception as exc:
            _dlog(knn_model_id, f"ERROR  Stock #{stock_id}: pattern extraction failed: {exc}")
            continue

        # Tag patterns with rl_model_id, stock_id and interval
        for p in patterns:
            p["rl_model_id"] = rl_model_id
            p["stock_id"] = stock_id
            p["interval"] = interval

        _dlog(knn_model_id, f"INFO   Stock #{stock_id}: extracted {len(patterns)} golden patterns (BUY+SELL).")
        all_patterns.extend(patterns)

    if not all_patterns:
        _dlog(knn_model_id, "ERROR  No golden patterns extracted. Check RL model quality and data availability.")
        async with async_session_factory() as db:
            await crud.update_knn_model_status(db, knn_model_id, "failed")
            await crud.update_lstm_model_status(db, lstm_model_id, "failed")
        _distillation_active.pop(knn_model_id, None)
        return

    # ── Save patterns to DB ───────────────────────────────────────────
    _dlog(knn_model_id, f"INFO   Saving {len(all_patterns)} patterns to DB...")
    try:
        async with async_session_factory() as db:
            saved = await crud.bulk_insert_patterns(db, all_patterns)
        _dlog(knn_model_id, f"INFO   {saved} patterns saved.")
    except Exception as exc:
        _dlog(knn_model_id, f"ERROR  Failed to save patterns: {exc}")
        async with async_session_factory() as db:
            await crud.update_knn_model_status(db, knn_model_id, "failed")
            await crud.update_lstm_model_status(db, lstm_model_id, "failed")
        _distillation_active.pop(knn_model_id, None)
        return

    # ── Convert patterns to X, y ──────────────────────────────────────
    _dlog(knn_model_id, "INFO   Building training arrays from patterns...")
    try:
        from app.ml.pattern_extractor import patterns_to_training_data
        # Reload patterns from DB to get all (including prior runs)
        async with async_session_factory() as db:
            db_patterns = await crud.get_patterns_by_rl_model(db, rl_model_id)
        db_patterns_list = [
            {
                "dataset_filepath": p.dataset_filepath,
                "row_index": p.row_index,
                "label": p.label,
            }
            for p in db_patterns
        ]
        X, y = await loop.run_in_executor(
            _training_executor,
            lambda: patterns_to_training_data(db_patterns_list, include_hold=True, seq_len=seq_len),
        )
    except Exception as exc:
        _dlog(knn_model_id, f"ERROR  Failed to build training arrays: {exc}")
        async with async_session_factory() as db:
            await crud.update_knn_model_status(db, knn_model_id, "failed")
            await crud.update_lstm_model_status(db, lstm_model_id, "failed")
        _distillation_active.pop(knn_model_id, None)
        return

    if len(X) == 0:
        _dlog(knn_model_id, "ERROR  Training array is empty after pattern conversion.")
        async with async_session_factory() as db:
            await crud.update_knn_model_status(db, knn_model_id, "failed")
            await crud.update_lstm_model_status(db, lstm_model_id, "failed")
        _distillation_active.pop(knn_model_id, None)
        return

    import numpy as _np
    _uniq, _cnt = _np.unique(y, return_counts=True)
    _dlog(knn_model_id, f"INFO   Training arrays: X={X.shape}, y={y.shape}  Classes: {dict(zip(_uniq.tolist(), _cnt.tolist()))}")

    # Set up save directory (model_dir passed from caller to avoid module-scope resolution issues)
    _base = model_dir if model_dir is not None else settings.MODEL_DIR
    save_dir = Path(_base) / "distill"
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Train KNN ─────────────────────────────────────────────────────
    _dlog(knn_model_id, f"INFO   Training KNN (k={k_neighbors})...")
    try:
        from app.ml.knn_distiller import train_knn, save_knn_model
        knn_model_obj, knn_metrics = await loop.run_in_executor(
            _training_executor,
            lambda: train_knn(X, y, k_neighbors=k_neighbors, log_fn=lambda msg: _dlog(knn_model_id, msg)),
        )
        knn_artifacts = save_knn_model(
            knn_model_obj, knn_metrics,
            save_dir=save_dir,
            model_name=f"knn_{knn_model_id}",
        )
        _dlog(knn_model_id, f"DONE   KNN trained. Accuracy={knn_metrics['accuracy']:.4f}  prec_buy={knn_metrics['precision_buy']:.4f}  prec_sell={knn_metrics['precision_sell']:.4f}")
        async with async_session_factory() as db:
            await crud.update_knn_model_completed(
                db, knn_model_id,
                model_path=knn_artifacts["model_path"],
                norm_params_path=knn_artifacts.get("norm_params_path"),
                accuracy=knn_metrics["accuracy"],
                precision_buy=knn_metrics["precision_buy"],
                precision_sell=knn_metrics["precision_sell"],
            )
    except Exception as exc:
        _dlog(knn_model_id, f"ERROR  KNN training failed: {exc}")
        async with async_session_factory() as db:
            await crud.update_knn_model_status(db, knn_model_id, "failed")

    # ── Train LSTM ────────────────────────────────────────────────────
    _dlog(knn_model_id, f"INFO   Training LSTM (hidden={lstm_hidden_size} layers={lstm_num_layers} max_epochs={lstm_max_epochs})...")
    try:
        from app.ml.lstm_distiller import train_lstm, save_lstm_model

        def _do_lstm_train():
            return train_lstm(
                X, y,
                hidden_size=lstm_hidden_size,
                num_layers=lstm_num_layers,
                dropout=lstm_dropout,
                max_epochs=lstm_max_epochs,
                log_fn=lambda msg: _dlog(knn_model_id, msg),
            )

        lstm_model_obj, lstm_metrics = await loop.run_in_executor(_training_executor, _do_lstm_train)
        lstm_artifacts = save_lstm_model(
            lstm_model_obj, lstm_metrics,
            save_dir=save_dir,
            model_name=f"lstm_{lstm_model_id}",
        )
        _dlog(knn_model_id, f"DONE   LSTM trained. Accuracy={lstm_metrics['accuracy']:.4f}  epochs={lstm_metrics['epochs_trained']}  prec_buy={lstm_metrics['precision_buy']:.4f}  prec_sell={lstm_metrics['precision_sell']:.4f}")
        async with async_session_factory() as db:
            await crud.update_lstm_model_completed(
                db, lstm_model_id,
                model_path=lstm_artifacts["model_path"],
                accuracy=lstm_metrics["accuracy"],
                precision_buy=lstm_metrics["precision_buy"],
                precision_sell=lstm_metrics["precision_sell"],
            )
    except Exception as exc:
        _dlog(knn_model_id, f"ERROR  LSTM training failed: {exc}")
        async with async_session_factory() as db:
            await crud.update_lstm_model_status(db, lstm_model_id, "failed")

    _dlog(knn_model_id, "=" * 60)
    _dlog(knn_model_id, "  Distillation complete.")
    _dlog(knn_model_id, "=" * 60)
    _distillation_active.pop(knn_model_id, None)


async def _run_training_background(
    model_id: int,
    stock_id: int,
    interval: str,
    algorithm: str,
    hyperparams: dict,
    total_timesteps: int,
    min_quality: float,
    regime_ids: list[int] | None,
    reward_function: str,
    seq_len: int,
    model_name: str,
    device: str = "auto",
) -> None:
    """Background task: fetch data, run RL training in thread pool, update DB."""
    from app.db.database import async_session_factory

    _training_progress[model_id] = []

    # Create per-model stop/pause events
    stop_event = threading.Event()
    pause_event = threading.Event()
    _training_controls[model_id] = {"stop": stop_event, "pause": pause_event}

    # Write header to log file
    _write_log_line(model_id, f"{'=' * 60}")
    _write_log_line(model_id, f"  Model #{model_id}  {algorithm}  stock={stock_id}  {interval}  steps={total_timesteps:,}")
    _write_log_line(model_id, f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    _write_log_line(model_id, f"{'=' * 60}")

    # ── Fetch OHLCV data ──────────────────────────────────────────────
    try:
        async with async_session_factory() as db:
            await crud.update_rl_model_status(db, model_id, "training")
            ohlcv_rows = await crud.get_ohlcv(db, stock_id, interval)
    except Exception as exc:
        logger.exception("Failed to fetch data for model %d", model_id)
        _training_progress[model_id].append({"error": str(exc)})
        _write_log_line(model_id, _fmt_log_entry({"error": str(exc)}, total_timesteps))
        async with async_session_factory() as db:
            await crud.update_rl_model_status(db, model_id, "failed")
        return

    # ── GPU verification + memory flush ──────────────────────────────
    if device == "cuda":
        # Clear any stale allocations from previous runs before we start
        _flush_gpu_memory()
        try:
            import torch
            test_tensor = torch.zeros(1, device="cuda")
            del test_tensor
            _flush_gpu_memory()  # toss the test tensor immediately
            gpu_info = _get_device_info()
            mem_free = round(gpu_info.get("gpu_memory_gb", 0), 1)
            gpu_msg = f"GPU verified: {gpu_info.get('gpu_name', 'CUDA')} ({mem_free} GB · CUDA {gpu_info.get('cuda_version')}) — memory cleared"
            _training_progress[model_id].append({"info": gpu_msg, "gpu_ok": True})
            _write_log_line(model_id, _fmt_log_entry({"info": gpu_msg}, total_timesteps))
            logger.info(gpu_msg)
        except Exception as cuda_err:
            err_msg = f"GPU requested but CUDA init failed: {cuda_err} — falling back to CPU"
            _training_progress[model_id].append({"info": err_msg, "gpu_ok": False})
            _write_log_line(model_id, _fmt_log_entry({"info": err_msg}, total_timesteps))
            logger.warning(err_msg)
            device = "cpu"
    else:
        _training_progress[model_id].append({"info": f"Using device: {device}"})
        _write_log_line(model_id, _fmt_log_entry({"info": f"Using device: {device}"}, total_timesteps))

    if not ohlcv_rows:
        msg = f"No OHLCV data for stock {stock_id} / {interval}"
        logger.error(msg)
        _training_progress[model_id].append({"error": msg})
        _write_log_line(model_id, _fmt_log_entry({"error": msg}, total_timesteps))
        async with async_session_factory() as db:
            await crud.update_rl_model_status(db, model_id, "failed")
        return

    df = pd.DataFrame(
        [
            {
                "date": r.date,
                "open": float(r.open),
                "high": float(r.high),
                "low": float(r.low),
                "close": float(r.close),
                "volume": float(r.volume),
            }
            for r in ohlcv_rows
        ]
    )

    # ── Progress callback (thread-safe dict writes) ───────────────────
    def _on_progress(info: dict) -> None:
        _training_progress[model_id].append(info)
        _write_log_line(model_id, _fmt_log_entry(info, total_timesteps))

    # ── Run CPU-bound training in thread pool ─────────────────────────
    loop = asyncio.get_event_loop()
    try:
        from app.ml.rl_trainer import train_rl_model  # lazy: requires stable_baselines3
        result: dict = await loop.run_in_executor(
            _training_executor,
            lambda: train_rl_model(
                ohlcv_df=df,
                algorithm=algorithm,
                hyperparams=hyperparams,
                total_timesteps=total_timesteps,
                min_quality=min_quality,
                regime_ids=regime_ids,
                reward_function=reward_function,
                seq_len=seq_len,
                model_name=model_name,
                on_progress=_on_progress,
                device=device,
                stop_event=stop_event,
                pause_event=pause_event,
            ),
        )
    except Exception as exc:
        logger.exception("Training failed for model %d", model_id)
        _training_progress[model_id].append({"error": str(exc)})
        _write_log_line(model_id, _fmt_log_entry({"error": str(exc)}, total_timesteps))
        async with async_session_factory() as db:
            await crud.update_rl_model_status(db, model_id, "failed")
        _training_controls.pop(model_id, None)
        return

    # ── Persist results to DB ─────────────────────────────────────────
    # If stopped by user, mark as "stopped" rather than completed
    final_status = "stopped" if stop_event.is_set() else "completed"
    _write_log_line(model_id, f"[{datetime.now().strftime('%H:%M:%S')}] {'STOPPED' if final_status == 'stopped' else 'DONE   '}  Training {final_status}. Reward={result.get('final_reward')}")

    # ── Step 1: Update model status immediately so the UI stops showing "training" ──
    try:
        async with async_session_factory() as db:
            try:
                if final_status == "stopped":
                    await crud.update_rl_model_status(db, model_id, "stopped")
                else:
                    await crud.update_rl_model_completed(
                        db,
                        model_id=model_id,
                        total_reward=_safe_float(result.get("final_reward")),
                        sharpe_ratio=_safe_float(result.get("best_reward")),
                        model_path=result.get("model_path"),
                    )
            except Exception as inner_exc:
                logger.exception("Failed to update model completed metrics for model %d, marking as failed", model_id)
                await db.rollback()
                await crud.update_rl_model_status(db, model_id, "failed")
    except Exception as exc:
        logger.exception("Failed to update model status for model %d", model_id)

    # Remove from live progress immediately after status update so the UI reflects completion
    _training_progress.pop(model_id, None)
    _training_controls.pop(model_id, None)
    logger.info("Training completed for model %d", model_id)

    # ── Step 2: Save training run log entries (slow; does not affect UI state) ──
    try:
        async with async_session_factory() as db:
            entries = [
                {
                    "rl_model_id": model_id,
                    "timestep": entry.get("timestep", 0),
                    "reward": _safe_float(entry.get("ep_rew_mean")),
                    "loss": _safe_float(entry.get("loss")),
                    "metrics": entry,
                }
                for entry in result.get("training_log", [])
            ]
            await crud.bulk_save_training_runs(db, entries)
    except Exception as exc:
        logger.exception("Failed to persist training logs for model %d", model_id)


# ── Schemas ────────────────────────────────────────────────────────────

class AlgorithmInfo(BaseModel):
    name: str
    policy: str
    obs_mode: str
    continuous: bool
    defaults: dict


class TrainRequest(BaseModel):
    stock_ids: list[int]
    algorithm: str = "PPO"
    hyperparams: dict | None = None
    total_timesteps: int = 100_000
    interval: str = "day"
    reward_function: str = "risk_adjusted_pnl"
    seq_len: int = 15
    min_quality: float = 0.8
    regime_ids: list[int] | None = None
    model_name: str | None = None
    device: str | None = None  # "cuda" | "cpu" | None (auto)


class DistillRequest(BaseModel):
    rl_model_id: int
    stock_ids: list[int] | None = None
    interval: str = "day"
    k_neighbors: int = 5
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.3
    lstm_max_epochs: int = 100
    min_profit_threshold: float = 1.2
    profit_horizon: int = 1
    knn_weight: float = 0.5
    lstm_weight: float = 0.5
    agreement_required: bool = True
    knn_name: str | None = None
    lstm_name: str | None = None
    ensemble_name: str | None = None


class RLModelOut(BaseModel):
    id: int
    name: str
    algorithm: str
    status: str
    interval: str | None = None
    total_reward: float | None = None
    sharpe_ratio: float | None = None
    model_path: str | None = None
    training_config: dict | None = None

    class Config:
        from_attributes = True


class KNNModelOut(BaseModel):
    id: int
    name: str
    source_rl_model_id: int
    k_neighbors: int
    accuracy: float | None = None
    precision_buy: float | None = None
    precision_sell: float | None = None
    status: str

    class Config:
        from_attributes = True


class LSTMModelOut(BaseModel):
    id: int
    name: str
    source_rl_model_id: int
    hidden_size: int
    num_layers: int
    accuracy: float | None = None
    precision_buy: float | None = None
    precision_sell: float | None = None
    status: str

    class Config:
        from_attributes = True


class EnsembleConfigOut(BaseModel):
    id: int
    name: str
    knn_model_id: int
    lstm_model_id: int
    knn_weight: float
    lstm_weight: float
    agreement_required: bool
    interval: str
    backtest_accuracy: float | None = None

    class Config:
        from_attributes = True


class EnsembleUpdateRequest(BaseModel):
    knn_weight: float
    lstm_weight: float
    agreement_required: bool


# ── Endpoints ──────────────────────────────────────────────────────────

@router.get("/algorithms", response_model=list[AlgorithmInfo])
async def get_algorithms():
    """List available RL algorithms and their configs."""
    return list_algorithms()


@router.get("/device")
async def get_device():
    """Return the compute device available for training (cuda or cpu)."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _get_device_info)


@router.get("/rl", response_model=list[RLModelOut])
async def list_rl_models(db: AsyncSession = Depends(get_db)):
    """List all RL models."""
    models = await crud.list_rl_models(db)
    return models


@router.get("/rl/{model_id}", response_model=RLModelOut)
async def get_rl_model(model_id: int, db: AsyncSession = Depends(get_db)):
    """Get RL model details."""
    model = await crud.get_rl_model(db, model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model


@router.post("/train")
async def train_model(
    req: TrainRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """Trigger RL model training (runs in background)."""
    if req.algorithm not in ALGORITHM_CONFIGS:
        raise HTTPException(status_code=400, detail=f"Unknown algorithm: {req.algorithm}")

    # Resolve device before creating the DB record so we can reject bad GPU requests early
    resolved_device = _resolve_device(req.device)
    if resolved_device == "cuda":
        try:
            import torch
            t = torch.zeros(1, device="cuda")
            del t
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"GPU selected but CUDA is not usable: {e}. Switch to CPU or check your PyTorch install.",
            )

    # Create model record
    model_name = req.model_name or f"{req.algorithm}_{req.interval}"
    rl_model = await crud.create_rl_model(
        db,
        name=model_name,
        algorithm=req.algorithm,
        hyperparams=req.hyperparams or ALGORITHM_CONFIGS[req.algorithm]["defaults"],
        training_config={
            "total_timesteps": req.total_timesteps,
            "stock_ids": req.stock_ids,
            "min_quality": req.min_quality,
            "regime_ids": req.regime_ids,
            "reward_function": req.reward_function,
            "seq_len": req.seq_len,
        },
        features=None,
        regime_filter={"regime_ids": req.regime_ids} if req.regime_ids else None,
        interval=req.interval,
        status="pending",
    )

    # Launch training in background
    background_tasks.add_task(
        _run_training_background,
        model_id=rl_model.id,
        stock_id=req.stock_ids[0],
        interval=req.interval,
        algorithm=req.algorithm,
        hyperparams=req.hyperparams or ALGORITHM_CONFIGS[req.algorithm]["defaults"],
        total_timesteps=req.total_timesteps,
        min_quality=req.min_quality,
        regime_ids=req.regime_ids,
        reward_function=req.reward_function,
        seq_len=req.seq_len,
        model_name=model_name,
        device=resolved_device,
    )

    return {
        "model_id": rl_model.id,
        "status": "pending",
        "message": f"RL model '{model_name}' training started in background.",
    }


@router.post("/distill")
async def distill_model(
    req: DistillRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """Trigger KNN + LSTM distillation from RL patterns."""
    rl_model = await crud.get_rl_model(db, req.rl_model_id)
    if not rl_model:
        raise HTTPException(status_code=404, detail="RL model not found")

    # Derive stock_ids from the RL model's training config if not provided
    stock_ids = req.stock_ids
    if not stock_ids and rl_model.training_config:
        stock_ids = rl_model.training_config.get("stock_ids") or []
    if not stock_ids:
        raise HTTPException(status_code=422, detail="stock_ids could not be determined from the RL model")

    # Create KNN model record
    knn_model = await crud.create_knn_model(
        db,
        name=req.knn_name or f"KNN_from_{rl_model.name}",
        source_rl_model_id=rl_model.id,
        k_neighbors=req.k_neighbors,
        seq_len=rl_model.training_config.get("seq_len", 15) if rl_model.training_config else 15,
        interval=req.interval,
        regime_filter=rl_model.regime_filter,
        status="pending",
    )

    # Create LSTM model record
    lstm_model = await crud.create_lstm_model(
        db,
        name=req.lstm_name or f"LSTM_from_{rl_model.name}",
        source_rl_model_id=rl_model.id,
        hidden_size=req.lstm_hidden_size,
        num_layers=req.lstm_num_layers,
        dropout=req.lstm_dropout,
        seq_len=rl_model.training_config.get("seq_len", 15) if rl_model.training_config else 15,
        interval=req.interval,
        regime_filter=rl_model.regime_filter,
        status="pending",
    )

    # Create ensemble config
    ensemble = await crud.create_ensemble_config(
        db,
        name=req.ensemble_name or f"Ensemble_{rl_model.name}",
        knn_model_id=knn_model.id,
        lstm_model_id=lstm_model.id,
        knn_weight=req.knn_weight,
        lstm_weight=req.lstm_weight,
        agreement_required=req.agreement_required,
        interval=req.interval,
    )

    # Launch distillation in background
    background_tasks.add_task(
        _run_distillation_background,
        knn_model_id=knn_model.id,
        lstm_model_id=lstm_model.id,
        rl_model_id=rl_model.id,
        stock_ids=stock_ids,
        interval=req.interval,
        k_neighbors=req.k_neighbors,
        lstm_hidden_size=req.lstm_hidden_size,
        lstm_num_layers=req.lstm_num_layers,
        lstm_dropout=req.lstm_dropout,
        lstm_max_epochs=req.lstm_max_epochs,
        min_profit_threshold=req.min_profit_threshold,
        profit_horizon=req.profit_horizon,
        model_dir=settings.MODEL_DIR,  # resolve eagerly to avoid NameError in background task
    )

    return {
        "rl_model_id": rl_model.id,
        "knn_model_id": knn_model.id,
        "lstm_model_id": lstm_model.id,
        "ensemble_config_id": ensemble.id,
        "status": "pending",
        "message": "Distillation started in background. Watch the log for progress.",
    }


@router.get("/knn/{knn_model_id}/log")
async def get_distill_log(knn_model_id: int):
    """Return the distillation log for a KNN model run."""
    p = _distill_log_path(knn_model_id)
    content = ""
    if p.exists():
        content = p.read_text(encoding="utf-8", errors="replace")
    elif knn_model_id not in _distillation_active:
        raise HTTPException(status_code=404, detail="Distillation log not found.")
    size = p.stat().st_size if p.exists() else len(content.encode())
    return {
        "knn_model_id": knn_model_id,
        "content": content,
        "size_bytes": size,
        "is_active": knn_model_id in _distillation_active,
    }



async def list_training_log_files():
    """List all training log files with sizes."""
    files = []
    for p in sorted(TRAINING_LOG_DIR.glob("model_*.log")):
        try:
            stat = p.stat()
            model_id = int(p.stem.split("_")[1])
            files.append({
                "model_id": model_id,
                "filename": p.name,
                "size_bytes": stat.st_size,
                "size_human": _human_size(stat.st_size),
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })
        except (ValueError, OSError):
            continue
    return files


@router.get("/rl/{model_id}/log-file")
async def get_training_log_file(model_id: int, db: AsyncSession = Depends(get_db)):
    """Return the full content of a model's training log file."""
    p = _log_path(model_id)
    model = await crud.get_rl_model(db, model_id)
    status = model.status.value if model and hasattr(model.status, "value") else getattr(model, "status", None)
    # While training is active, also append any in-memory entries not yet flushed
    content = ""
    if p.exists():
        content = p.read_text(encoding="utf-8", errors="replace")
    elif model_id in _training_progress:
        # No file yet but in-memory data exists — build content on the fly
        lines = [_fmt_log_entry(e) for e in _training_progress[model_id]]
        content = "\n".join(lines)
    else:
        # Instead of 404, return empty if it's potentially starting
        content = ""
    
    size = p.stat().st_size if p.exists() else len(content.encode())
    return {
        "model_id": model_id,
        "content": content,
        "size_bytes": size,
        "size_human": _human_size(size),
        "is_active": model_id in _training_controls or model_id in _training_progress,
        "status": status,
    }


@router.delete("/rl/{model_id}/log-file")
async def delete_training_log_file(model_id: int):
    """Delete the log file for a model."""
    p = _log_path(model_id)
    if not p.exists():
        raise HTTPException(status_code=404, detail="Log file not found.")
    p.unlink()
    return {"message": "Log file deleted."}


@router.get("/training-logs/total-size")
async def training_logs_total_size():
    """Return total size of all training log files."""
    total = sum(p.stat().st_size for p in TRAINING_LOG_DIR.glob("model_*.log") if p.exists())
    return {"total_bytes": total, "total_human": _human_size(total)}


@router.delete("/training-logs")
async def delete_all_training_logs():
    """Delete all training log files."""
    count = 0
    for p in TRAINING_LOG_DIR.glob("model_*.log"):
        try:
            p.unlink()
            count += 1
        except OSError:
            pass
    return {"deleted": count, "message": f"Deleted {count} log file(s)."}


@router.get("/rl/{model_id}/logs")
async def get_training_logs(model_id: int, db: AsyncSession = Depends(get_db)):
    """Get training progress logs for a model (in-memory while training, DB runs after)."""
    model = await crud.get_rl_model(db, model_id)
    status = model.status.value if model and hasattr(model.status, "value") else getattr(model, "status", None)
    # Return in-memory progress if training is active
    if model_id in _training_progress:
        return {
            "source": "live",
            "logs": _training_progress[model_id],
            "is_active": model_id in _training_controls,
            "status": status,
        }

    # Fall back to DB training runs
    runs = await crud.list_training_runs(db, model_id)
    db_logs = [
        {
            "timestep": r.timestep,
            "reward": r.reward,
            "loss": r.loss,
            "metrics": r.metrics or {},
        }
        for r in runs
    ]
    return {
        "source": "db",
        "logs": db_logs,
        "is_active": model_id in _training_controls,
        "status": status,
    }


@router.post("/rl/{model_id}/stop")
async def stop_training(model_id: int, db: AsyncSession = Depends(get_db)):
    """Signal the training loop to stop."""
    if model_id in _training_controls:
        _training_controls[model_id]["stop"].set()
        # Also mark DB immediately so UI reflects the stop even if the thread
        # is stuck or died before it could update the status itself.
        model = await crud.get_rl_model(db, model_id)
        if model and model.status in ("pending", "training", "paused"):
            await crud.update_rl_model_status(db, model_id, "stopped")
        _training_controls.pop(model_id, None)
        # Schedule GPU memory cleanup on a background thread so we don't block the response
        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, _flush_gpu_memory)
        return {"message": "Stop signal sent."}
    # Not currently training — just mark stopped if pending/training
    model = await crud.get_rl_model(db, model_id)
    if model and model.status in ("pending", "training", "paused"):
        await crud.update_rl_model_status(db, model_id, "stopped")
    return {"message": "Model was not training; status updated."}


@router.post("/rl/{model_id}/pause")
async def pause_training(model_id: int, db: AsyncSession = Depends(get_db)):
    """Signal the training loop to pause."""
    if model_id not in _training_controls:
        raise HTTPException(status_code=400, detail="Model is not currently training.")
    _training_controls[model_id]["pause"].set()
    await crud.update_rl_model_status(db, model_id, "paused")
    return {"message": "Pause signal sent."}


@router.post("/rl/{model_id}/resume")
async def resume_training(model_id: int, db: AsyncSession = Depends(get_db)):
    """Clear the pause signal so training resumes."""
    if model_id not in _training_controls:
        raise HTTPException(status_code=400, detail="Model is not currently training.")
    _training_controls[model_id]["pause"].clear()
    await crud.update_rl_model_status(db, model_id, "training")
    return {"message": "Resume signal sent."}


# ── Shared deletion helpers ────────────────────────────────────────────

async def _cascade_delete_ensemble(db: AsyncSession, config_id: int) -> None:
    """Delete one EnsembleConfig and all its child rows, preserving trade_order history."""
    from sqlalchemy import select, delete, update
    from app.db.models import EnsembleConfig, EnsemblePrediction, TradeOrder

    # Null-out the FK on any trade orders that reference these predictions
    pred_ids_result = await db.execute(
        select(EnsemblePrediction.id).where(EnsemblePrediction.ensemble_config_id == config_id)
    )
    pred_ids = [r[0] for r in pred_ids_result]
    if pred_ids:
        await db.execute(
            update(TradeOrder)
            .where(TradeOrder.ensemble_prediction_id.in_(pred_ids))
            .values(ensemble_prediction_id=None)
        )

    await db.execute(
        delete(EnsemblePrediction).where(EnsemblePrediction.ensemble_config_id == config_id)
    )
    await db.execute(delete(EnsembleConfig).where(EnsembleConfig.id == config_id))


@router.delete("/rl/{model_id}")
async def delete_rl_model(model_id: int, db: AsyncSession = Depends(get_db)):
    """Stop training (if active) then cascade-delete the full RL model tree."""
    from sqlalchemy import select, delete
    from app.db.models import KNNModel, LSTMModel, KNNPrediction, LSTMPrediction

    # Stop training if running
    if model_id in _training_controls:
        _training_controls[model_id]["stop"].set()
    _training_progress.pop(model_id, None)
    _training_controls.pop(model_id, None)

    model = await crud.get_rl_model(db, model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found.")

    artifact_dir = Path(model.model_path).parent if model.model_path else None

    # ── Cascade through KNN children ────────────────────────────────
    knn_rows = (await db.execute(
        select(KNNModel).where(KNNModel.source_rl_model_id == model_id)
    )).scalars().all()
    for knn in knn_rows:
        from sqlalchemy import select as sa_select
        from app.db.models import EnsembleConfig as EC
        cfg_result = await db.execute(sa_select(EC.id).where(EC.knn_model_id == knn.id))
        for (cfg_id,) in cfg_result:
            await _cascade_delete_ensemble(db, cfg_id)
        await db.execute(delete(KNNPrediction).where(KNNPrediction.knn_model_id == knn.id))
        await db.execute(delete(KNNModel).where(KNNModel.id == knn.id))
        knn_artifact = Path(knn.model_path).parent if knn.model_path else None
        if knn_artifact and knn_artifact.exists():
            shutil.rmtree(knn_artifact, ignore_errors=True)
        _distill_log_path(knn.id).unlink(missing_ok=True)

    # ── Cascade through LSTM children ───────────────────────────────
    lstm_rows = (await db.execute(
        select(LSTMModel).where(LSTMModel.source_rl_model_id == model_id)
    )).scalars().all()
    for lstm in lstm_rows:
        from app.db.models import EnsembleConfig as EC2
        cfg_result2 = await db.execute(sa_select(EC2.id).where(EC2.lstm_model_id == lstm.id))
        for (cfg_id,) in cfg_result2:
            await _cascade_delete_ensemble(db, cfg_id)
        await db.execute(delete(LSTMPrediction).where(LSTMPrediction.lstm_model_id == lstm.id))
        await db.execute(delete(LSTMModel).where(LSTMModel.id == lstm.id))
        lstm_artifact = Path(lstm.model_path).parent if lstm.model_path else None
        if lstm_artifact and lstm_artifact.exists():
            shutil.rmtree(lstm_artifact, ignore_errors=True)

    # ── Delete RL model itself (golden_patterns + training_runs inside) ─
    # Commit cascades above first so FK constraints are satisfied
    await db.commit()
    await crud.delete_rl_model(db, model_id)

    # ── Disk cleanup ────────────────────────────────────────────────
    if artifact_dir and artifact_dir.exists():
        shutil.rmtree(artifact_dir, ignore_errors=True)
    _log_path(model_id).unlink(missing_ok=True)

    return {"message": "Model deleted."}


@router.get("/knn", response_model=list[KNNModelOut])
async def list_knn_models(db: AsyncSession = Depends(get_db)):
    """List all KNN models."""
    from sqlalchemy import select
    from app.db.models import KNNModel
    result = await db.execute(select(KNNModel).order_by(KNNModel.created_at.desc()))
    return result.scalars().all()


@router.delete("/knn/{model_id}")
async def delete_knn_model(model_id: int, db: AsyncSession = Depends(get_db)):
    """Cascade-delete a KNN model and all its dependent data."""
    from sqlalchemy import select, delete
    from app.db.models import KNNModel, EnsembleConfig, KNNPrediction

    result = await db.execute(select(KNNModel).where(KNNModel.id == model_id))
    model = result.scalar_one_or_none()
    if not model:
        raise HTTPException(status_code=404, detail="KNN model not found")

    # Cascade through ensemble configs (predictions → trade_order nulling → config)
    cfg_result = await db.execute(select(EnsembleConfig.id).where(EnsembleConfig.knn_model_id == model_id))
    for (cfg_id,) in cfg_result:
        await _cascade_delete_ensemble(db, cfg_id)

    await db.execute(delete(KNNPrediction).where(KNNPrediction.knn_model_id == model_id))
    await db.execute(delete(KNNModel).where(KNNModel.id == model_id))
    await db.commit()

    # Disk cleanup — remove entire artifact directory
    if model.model_path:
        artifact_dir = Path(model.model_path).parent
        if artifact_dir.exists():
            shutil.rmtree(artifact_dir, ignore_errors=True)
    _distill_log_path(model_id).unlink(missing_ok=True)

    return {"message": "KNN model deleted"}


@router.get("/lstm", response_model=list[LSTMModelOut])
async def list_lstm_models(db: AsyncSession = Depends(get_db)):
    """List all LSTM models."""
    from sqlalchemy import select
    from app.db.models import LSTMModel
    result = await db.execute(select(LSTMModel).order_by(LSTMModel.created_at.desc()))
    return result.scalars().all()


@router.delete("/lstm/{model_id}")
async def delete_lstm_model(model_id: int, db: AsyncSession = Depends(get_db)):
    """Cascade-delete an LSTM model and all its dependent data."""
    from sqlalchemy import select, delete
    from app.db.models import LSTMModel, EnsembleConfig, LSTMPrediction

    result = await db.execute(select(LSTMModel).where(LSTMModel.id == model_id))
    model = result.scalar_one_or_none()
    if not model:
        raise HTTPException(status_code=404, detail="LSTM model not found")

    # Cascade through ensemble configs
    cfg_result = await db.execute(select(EnsembleConfig.id).where(EnsembleConfig.lstm_model_id == model_id))
    for (cfg_id,) in cfg_result:
        await _cascade_delete_ensemble(db, cfg_id)

    await db.execute(delete(LSTMPrediction).where(LSTMPrediction.lstm_model_id == model_id))
    await db.execute(delete(LSTMModel).where(LSTMModel.id == model_id))
    await db.commit()

    # Disk cleanup — remove entire artifact directory
    if model.model_path:
        artifact_dir = Path(model.model_path).parent
        if artifact_dir.exists():
            shutil.rmtree(artifact_dir, ignore_errors=True)

    return {"message": "LSTM model deleted"}


@router.get("/ensemble", response_model=list[EnsembleConfigOut])
async def list_ensemble_configs(db: AsyncSession = Depends(get_db)):
    """List all ensemble configurations."""
    from sqlalchemy import select
    from app.db.models import EnsembleConfig
    result = await db.execute(select(EnsembleConfig).order_by(EnsembleConfig.created_at.desc()))
    return result.scalars().all()


@router.put("/ensemble/{config_id}")
async def update_ensemble_config(
    config_id: int,
    body: EnsembleUpdateRequest,
    db: AsyncSession = Depends(get_db),
):
    """Update ensemble weights and agreement flag."""
    from sqlalchemy import select, update
    from app.db.models import EnsembleConfig
    result = await db.execute(select(EnsembleConfig).where(EnsembleConfig.id == config_id))
    cfg = result.scalar_one_or_none()
    if not cfg:
        raise HTTPException(status_code=404, detail="Ensemble config not found.")
    await db.execute(
        update(EnsembleConfig).where(EnsembleConfig.id == config_id).values(
            knn_weight=body.knn_weight,
            lstm_weight=body.lstm_weight,
            agreement_required=body.agreement_required,
        )
    )
    await db.commit()
    return {"message": "Updated."}


@router.delete("/ensemble/{config_id}")
async def delete_ensemble_config(config_id: int, db: AsyncSession = Depends(get_db)):
    """Cascade-delete an ensemble config and all its predictions."""
    from sqlalchemy import select
    from app.db.models import EnsembleConfig
    result = await db.execute(select(EnsembleConfig).where(EnsembleConfig.id == config_id))
    cfg = result.scalar_one_or_none()
    if not cfg:
        raise HTTPException(status_code=404, detail="Ensemble config not found.")
    await _cascade_delete_ensemble(db, config_id)
    await db.commit()
    return {"message": "Deleted."}


@router.get("/")
async def list_all_models(db: AsyncSession = Depends(get_db)):
    """List all model types."""
    rl_models = await crud.list_rl_models(db)
    from sqlalchemy import select
    from app.db.models import KNNModel, LSTMModel, EnsembleConfig
    knn_result = await db.execute(select(KNNModel))
    lstm_result = await db.execute(select(LSTMModel))
    ensemble_result = await db.execute(select(EnsembleConfig))

    return {
        "rl_models": len(list(rl_models)),
        "knn_models": len(knn_result.scalars().all()),
        "lstm_models": len(lstm_result.scalars().all()),
        "ensembles": len(ensemble_result.scalars().all()),
    }
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.db.models import GoldenPattern

@router.get("/patterns")
async def get_golden_patterns(
    rl_model_id: int | None = Query(None),
    limit: int = 50,
    db: AsyncSession = Depends(get_db)
):
    q = select(GoldenPattern).order_by(GoldenPattern.id.desc())
    if rl_model_id:
        q = q.where(GoldenPattern.rl_model_id == rl_model_id)
    q = q.limit(limit)
    res = await db.execute(q)
    patterns = res.scalars().all()
    
    out = []
    for p in patterns:
        out.append({
            "id": p.id,
            "rl_model_id": p.rl_model_id,
            "stock_id": p.stock_id,
            "date": p.date,
            "label": p.label,
            "pnl_percent": p.pnl_percent,
            "confidence": p.confidence,
            "interval": p.interval,
            "regime_id": p.regime_id
        })
    return out
