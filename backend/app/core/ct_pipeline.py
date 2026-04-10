"""Continuous Training (CT) Pipeline — rolling window KNN + LSTM retraining.

Trains on GoldenPatterns captured within the last `lookback_years` years.
New models are *staged* (inserted into DB) but do NOT replace the active
EnsembleConfig — the operator activates them manually after reviewing accuracy.

Typical call-site: POST /api/training/auto-retrain (monthly recommended).
"""
from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from pathlib import Path
from typing import Callable

import numpy as np
from dateutil.relativedelta import relativedelta
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db import crud
from app.db.models import GoldenPattern, IntervalEnum
from app.db.database import async_session_factory

logger = logging.getLogger(__name__)

# Dedicated single-worker executor so CT training never starves the event loop
_ct_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ct_train")


def _build_training_arrays(
    patterns: list[dict],
    seq_len: int = 15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert pattern dicts to (X, y, roc5_targets).

    Extracted from the inline logic in routes/models.py so both the manual
    distillation route and the CT pipeline share the same array construction.

    Pattern dicts require: dataset_filepath, row_index, label, pnl_percent.
    """
    from app.ml.pattern_extractor import patterns_to_training_data

    X, y = patterns_to_training_data(patterns, include_hold=True, seq_len=seq_len)
    roc5_targets = np.array(
        [float(p.get("pnl_percent") or 0) / 100.0 for p in patterns],
        dtype=np.float32,
    )
    return X, y, roc5_targets


async def auto_retrain(
    db: AsyncSession,
    lookback_years: int = 2,
    log_fn: Callable[[str], None] | None = None,
) -> dict:
    """Retrain KNN + LSTM on a rolling lookback window of GoldenPatterns.

    Steps:
      1. Query GoldenPattern WHERE date >= cutoff
      2. Build X, y, roc5_targets
      3. Train KNN → save artifacts → update KNNModel row
      4. Train LSTM (with LSTM pre-training) → save artifacts → update LSTMModel row
      5. Insert a new EnsembleConfig (staged, not activated)
      6. Store last_auto_retrain_at in AppSetting

    Returns a summary dict { knn_model_id, lstm_model_id, ensemble_config_id,
                              n_patterns, date_range, knn_accuracy, lstm_accuracy }.
    """

    def _log(msg: str) -> None:
        logger.info(msg)
        if log_fn:
            log_fn(msg)

    cutoff = date.today() - relativedelta(years=lookback_years)
    _log(f"Auto-retrain: loading patterns since {cutoff} (lookback={lookback_years}y)")

    # ── Load patterns in date range ────────────────────────────────────
    result = await db.execute(
        select(GoldenPattern)
        .where(GoldenPattern.date >= cutoff)
        .order_by(GoldenPattern.date)
    )
    patterns = result.scalars().all()

    if not patterns:
        raise ValueError(
            f"No golden patterns found since {cutoff}. "
            "Run RL training first to generate patterns."
        )

    # Use rl_model_id of the most recent pattern as the FK source reference
    source_rl_model_id = patterns[-1].rl_model_id

    db_patterns_list = [
        {
            "dataset_filepath": p.dataset_filepath,
            "row_index": p.row_index,
            "label": p.label,
            "pnl_percent": p.pnl_percent,
        }
        for p in patterns
    ]

    n_patterns = len(db_patterns_list)
    date_range = f"{patterns[0].date} → {patterns[-1].date}"
    _log(f"Auto-retrain: {n_patterns} patterns ({date_range})")

    # ── Build training arrays (CPU-bound) ──────────────────────────────
    loop = asyncio.get_running_loop()
    seq_len = settings.DEFAULT_SEQ_LEN_DAILY

    try:
        X, y, roc5_targets = await loop.run_in_executor(
            _ct_executor,
            lambda: _build_training_arrays(db_patterns_list, seq_len=seq_len),
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to build training arrays: {exc}") from exc

    if len(X) == 0:
        raise ValueError("Training array is empty after pattern conversion — nothing to train on")

    uniq, cnt = np.unique(y, return_counts=True)
    _log(f"Arrays: X={X.shape}  y={y.shape}  classes={dict(zip(uniq.tolist(), cnt.tolist()))}")

    # ── Timestamped save directories ───────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path(settings.MODEL_DIR)
    knn_save_dir = base_dir / "knn" / f"auto_{timestamp}"
    lstm_save_dir = base_dir / "lstm" / f"auto_{timestamp}"
    knn_save_dir.mkdir(parents=True, exist_ok=True)
    lstm_save_dir.mkdir(parents=True, exist_ok=True)

    # ── Create KNNModel DB record (status=training) ───────────────────
    async with async_session_factory() as sdb:
        knn_rec = await crud.create_knn_model(
            sdb,
            name=f"auto_knn_{timestamp}",
            source_rl_model_id=source_rl_model_id,
            k_neighbors=5,
            seq_len=seq_len,
            interval=IntervalEnum.day,
            status="training",
        )
        knn_model_id: int = knn_rec.id

    _log(f"Training KNN (model_id={knn_model_id})...")
    knn_metrics: dict = {}
    try:
        from app.ml.knn_distiller import train_knn, save_knn_model

        knn_obj, knn_metrics = await loop.run_in_executor(
            _ct_executor,
            lambda: train_knn(X, y, k_neighbors=5, log_fn=_log),
        )
        knn_artifacts = save_knn_model(
            knn_obj, knn_metrics,
            save_dir=knn_save_dir,
            model_name=f"knn_{knn_model_id}",
        )
        async with async_session_factory() as sdb:
            await crud.update_knn_model_completed(
                sdb, knn_model_id,
                model_path=knn_artifacts["model_path"],
                norm_params_path=knn_artifacts.get("norm_params_path"),
                accuracy=knn_metrics["accuracy"],
                precision_buy=knn_metrics["precision_buy"],
                precision_sell=knn_metrics["precision_sell"],
            )
        _log(
            f"KNN done: acc={knn_metrics['accuracy']:.4f}  "
            f"prec_buy={knn_metrics['precision_buy']:.4f}  "
            f"prec_sell={knn_metrics['precision_sell']:.4f}"
        )
    except Exception as exc:
        _log(f"KNN training failed: {exc}")
        async with async_session_factory() as sdb:
            await crud.update_knn_model_status(sdb, knn_model_id, "failed")
        raise

    # ── Create LSTMModel DB record (status=training) ──────────────────
    async with async_session_factory() as sdb:
        lstm_rec = await crud.create_lstm_model(
            sdb,
            name=f"auto_lstm_{timestamp}",
            source_rl_model_id=source_rl_model_id,
            hidden_size=128,
            num_layers=2,
            dropout=0.3,
            seq_len=seq_len,
            interval=IntervalEnum.day,
            status="training",
        )
        lstm_model_id: int = lstm_rec.id

    _log(f"Training LSTM (model_id={lstm_model_id})...")
    lstm_metrics: dict = {}
    try:
        from app.ml.lstm_distiller import train_lstm, save_lstm_model

        lstm_obj, lstm_metrics = await loop.run_in_executor(
            _ct_executor,
            lambda: train_lstm(
                X, y,
                hidden_size=128,
                num_layers=2,
                dropout=0.3,
                roc5_targets=roc5_targets,
                pretrain_epochs=10,
                log_fn=_log,
            ),
        )
        lstm_artifacts = save_lstm_model(
            lstm_obj, lstm_metrics,
            save_dir=lstm_save_dir,
            model_name=f"lstm_{lstm_model_id}",
        )
        async with async_session_factory() as sdb:
            await crud.update_lstm_model_completed(
                sdb, lstm_model_id,
                model_path=lstm_artifacts["model_path"],
                accuracy=lstm_metrics["accuracy"],
                precision_buy=lstm_metrics["precision_buy"],
                precision_sell=lstm_metrics["precision_sell"],
            )
        _log(
            f"LSTM done: acc={lstm_metrics['accuracy']:.4f}  "
            f"prec_buy={lstm_metrics['precision_buy']:.4f}  "
            f"prec_sell={lstm_metrics['precision_sell']:.4f}"
        )
    except Exception as exc:
        _log(f"LSTM training failed: {exc}")
        async with async_session_factory() as sdb:
            await crud.update_lstm_model_status(sdb, lstm_model_id, "failed")
        raise

    # ── Stage new EnsembleConfig (inactive — operator activates manually) ──
    async with async_session_factory() as sdb:
        ensemble_rec = await crud.create_ensemble_config(
            sdb,
            name=f"auto_ensemble_{timestamp}",
            knn_model_id=knn_model_id,
            lstm_model_id=lstm_model_id,
            knn_weight=0.5,
            lstm_weight=0.5,
            agreement_required=True,
            interval=IntervalEnum.day,
        )
        ensemble_config_id: int = ensemble_rec.id

    # ── Record completion timestamp in AppSetting ─────────────────────
    async with async_session_factory() as sdb:
        await crud.set_setting(sdb, "last_auto_retrain_at", datetime.utcnow().isoformat())

    summary = {
        "knn_model_id": knn_model_id,
        "lstm_model_id": lstm_model_id,
        "ensemble_config_id": ensemble_config_id,
        "n_patterns": n_patterns,
        "date_range": date_range,
        "knn_accuracy": knn_metrics.get("accuracy"),
        "lstm_accuracy": lstm_metrics.get("accuracy"),
    }
    _log(f"Auto-retrain complete: {summary}")
    return summary
