"""Regime-Stratified Training Pipeline.

For each of the 6 regime IDs (0-5), trains a dedicated KNN + LSTM pair using
only the GoldenPatterns that were captured during that regime. The resulting
EnsembleConfig is written to ``regime_ensemble_map`` so the prediction engine
can route each live bar to its regime-specific model.

Usage:
    from app.ml.regime_trainer import train_all_regime_models
    summary = asyncio.run(train_all_regime_models(db))
"""
from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Callable

import numpy as np
from dateutil.relativedelta import relativedelta
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db import crud
from app.db.database import async_session_factory
from app.db.models import GoldenPattern, IntervalEnum, RegimeEnsembleMap

logger = logging.getLogger(__name__)

NUM_REGIMES = 6
MIN_PATTERNS_PER_REGIME = 20


async def train_regime_model(
    regime_id: int,
    lookback_years: int = 2,
    log_fn: Callable[[str], None] | None = None,
) -> dict:
    """Train one KNN + LSTM pair for a single regime_id.

    Returns a dict with keys: regime_id, knn_model_id, lstm_model_id,
    ensemble_config_id, n_patterns, knn_accuracy, lstm_accuracy.
    Raises ValueError if not enough patterns exist for this regime.
    """

    def _log(msg: str) -> None:
        logger.info("[regime=%d] %s", regime_id, msg)
        if log_fn:
            log_fn(f"[regime={regime_id}] {msg}")

    cutoff: date = date.today() - relativedelta(years=lookback_years)
    _log(f"Loading patterns since {cutoff}")

    async with async_session_factory() as db:
        result = await db.execute(
            select(GoldenPattern)
            .where(GoldenPattern.date >= cutoff)
            .where(GoldenPattern.regime_id == regime_id)
            .order_by(GoldenPattern.date)
        )
        patterns = result.scalars().all()

    if len(patterns) < MIN_PATTERNS_PER_REGIME:
        raise ValueError(
            f"Only {len(patterns)} patterns for regime {regime_id} — "
            f"minimum required is {MIN_PATTERNS_PER_REGIME}. Skipping."
        )

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
    _log(f"{n_patterns} patterns ({date_range})")

    seq_len = settings.DEFAULT_SEQ_LEN_DAILY
    loop = asyncio.get_running_loop()

    from app.core.ct_pipeline import _build_training_arrays

    X, y, roc5_targets = await loop.run_in_executor(
        None,
        lambda: _build_training_arrays(db_patterns_list, seq_len=seq_len),
    )

    if len(X) == 0:
        raise ValueError(f"Empty training array for regime {regime_id}")

    uniq, cnt = np.unique(y, return_counts=True)
    _log(f"X={X.shape}  classes={dict(zip(uniq.tolist(), cnt.tolist()))}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path(settings.MODEL_DIR)
    knn_save_dir = base_dir / "knn" / f"regime{regime_id}_{timestamp}"
    lstm_save_dir = base_dir / "lstm" / f"regime{regime_id}_{timestamp}"
    knn_save_dir.mkdir(parents=True, exist_ok=True)
    lstm_save_dir.mkdir(parents=True, exist_ok=True)

    # ── KNN ────────────────────────────────────────────────────────────
    async with async_session_factory() as db:
        knn_rec = await crud.create_knn_model(
            db,
            name=f"regime{regime_id}_knn_{timestamp}",
            source_rl_model_id=source_rl_model_id,
            k_neighbors=11,
            seq_len=seq_len,
            interval=IntervalEnum.day,
            status="training",
        )
        knn_model_id: int = knn_rec.id

    _log(f"Training KNN (model_id={knn_model_id})")
    try:
        from app.ml.knn_distiller import train_knn, save_knn_model

        knn_obj, knn_metrics = await loop.run_in_executor(
            None,
            lambda: train_knn(
                X, y,
                k_neighbors=11,
                train_ratio=0.7,
                smote_k_neighbors=11,
                augment_jitter=True,
                jitter_copies=3,
                use_pca=True,
                pca_components=50,
                log_fn=_log,
            ),
        )
        knn_artifacts = save_knn_model(
            knn_obj, knn_metrics,
            save_dir=knn_save_dir,
            model_name=f"knn_{knn_model_id}",
        )
        async with async_session_factory() as db:
            await crud.update_knn_model_completed(
                db, knn_model_id,
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
        _log(f"KNN failed: {exc}")
        async with async_session_factory() as db:
            await crud.update_knn_model_status(db, knn_model_id, "failed")
        raise

    # ── LSTM ───────────────────────────────────────────────────────────
    async with async_session_factory() as db:
        lstm_rec = await crud.create_lstm_model(
            db,
            name=f"regime{regime_id}_lstm_{timestamp}",
            source_rl_model_id=source_rl_model_id,
            hidden_size=256,
            num_layers=2,
            dropout=0.3,
            seq_len=seq_len,
            interval=IntervalEnum.day,
            status="training",
        )
        lstm_model_id: int = lstm_rec.id

    _log(f"Training LSTM (model_id={lstm_model_id})")
    try:
        from app.ml.lstm_distiller import train_lstm, save_lstm_model

        lstm_obj, lstm_metrics = await loop.run_in_executor(
            None,
            lambda: train_lstm(
                X, y,
                hidden_size=256,
                num_layers=2,
                dropout=0.3,
                lr=5e-4,
                patience=20,
                train_ratio=0.7,
                augment_jitter=True,
                jitter_copies=3,
                roc5_targets=roc5_targets,
                pretrain_epochs=25,
                log_fn=_log,
            ),
        )
        lstm_artifacts = save_lstm_model(
            lstm_obj, lstm_metrics,
            save_dir=lstm_save_dir,
            model_name=f"lstm_{lstm_model_id}",
        )
        async with async_session_factory() as db:
            await crud.update_lstm_model_completed(
                db, lstm_model_id,
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
        _log(f"LSTM failed: {exc}")
        async with async_session_factory() as db:
            await crud.update_lstm_model_status(db, lstm_model_id, "failed")
        raise

    # ── EnsembleConfig + RegimeEnsembleMap ─────────────────────────────
    async with async_session_factory() as db:
        ensemble_rec = await crud.create_ensemble_config(
            db,
            name=f"regime{regime_id}_ensemble_{timestamp}",
            knn_model_id=knn_model_id,
            lstm_model_id=lstm_model_id,
            knn_weight=0.5,
            lstm_weight=0.5,
            agreement_required=True,
            interval=IntervalEnum.day,
        )
        ensemble_config_id: int = ensemble_rec.id

        # Upsert into regime_ensemble_map
        await db.execute(
            delete(RegimeEnsembleMap).where(RegimeEnsembleMap.regime_id == regime_id)
        )
        db.add(
            RegimeEnsembleMap(
                regime_id=regime_id,
                ensemble_config_id=ensemble_config_id,
                updated_at=datetime.utcnow(),
            )
        )
        await db.commit()

    _log(f"Mapped regime {regime_id} → ensemble {ensemble_config_id}")

    return {
        "regime_id": regime_id,
        "knn_model_id": knn_model_id,
        "lstm_model_id": lstm_model_id,
        "ensemble_config_id": ensemble_config_id,
        "n_patterns": n_patterns,
        "knn_accuracy": knn_metrics.get("accuracy"),
        "lstm_accuracy": lstm_metrics.get("accuracy"),
    }


async def train_all_regime_models(
    lookback_years: int = 2,
    log_fn: Callable[[str], None] | None = None,
) -> dict:
    """Train KNN + LSTM pairs for all 6 regimes sequentially.

    Skips regimes with insufficient data and collects errors without aborting
    the full run.

    Returns:
        {
          "completed": [<per-regime summary dicts>],
          "skipped":   [{"regime_id": int, "reason": str}],
          "failed":    [{"regime_id": int, "error": str}],
        }
    """
    completed: list[dict] = []
    skipped: list[dict] = []
    failed: list[dict] = []

    for regime_id in range(NUM_REGIMES):
        try:
            result = await train_regime_model(regime_id, lookback_years=lookback_years, log_fn=log_fn)
            completed.append(result)
        except ValueError as exc:
            logger.warning("Regime %d skipped: %s", regime_id, exc)
            skipped.append({"regime_id": regime_id, "reason": str(exc)})
        except Exception as exc:
            logger.error("Regime %d FAILED: %s", regime_id, exc, exc_info=True)
            failed.append({"regime_id": regime_id, "error": str(exc)})

    # Persist completion timestamp
    async with async_session_factory() as db:
        await crud.set_setting(db, "last_regime_retrain_at", datetime.utcnow().isoformat())

    return {"completed": completed, "skipped": skipped, "failed": failed}
