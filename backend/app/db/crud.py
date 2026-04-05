"""DB CRUD operations — bulk inserts, upserts, cache queries.

Ported from pytrade's db_common.py patterns:
- ON DUPLICATE KEY UPDATE for upserts
- Batch processing (5000 for OHLCV, 10000 for indicators)
- FK population before indicator inserts
"""
from __future__ import annotations

from datetime import date
from typing import Any, Sequence

import pandas as pd
from sqlalchemy import select, text, func, delete, update
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    AppSetting,
    EnsembleConfig,
    EnsemblePrediction,
    GoldenPattern,
    KNNModel,
    KNNPrediction,
    LSTMModel,
    LSTMPrediction,
    NSEHoliday,
    RLModel,
    RLTrainingRun,
    Stock,
    StockEnsembleWeights,
    StockIndicator,
    StockOHLCV,
    StockRegime,
    IntervalEnum,
)

OHLCV_BATCH_SIZE = 5000
INDICATOR_BATCH_SIZE = 10000
STOCK_BATCH_SIZE = 1000


# ── Stock CRUD ─────────────────────────────────────────────────────────

async def get_stock_by_symbol(db: AsyncSession, symbol: str) -> Stock | None:
    result = await db.execute(select(Stock).where(Stock.symbol == symbol))
    return result.scalar_one_or_none()


async def get_stock_by_id(db: AsyncSession, stock_id: int) -> Stock | None:
    result = await db.execute(select(Stock).where(Stock.id == stock_id))
    return result.scalar_one_or_none()


async def get_all_active_stocks(db: AsyncSession) -> Sequence[Stock]:
    result = await db.execute(
        select(Stock).where(Stock.is_active == True).order_by(Stock.symbol)
    )
    return result.scalars().all()


async def get_stocks_by_sector(db: AsyncSession, sector: str) -> Sequence[Stock]:
    result = await db.execute(
        select(Stock).where(Stock.sector == sector, Stock.is_active == True).order_by(Stock.symbol)
    )
    return result.scalars().all()


async def upsert_stock(db: AsyncSession, symbol: str, **kwargs) -> Stock:
    stmt = mysql_insert(Stock).values(symbol=symbol, **kwargs)
    stmt = stmt.on_duplicate_key_update(
        **{k: v for k, v in kwargs.items() if v is not None}
    )
    await db.execute(stmt)
    await db.commit()
    return await get_stock_by_symbol(db, symbol)


async def bulk_upsert_stocks(db: AsyncSession, stocks: list[dict]) -> int:
    if not stocks:
        return 0

    total = 0
    for i in range(0, len(stocks), STOCK_BATCH_SIZE):
        batch = stocks[i : i + STOCK_BATCH_SIZE]
        stmt = mysql_insert(Stock).values(batch)
        batch_cols = set(batch[0].keys())
        update_cols = {
            c.name: c for c in stmt.inserted if c.name != "id" and c.name in batch_cols
        }
        stmt = stmt.on_duplicate_key_update(**update_cols)
        await db.execute(stmt)
        total += len(batch)

    await db.commit()
    return total


# ── OHLCV CRUD ─────────────────────────────────────────────────────────

async def bulk_upsert_ohlcv(db: AsyncSession, rows: list[dict]) -> int:
    """Bulk insert OHLCV with ON DUPLICATE KEY UPDATE. Batched at 5000 rows."""
    if not rows:
        return 0
    total = 0
    for i in range(0, len(rows), OHLCV_BATCH_SIZE):
        batch = rows[i : i + OHLCV_BATCH_SIZE]
        stmt = mysql_insert(StockOHLCV).values(batch)
        stmt = stmt.on_duplicate_key_update(
            open=stmt.inserted.open,
            high=stmt.inserted.high,
            low=stmt.inserted.low,
            close=stmt.inserted.close,
            adj_close=stmt.inserted.adj_close,
            volume=stmt.inserted.volume,
        )
        await db.execute(stmt)
        total += len(batch)
    await db.commit()
    return total


async def get_ohlcv(
    db: AsyncSession,
    stock_id: int,
    interval: str,
    start_date: date | None = None,
    end_date: date | None = None,
) -> Sequence[StockOHLCV]:
    q = select(StockOHLCV).where(
        StockOHLCV.stock_id == stock_id,
        StockOHLCV.interval == interval,
    )
    if start_date:
        q = q.where(StockOHLCV.date >= start_date)
    if end_date:
        q = q.where(StockOHLCV.date <= end_date)
    q = q.order_by(StockOHLCV.date)
    result = await db.execute(q)
    return result.scalars().all()


async def get_ohlcv_max_date(db: AsyncSession, stock_id: int, interval: str) -> date | None:
    result = await db.execute(
        select(func.max(StockOHLCV.date)).where(
            StockOHLCV.stock_id == stock_id,
            StockOHLCV.interval == interval,
        )
    )
    return result.scalar_one_or_none()


# ── Indicators CRUD ────────────────────────────────────────────────────

async def bulk_upsert_indicators(db: AsyncSession, rows: list[dict]) -> int:
    """Bulk insert indicators with ON DUPLICATE KEY UPDATE. Batched at 10000 rows."""
    if not rows:
        return 0
    total = 0
    for i in range(0, len(rows), INDICATOR_BATCH_SIZE):
        batch = rows[i : i + INDICATOR_BATCH_SIZE]
        stmt = mysql_insert(StockIndicator).values(batch)
        # Update all indicator columns on conflict
        update_cols = {
            c.name: c
            for c in stmt.inserted
            if c.name not in ("id", "stock_id", "date", "interval")
        }
        stmt = stmt.on_duplicate_key_update(**update_cols)
        await db.execute(stmt)
        total += len(batch)
    await db.commit()
    return total


async def get_indicators(
    db: AsyncSession,
    stock_id: int,
    interval: str,
    start_date: date | None = None,
    end_date: date | None = None,
) -> Sequence[StockIndicator]:
    q = select(StockIndicator).where(
        StockIndicator.stock_id == stock_id,
        StockIndicator.interval == interval,
    )
    if start_date:
        q = q.where(StockIndicator.date >= start_date)
    if end_date:
        q = q.where(StockIndicator.date <= end_date)
    q = q.order_by(StockIndicator.date)
    result = await db.execute(q)
    return result.scalars().all()


# ── Regime CRUD ────────────────────────────────────────────────────────

async def bulk_upsert_regimes(db: AsyncSession, rows: list[dict]) -> int:
    if not rows:
        return 0
    total = 0
    for i in range(0, len(rows), INDICATOR_BATCH_SIZE):
        batch = rows[i : i + INDICATOR_BATCH_SIZE]
        stmt = mysql_insert(StockRegime).values(batch)
        update_cols = {
            c.name: c
            for c in stmt.inserted
            if c.name not in ("id", "stock_id", "date", "interval")
        }
        stmt = stmt.on_duplicate_key_update(**update_cols)
        await db.execute(stmt)
        total += len(batch)
    await db.commit()
    return total


async def get_regimes(
    db: AsyncSession,
    stock_id: int,
    interval: str,
    start_date: date | None = None,
    end_date: date | None = None,
) -> Sequence[StockRegime]:
    q = select(StockRegime).where(
        StockRegime.stock_id == stock_id,
        StockRegime.interval == interval,
    )
    if start_date:
        q = q.where(StockRegime.date >= start_date)
    if end_date:
        q = q.where(StockRegime.date <= end_date)
    q = q.order_by(StockRegime.date)
    result = await db.execute(q)
    return result.scalars().all()


# ── Holiday CRUD ───────────────────────────────────────────────────────

async def upsert_holidays(db: AsyncSession, holidays: list[dict]) -> int:
    if not holidays:
        return 0
    stmt = mysql_insert(NSEHoliday).values(holidays)
    stmt = stmt.on_duplicate_key_update(
        week_day=stmt.inserted.week_day,
        description=stmt.inserted.description,
    )
    await db.execute(stmt)
    await db.commit()
    return len(holidays)


async def get_holidays(db: AsyncSession, year: int | None = None) -> Sequence[NSEHoliday]:
    q = select(NSEHoliday)
    if year:
        q = q.where(func.year(NSEHoliday.trading_date) == year)
    q = q.order_by(NSEHoliday.trading_date)
    result = await db.execute(q)
    return result.scalars().all()


# ── RL Model CRUD ──────────────────────────────────────────────────────

async def create_rl_model(db: AsyncSession, **kwargs) -> RLModel:
    model = RLModel(**kwargs)
    db.add(model)
    await db.commit()
    await db.refresh(model)
    return model


async def get_rl_model(db: AsyncSession, model_id: int) -> RLModel | None:
    result = await db.execute(select(RLModel).where(RLModel.id == model_id))
    return result.scalar_one_or_none()


async def list_rl_models(db: AsyncSession) -> Sequence[RLModel]:
    result = await db.execute(select(RLModel).order_by(RLModel.created_at.desc()))
    return result.scalars().all()


async def update_rl_model_status(db: AsyncSession, model_id: int, status: str) -> None:
    await db.execute(update(RLModel).where(RLModel.id == model_id).values(status=status))
    await db.commit()


async def update_rl_model_completed(
    db: AsyncSession,
    model_id: int,
    total_reward: float | None,
    sharpe_ratio: float | None,
    model_path: str | None,
) -> None:
    await db.execute(
        update(RLModel).where(RLModel.id == model_id).values(
            status="completed",
            total_reward=total_reward,
            sharpe_ratio=sharpe_ratio,
            model_path=model_path,
        )
    )
    await db.commit()


async def list_training_runs(db: AsyncSession, rl_model_id: int) -> Sequence[RLTrainingRun]:
    result = await db.execute(
        select(RLTrainingRun)
        .where(RLTrainingRun.rl_model_id == rl_model_id)
        .order_by(RLTrainingRun.timestamp)
    )
    return result.scalars().all()


async def delete_rl_model(db: AsyncSession, model_id: int) -> bool:
    """Delete an RL model and all directly associated rows. Returns True if found.
    Caller is responsible for cascading through knn_models/lstm_models first.
    """
    model = await get_rl_model(db, model_id)
    if not model:
        return False
    # golden_patterns references rl_models via FK — must go before rl_models
    await db.execute(delete(GoldenPattern).where(GoldenPattern.rl_model_id == model_id))
    await db.execute(delete(RLTrainingRun).where(RLTrainingRun.rl_model_id == model_id))
    await db.execute(delete(RLModel).where(RLModel.id == model_id))
    await db.commit()
    return True


async def save_training_run(db: AsyncSession, **kwargs) -> RLTrainingRun:
    run = RLTrainingRun(**kwargs)
    db.add(run)
    await db.commit()
    return run


async def bulk_save_training_runs(db: AsyncSession, entries: list[dict]) -> int:
    """Insert all training run entries in a single transaction."""
    if not entries:
        return 0
    db.add_all([RLTrainingRun(**e) for e in entries])
    await db.commit()
    return len(entries)


# ── Golden Pattern CRUD ────────────────────────────────────────────────

async def bulk_insert_patterns(db: AsyncSession, patterns: list[dict]) -> int:
    if not patterns:
        return 0
    db.add_all([GoldenPattern(**p) for p in patterns])
    await db.commit()
    return len(patterns)


async def get_patterns_by_rl_model(
    db: AsyncSession, rl_model_id: int
) -> Sequence[GoldenPattern]:
    result = await db.execute(
        select(GoldenPattern)
        .where(GoldenPattern.rl_model_id == rl_model_id)
        .order_by(GoldenPattern.date)
    )
    return result.scalars().all()


# ── KNN / LSTM / Ensemble Model CRUD ──────────────────────────────────

async def create_knn_model(db: AsyncSession, **kwargs) -> KNNModel:
    m = KNNModel(**kwargs)
    db.add(m)
    await db.commit()
    await db.refresh(m)
    return m


async def update_knn_model_status(db: AsyncSession, model_id: int, status: str) -> None:
    await db.execute(update(KNNModel).where(KNNModel.id == model_id).values(status=status))
    await db.commit()


async def update_knn_model_completed(
    db: AsyncSession,
    model_id: int,
    model_path: str | None,
    norm_params_path: str | None,
    accuracy: float | None,
    precision_buy: float | None,
    precision_sell: float | None,
) -> None:
    await db.execute(
        update(KNNModel).where(KNNModel.id == model_id).values(
            status="completed",
            model_path=model_path,
            norm_params_path=norm_params_path,
            accuracy=accuracy,
            precision_buy=precision_buy,
            precision_sell=precision_sell,
        )
    )
    await db.commit()


async def create_lstm_model(db: AsyncSession, **kwargs) -> LSTMModel:
    m = LSTMModel(**kwargs)
    db.add(m)
    await db.commit()
    await db.refresh(m)
    return m


async def update_lstm_model_status(db: AsyncSession, model_id: int, status: str) -> None:
    await db.execute(update(LSTMModel).where(LSTMModel.id == model_id).values(status=status))
    await db.commit()


async def update_lstm_model_completed(
    db: AsyncSession,
    model_id: int,
    model_path: str | None,
    accuracy: float | None,
    precision_buy: float | None,
    precision_sell: float | None,
) -> None:
    await db.execute(
        update(LSTMModel).where(LSTMModel.id == model_id).values(
            status="completed",
            model_path=model_path,
            accuracy=accuracy,
            precision_buy=precision_buy,
            precision_sell=precision_sell,
        )
    )
    await db.commit()


async def create_ensemble_config(db: AsyncSession, **kwargs) -> EnsembleConfig:
    m = EnsembleConfig(**kwargs)
    db.add(m)
    await db.commit()
    await db.refresh(m)
    return m


# ── Prediction CRUD ───────────────────────────────────────────────────

async def bulk_upsert_knn_predictions(db: AsyncSession, rows: list[dict]) -> int:
    if not rows:
        return 0
    stmt = mysql_insert(KNNPrediction).values(rows)
    update_cols = {
        c.name: c
        for c in stmt.inserted
        if c.name not in ("id", "knn_model_id", "stock_id", "date", "interval")
    }
    stmt = stmt.on_duplicate_key_update(**update_cols)
    await db.execute(stmt)
    await db.commit()
    return len(rows)


async def bulk_upsert_lstm_predictions(db: AsyncSession, rows: list[dict]) -> int:
    if not rows:
        return 0
    stmt = mysql_insert(LSTMPrediction).values(rows)
    update_cols = {
        c.name: c
        for c in stmt.inserted
        if c.name not in ("id", "lstm_model_id", "stock_id", "date", "interval")
    }
    stmt = stmt.on_duplicate_key_update(**update_cols)
    await db.execute(stmt)
    await db.commit()
    return len(rows)


async def bulk_upsert_ensemble_predictions(db: AsyncSession, rows: list[dict]) -> int:
    if not rows:
        return 0
    stmt = mysql_insert(EnsemblePrediction).values(rows)
    update_cols = {
        c.name: c
        for c in stmt.inserted
        if c.name not in ("id", "ensemble_config_id", "stock_id", "date", "interval")
    }
    stmt = stmt.on_duplicate_key_update(**update_cols)
    await db.execute(stmt)
    await db.commit()
    return len(rows)


async def get_ensemble_predictions_for_date(
    db: AsyncSession,
    target_date: date,
    interval: str = "day",
    min_confidence: float = 0.65,
    agreement_only: bool = True,
) -> Sequence[EnsemblePrediction]:
    """Primary trading query — get ensemble predictions from DB."""
    q = select(EnsemblePrediction).where(
        EnsemblePrediction.date == target_date,
        EnsemblePrediction.interval == interval,
        EnsemblePrediction.confidence >= min_confidence,
    )
    if agreement_only:
        q = q.where(EnsemblePrediction.agreement == True)
    q = q.order_by(EnsemblePrediction.confidence.desc())
    result = await db.execute(q)
    return result.scalars().all()


# ── Per-Stock Ensemble Weights CRUD ──────────────────────────────────

async def upsert_stock_ensemble_weight(
    db: AsyncSession,
    ensemble_config_id: int,
    stock_id: int,
    knn_weight: float,
    lstm_weight: float,
    calibration_precision: float | None = None,
) -> StockEnsembleWeights:
    """Insert or replace the calibrated weight row for a (config, stock) pair."""
    from datetime import datetime
    row = StockEnsembleWeights(
        ensemble_config_id=ensemble_config_id,
        stock_id=stock_id,
        knn_weight=knn_weight,
        lstm_weight=lstm_weight,
        calibration_precision=calibration_precision,
        calibrated_at=datetime.utcnow(),
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return row


async def get_stock_ensemble_weight(
    db: AsyncSession,
    ensemble_config_id: int,
    stock_id: int,
) -> StockEnsembleWeights | None:
    """Return the most-recently calibrated weight row for a (config, stock) pair."""
    q = (
        select(StockEnsembleWeights)
        .where(
            StockEnsembleWeights.ensemble_config_id == ensemble_config_id,
            StockEnsembleWeights.stock_id == stock_id,
        )
        .order_by(StockEnsembleWeights.calibrated_at.desc())
        .limit(1)
    )
    result = await db.execute(q)
    return result.scalar_one_or_none()


# ── Settings CRUD ──────────────────────────────────────────────────────

async def get_setting(db: AsyncSession, key: str) -> str | None:
    result = await db.execute(
        select(AppSetting.value).where(AppSetting.property == key)
    )
    return result.scalar_one_or_none()


async def set_setting(db: AsyncSession, key: str, value: str) -> None:
    stmt = mysql_insert(AppSetting).values(property=key, value=value)
    stmt = stmt.on_duplicate_key_update(value=stmt.inserted.value)
    await db.execute(stmt)
    await db.commit()
