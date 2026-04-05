"""DB-first data access layer.

Implements the decision logic: DB first → if stale/missing → fetch from Zerodha → upsert into DB.
Ported from pytrade's get_dfs() / db_common pattern.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession

from app.db import crud

logger = logging.getLogger(__name__)


async def get_last_trading_day(db: AsyncSession, ref_date: date | None = None) -> date:
    """Get the most recent trading day (excludes weekends & NSE holidays)."""
    if ref_date is None:
        ref_date = date.today()

    holidays = await crud.get_holidays(db, year=ref_date.year)
    holiday_dates = {h.trading_date for h in holidays}

    d = ref_date
    while True:
        if d.weekday() >= 5:  # Sat/Sun
            d -= timedelta(days=1)
            continue
        if d in holiday_dates:
            d -= timedelta(days=1)
            continue
        return d


async def is_data_stale(
    db: AsyncSession, stock_id: int, interval: str
) -> tuple[bool, date | None]:
    """Check if cached OHLCV is stale. Returns (is_stale, max_date_in_db)."""
    max_date = await crud.get_ohlcv_max_date(db, stock_id, interval)
    if max_date is None:
        return True, None

    # MySQL may return datetime instead of date for MAX(date) — normalise
    if isinstance(max_date, datetime):
        max_date = max_date.date()

    last_trading = await get_last_trading_day(db)
    return max_date < last_trading, max_date


async def get_ohlcv_cached(
    db: AsyncSession,
    stock_id: int,
    interval: str,
    start_date: date | None = None,
    end_date: date | None = None,
    force_refresh: bool = False,
) -> list:
    """
    DB-first OHLCV fetch.

    1. Check DB freshness
    2. If stale → caller must invoke data_pipeline.sync_stock() first
    3. Return data from DB
    """
    stale, max_db_date = await is_data_stale(db, stock_id, interval)

    if stale or force_refresh:
        logger.info(
            "Stock %s interval=%s data is stale (max_date=%s). "
            "Caller should trigger sync.",
            stock_id,
            interval,
            max_db_date,
        )

    rows = await crud.get_ohlcv(db, stock_id, interval, start_date, end_date)
    return list(rows)


async def get_indicators_cached(
    db: AsyncSession,
    stock_id: int,
    interval: str,
    start_date: date | None = None,
    end_date: date | None = None,
) -> list:
    """Return cached indicators from DB."""
    rows = await crud.get_indicators(db, stock_id, interval, start_date, end_date)
    return list(rows)


async def get_regimes_cached(
    db: AsyncSession,
    stock_id: int,
    interval: str,
    start_date: date | None = None,
    end_date: date | None = None,
) -> list:
    """Return cached regime classifications from DB."""
    rows = await crud.get_regimes(db, stock_id, interval, start_date, end_date)
    return list(rows)
