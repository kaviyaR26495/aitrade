"""Data pipeline — fetch, clean, DB cache, incremental sync.

Ported from pytrade's df_creator.py + db_common.py patterns:
- DB-first fetch logic (get_dfs pattern)
- 2000-day chunked Kite API calls
- Incremental sync (only fetch since MAX(date))
- Bulk upsert into stock_ohlcv
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import zerodha
from app.db import crud
from app.db.cache import get_last_trading_day, is_data_stale
from app.db.models import Stock, IntervalEnum

logger = logging.getLogger(__name__)

DEFAULT_HISTORY_YEARS = 8  # fetch ~8 years of history for new stocks


async def sync_stock_ohlcv(
    db: AsyncSession,
    stock: Stock,
    interval: str = "day",
    force_full: bool = False,
) -> int:
    """
    Sync OHLCV data for a stock. DB-first with incremental updates.

    1. Check MAX(date) in DB
    2. If stale → fetch new data from Kite API since MAX(date)
    3. Bulk upsert into stock_ohlcv
    """
    if not stock.kite_id:
        logger.debug("Stock %s has no kite_id — attempting live lookup from Zerodha", stock.symbol)
        kite_id = await _resolve_kite_id(db, stock.symbol)
        if kite_id:
            # Persist the resolved kite_id so future syncs skip this lookup
            from sqlalchemy import update as sa_update
            await db.execute(
                sa_update(Stock)
                .where(Stock.id == stock.id)
                .values(kite_id=kite_id)
            )
            await db.commit()
            await db.refresh(stock)
        else:
            logger.warning("Could not resolve kite_id for %s, will rely on Yahoo Finance fallback.", stock.symbol)

    # Check if we can use Zerodha
    use_zerodha = zerodha.is_authenticated()
    if not use_zerodha:
        logger.info("Zerodha is not authenticated, will fallback to Yahoo Finance for %s.", stock.symbol)

    stale, max_db_date = await is_data_stale(db, stock.id, interval)

    if not stale and not force_full:
        logger.debug("Stock %s interval=%s is fresh (max=%s)", stock.symbol, interval, max_db_date)
        return 0

    # Determine start date — always use datetime so Kite API chunking
    # (which does `current_from < to_date`) never hits a date/datetime
    # type mismatch, regardless of what the DB driver returned.
    if max_db_date and not force_full:
        # max_db_date is guaranteed to be a date (normalised in is_data_stale)
        next_day = max_db_date if isinstance(max_db_date, datetime) else datetime(
            max_db_date.year, max_db_date.month, max_db_date.day
        )
        from_date = next_day + timedelta(days=1)
    else:
        from_date = datetime.now() - timedelta(days=365 * DEFAULT_HISTORY_YEARS)

    to_date = datetime.now()

    logger.info("Syncing %s interval=%s from=%s to=%s", stock.symbol, interval, from_date, to_date)

    raw_data = None
    if use_zerodha and stock.kite_id:
        try:
            # Fetch from Kite API (with 2000-day chunking)
            raw_data = zerodha.fetch_historical_data(
                instrument_token=stock.kite_id,
                from_date=from_date,
                to_date=to_date,
                interval=interval,
            )
        except Exception as e:
            logger.warning("Zerodha fetch failed for %s: %s. Falling back to yfinance.", stock.symbol, e)
            raw_data = None
            
    if not raw_data:
        # Fallback to Yahoo Finance
        from app.core import yfinance_api
        logger.info("Fetching data for %s using Yahoo Finance fallback", stock.symbol)
        raw_data = yfinance_api.fetch_historical_data(
            symbol=stock.symbol,
            from_date=from_date,
            to_date=to_date,
            interval=interval,
        )

    if not raw_data:
        logger.warning("No data returned for %s (from both Zerodha and yfinance)", stock.symbol)
        return 0

    # Convert to DB rows
    rows = []
    for record in raw_data:
        dt = record["date"]
        if isinstance(dt, datetime):
            dt = dt.date()
        rows.append({
            "stock_id": stock.id,
            "date": dt,
            "open": float(record["open"]),
            "high": float(record["high"]),
            "low": float(record["low"]),
            "close": float(record["close"]),
            "adj_close": float(record.get("close", record["close"])),
            "volume": float(record["volume"]),
            "interval": interval,
        })

    # Bulk upsert (5000 rows/batch)
    count = await crud.bulk_upsert_ohlcv(db, rows)
    logger.info("Upserted %d OHLCV rows for %s interval=%s", count, stock.symbol, interval)
    return count


async def sync_all_stocks(
    db: AsyncSession,
    interval: str = "day",
    stock_ids: list[int] | None = None,
    concurrency: int = 8,
) -> dict[str, int]:
    """Sync OHLCV for all active stocks (or specified subset).

    Uses ``asyncio.Semaphore(concurrency)`` to run up to *concurrency* syncs
    concurrently (default 8), keeping well within Zerodha API rate limits and
    the SQLAlchemy connection pool (pool_size=20).  Sequential execution for
    500 stocks at ~0.5 s/stock takes > 4 minutes; parallel at 8 takes ~30 s.
    """
    import asyncio

    if stock_ids:
        from sqlalchemy import select as _select
        from app.db.models import Stock as StockModel
        result = await db.execute(
            _select(StockModel).where(StockModel.id.in_(stock_ids))
        )
        stocks = result.scalars().all()
    else:
        stocks = await crud.get_all_active_stocks(db)

    sem = asyncio.Semaphore(concurrency)

    async def _sync_one(stock: Stock) -> tuple[str, int]:
        async with sem:
            try:
                count = await sync_stock_ohlcv(db, stock, interval)
                return stock.symbol, count
            except Exception as e:
                logger.error("Failed to sync %s: %s", stock.symbol, e)
                return stock.symbol, -1

    completed = await asyncio.gather(*[_sync_one(s) for s in stocks])
    return dict(completed)


async def _resolve_kite_id(db: AsyncSession, symbol: str) -> int | None:
    """Look up instrument_token for a symbol directly from Zerodha."""
    try:
        instruments = zerodha.get_instruments("NSE")
    except Exception as exc:
        logger.error("Cannot fetch instruments from Zerodha: %s", exc)
        return None
    for inst in instruments:
        if (
            str(inst.get("tradingsymbol", "")).upper() == symbol.upper()
            and inst.get("instrument_type") == "EQ"
        ):
            return int(inst["instrument_token"])
    return None


async def populate_stock_list(db: AsyncSession) -> int:
    """Fetch NSE EQ instruments from Kite API and populate stocks_list table.

    Accepts both segment='NSE' and segment='NSE-EQ' so this works across
    all kiteconnect versions that parse the segment column differently.
    """
    instruments = zerodha.get_instruments("NSE")

    stocks = []
    for inst in instruments:
        # Accept EQ instruments regardless of how the segment column is labelled;
        # exchange='NSE' (guaranteed by the get_instruments call) is enough.
        if inst.get("instrument_type") == "EQ":
            stocks.append({
                "symbol": str(inst["tradingsymbol"]),
                "exchange": str(inst.get("exchange") or "NSE"),
                "kite_id": int(inst["instrument_token"]),
                "tick_size": float(inst["tick_size"]) if inst.get("tick_size") else None,
                "lot_size": int(inst["lot_size"]) if inst.get("lot_size") else 1,
                "is_active": True,
            })

    if not stocks:
        logger.error(
            "populate_stock_list: zero instruments matched — "
            "raw sample: %s",
            instruments[:3] if instruments else "(empty)",
        )
        return 0

    count = await crud.bulk_upsert_stocks(db, stocks)
    logger.info("Populated/updated %d NSE EQ stocks", count)
    return count


async def sync_holidays(db: AsyncSession) -> int:
    """Fetch NSE holidays and populate nse_holidays table."""
    import httpx

    url = "https://www.nseindia.com/api/holiday-master?type=trading"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
    }
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.error("Failed to fetch NSE holidays: %s", e)
        return 0

    holidays = []
    for entry in data.get("CM", []):
        try:
            dt = datetime.strptime(entry["tradingDate"], "%d-%b-%Y").date()
            holidays.append({
                "trading_date": dt,
                "week_day": entry.get("weekDay"),
                "description": entry.get("description", ""),
            })
        except (KeyError, ValueError) as e:
            logger.warning("Skipping holiday entry: %s", e)

    if holidays:
        count = await crud.upsert_holidays(db, holidays)
        logger.info("Upserted %d holidays", count)
        return count
    return 0


def ohlcv_to_dataframe(rows: list) -> pd.DataFrame:
    """Convert StockOHLCV ORM objects (or plain dicts) to a pandas DataFrame."""
    if not rows:
        return pd.DataFrame()

    # Accept both ORM objects (attribute access) and dicts (mapping access)
    if isinstance(rows[0], dict):
        df = pd.DataFrame(rows)
    else:
        data = [
            {
                "date": r.date,
                "open": r.open,
                "high": r.high,
                "low": r.low,
                "close": r.close,
                "adj_close": r.adj_close,
                "volume": r.volume,
            }
            for r in rows
        ]
        df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    # Cast price/volume columns to float32 immediately so all downstream
    # pandas indicator calculations and numpy feature arrays are 32-bit.
    # Eliminates silent float64 → float32 coercion inside PyTorch/SB3.
    _f32 = ["open", "high", "low", "close", "adj_close", "volume"]
    for col in _f32:
        if col in df.columns:
            df[col] = df[col].astype(np.float32)
    return df
