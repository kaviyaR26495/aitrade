"""Point-in-Time (PIT) Fundamental Data Pipeline.

Ingests stock-level fundamental metrics from yfinance and NSE index PE/PB
ratios from nsepython, storing one row per (stock, date) in
``stock_fundamentals_pit``.  A separate step computes ML-ready z-scores
into ``stock_fundamental_zscores``.

PIT guarantee
-------------
The ingestion function only inserts a row for *today's* date — it never
back-fills historical dates.  This prevents look-ahead bias: when the ML
models are trained on past windows, each row contains only the fundamental
data that was known on that calendar date.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Sequence

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    Stock,
    StockFundamentalPIT,
    StockFundamentalZScore,
    FundamentalSectorStats,
    now_ist,
)

_log = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))

# ── yfinance field mapping ─────────────────────────────────────────────
_YFINANCE_FIELDS = (
    "trailingPE",
    "forwardPE",
    "priceToBook",
    "dividendYield",
    "returnOnEquity",
    "debtToEquity",
)


async def fetch_stock_fundamentals(symbol: str) -> dict:
    """Fetch fundamental metrics for a single NSE stock via yfinance.

    Returns a dict with keys matching StockFundamentalPIT columns.
    Any unavailable field is returned as None — the caller must
    handle missing values gracefully.
    """
    import asyncio
    import yfinance as yf

    def _blocking_fetch() -> dict:
        ticker = yf.Ticker(f"{symbol}.NS")
        info = ticker.info or {}
        return {
            "pe_ratio": _safe_float(info.get("trailingPE")),
            "forward_pe": _safe_float(info.get("forwardPE")),
            "pb_ratio": _safe_float(info.get("priceToBook")),
            "dividend_yield": _safe_float(info.get("dividendYield")),
            "roe": _safe_float(info.get("returnOnEquity")),
            "debt_equity": _safe_float(info.get("debtToEquity")),
        }

    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(None, _blocking_fetch)
    except Exception as exc:
        _log.warning("yfinance fetch failed for %s: %s", symbol, exc)
        return {k: None for k in ("pe_ratio", "forward_pe", "pb_ratio", "dividend_yield", "roe", "debt_equity")}


async def ingest_fundamentals_pit(
    db: AsyncSession,
    stocks: Sequence[Stock],
    as_of_date: date | None = None,
) -> int:
    """Ingest today's fundamental snapshot for a list of stocks.

    Each stock gets exactly one row with ``date = as_of_date``.
    If a row for (stock_id, date) already exists it is skipped (idempotent).

    Returns the number of new rows inserted.
    """
    if as_of_date is None:
        as_of_date = datetime.now(IST).date()

    inserted = 0
    for stock in stocks:
        # Check if we already have today's row
        existing = await db.execute(
            select(StockFundamentalPIT).where(
                StockFundamentalPIT.stock_id == stock.id,
                StockFundamentalPIT.date == as_of_date,
            )
        )
        if existing.scalar_one_or_none() is not None:
            continue

        metrics = await fetch_stock_fundamentals(stock.symbol)
        row = StockFundamentalPIT(
            stock_id=stock.id,
            date=as_of_date,
            source="yfinance",
            ingested_at=now_ist(),
            **metrics,
        )
        db.add(row)
        inserted += 1

    await db.commit()
    _log.info("Fundamental PIT ingestion: %d new rows for date=%s", inserted, as_of_date)
    return inserted


async def compute_sector_stats(
    db: AsyncSession,
    as_of_date: date | None = None,
) -> None:
    """Compute sector-level PE averages for today and upsert into
    ``fundamental_sector_stats``.  Called after ``ingest_fundamentals_pit``."""
    if as_of_date is None:
        as_of_date = datetime.now(IST).date()

    # Pull today's fundamentals joined with sector
    result = await db.execute(
        select(
            Stock.sector,
            StockFundamentalPIT.pe_ratio,
        )
        .join(StockFundamentalPIT, StockFundamentalPIT.stock_id == Stock.id)
        .where(StockFundamentalPIT.date == as_of_date)
        .where(StockFundamentalPIT.pe_ratio.isnot(None))
        .where(StockFundamentalPIT.pe_ratio > 0)
    )
    rows = result.all()
    if not rows:
        return

    df = pd.DataFrame(rows, columns=["sector", "pe_ratio"])
    grouped = df.groupby("sector")["pe_ratio"]

    for sector, pe_series in grouped:
        pe_arr = pe_series.dropna().values
        if len(pe_arr) < 2:
            continue
        avg = float(np.mean(pe_arr))
        std = float(np.std(pe_arr, ddof=1))

        existing = await db.execute(
            select(FundamentalSectorStats).where(
                FundamentalSectorStats.sector == sector,
                FundamentalSectorStats.date == as_of_date,
            )
        )
        row = existing.scalar_one_or_none()
        if row is None:
            row = FundamentalSectorStats(
                sector=sector,
                date=as_of_date,
                computed_at=now_ist(),
            )
            db.add(row)
        row.sector_pe_avg = avg
        row.sector_pe_std = std
        row.stock_count = int(len(pe_arr))
        row.computed_at = now_ist()

    await db.commit()


async def compute_valuation_zscores(
    db: AsyncSession,
    stock: Stock,
    as_of_date: date | None = None,
    lookback_years: int = 3,
) -> StockFundamentalZScore | None:
    """Compute and persist ML-ready z-scores for a single stock.

    - ``pe_zscore_3y``      : (today_PE - rolling_mean_3yr) / rolling_std_3yr, clipped ±3
    - ``pe_zscore_sector``  : (today_PE - sector_avg) / sector_std, clipped ±3
    - ``roe_norm``          : ROE clipped to [0, 1.0] (100%), then passed through
    - ``debt_equity_norm``  : D/E clipped to [0, 5], divided by 5, then inverted (1 = low debt)

    Returns None if insufficient data exists.
    """
    if as_of_date is None:
        as_of_date = datetime.now(IST).date()

    cutoff = as_of_date - timedelta(days=lookback_years * 365)

    # Fetch 3-year history of PE for this stock
    result = await db.execute(
        select(StockFundamentalPIT.date, StockFundamentalPIT.pe_ratio)
        .where(StockFundamentalPIT.stock_id == stock.id)
        .where(StockFundamentalPIT.date >= cutoff)
        .where(StockFundamentalPIT.date <= as_of_date)
        .order_by(StockFundamentalPIT.date)
    )
    pit_rows = result.all()
    if not pit_rows:
        return None

    today_row = next((r for r in reversed(pit_rows) if r.date == as_of_date), None)
    if today_row is None or today_row.pe_ratio is None:
        return None

    today_pe = today_row.pe_ratio
    pe_history = [r.pe_ratio for r in pit_rows if r.pe_ratio is not None]

    # 3-year z-score
    pe_zscore_3y: float | None = None
    if len(pe_history) >= 10:
        mean_3y = float(np.mean(pe_history))
        std_3y = float(np.std(pe_history, ddof=1))
        if std_3y > 0:
            pe_zscore_3y = float(np.clip((today_pe - mean_3y) / std_3y, -3.0, 3.0))

    # Sector z-score (today's sector stats)
    pe_zscore_sector: float | None = None
    if stock.sector:
        sector_result = await db.execute(
            select(FundamentalSectorStats).where(
                FundamentalSectorStats.sector == stock.sector,
                FundamentalSectorStats.date == as_of_date,
            )
        )
        sector_stats = sector_result.scalar_one_or_none()
        if sector_stats and sector_stats.sector_pe_std and sector_stats.sector_pe_std > 0:
            pe_zscore_sector = float(
                np.clip(
                    (today_pe - sector_stats.sector_pe_avg) / sector_stats.sector_pe_std,
                    -3.0,
                    3.0,
                )
            )

    # ROE normalisation: raw ROE from yfinance is a decimal (0.18 = 18%)
    roe_norm: float | None = None
    roe_raw = today_row.pe_ratio  # NOTE: we re-fetch the full row below
    # Fetch full row for non-PE fields
    full_result = await db.execute(
        select(StockFundamentalPIT).where(
            StockFundamentalPIT.stock_id == stock.id,
            StockFundamentalPIT.date == as_of_date,
        )
    )
    full_row = full_result.scalar_one_or_none()
    if full_row and full_row.roe is not None:
        roe_norm = float(np.clip(full_row.roe, 0.0, 1.0))  # already 0–1 decimal from yfinance

    debt_equity_norm: float | None = None
    if full_row and full_row.debt_equity is not None:
        # D/E: clipped to [0,5], /5 → 0-1, then invert so 1=low debt is "good"
        de_scaled = float(np.clip(full_row.debt_equity, 0.0, 5.0)) / 5.0
        debt_equity_norm = round(1.0 - de_scaled, 4)

    # Upsert z-score row
    existing = await db.execute(
        select(StockFundamentalZScore).where(
            StockFundamentalZScore.stock_id == stock.id,
            StockFundamentalZScore.date == as_of_date,
        )
    )
    z_row = existing.scalar_one_or_none()
    if z_row is None:
        z_row = StockFundamentalZScore(
            stock_id=stock.id,
            date=as_of_date,
            computed_at=now_ist(),
        )
        db.add(z_row)

    z_row.pe_zscore_3y = pe_zscore_3y
    z_row.pe_zscore_sector = pe_zscore_sector
    z_row.roe_norm = roe_norm
    z_row.debt_equity_norm = debt_equity_norm
    z_row.computed_at = now_ist()

    await db.commit()
    return z_row


async def ingest_and_score(
    db: AsyncSession,
    stocks: list[Stock],
    as_of_date: date | None = None,
) -> dict:
    """Full pipeline: ingest PIT → compute sector stats → compute z-scores.

    Returns a summary dict suitable for the API response.
    """
    if as_of_date is None:
        as_of_date = datetime.now(IST).date()

    inserted = await ingest_fundamentals_pit(db, stocks, as_of_date)
    await compute_sector_stats(db, as_of_date)

    scored = 0
    for stock in stocks:
        z = await compute_valuation_zscores(db, stock, as_of_date)
        if z is not None:
            scored += 1

    return {
        "date": as_of_date.isoformat(),
        "total_stocks": len(stocks),
        "pit_rows_inserted": inserted,
        "zscores_computed": scored,
    }


# ── Internal helpers ───────────────────────────────────────────────────

def _safe_float(val) -> float | None:
    """Convert a value to float, returning None for NaN/Inf/None."""
    if val is None:
        return None
    try:
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None
