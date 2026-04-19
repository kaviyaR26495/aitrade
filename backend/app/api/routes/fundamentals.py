"""Fundamentals API routes.

POST /api/fundamentals/sync          — trigger PIT ingestion + z-score computation
GET  /api/fundamentals/{stock_id}    — return PIT history with z-scores
GET  /api/fundamentals/zscore/{stock_id} — latest z-scores for a stock
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.db.database import async_session_factory
from app.db.models import Stock, StockFundamentalPIT, StockFundamentalZScore

router = APIRouter()
_log = logging.getLogger(__name__)
IST = timezone(timedelta(hours=5, minutes=30))


@router.post("/sync")
async def trigger_fundamental_sync(
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """Trigger PIT fundamental ingestion for all active universe stocks.

    Runs asynchronously in the background.  The job is idempotent — running
    twice on the same day safely skips already-ingested stocks.
    """
    from app.core.fundamental_pipeline import ingest_and_score

    result = await db.execute(
        select(Stock).where(Stock.is_active == True)  # noqa: E712
    )
    stocks = result.scalars().all()

    async def _run():
        async with async_session_factory() as bdb:
            try:
                summary = await ingest_and_score(bdb, list(stocks))
                _log.info("Fundamental sync completed: %s", summary)
            except Exception as exc:
                _log.error("Fundamental sync failed: %s", exc, exc_info=True)

    background_tasks.add_task(_run)
    return {
        "message": "Fundamental sync started in background",
        "stock_count": len(stocks),
    }


@router.get("/zscore/{stock_id}")
async def get_latest_zscore(
    stock_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Return the most recent fundamental z-scores for a stock."""
    result = await db.execute(
        select(StockFundamentalZScore)
        .where(StockFundamentalZScore.stock_id == stock_id)
        .order_by(StockFundamentalZScore.date.desc())
        .limit(1)
    )
    row = result.scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="No z-scores found for this stock")

    return {
        "stock_id": stock_id,
        "date": row.date.isoformat(),
        "pe_zscore_3y": row.pe_zscore_3y,
        "pe_zscore_sector": row.pe_zscore_sector,
        "roe_norm": row.roe_norm,
        "debt_equity_norm": row.debt_equity_norm,
        "computed_at": row.computed_at.isoformat() if row.computed_at else None,
    }


@router.get("/{stock_id}")
async def get_fundamental_history(
    stock_id: int,
    days: int = 90,
    db: AsyncSession = Depends(get_db),
):
    """Return PIT fundamental snapshots for a stock (last N days)."""
    from datetime import date

    cutoff = datetime.now(IST).date() - timedelta(days=days)

    pit_result = await db.execute(
        select(StockFundamentalPIT)
        .where(
            StockFundamentalPIT.stock_id == stock_id,
            StockFundamentalPIT.date >= cutoff,
        )
        .order_by(StockFundamentalPIT.date.desc())
    )
    pit_rows = pit_result.scalars().all()

    zscore_result = await db.execute(
        select(StockFundamentalZScore)
        .where(
            StockFundamentalZScore.stock_id == stock_id,
            StockFundamentalZScore.date >= cutoff,
        )
        .order_by(StockFundamentalZScore.date.desc())
    )
    zscore_rows = {r.date: r for r in zscore_result.scalars().all()}

    records = []
    for pit in pit_rows:
        z = zscore_rows.get(pit.date)
        records.append({
            "date": pit.date.isoformat(),
            "pe_ratio": pit.pe_ratio,
            "forward_pe": pit.forward_pe,
            "pb_ratio": pit.pb_ratio,
            "dividend_yield": pit.dividend_yield,
            "roe": pit.roe,
            "debt_equity": pit.debt_equity,
            "source": pit.source,
            "pe_zscore_3y": z.pe_zscore_3y if z else None,
            "pe_zscore_sector": z.pe_zscore_sector if z else None,
            "roe_norm": z.roe_norm if z else None,
            "debt_equity_norm": z.debt_equity_norm if z else None,
        })

    return {
        "stock_id": stock_id,
        "days": days,
        "count": len(records),
        "data": records,
    }
