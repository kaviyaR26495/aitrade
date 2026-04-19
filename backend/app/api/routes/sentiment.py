"""Sentiment API routes.

POST /api/sentiment/sync             — trigger FinBERT + LLM sentiment run
GET  /api/sentiment/{stock_id}       — return sentiment history for a stock
GET  /api/sentiment/latest/{stock_id} — latest single day score
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.db.database import async_session_factory
from app.db.models import Stock, StockSentiment

router = APIRouter()
_log = logging.getLogger(__name__)
IST = timezone(timedelta(hours=5, minutes=30))


@router.post("/sync")
async def trigger_sentiment_sync(
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """Trigger the full FinBERT + LLM sentiment pipeline for all active stocks.

    Designed to run at 08:45 IST before market open.
    Idempotent: re-running updates the existing row for today's date.
    """
    from app.core.sentiment_pipeline import run_sentiment_batch

    result = await db.execute(
        select(Stock).where(Stock.is_active == True)  # noqa: E712
    )
    stocks = result.scalars().all()

    async def _run():
        async with async_session_factory() as bdb:
            try:
                summaries = await run_sentiment_batch(bdb, list(stocks), concurrency=5)
                total = len(summaries)
                with_score = sum(1 for s in summaries if s["llm_impact_score"] is not None)
                _log.info("Sentiment sync done: %d/%d stocks scored", with_score, total)
            except Exception as exc:
                _log.error("Sentiment sync failed: %s", exc, exc_info=True)

    background_tasks.add_task(_run)
    return {
        "message": "Sentiment sync started in background",
        "stock_count": len(stocks),
    }


@router.get("/latest/{stock_id}")
async def get_latest_sentiment(
    stock_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Return the most recent sentiment score for a stock."""
    result = await db.execute(
        select(StockSentiment)
        .where(StockSentiment.stock_id == stock_id)
        .order_by(StockSentiment.date.desc())
        .limit(1)
    )
    row = result.scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="No sentiment data found for this stock")

    return _row_to_dict(row)


@router.get("/{stock_id}")
async def get_sentiment_history(
    stock_id: int,
    days: int = 30,
    db: AsyncSession = Depends(get_db),
):
    """Return sentiment history for a stock (last N days)."""
    cutoff = datetime.now(IST).date() - timedelta(days=days)

    result = await db.execute(
        select(StockSentiment)
        .where(
            StockSentiment.stock_id == stock_id,
            StockSentiment.date >= cutoff,
        )
        .order_by(StockSentiment.date.desc())
    )
    rows = result.scalars().all()

    return {
        "stock_id": stock_id,
        "days": days,
        "count": len(rows),
        "data": [_row_to_dict(r) for r in rows],
    }


def _row_to_dict(row: StockSentiment) -> dict:
    return {
        "stock_id": row.stock_id,
        "date": row.date.isoformat(),
        "headline_count": row.headline_count,
        "neutral_filtered_count": row.neutral_filtered_count,
        "pass_through_rate": (
            round((row.headline_count - row.neutral_filtered_count) / row.headline_count, 3)
            if row.headline_count > 0
            else None
        ),
        "avg_finbert_score": row.avg_finbert_score,
        "llm_impact_score": row.llm_impact_score,
        "llm_summary": row.llm_summary,
        "ingested_at": row.ingested_at.isoformat() if row.ingested_at else None,
    }
