"""Index OHLCV ingest endpoint.

POST /api/indices/ingest — fetch historical OHLCV for a market index
(e.g. NIFTY 50) from Zerodha Kite and store it in the index_ohlcv table.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.config import settings
from app.db import crud

logger = logging.getLogger(__name__)

router = APIRouter()


class IngestRequest(BaseModel):
    symbol: str = Field(default_factory=lambda: settings.BENCHMARK_SYMBOL)
    kite_token: int = Field(default_factory=lambda: settings.BENCHMARK_KITE_TOKEN)
    from_date: date
    to_date: date
    interval: str = "day"


class IngestResponse(BaseModel):
    symbol: str
    interval: str
    rows_upserted: int


@router.post("/ingest", response_model=IngestResponse)
async def ingest_index_ohlcv(
    req: IngestRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> IngestResponse:
    """Fetch index OHLCV from Zerodha Kite and upsert into index_ohlcv table."""
    from datetime import datetime

    try:
        from app.core import zerodha
        raw = zerodha.fetch_historical_data(
            instrument_token=req.kite_token,
            from_date=datetime.combine(req.from_date, datetime.min.time()),
            to_date=datetime.combine(req.to_date, datetime.min.time()),
            interval=req.interval,
        )
    except Exception as exc:
        logger.error("Zerodha fetch failed for %s: %s", req.symbol, exc)
        raise HTTPException(status_code=502, detail=f"Zerodha fetch failed: {exc}")

    if not raw:
        raise HTTPException(status_code=404, detail="No data returned for the requested date range")

    rows = [
        {
            "symbol": req.symbol,
            "date": r["date"].date() if hasattr(r["date"], "date") else r["date"],
            "interval": req.interval,
            "open": r.get("open"),
            "high": r.get("high"),
            "low": r.get("low"),
            "close": r.get("close"),
            "volume": r.get("volume"),
        }
        for r in raw
    ]

    n = await crud.upsert_index_ohlcv(db, rows)
    logger.info("Upserted %d rows for index %s (%s)", n, req.symbol, req.interval)
    return IngestResponse(symbol=req.symbol, interval=req.interval, rows_upserted=n)
