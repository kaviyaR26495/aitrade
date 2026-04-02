from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.db import crud
from app.core import data_service
from app.core.regime_classifier import classify_and_score, regime_summary, get_quality_filtered_data
from app.core.data_pipeline import ohlcv_to_dataframe
from app.core.indicators import compute_all_indicators

router = APIRouter()


class RegimeOut(BaseModel):
    date: date
    trend: str
    volatility: str
    regime_id: int
    regime_confidence: float
    quality_score: float
    is_transition: bool
    close: float = 0.0

    class Config:
        from_attributes = True


class ClassifyRequest(BaseModel):
    stock_ids: list[int] | None = None
    interval: str = "day"


class ClassifyResult(BaseModel):
    stock_id: int
    symbol: str
    rows_classified: int
    rows_stored: int


@router.get("/{stock_id}", response_model=list[RegimeOut])
async def get_regime(
    stock_id: int,
    db: AsyncSession = Depends(get_db),
    interval: str = Query("day"),
    start_date: date | None = Query(None),
    end_date: date | None = Query(None),
):
    """Get regime classification for a stock."""
    stock = await crud.get_stock_by_id(db, stock_id)
    if not stock:
        raise HTTPException(status_code=404, detail="Stock not found")

    rows = await crud.get_regimes(db, stock_id, interval, start_date, end_date)
    
    # Fetch closely matched OHLCV data to serve the close price for charts
    ohlcv_rows = await crud.get_ohlcv(db, stock_id, interval, start_date, end_date)
    ohlcv_dict = {o.date: o.close for o in ohlcv_rows}
    
    results = []
    for r in rows:
        r_dict = {
            "date": r.date,
            "trend": r.trend,
            "volatility": r.volatility,
            "regime_id": r.regime_id,
            "regime_confidence": r.regime_confidence,
            "quality_score": r.quality_score,
            "is_transition": r.is_transition,
            "close": ohlcv_dict.get(r.date, 0.0)
        }
        results.append(r_dict)
        
    return results


@router.get("/{stock_id}/summary")
async def get_regime_summary(
    stock_id: int,
    db: AsyncSession = Depends(get_db),
    interval: str = Query("day"),
):
    """Get regime distribution summary for a stock."""
    stock = await crud.get_stock_by_id(db, stock_id)
    if not stock:
        raise HTTPException(status_code=404, detail="Stock not found")

    # Compute live from OHLCV data
    ohlcv_rows = await crud.get_ohlcv(db, stock_id, interval)
    if not ohlcv_rows:
        return {"error": "No OHLCV data available"}

    df = ohlcv_to_dataframe(list(ohlcv_rows))
    if len(df) < 50:
        return {"error": "Insufficient data for regime classification"}

    df = compute_all_indicators(df)
    df = classify_and_score(df)
    return regime_summary(df)


@router.post("/classify", response_model=list[ClassifyResult])
async def trigger_classification(
    req: ClassifyRequest,
    db: AsyncSession = Depends(get_db),
):
    """Classify regimes for stocks and store in DB."""
    if req.stock_ids:
        stock_ids = req.stock_ids
    else:
        stocks = await crud.get_all_active_stocks(db)
        stock_ids = [s.id for s in stocks]

    results = []
    for sid in stock_ids:
        stock = await crud.get_stock_by_id(db, sid)
        if not stock:
            continue

        ohlcv_rows = await crud.get_ohlcv(db, sid, req.interval)
        if not ohlcv_rows or len(ohlcv_rows) < 50:
            results.append(ClassifyResult(stock_id=sid, symbol=stock.symbol, rows_classified=0, rows_stored=0))
            continue

        df = ohlcv_to_dataframe(list(ohlcv_rows))
        df = compute_all_indicators(df)
        df = classify_and_score(df)

        # Store in DB
        import pandas as pd
        rows = []
        for _, row in df.iterrows():
            dt = row["date"].date() if isinstance(row["date"], pd.Timestamp) else row["date"]
            rows.append({
                "stock_id": sid,
                "date": dt,
                "interval": req.interval,
                "trend": row["trend"],
                "volatility": row["volatility"],
                "regime_id": int(row["regime_id"]),
                "regime_confidence": float(row["regime_confidence"]),
                "quality_score": float(row["quality_score"]),
                "is_transition": bool(row["is_transition"]),
            })

        stored = await crud.bulk_upsert_regimes(db, rows)
        results.append(ClassifyResult(
            stock_id=sid,
            symbol=stock.symbol,
            rows_classified=len(df),
            rows_stored=stored,
        ))

    return results
