from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.api.deps import get_db
from app.db import crud
from app.db.models import BacktestResult as BacktestResultModel

router = APIRouter()


class BacktestRequest(BaseModel):
    model_type: str  # "rl", "knn", "lstm", "ensemble"
    model_id: int
    stock_ids: list[int]
    interval: str = "day"
    start_date: date | None = None
    end_date: date | None = None
    initial_capital: float = 100_000.0
    stoploss_pct: float = 5.0
    target_pct: float | None = None
    min_confidence: float = 0.65
    max_positions: int = 10


class BacktestSummary(BaseModel):
    id: int
    model_type: str
    model_id: int
    total_return: float | None
    win_rate: float | None
    max_drawdown: float | None
    sharpe: float | None
    profit_factor: float | None
    trades_count: int | None

    class Config:
        from_attributes = True


@router.post("/run")
async def run_backtest(
    req: BacktestRequest,
    db: AsyncSession = Depends(get_db),
):
    """Run backtest for a model on given date range."""
    valid_types = {"rl", "knn", "lstm", "ensemble"}
    if req.model_type not in valid_types:
        raise HTTPException(status_code=400, detail=f"model_type must be one of {valid_types}")

    # Store placeholder result (actual backtesting runs via worker)
    from app.db.models import BacktestResult as BRM
    result = BRM(
        model_type=req.model_type,
        model_id=req.model_id,
        stock_id=req.stock_ids[0] if req.stock_ids else None,
        interval=req.interval,
        start_date=req.start_date or date(2020, 1, 1),
        end_date=req.end_date or date.today(),
    )
    db.add(result)
    await db.commit()
    await db.refresh(result)

    return {
        "backtest_id": result.id,
        "status": "pending",
        "message": f"Backtest created for {req.model_type} model {req.model_id}. Trigger execution via worker.",
    }


@router.get("/results/{backtest_id}")
async def get_backtest_results(
    backtest_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Get backtest results."""
    result = await db.execute(
        select(BacktestResultModel).where(BacktestResultModel.id == backtest_id)
    )
    bt = result.scalar_one_or_none()
    if not bt:
        raise HTTPException(status_code=404, detail="Backtest not found")

    return {
        "id": bt.id,
        "model_type": bt.model_type,
        "model_id": bt.model_id,
        "stock_id": bt.stock_id,
        "interval": bt.interval,
        "start_date": str(bt.start_date),
        "end_date": str(bt.end_date),
        "total_return": bt.total_return,
        "win_rate": bt.win_rate,
        "max_drawdown": bt.max_drawdown,
        "sharpe": bt.sharpe,
        "profit_factor": bt.profit_factor,
        "trades_count": bt.trades_count,
        "trade_log": bt.trade_log,
    }


@router.get("/results", response_model=list[BacktestSummary])
async def list_backtest_results(
    db: AsyncSession = Depends(get_db),
    model_type: str | None = Query(None),
):
    """List all backtest results."""
    q = select(BacktestResultModel).order_by(BacktestResultModel.id.desc())
    if model_type:
        q = q.where(BacktestResultModel.model_type == model_type)
    result = await db.execute(q)
    return result.scalars().all()
