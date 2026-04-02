from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.db import crud
from app.core import zerodha

router = APIRouter()


class OrderRequest(BaseModel):
    stock_id: int
    ensemble_prediction_id: int | None = None
    transaction_type: str  # BUY or SELL
    quantity: int
    price: float | None = None
    sl_price: float | None = None
    target_price: float | None = None
    variety: str = "regular"  # regular, amo, gtt
    tag: str | None = None


@router.get("/predictions")
async def get_predictions(
    db: AsyncSession = Depends(get_db),
    target_date: date | None = Query(None),
    interval: str = Query("day"),
    min_confidence: float = Query(0.65),
    agreement_only: bool = Query(True),
):
    """Get ensemble predictions from DB."""
    if target_date is None:
        target_date = date.today()

    predictions = await crud.get_ensemble_predictions_for_date(
        db, target_date, interval, min_confidence, agreement_only
    )

    results = []
    for p in predictions:
        stock = await crud.get_stock_by_id(db, p.stock_id)
        results.append({
            "id": p.id,
            "stock_id": p.stock_id,
            "symbol": stock.symbol if stock else "?",
            "date": str(p.date),
            "action": p.action,
            "confidence": p.confidence,
            "knn_action": p.knn_action,
            "knn_confidence": p.knn_confidence,
            "lstm_action": p.lstm_action,
            "lstm_confidence": p.lstm_confidence,
            "agreement": p.agreement,
            "regime_id": p.regime_id,
        })

    return results


class RunPredictionsRequest(BaseModel):
    knn_name: str = "latest"
    lstm_name: str = "latest"
    knn_weight: float = 0.5
    lstm_weight: float = 0.5
    agreement_required: bool = True
    stock_ids: list[int] | None = None
    interval: str = "day"


@router.post("/run-predictions")
async def run_predictions(
    req: RunPredictionsRequest,
    db: AsyncSession = Depends(get_db),
):
    """Run ensemble predictions for all active stocks (or specified subset)."""
    from app.ml.predictor import run_daily_predictions

    result = await run_daily_predictions(
        db=db,
        knn_name=req.knn_name,
        lstm_name=req.lstm_name,
        knn_weight=req.knn_weight,
        lstm_weight=req.lstm_weight,
        agreement_required=req.agreement_required,
        interval=req.interval,
        stock_ids=req.stock_ids,
    )
    return result


@router.post("/order")
async def place_order(
    req: OrderRequest,
    db: AsyncSession = Depends(get_db),
):
    """Place a Zerodha CNC order."""
    stock = await crud.get_stock_by_id(db, req.stock_id)
    if not stock:
        raise HTTPException(status_code=404, detail="Stock not found")

    if req.transaction_type not in ("BUY", "SELL"):
        raise HTTPException(status_code=400, detail="transaction_type must be BUY or SELL")

    # Place order via Zerodha
    try:
        if req.variety == "gtt" and req.sl_price and req.target_price:
            order_id = zerodha.place_gtt_order(
                symbol=stock.symbol,
                transaction_type=req.transaction_type,
                quantity=req.quantity,
                price=req.price,
                sl_price=req.sl_price,
                target_price=req.target_price,
            )
        else:
            order_id = zerodha.place_cnc_order(
                symbol=stock.symbol,
                transaction_type=req.transaction_type,
                quantity=req.quantity,
                price=req.price,
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Order placement failed: {e}")

    # Store order in DB
    from app.db.models import TradeOrder
    order = TradeOrder(
        stock_id=req.stock_id,
        ensemble_prediction_id=req.ensemble_prediction_id,
        variety=req.variety,
        transaction_type=req.transaction_type,
        quantity=req.quantity,
        price=req.price,
        sl_price=req.sl_price,
        target_price=req.target_price,
        status="placed",
        zerodha_order_id=str(order_id) if order_id else None,
        tag=req.tag,
    )
    db.add(order)
    await db.commit()
    await db.refresh(order)

    return {
        "order_id": order.id,
        "zerodha_order_id": order.zerodha_order_id,
        "status": "placed",
    }


@router.get("/orders")
async def list_orders(
    db: AsyncSession = Depends(get_db),
    limit: int = Query(50),
):
    """List recent trade orders."""
    from sqlalchemy import select
    from app.db.models import TradeOrder
    result = await db.execute(
        select(TradeOrder).order_by(TradeOrder.timestamp.desc()).limit(limit)
    )
    orders = result.scalars().all()
    return [
        {
            "id": o.id,
            "stock_id": o.stock_id,
            "transaction_type": o.transaction_type,
            "quantity": o.quantity,
            "price": o.price,
            "status": o.status,
            "zerodha_order_id": o.zerodha_order_id,
            "timestamp": str(o.timestamp),
        }
        for o in orders
    ]
