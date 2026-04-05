from __future__ import annotations

import asyncio
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


class ExecuteSignalRequest(BaseModel):
    stock_id: int
    ensemble_prediction_id: int | None = None
    quantity: int
    exchange: str = "NSE"
    product: str = "CNC"
    gtt_sell_pct: float = 15.0
    gtt_stoploss_pct: float = 5.0
    max_attempts: int = 5


@router.post("/execute-signal")
async def execute_signal(
    req: ExecuteSignalRequest,
    db: AsyncSession = Depends(get_db),
):
    """Limit-chase a BUY then immediately place protective GTT OCO stoploss+target.

    The GTT trigger prices are anchored to the *actual fill price* returned by
    Zerodha order_history so slippage is automatically absorbed.  Returns the
    chase order id and GTT id so the caller can track both legs.
    """
    stock = await crud.get_stock_by_id(db, req.stock_id)
    if not stock:
        raise HTTPException(status_code=404, detail="Stock not found")

    # --- Ex-date / corporate action guard ---
    # Check DB for an existing active block first (persists across restarts).
    # If none exists, run a live OHLC gap check; if a gap is detected, create a
    # 48-hour block in the DB and refuse the trade.
    from datetime import timedelta
    from app.db.models import now_ist

    active_block = await crud.get_active_ca_block(db, stock.symbol, req.exchange)
    if active_block:
        raise HTTPException(
            status_code=423,
            detail=(
                f"{stock.symbol} is blocked until {active_block.blocked_until.isoformat()} "
                f"due to a suspected corporate action (gap={active_block.gap_pct:.1f}%). "
                "Trading resumes automatically after the 48-hour cooldown."
            ),
        )

    gap_detected, gap_pct = await zerodha.async_detect_corporate_action_gap(
        stock.symbol, req.exchange
    )
    if gap_detected:
        blocked_until = now_ist() + timedelta(hours=48)
        await crud.create_ca_block(
            db,
            symbol=stock.symbol,
            exchange=req.exchange,
            gap_pct=gap_pct,
            blocked_until=blocked_until,
            reason=f"Open/prev-close gap {gap_pct:.1f}% ≥ threshold on ex-date",
        )
        raise HTTPException(
            status_code=423,
            detail=(
                f"{stock.symbol} open/prev-close gap is {gap_pct:.1f}% — suspected ex-date "
                f"(dividend / split / bonus). Trading blocked for 48 hours until "
                f"{blocked_until.isoformat()} to allow adj_close to normalise."
            ),
        )

    # --- BUY leg: limit-chase until filled or MARKET fallback ---
    try:
        result = await zerodha.async_place_limit_chase_order(
            symbol=stock.symbol,
            transaction_type="BUY",
            quantity=req.quantity,
            exchange=req.exchange,
            product=req.product,
            max_attempts=req.max_attempts,
        )
    except zerodha.CorporateActionGapError as exc:
        # Last-resort guard fired inside the chase loop (race condition).
        # Write a 48-h DB block so restarts honour the cooldown too.
        from datetime import timedelta
        from app.db.models import now_ist
        blocked_until = now_ist() + timedelta(hours=48)
        await crud.create_ca_block(
            db,
            symbol=stock.symbol,
            exchange=req.exchange,
            gap_pct=exc.gap_pct,
            blocked_until=blocked_until,
            reason=f"CA gap {exc.gap_pct:.1f}% detected during limit-chase (last-resort guard)",
        )
        raise HTTPException(status_code=423, detail=str(exc))
    if not result:
        raise HTTPException(status_code=502, detail="Limit-chase order failed — no fill obtained")

    chase_order_id: str = result["order_id"]
    fill_price: float = result["fill_price"]

    # --- Persist the BUY leg immediately so the position is never invisible ---
    from app.db.models import TradeOrder
    chase_record = TradeOrder(
        stock_id=req.stock_id,
        ensemble_prediction_id=req.ensemble_prediction_id,
        variety="regular",
        transaction_type="BUY",
        quantity=req.quantity,
        price=fill_price,
        status="complete",
        zerodha_order_id=chase_order_id,
    )
    db.add(chase_record)
    await db.commit()
    await db.refresh(chase_record)

    # --- GTT leg: 3-attempt retry so a transient network blip can't leave
    #     the position naked. If all retries fail, fire an emergency alert.   ---
    gtt_id: int | None = None
    last_gtt_exc: Exception | None = None
    for attempt in range(1, 4):
        try:
            gtt_id = await zerodha.async_place_gtt_order(
                symbol=stock.symbol,
                exchange=req.exchange,
                avg_price=fill_price,
                quantity=req.quantity,
                sell_pct=req.gtt_sell_pct,
                stoploss_pct=req.gtt_stoploss_pct,
            )
            if gtt_id:
                last_gtt_exc = None
                break
        except Exception as exc:
            last_gtt_exc = exc
            logger.error(
                "GTT placement attempt %d/3 failed for %s @ %.2f: %s",
                attempt, stock.symbol, fill_price, exc,
            )
            if attempt < 3:
                await asyncio.sleep(2)

    if not gtt_id:
        # Position held with zero stoploss protection — alert immediately.
        alert_msg = (
            f"URGENT: Bought {req.quantity} {stock.symbol} @ {fill_price:.2f} "
            f"(order {chase_order_id}) but GTT stoploss failed after 3 attempts. "
            f"Last error: {last_gtt_exc}"
        )
        zerodha.send_emergency_alert(alert_msg)
        raise HTTPException(
            status_code=502,
            detail=(
                f"BUY filled (order_id={chase_order_id}, fill_price={fill_price}) "
                f"but GTT placement failed after 3 attempts: {last_gtt_exc}"
            ),
        )

    # --- Persist the GTT leg ---
    gtt_record = TradeOrder(
        stock_id=req.stock_id,
        ensemble_prediction_id=req.ensemble_prediction_id,
        variety="gtt",
        transaction_type="SELL",
        quantity=req.quantity,
        price=fill_price,
        status="placed",
        zerodha_order_id=str(gtt_id) if gtt_id else None,
    )
    db.add(gtt_record)
    await db.commit()
    await db.refresh(chase_record)
    await db.refresh(gtt_record)

    return {
        "chase_order_id": chase_order_id,
        "fill_price": fill_price,
        "gtt_id": gtt_id,
        "chase_db_id": chase_record.id,
        "gtt_db_id": gtt_record.id,
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
