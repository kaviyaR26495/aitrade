from __future__ import annotations
import asyncio
from datetime import date
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
import logging
from app.api.deps import get_db
from app.db import crud
from app.core import zerodha
from app.ml.predictor import ACTION_MAP
from app.ml import predictor
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks

logger = logging.getLogger(__name__)
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


class RunPredictionsRequest(BaseModel):
    interval: str = "day"
    agreement_required: bool = True
    stock_ids: list[int] | None = None
    target_date: date | None = None


@router.get("/predictions")
async def get_predictions(
    db: AsyncSession = Depends(get_db),
    target_date: date | None = Query(None),
    batch_id: str | None = Query(None),
    interval: str = Query("day"),
    min_confidence: float = Query(0.65),
    agreement_only: bool = Query(True),
):
    """Get ensemble predictions from DB with bulk stock resolution."""
    predictions = await crud.get_ensemble_predictions_for_date(
        db, 
        target_date=target_date, 
        batch_id=batch_id, 
        interval=interval, 
        min_confidence=min_confidence, 
        agreement_only=agreement_only
    )
    
    if not predictions:
        return []

    # Optimize: Bulk fetch all required stocks in one query instead of sequentially
    from app.db.models import Stock
    from sqlalchemy import select as sqla_select
    stock_ids = list({p.stock_id for p in predictions})
    stock_res = await db.execute(sqla_select(Stock).where(Stock.id.in_(stock_ids)))
    stocks_map = {s.id: s for s in stock_res.scalars().all()}

    results = []
    for p in predictions:
        stock = stocks_map.get(p.stock_id)
        results.append({
            "id": p.id,
            "stock_id": p.stock_id,
            "symbol": stock.symbol if stock else f"#{p.stock_id}",
            "date": str(p.date),
            "interval": p.interval,
            "action": ACTION_MAP.get(p.action, "HOLD"),
            "confidence": p.confidence,
            "knn_action": ACTION_MAP.get(p.knn_action, "HOLD"),
            "knn_confidence": p.knn_confidence,
            "lstm_action": ACTION_MAP.get(p.lstm_action, "HOLD"),
            "lstm_confidence": p.lstm_confidence,
            "agreement": p.agreement,
            "regime_id": p.regime_id,
        })
    return results


@router.get("/predictions/forward-look")
async def get_forward_look(
    stock_id: int,
    after_date: date,
    interval: str = Query("day"),
    count: int = Query(5),
    db: AsyncSession = Depends(get_db),
):
    """Fetch consecutive OHLCV rows starting after after_date (for performance visualization)."""
    from sqlalchemy import select
    from app.db.models import StockOHLCV
    
    stmt = (
        select(StockOHLCV)
        .where(
            StockOHLCV.stock_id == stock_id,
            StockOHLCV.interval == interval,
            StockOHLCV.date > after_date
        )
        .order_by(StockOHLCV.date.asc())
        .limit(count)
    )
    
    result = await db.execute(stmt)
    rows = result.scalars().all()
    
    return [
        {
            "date": str(r.date),
            "open": r.open,
            "high": r.high,
            "low": r.low,
            "close": r.close,
            "volume": r.volume
        }
        for r in rows
    ]


@router.get("/predictions/jobs/{job_id}")
async def get_prediction_job(
    job_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get status of an async prediction run."""
    job = await crud.get_prediction_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.delete("/predictions/jobs/{job_id}")
async def cancel_prediction_job(
    job_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Mark a prediction run for cancellation."""
    await crud.update_prediction_job(db, job_id, status="cancelled")
    return {"status": "requesting_cancellation", "job_id": job_id}


@router.post("/run-predictions")
async def run_predictions(
    background_tasks: BackgroundTasks,
    req: RunPredictionsRequest,
    db: AsyncSession = Depends(get_db),
):
    """Kick off ensemble predictions as a background task."""
    from app.db.database import async_session_factory
    from app.core import data_service
    import uuid
    
    # 1. Resolve stocks to be predicted (from request or universe) — bulk fetch
    if req.stock_ids:
        from app.db.models import Stock
        from sqlalchemy import select as sqla_select
        result = await db.execute(sqla_select(Stock).where(Stock.id.in_(req.stock_ids)))
        stocks = result.scalars().all()
    else:
        stocks = await data_service.get_universe_stocks(db)
    
    if not stocks:
        raise HTTPException(status_code=400, detail="No stocks identified for prediction")

    # 2. Create Job record
    job_id = str(uuid.uuid4())
    batch_id = str(uuid.uuid4())

    # For partial re-runs (specific stocks selected), reuse the latest existing
    # batch_id for the same target_date so predictions stay grouped in the same
    # batch instead of fragmenting into a separate 1-stock batch.
    if req.stock_ids:
        from app.db.models import EnsemblePrediction
        from sqlalchemy import select as sqla_select, desc as sqla_desc
        from datetime import date as date_type
        tdate = req.target_date or date_type.today()
        q_existing = (
            sqla_select(EnsemblePrediction.batch_id)
            .where(
                EnsemblePrediction.date == tdate,
                EnsemblePrediction.interval == req.interval,
            )
            .order_by(sqla_desc(EnsemblePrediction.run_at))
            .limit(1)
        )
        existing_row = await db.execute(q_existing)
        existing_batch_id = existing_row.scalar_one_or_none()
        if existing_batch_id:
            batch_id = existing_batch_id

    await crud.create_prediction_job(db, job_id, total_stocks=len(stocks), batch_id=batch_id)

    # 3. Define background task wrapper
    async def prediction_wrapper(jid, b_id, s_ids):
        async with async_session_factory() as b_db:
            try:
                await predictor.run_daily_predictions(
                    b_db,
                    stock_ids=s_ids,
                    interval=req.interval,
                    agreement_required=req.agreement_required,
                    target_date=req.target_date,
                    job_id=jid,
                    batch_id=b_id
                )
            except Exception as e:
                logger.error(f"Background prediction job {jid} failed: {e}")
                # Use a FRESH session — b_db may be in a broken/rolled-back state
                try:
                    async with async_session_factory() as err_db:
                        await crud.update_prediction_job(err_db, jid, status="failed", error=str(e)[:2000])
                except Exception as db_err:
                    logger.error(f"Failed to mark prediction job {jid} as failed: {db_err}")

    # 4. Launch!
    stock_ids = [s.id for s in stocks]
    background_tasks.add_task(prediction_wrapper, job_id, batch_id, stock_ids)

    return {"job_id": job_id, "batch_id": batch_id, "total_stocks": len(stocks)}



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
        status="submitted",
        zerodha_order_id=str(order_id) if order_id else None,
        tag=req.tag,
    )
    db.add(order)
    await db.commit()
    await db.refresh(order)

    return {
        "order_id": order.id,
        "zerodha_order_id": order.zerodha_order_id,
        "status": "submitted",
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
        status="filled",
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
        status="submitted",
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
@router.get("/batches")
async def get_batches(
    interval: str = Query("day"),
    db: AsyncSession = Depends(get_db),
):
    """List unique prediction runs for the given interval."""
    batches = await crud.get_prediction_batches(db, interval)
    return batches


@router.delete("/batches/{batch_id}")
async def delete_batch(
    batch_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Delete a specific prediction session."""
    await crud.delete_prediction_batch(db, batch_id)
    return {"status": "deleted", "batch_id": batch_id}


