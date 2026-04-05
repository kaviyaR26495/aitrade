from datetime import date

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.core import zerodha
from app.db import crud

router = APIRouter()


@router.get("/holdings")
async def get_holdings():
    """Get current holdings from Zerodha."""
    try:
        holdings = zerodha.get_holdings()
        return {"holdings": holdings}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch holdings: {e}")


@router.get("/positions")
async def get_positions():
    """Get current positions from Zerodha."""
    try:
        positions = zerodha.get_positions()
        return {"positions": positions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch positions: {e}")


@router.get("/ltp/{symbol}")
async def get_ltp(symbol: str):
    """Get last traded price for a symbol."""
    try:
        ltp = zerodha.get_ltp(symbol)
        return {"symbol": symbol, "ltp": ltp}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch LTP: {e}")


@router.post("/exit-all")
async def exit_all():
    """Emergency kill switch — sell all holdings at market."""
    try:
        holdings = zerodha.get_holdings()
        orders = []
        for h in holdings:
            if h.get("quantity", 0) > 0:
                order_id = zerodha.place_cnc_order(
                    symbol=h["tradingsymbol"],
                    transaction_type="SELL",
                    quantity=h["quantity"],
                    price=None,  # market order
                )
                orders.append({
                    "symbol": h["tradingsymbol"],
                    "quantity": h["quantity"],
                    "order_id": order_id,
                })
        return {"orders_placed": len(orders), "orders": orders}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Exit all failed: {e}")


# ── Daily Morning Reconciliation (08:30 IST) ───────────────────────────
# Fetches the authoritative cash + holdings state from Zerodha and writes
# it to `portfolio_snapshots`.  Position-sizing algorithms (Kelly,
# Vol-Target) must derive their cash figure exclusively from this table
# to avoid T+1 settlement drift, dividend adjustments, or broker rejections
# causing the local ledger to diverge from reality.

@router.post("/reconcile")
async def reconcile_portfolio(db: AsyncSession = Depends(get_db)):
    """Fetch live broker state and persist as today's reconciliation snapshot.

    Run this at 08:30 IST every trading day (via cron or the nightly
    scheduler) *before* any orders are placed.  Can also be triggered
    manually from the Frontend when you suspect ledger drift.
    """
    try:
        snapshot = zerodha.build_portfolio_snapshot()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Zerodha API error: {e}")

    today = date.today()
    row = await crud.upsert_portfolio_snapshot(
        db,
        snapshot_date=today,
        cash_available=snapshot["cash_available"],
        opening_balance=snapshot["opening_balance"],
        holdings_value=snapshot["holdings_value"],
        unrealized_pnl=snapshot["unrealized_pnl"],
        holdings_json=snapshot["holdings"],
        positions_json=snapshot["positions"],
    )

    return {
        "snapshot_date": str(row.snapshot_date),
        "cash_available": row.cash_available,
        "opening_balance": row.opening_balance,
        "holdings_value": row.holdings_value,
        "unrealized_pnl": row.unrealized_pnl,
        "total_equity": row.cash_available + row.holdings_value,
        "reconciled_at": row.reconciled_at.isoformat(),
        "holdings_count": len(snapshot["holdings"]),
        "positions_count": len(snapshot["positions"]),
    }


@router.get("/snapshot")
async def get_latest_snapshot(db: AsyncSession = Depends(get_db)):
    """Return the most recent reconciliation snapshot from the database."""
    row = await crud.get_latest_portfolio_snapshot(db)
    if row is None:
        raise HTTPException(
            status_code=404,
            detail="No reconciliation snapshot found. Run POST /portfolio/reconcile first.",
        )
    return {
        "snapshot_date": str(row.snapshot_date),
        "cash_available": row.cash_available,
        "opening_balance": row.opening_balance,
        "holdings_value": row.holdings_value,
        "unrealized_pnl": row.unrealized_pnl,
        "total_equity": row.cash_available + row.holdings_value,
        "reconciled_at": row.reconciled_at.isoformat(),
        "holdings": row.holdings_json,
        "positions": row.positions_json,
    }


@router.get("/snapshot/{snapshot_date}")
async def get_snapshot_by_date(snapshot_date: date, db: AsyncSession = Depends(get_db)):
    """Return the reconciliation snapshot for a specific date."""
    row = await crud.get_portfolio_snapshot_by_date(db, snapshot_date)
    if row is None:
        raise HTTPException(status_code=404, detail=f"No snapshot for {snapshot_date}")
    return {
        "snapshot_date": str(row.snapshot_date),
        "cash_available": row.cash_available,
        "opening_balance": row.opening_balance,
        "holdings_value": row.holdings_value,
        "unrealized_pnl": row.unrealized_pnl,
        "total_equity": row.cash_available + row.holdings_value,
        "reconciled_at": row.reconciled_at.isoformat(),
        "holdings": row.holdings_json,
        "positions": row.positions_json,
    }
