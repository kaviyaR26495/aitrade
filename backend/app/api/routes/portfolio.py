from fastapi import APIRouter, HTTPException

from app.core import zerodha

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
