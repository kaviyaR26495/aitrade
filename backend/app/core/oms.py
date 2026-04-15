"""Order Management System — reconcile open orders against Zerodha broker state.

Called at the start of every prediction run so the portfolio state is current
before any new signals are generated.  Every order is polled individually via
order_history() to avoid exhausting Kite's rate limit (3 req/s).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from app.db import crud
from app.db.models import OrderStatus

logger = logging.getLogger(__name__)

# Map Zerodha status strings → internal OrderStatus
_ZERODHA_STATUS_MAP: dict[str, OrderStatus] = {
    "COMPLETE":                   OrderStatus.filled,
    "OPEN":                       OrderStatus.submitted,
    "TRIGGER PENDING":            OrderStatus.submitted,
    "AMO REQ RECEIVED":           OrderStatus.submitted,
    "PUT ORDER REQ RECEIVED":     OrderStatus.submitted,
    "VALIDATION PENDING":         OrderStatus.submitted,
    "OPEN PENDING":               OrderStatus.submitted,
    "MODIFY PENDING":             OrderStatus.submitted,
    "MODIFY VALIDATION PENDING":  OrderStatus.submitted,
    "CANCELLED":                  OrderStatus.cancelled,
    "CANCELLED AMO":              OrderStatus.cancelled,
    "REJECTED":                   OrderStatus.rejected,
}


async def reconcile_open_orders(db: AsyncSession) -> int:
    """Poll Zerodha for each open order and update the DB with the current state.

    Transitions:
      placed / submitted / partial_fill  →  filled | partial_fill | submitted | cancelled | rejected

    Returns the count of orders whose status was updated.
    Never raises — a broker outage must never block the prediction run.
    """
    from app.core import zerodha  # lazy import — avoids circular at module load

    orders = await crud.get_open_trade_orders(db)
    if not orders:
        return 0

    reconciled = 0
    for order in orders:
        if not order.zerodha_order_id:
            continue
        try:
            raw_status = zerodha._order_status(order.zerodha_order_id)
            new_status = _ZERODHA_STATUS_MAP.get(raw_status.upper())

            if new_status is None:
                logger.warning(
                    "OMS: unknown Zerodha status '%s' for order %s — skipping",
                    raw_status, order.zerodha_order_id,
                )
                continue

            if new_status == order.status:
                continue  # no change, skip DB write

            fill_qty: int | None = None
            fill_price: float | None = None

            if new_status == OrderStatus.filled:
                # Fetch fill details from order history to populate fill columns
                try:
                    history = zerodha.get_kite().order_history(order_id=order.zerodha_order_id)
                    if history:
                        last = history[-1]
                        fill_qty = int(last.get("filled_quantity") or order.quantity)
                        raw_price = float(last.get("average_price") or 0)
                        fill_price = raw_price if raw_price > 0 else None
                        # Demote to partial_fill if broker only filled part of the qty
                        if fill_qty < order.quantity:
                            new_status = OrderStatus.partial_fill
                except Exception as hist_exc:
                    logger.warning(
                        "OMS: could not fetch fill details for order %s: %s",
                        order.zerodha_order_id, hist_exc,
                    )

            await crud.update_trade_order_fill(
                db, order.id,
                status=new_status.value,
                filled_quantity=fill_qty,
                avg_fill_price=fill_price,
            )
            logger.info(
                "OMS: order %s (db_id=%d) %s → %s",
                order.zerodha_order_id, order.id, order.status, new_status.value,
            )
            reconciled += 1

        except Exception as exc:
            logger.warning(
                "OMS: failed to reconcile order %s: %s",
                getattr(order, "zerodha_order_id", "?"), exc,
            )

    return reconciled
