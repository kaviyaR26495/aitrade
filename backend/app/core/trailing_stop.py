"""Dynamic Trailing Stop Engine — S/R-aware GTT management.

When price moves favourably past key S/R zones, the stoploss is ratcheted
up to lock in profits via a cancel + re-place GTT cycle (Kite API does
not support GTT modification).

State Machine
─────────────
  INACTIVE → ACTIVE (price > entry + 1 ATR)
  ACTIVE: on each S/R zone breach upward
    1. Cancel existing GTT
    2. Compute new SL = max(prev_SL, breached_zone.price_low - buffer)
    3. Place new GTT OCO with updated SL + original target
    4. Update TradeSignal.current_stoploss, trailing_updates_count

Triggered by a periodic Celery Beat task (every 5 min during market hours).
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TrailingStopState:
    """Current state of a trailing stop for a position."""
    signal_id: int
    stock_id: int
    symbol: str
    exchange: str
    quantity: int
    entry_price: float
    target_price: float
    original_sl: float
    current_sl: float
    last_gtt_id: Optional[str]
    is_active: bool
    updates_count: int


@dataclass
class TrailingStopUpdate:
    """Result of a trailing stop evaluation."""
    should_update: bool
    new_sl: float
    reason: str
    old_sl: float


def evaluate_trailing_stop(
    state: TrailingStopState,
    current_price: float,
    atr: float,
    nearest_support_below_price: Optional[float],
    activation_atr_multiple: float = 1.0,
    buffer_pct: float = 0.002,
) -> TrailingStopUpdate:
    """Evaluate whether the trailing stop should be updated.

    Parameters
    ----------
    state : TrailingStopState
        Current trailing stop state.
    current_price : float
        Latest traded price.
    atr : float
        Current 14-day ATR.
    nearest_support_below_price : float | None
        Midpoint of the nearest support zone below current_price.
        Comes from ``compute_sr_zones()`` re-run on latest data.
    activation_atr_multiple : float
        Trailing stop activates when price is > entry + N × ATR.
    buffer_pct : float
        Buffer below support zone for the new SL placement.

    Returns
    -------
    TrailingStopUpdate
    """
    old_sl = state.current_sl

    # Check activation threshold
    activation_price = state.entry_price + activation_atr_multiple * atr
    if current_price < activation_price:
        return TrailingStopUpdate(
            should_update=False,
            new_sl=old_sl,
            reason=f"price {current_price:.2f} < activation {activation_price:.2f}",
            old_sl=old_sl,
        )

    # Compute candidate new SL
    candidate_sl = old_sl  # never lower than existing

    # Option 1: S/R-aware SL — place just below nearest support
    if nearest_support_below_price is not None:
        sr_sl = nearest_support_below_price * (1 - buffer_pct)
        if sr_sl > candidate_sl:
            candidate_sl = sr_sl

    # Option 2: ATR trail — price minus 1.5 ATR
    atr_trail_sl = current_price - 1.5 * atr
    if atr_trail_sl > candidate_sl:
        candidate_sl = atr_trail_sl

    # Round to tick size
    candidate_sl = round(round(candidate_sl / 0.05) * 0.05, 2)

    # Only update if new SL is meaningfully higher (> 0.5% improvement)
    if candidate_sl <= old_sl * 1.005:
        return TrailingStopUpdate(
            should_update=False,
            new_sl=old_sl,
            reason=f"new SL {candidate_sl:.2f} not meaningfully above {old_sl:.2f}",
            old_sl=old_sl,
        )

    # Ensure new SL doesn't exceed target (would make no sense)
    if candidate_sl >= state.target_price:
        return TrailingStopUpdate(
            should_update=False,
            new_sl=old_sl,
            reason=f"new SL {candidate_sl:.2f} would exceed target {state.target_price:.2f}",
            old_sl=old_sl,
        )

    return TrailingStopUpdate(
        should_update=True,
        new_sl=candidate_sl,
        reason=f"trail SL {old_sl:.2f} → {candidate_sl:.2f} (price={current_price:.2f})",
        old_sl=old_sl,
    )


async def execute_trailing_stop_update(
    state: TrailingStopState,
    new_sl: float,
) -> Optional[str]:
    """Cancel existing GTT and place a new one with updated SL.

    Parameters
    ----------
    state : TrailingStopState
        Current trailing stop state (includes last_gtt_id).
    new_sl : float
        New stoploss price.

    Returns
    -------
    New GTT ID string, or None on failure.
    """
    from app.core.zerodha import get_kite

    kite = get_kite()

    # Step 1: Cancel existing GTT
    if state.last_gtt_id:
        try:
            await asyncio.to_thread(
                kite.delete_gtt, trigger_id=int(state.last_gtt_id)
            )
            logger.info(
                "Cancelled GTT %s for %s (trailing update)",
                state.last_gtt_id, state.symbol,
            )
        except Exception as e:
            logger.warning(
                "Failed to cancel GTT %s for %s: %s",
                state.last_gtt_id, state.symbol, e,
            )
            # Continue anyway — old GTT may have already been triggered

    # Step 2: Place new GTT OCO with updated SL
    try:
        gtt_id = await asyncio.to_thread(
            kite.place_gtt,
            trigger_type=kite.GTT_TYPE_OCO,
            tradingsymbol=state.symbol,
            exchange=state.exchange,
            trigger_values=[new_sl, state.target_price],
            last_price=state.entry_price,
            orders=[
                {
                    "exchange": state.exchange,
                    "tradingsymbol": state.symbol,
                    "transaction_type": "SELL",
                    "quantity": state.quantity,
                    "order_type": "LIMIT",
                    "product": "CNC",
                    "price": new_sl,
                },
                {
                    "exchange": state.exchange,
                    "tradingsymbol": state.symbol,
                    "transaction_type": "SELL",
                    "quantity": state.quantity,
                    "order_type": "LIMIT",
                    "product": "CNC",
                    "price": state.target_price,
                },
            ],
        )
        logger.info(
            "New trailing GTT placed for %s: SL=%.2f → %.2f, target=%.2f → %s",
            state.symbol, state.current_sl, new_sl, state.target_price, gtt_id,
        )
        return str(gtt_id)
    except Exception as e:
        logger.error("Failed to place trailing GTT for %s: %s", state.symbol, e)
        return None
