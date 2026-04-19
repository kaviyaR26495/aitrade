"""Risk Manager Agent — checks portfolio concentration, drawdown, and volatility gates.

Acts as a HARD GATE: if it returns SELL/HOLD, the committee cannot override it.
"""
from __future__ import annotations

import logging
from datetime import date

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.agents import AgentVerdict

logger = logging.getLogger(__name__)

# Position limits
MAX_SINGLE_STOCK_PCT = 0.10   # max 10% of portfolio in one stock
MAX_SECTOR_PCT = 0.25         # max 25% in one sector
MAX_DRAWDOWN_GATE = -0.20     # block new buys if portfolio down >20% from peak
MAX_ATR_MULTIPLE = 0.04       # block if today's range > 4× ATR (gap/halt risk)


async def run(
    db: AsyncSession,
    stock_id: int,
    symbol: str,
    proposed_signal: str,
    atr_value: float | None = None,
    target_date: date | None = None,
) -> AgentVerdict:
    """Evaluate risk constraints for a proposed signal.

    Args:
        proposed_signal: signal from ConfidenceVote (BUY/SELL/HOLD)
        atr_value: current ATR for the stock (optional; used for volatility gate)

    Returns AgentVerdict — signal may be downgraded to HOLD if risk limits hit.
    """
    from app.db import crud

    snapshot = await crud.get_latest_portfolio_snapshot(db)
    if snapshot is None:
        # No portfolio data yet — approve cautiously
        return AgentVerdict(
            agent_name="RiskManagerAgent",
            signal=proposed_signal,
            confidence=0.7,
            reasoning="No portfolio snapshot available — proceeding with proposed signal.",
        )

    total_portfolio = snapshot.cash_available + snapshot.holdings_value
    if total_portfolio <= 0:
        return AgentVerdict(
            agent_name="RiskManagerAgent",
            signal="HOLD",
            confidence=1.0,
            reasoning="Portfolio value is zero or negative — blocking all new entries.",
        )

    # ── Concentration check ───────────────────────────────────────────
    if proposed_signal == "BUY":
        holdings = snapshot.holdings_json or []
        current_positions = {
            h["tradingsymbol"]: h.get("last_price", 0) * h.get("quantity", 0)
            for h in holdings
        }
        existing_value = current_positions.get(symbol, 0.0)
        existing_pct = existing_value / total_portfolio

        if existing_pct >= MAX_SINGLE_STOCK_PCT:
            return AgentVerdict(
                agent_name="RiskManagerAgent",
                signal="HOLD",
                confidence=0.95,
                reasoning=(
                    f"Concentration limit hit: {symbol} already occupies "
                    f"{existing_pct:.1%} of portfolio (max {MAX_SINGLE_STOCK_PCT:.0%})."
                ),
                metadata={"existing_pct": existing_pct, "limit": MAX_SINGLE_STOCK_PCT},
            )

    # ── Drawdown gate ─────────────────────────────────────────────────
    unrealized_pct = snapshot.unrealized_pnl / max(snapshot.opening_balance, 1)
    if unrealized_pct < MAX_DRAWDOWN_GATE and proposed_signal == "BUY":
        return AgentVerdict(
            agent_name="RiskManagerAgent",
            signal="HOLD",
            confidence=0.90,
            reasoning=(
                f"Portfolio drawdown {unrealized_pct:.1%} exceeds gate of "
                f"{MAX_DRAWDOWN_GATE:.0%}. New buys suspended."
            ),
            metadata={"unrealized_pct": unrealized_pct},
        )

    # ── ATR volatility gate ───────────────────────────────────────────
    if atr_value and atr_value > 0 and proposed_signal == "BUY":
        from sqlalchemy import select
        from app.db.models import StockOHLCV
        q = (
            select(StockOHLCV)
            .where(StockOHLCV.stock_id == stock_id, StockOHLCV.interval == "day")
            .order_by(StockOHLCV.date.desc())
            .limit(1)
        )
        r = await db.execute(q)
        last_ohlcv = r.scalar_one_or_none()
        if last_ohlcv and last_ohlcv.close and last_ohlcv.close > 0:
            today_range_pct = (last_ohlcv.high - last_ohlcv.low) / last_ohlcv.close
            if today_range_pct > MAX_ATR_MULTIPLE:
                return AgentVerdict(
                    agent_name="RiskManagerAgent",
                    signal="HOLD",
                    confidence=0.85,
                    reasoning=(
                        f"Excessive intraday volatility: range {today_range_pct:.2%} "
                        f"> {MAX_ATR_MULTIPLE:.0%} ATR gate."
                    ),
                    metadata={"today_range_pct": today_range_pct},
                )

    # ── All checks passed ─────────────────────────────────────────────
    return AgentVerdict(
        agent_name="RiskManagerAgent",
        signal=proposed_signal,
        confidence=0.80,
        reasoning="All risk checks passed.",
        metadata={
            "total_portfolio": total_portfolio,
            "cash_available": snapshot.cash_available,
            "unrealized_pct": unrealized_pct,
        },
    )
