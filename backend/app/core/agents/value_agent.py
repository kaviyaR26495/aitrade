"""Value Agent — reads fundamental z-scores (PE, ROE, D/E) to produce a valuation verdict."""
from __future__ import annotations

import json
import logging
from datetime import date

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.agents import AgentVerdict, llm_complete

logger = logging.getLogger(__name__)


async def run(
    db: AsyncSession,
    stock_id: int,
    symbol: str,
    target_date: date | None = None,
) -> AgentVerdict:
    from sqlalchemy import select
    from app.db.models import StockFundamentalZScore, StockFundamentalPIT

    # Latest z-score row (closest to or before target_date)
    q = select(StockFundamentalZScore).where(
        StockFundamentalZScore.stock_id == stock_id
    )
    if target_date:
        q = q.where(StockFundamentalZScore.date <= target_date)
    q = q.order_by(StockFundamentalZScore.date.desc()).limit(1)
    result = await db.execute(q)
    zscore = result.scalar_one_or_none()

    # Also grab raw PIT for sector context
    q2 = select(StockFundamentalPIT).where(
        StockFundamentalPIT.stock_id == stock_id
    )
    if target_date:
        q2 = q2.where(StockFundamentalPIT.date <= target_date)
    q2 = q2.order_by(StockFundamentalPIT.date.desc()).limit(1)
    result2 = await db.execute(q2)
    pit = result2.scalar_one_or_none()

    if zscore is None and pit is None:
        return AgentVerdict(
            agent_name="ValueAgent",
            signal="ABSTAIN",
            confidence=0.0,
            reasoning="No fundamental data available.",
        )

    summary = {
        "symbol": symbol,
        "pe_ratio": pit.pe_ratio if pit else None,
        "forward_pe": pit.forward_pe if pit else None,
        "pb_ratio": pit.pb_ratio if pit else None,
        "roe": pit.roe if pit else None,
        "debt_equity": pit.debt_equity if pit else None,
        "pe_zscore_3y": round(zscore.pe_zscore_3y, 3) if zscore and zscore.pe_zscore_3y else None,
        "pe_zscore_sector": round(zscore.pe_zscore_sector, 3) if zscore and zscore.pe_zscore_sector else None,
        "roe_norm": round(zscore.roe_norm, 3) if zscore and zscore.roe_norm else None,
        "debt_equity_norm": round(zscore.debt_equity_norm, 3) if zscore and zscore.debt_equity_norm else None,
    }

    system = (
        "You are an expert value investor focused on Indian equities. "
        "Z-scores > +1.5 = expensive (bearish), < -1.5 = cheap (bullish). "
        "Return ONLY a JSON object: signal (BUY|SELL|HOLD), confidence (0-1), reasoning (1-2 sentences)."
    )
    prompt = f"Fundamental data for {symbol}:\n{json.dumps(summary, indent=2)}"

    try:
        raw = await llm_complete(db, [{"role": "user", "content": prompt}], system_prompt=system)
        start = raw.find("{")
        end = raw.rfind("}") + 1
        result_dict = json.loads(raw[start:end]) if start >= 0 and end > start else {}
        signal = result_dict.get("signal", "HOLD")
        confidence = float(result_dict.get("confidence", 0.5))
        reasoning = result_dict.get("reasoning", raw[:300])
    except Exception as exc:
        logger.warning("ValueAgent LLM failed for %s: %s — rule fallback", symbol, exc)
        # Rule-based fallback: use pe_zscore_3y
        pe_z = zscore.pe_zscore_3y if zscore and zscore.pe_zscore_3y else 0.0
        if pe_z < -1.5 and zscore and zscore.roe_norm and zscore.roe_norm > 0:
            signal, confidence = "BUY", 0.65
        elif pe_z > 1.5:
            signal, confidence = "SELL", 0.60
        else:
            signal, confidence = "HOLD", 0.50
        reasoning = f"Rule fallback: pe_zscore_3y={pe_z:.2f}"

    return AgentVerdict(
        agent_name="ValueAgent",
        signal=signal,
        confidence=max(0.0, min(1.0, confidence)),
        reasoning=reasoning,
        metadata=summary,
    )
