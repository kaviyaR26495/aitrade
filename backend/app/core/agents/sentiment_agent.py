"""Sentiment Agent — reads FinBERT + LLM impact scores for the given stock."""
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
    from app.db.models import StockSentiment

    q = select(StockSentiment).where(StockSentiment.stock_id == stock_id)
    if target_date:
        q = q.where(StockSentiment.date <= target_date)
    q = q.order_by(StockSentiment.date.desc()).limit(1)
    result = await db.execute(q)
    row = result.scalar_one_or_none()

    if row is None:
        return AgentVerdict(
            agent_name="SentimentAgent",
            signal="ABSTAIN",
            confidence=0.0,
            reasoning="No sentiment data available.",
        )

    summary = {
        "symbol": symbol,
        "date": str(row.date),
        "headline_count": row.headline_count,
        "avg_finbert_score": round(row.avg_finbert_score, 4) if row.avg_finbert_score else None,
        "llm_impact_score": round(row.llm_impact_score, 4) if row.llm_impact_score else None,
        "llm_summary": row.llm_summary,
    }

    system = (
        "You are a news-sentiment analyst for Indian equities. "
        "avg_finbert_score: -1 (very negative) to +1 (very positive). "
        "llm_impact_score: -1 (severe negative event) to +1 (major catalyst). "
        "Return ONLY a JSON object: signal (BUY|SELL|HOLD), confidence (0-1), reasoning (1-2 sentences)."
    )
    prompt = f"Sentiment data for {symbol}:\n{json.dumps(summary, indent=2)}"

    try:
        raw = await llm_complete(db, [{"role": "user", "content": prompt}], system_prompt=system)
        start = raw.find("{")
        end = raw.rfind("}") + 1
        result_dict = json.loads(raw[start:end]) if start >= 0 and end > start else {}
        signal = result_dict.get("signal", "HOLD")
        confidence = float(result_dict.get("confidence", 0.5))
        reasoning = result_dict.get("reasoning", raw[:300])
    except Exception as exc:
        logger.warning("SentimentAgent LLM failed for %s: %s — rule fallback", symbol, exc)
        # Rule-based fallback
        fs = row.avg_finbert_score or 0.0
        ls = row.llm_impact_score or 0.0
        combined = (fs + ls) / 2
        if combined > 0.3:
            signal, confidence = "BUY", 0.60
        elif combined < -0.3:
            signal, confidence = "SELL", 0.60
        else:
            signal, confidence = "HOLD", 0.45
        reasoning = f"Rule fallback: finbert={fs:.3f}, llm_impact={ls:.3f}"

    return AgentVerdict(
        agent_name="SentimentAgent",
        signal=signal,
        confidence=max(0.0, min(1.0, confidence)),
        reasoning=reasoning,
        metadata=summary,
    )
