"""Investment Committee — orchestrates 4 specialised agents and produces a final decision.

Voting weights (sum = 1.0):
  Technical : 0.40
  Value     : 0.20
  Sentiment : 0.20
  (Risk is a HARD GATE, not a vote — it can veto any BUY)

A BUY/SELL verdict requires:
  - Weighted confidence ≥ 0.65
  - TechnicalAgent must not be HOLD/ABSTAIN (anchor requirement)
  - RiskManagerAgent must APPROVE (not HOLD / veto the entry)

POST /api/agents/evaluate   →  CommitteeDecision JSON
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Literal

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.agents import AgentVerdict

logger = logging.getLogger(__name__)

WEIGHTS = {
    "TechnicalAgent": 0.40,
    "ValueAgent": 0.20,
    "SentimentAgent": 0.20,
}
CONFIDENCE_GATE = 0.65
Signal = Literal["BUY", "SELL", "HOLD"]


@dataclass
class CommitteeDecision:
    symbol: str
    final_signal: Signal
    weighted_confidence: float
    regime_id: int | None
    verdicts: list[AgentVerdict]
    risk_verdict: AgentVerdict
    reasoning: str
    metadata: dict = field(default_factory=dict)


def _weighted_vote(verdicts: list[AgentVerdict]) -> tuple[Signal, float]:
    """Compute weighted signal vote from non-risk agents."""
    scores: dict[str, float] = {"BUY": 0.0, "SELL": 0.0, "HOLD": 0.0}

    for v in verdicts:
        w = WEIGHTS.get(v.agent_name, 0.0)
        if v.signal in scores:
            scores[v.signal] += w * v.confidence
        # ABSTAIN contributes to HOLD
        elif v.signal == "ABSTAIN":
            scores["HOLD"] += w * 0.5

    total = sum(scores.values()) or 1.0
    normalised = {k: v / total for k, v in scores.items()}
    top_signal = max(normalised, key=lambda k: normalised[k])
    top_confidence = normalised[top_signal]
    return top_signal, top_confidence


async def evaluate(
    db: AsyncSession,
    stock_id: int,
    symbol: str,
    target_date: date | None = None,
    atr_value: float | None = None,
) -> CommitteeDecision:
    """Run all agents in parallel and synthesise a final trading signal."""
    import app.core.agents.technical_agent as technical_agent
    import app.core.agents.value_agent as value_agent
    import app.core.agents.sentiment_agent as sentiment_agent
    import app.core.agents.risk_agent as risk_agent
    from app.db import crud

    # ── Run voting agents in parallel ─────────────────────────────────
    tech_task = technical_agent.run(db, stock_id, symbol, target_date)
    val_task = value_agent.run(db, stock_id, symbol, target_date)
    sent_task = sentiment_agent.run(db, stock_id, symbol, target_date)

    tech_v, val_v, sent_v = await asyncio.gather(tech_task, val_task, sent_task)
    voting_verdicts = [tech_v, val_v, sent_v]

    # ── Weighted vote ─────────────────────────────────────────────────
    raw_signal, weighted_conf = _weighted_vote(voting_verdicts)

    # ── Technical anchor check ────────────────────────────────────────
    if tech_v.signal in ("HOLD", "ABSTAIN") and raw_signal == "BUY":
        raw_signal = "HOLD"
        reasoning = "TechnicalAgent anchor failed — overriding committee BUY to HOLD."
    elif weighted_conf < CONFIDENCE_GATE:
        raw_signal = "HOLD"
        reasoning = f"Confidence gate not met ({weighted_conf:.2%} < {CONFIDENCE_GATE:.0%})."
    else:
        reasoning = (
            f"Committee vote: {raw_signal} with {weighted_conf:.2%} weighted confidence. "
            f"Tech={tech_v.signal}({tech_v.confidence:.2f}), "
            f"Value={val_v.signal}({val_v.confidence:.2f}), "
            f"Sentiment={sent_v.signal}({sent_v.confidence:.2f})."
        )

    # ── Risk gate (hard veto) ─────────────────────────────────────────
    risk_v = await risk_agent.run(
        db, stock_id, symbol,
        proposed_signal=raw_signal,
        atr_value=atr_value,
        target_date=target_date,
    )
    final_signal = risk_v.signal  # risk may downgrade to HOLD

    if final_signal != raw_signal:
        reasoning += f" RiskManager vetoed to HOLD: {risk_v.reasoning}"

    # ── Regime lookup for metadata ────────────────────────────────────
    regime_row = await crud.get_latest_regime(db, stock_id, "day")
    regime_id = regime_row.regime_id if regime_row else None

    return CommitteeDecision(
        symbol=symbol,
        final_signal=final_signal,
        weighted_confidence=weighted_conf,
        regime_id=regime_id,
        verdicts=voting_verdicts,
        risk_verdict=risk_v,
        reasoning=reasoning,
        metadata={"stock_id": stock_id, "target_date": str(target_date)},
    )
