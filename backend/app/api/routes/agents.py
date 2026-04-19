"""Agents API — exposes the Investment Committee evaluation endpoint."""
from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.core.agents.investment_committee import CommitteeDecision, evaluate

router = APIRouter(prefix="/api/agents", tags=["agents"])


class EvaluateRequest(BaseModel):
    stock_id: int
    symbol: str
    target_date: date | None = None
    atr_value: float | None = None


class Verdict(BaseModel):
    agent_name: str
    signal: str
    confidence: float
    reasoning: str
    metadata: dict = {}


class EvaluateResponse(BaseModel):
    symbol: str
    final_signal: str
    weighted_confidence: float
    regime_id: int | None
    reasoning: str
    verdicts: list[Verdict]
    risk_verdict: Verdict
    metadata: dict = {}


@router.post("/evaluate", response_model=EvaluateResponse)
async def evaluate_stock(
    req: EvaluateRequest,
    db: AsyncSession = Depends(get_db),
) -> EvaluateResponse:
    """Run the full Investment Committee evaluation for a single stock."""
    decision: CommitteeDecision = await evaluate(
        db,
        stock_id=req.stock_id,
        symbol=req.symbol,
        target_date=req.target_date,
        atr_value=req.atr_value,
    )
    return EvaluateResponse(
        symbol=decision.symbol,
        final_signal=decision.final_signal,
        weighted_confidence=decision.weighted_confidence,
        regime_id=decision.regime_id,
        reasoning=decision.reasoning,
        verdicts=[
            Verdict(
                agent_name=v.agent_name,
                signal=v.signal,
                confidence=v.confidence,
                reasoning=v.reasoning,
                metadata=v.metadata,
            )
            for v in decision.verdicts
        ],
        risk_verdict=Verdict(
            agent_name=decision.risk_verdict.agent_name,
            signal=decision.risk_verdict.signal,
            confidence=decision.risk_verdict.confidence,
            reasoning=decision.risk_verdict.reasoning,
            metadata=decision.risk_verdict.metadata,
        ),
        metadata=decision.metadata,
    )
