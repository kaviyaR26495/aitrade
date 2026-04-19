"""Technical Agent — reads ML ensemble signal + regime + multi-horizon forecast."""
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
    """Return a technical-analysis AgentVerdict for the given stock."""
    from sqlalchemy import select
    from app.db.models import (
        EnsemblePrediction,
        StockRegime,
        LSTMHorizonPrediction,
    )
    from app.db import crud

    # ── Latest ensemble prediction ────────────────────────────────────
    preds = await crud.get_ensemble_predictions_for_date(
        db,
        target_date=target_date,
        interval="day",
        min_confidence=0.0,
        agreement_only=False,
    )
    stock_pred = next((p for p in preds if p.stock_id == stock_id), None)

    # ── Latest regime ─────────────────────────────────────────────────
    regime_row = await crud.get_latest_regime(db, stock_id, "day")
    regime_id = regime_row.regime_id if regime_row else None
    regime_conf = regime_row.regime_confidence if regime_row else None

    # ── Multi-horizon prediction (if available) ───────────────────────
    horizon_pred = None
    try:
        from sqlalchemy import select as sa_select
        result = await db.execute(
            sa_select(LSTMHorizonPrediction)
            .where(LSTMHorizonPrediction.stock_id == stock_id)
            .order_by(LSTMHorizonPrediction.prediction_date.desc())
            .limit(1)
        )
        horizon_pred = result.scalar_one_or_none()
    except Exception:
        pass

    # ── Build summary dict ────────────────────────────────────────────
    summary = {
        "symbol": symbol,
        "ensemble_signal": stock_pred.action if stock_pred else None,
        "ensemble_confidence": round(stock_pred.confidence, 4) if stock_pred else None,
        "knn_action": stock_pred.knn_action if stock_pred else None,
        "lstm_action": stock_pred.lstm_action if stock_pred else None,
        "agreement": stock_pred.agreement if stock_pred else None,
        "regime_id": regime_id,
        "regime_confidence": round(regime_conf, 4) if regime_conf else None,
        "h1_action": horizon_pred.h1_action if horizon_pred else None,
        "h5_action": horizon_pred.h5_action if horizon_pred else None,
        "trend_durability": round(horizon_pred.trend_durability_score, 4) if horizon_pred and horizon_pred.trend_durability_score else None,
    }

    system = (
        "You are an expert quantitative technical analyst for Indian equities. "
        "Analyse the provided ML model signals and return ONLY a JSON object with keys: "
        'signal (BUY|SELL|HOLD), confidence (0-1), reasoning (1-2 sentences).'
    )
    prompt = f"Technical signals for {symbol}:\n{json.dumps(summary, indent=2)}"

    try:
        raw = await llm_complete(db, [{"role": "user", "content": prompt}], system_prompt=system)
        # Parse JSON from response
        start = raw.find("{")
        end = raw.rfind("}") + 1
        result_dict = json.loads(raw[start:end]) if start >= 0 and end > start else {}
        signal = result_dict.get("signal", "HOLD")
        confidence = float(result_dict.get("confidence", 0.5))
        reasoning = result_dict.get("reasoning", raw[:300])
    except Exception as exc:
        logger.warning("TechnicalAgent LLM failed for %s: %s — falling back to rule", symbol, exc)
        # Rule-based fallback
        if stock_pred and stock_pred.agreement and stock_pred.confidence >= 0.70:
            signal = "BUY" if stock_pred.action == 1 else "SELL" if stock_pred.action == 2 else "HOLD"
            confidence = stock_pred.confidence
        else:
            signal = "HOLD"
            confidence = 0.5
        reasoning = f"LLM unavailable — rule fallback from ensemble signal={stock_pred.action if stock_pred else 'n/a'}"

    return AgentVerdict(
        agent_name="TechnicalAgent",
        signal=signal,
        confidence=max(0.0, min(1.0, confidence)),
        reasoning=reasoning,
        metadata=summary,
    )
