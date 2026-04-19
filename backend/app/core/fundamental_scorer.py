"""Fundamental Quality Score (FQS) — a composite 0-1 score.

Combines six orthogonal factor groups:
1. Valuation    – PE z-score (3-yr own + sector-relative)
2. Quality      – ROE normalised, D/E inverse-normalised
3. Growth proxy – Forward PE discount vs trailing PE
4. Dividend     – Yield percentile (zero is neutral, not negative)
5. Earnings momentum – PE compression/expansion trend
6. Institutional signal – sentiment proxy (avg_finbert_score)

Each factor is scored 0-1 independently, then weighted and combined.
The final FQS is used in the signal synthesiser as the fundamental
confidence anchor.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class FundamentalScore:
    """Output of the fundamental scorer."""
    fqs: float  # composite 0-1 score
    valuation_score: float
    quality_score: float
    growth_score: float
    dividend_score: float
    earnings_momentum_score: float
    sentiment_score: float


# ── Factor weights (sum to 1.0) ──────────────────────────────────────

_WEIGHTS = {
    "valuation": 0.25,
    "quality": 0.25,
    "growth": 0.15,
    "dividend": 0.10,
    "earnings_momentum": 0.10,
    "sentiment": 0.15,
}


# ── Individual factor scorers ────────────────────────────────────────

def _score_valuation(
    pe_zscore_3y: Optional[float],
    pe_zscore_sector: Optional[float],
) -> float:
    """Lower PE z-score = more attractive.  Map z ∈ [-3, 3] → score ∈ [0, 1].

    Blends own-history (60%) and sector-relative (40%) z-scores.
    """
    def _z_to_score(z: Optional[float]) -> float:
        if z is None:
            return 0.5  # neutral
        # Clamp to [-3, 3], invert (low PE = high score)
        z_clamped = max(-3.0, min(3.0, z))
        return (3.0 - z_clamped) / 6.0  # maps -3→1.0, 0→0.5, 3→0.0

    own = _z_to_score(pe_zscore_3y)
    sector = _z_to_score(pe_zscore_sector)
    return 0.6 * own + 0.4 * sector


def _score_quality(
    roe_norm: Optional[float],
    debt_equity_norm: Optional[float],
) -> float:
    """Higher ROE and lower D/E = higher quality.

    roe_norm is already 0-1 (clipped ROE/100).
    debt_equity_norm is already inverted 0-1 (low debt = high score).
    """
    roe = roe_norm if roe_norm is not None else 0.5
    de = debt_equity_norm if debt_equity_norm is not None else 0.5
    return 0.6 * roe + 0.4 * de


def _score_growth(
    pe_ratio: Optional[float],
    forward_pe: Optional[float],
) -> float:
    """Forward PE discount signals expected earnings growth.

    discount = (trailing - forward) / trailing
    Positive discount = growth expected → higher score.
    """
    if pe_ratio is None or forward_pe is None or pe_ratio <= 0:
        return 0.5
    discount = (pe_ratio - forward_pe) / pe_ratio
    # Clamp to [-0.5, 0.5] → map to [0, 1]
    discount = max(-0.5, min(0.5, discount))
    return discount + 0.5


def _score_dividend(dividend_yield: Optional[float]) -> float:
    """Map yield to 0-1.  0% → 0.3 (neutral-ish), 5%+ → 1.0."""
    if dividend_yield is None or dividend_yield <= 0:
        return 0.3  # no dividend isn't strongly negative for growth stocks
    # Linear scale: 0→0.3, 5%→1.0
    return min(1.0, 0.3 + dividend_yield / 0.05 * 0.7)


def _score_earnings_momentum(
    pe_ratio: Optional[float],
    pe_ratio_prev: Optional[float],
) -> float:
    """PE compression (falling PE at stable/rising price) = positive momentum.

    Requires prior-quarter PE for comparison.
    """
    if pe_ratio is None or pe_ratio_prev is None or pe_ratio_prev <= 0:
        return 0.5
    change = (pe_ratio - pe_ratio_prev) / pe_ratio_prev
    # Negative change (compression) = good, positive = bad
    # Clamp to [-0.3, 0.3] → map to [0, 1]
    change = max(-0.3, min(0.3, change))
    return 0.5 - change / 0.6  # -0.3→1.0, 0→0.5, 0.3→0.0


def _score_sentiment(avg_finbert_score: Optional[float]) -> float:
    """Map FinBERT sentiment [-1, 1] → [0, 1]."""
    if avg_finbert_score is None:
        return 0.5
    return (max(-1.0, min(1.0, avg_finbert_score)) + 1.0) / 2.0


# ── Public API ────────────────────────────────────────────────────────

def compute_fundamental_score(
    *,
    pe_zscore_3y: Optional[float] = None,
    pe_zscore_sector: Optional[float] = None,
    roe_norm: Optional[float] = None,
    debt_equity_norm: Optional[float] = None,
    pe_ratio: Optional[float] = None,
    forward_pe: Optional[float] = None,
    dividend_yield: Optional[float] = None,
    pe_ratio_prev: Optional[float] = None,
    avg_finbert_score: Optional[float] = None,
) -> FundamentalScore:
    """Compute composite Fundamental Quality Score (FQS).

    All inputs are optional — missing data defaults to neutral (0.5).

    Parameters come from:
    - ``StockFundamentalZScore``: pe_zscore_3y, pe_zscore_sector, roe_norm, debt_equity_norm
    - ``StockFundamentalPIT``: pe_ratio, forward_pe, dividend_yield
    - ``StockFundamentalPIT`` (prior quarter): pe_ratio_prev
    - ``StockSentiment``: avg_finbert_score
    """
    val = _score_valuation(pe_zscore_3y, pe_zscore_sector)
    qual = _score_quality(roe_norm, debt_equity_norm)
    growth = _score_growth(pe_ratio, forward_pe)
    div = _score_dividend(dividend_yield)
    em = _score_earnings_momentum(pe_ratio, pe_ratio_prev)
    sent = _score_sentiment(avg_finbert_score)

    fqs = (
        _WEIGHTS["valuation"] * val
        + _WEIGHTS["quality"] * qual
        + _WEIGHTS["growth"] * growth
        + _WEIGHTS["dividend"] * div
        + _WEIGHTS["earnings_momentum"] * em
        + _WEIGHTS["sentiment"] * sent
    )

    return FundamentalScore(
        fqs=round(fqs, 4),
        valuation_score=round(val, 4),
        quality_score=round(qual, 4),
        growth_score=round(growth, 4),
        dividend_score=round(div, 4),
        earnings_momentum_score=round(em, 4),
        sentiment_score=round(sent, 4),
    )


def fundamental_features(score: FundamentalScore) -> dict[str, float]:
    """Flat dict of fundamental features for ML consumption (7 features)."""
    return {
        "fqs_composite": score.fqs,
        "fqs_valuation": score.valuation_score,
        "fqs_quality": score.quality_score,
        "fqs_growth": score.growth_score,
        "fqs_dividend": score.dividend_score,
        "fqs_earnings_momentum": score.earnings_momentum_score,
        "fqs_sentiment": score.sentiment_score,
    }
