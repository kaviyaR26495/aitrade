"""Signal Synthesiser — combines multiple model outputs into trade signals.

Takes the regression outputs from LSTM (μ, σ), KNN return statistics,
support/resistance zones, and fundamental quality score to produce a
``TradeSignalCandidate`` with entry, target, stoploss, and confidence.

Pipeline
────────
1. LSTM provides predicted return distribution (μ, σ) across 10 horizons
2. KNN provides historical analog return stats (median, std, win_rate)
3. S/R engine provides nearest support/resistance zones
4. Fundamental scorer provides FQS composite
5. Execution cost estimator deducts spread + slippage
6. κ-decay projects R:R degradation over time

Output: TradeSignalCandidate with BUY/NO_TRADE decision + price levels
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from app.core.support_resistance import SRResult

logger = logging.getLogger(__name__)


# ── Configuration ─────────────────────────────────────────────────────

@dataclass
class SynthesizerConfig:
    """Tunable parameters for the signal synthesiser."""
    # Minimum predicted return to consider a BUY (after cost deduction)
    min_net_return_pct: float = 0.02      # 2%
    # Minimum R:R ratio (reward / risk)
    min_rr_ratio: float = 2.0
    # κ-decay rate: R:R degrades by this factor per day
    kappa: float = 0.03
    # Maximum days before signal expires
    max_signal_days: int = 25
    # S/R confidence weight
    sr_weight: float = 0.2
    # FQS minimum for BUY consideration
    min_fqs: float = 0.35
    # KNN minimum win rate
    min_knn_win_rate: float = 0.45
    # LSTM-KNN agreement bonus
    agreement_bonus: float = 0.15
    # Execution cost default (bps) if not available from order book
    default_cost_bps: float = 15.0  # 0.15%
    # Stoploss ATR multiplier
    sl_atr_multiplier: float = 1.5
    # Target rounding to tick size
    tick_size: float = 0.05


# ── Data structures ───────────────────────────────────────────────────

@dataclass
class TradeSignalCandidate:
    """A potential trade signal before meta-classifier gating."""
    stock_id: int
    # Price levels
    entry_price: float
    target_price: float
    stoploss_price: float
    # Scores
    confluence_score: float       # raw 0-1 confidence before meta-classifier
    fqs_score: float
    execution_cost_pct: float     # spread + slippage as % of price
    # R:R
    initial_rr_ratio: float
    net_expected_return_pct: float  # after cost deduction
    # Model outputs
    lstm_mu: float                # LSTM predicted mean return
    lstm_sigma: float             # LSTM predicted uncertainty
    knn_median_return: float
    knn_win_rate: float
    # Metadata
    regime_id: Optional[int] = None
    is_buy: bool = False          # final BUY/NO_TRADE decision
    reject_reason: str = ""


# ── Utility ───────────────────────────────────────────────────────────

def _round_to_tick(price: float, tick_size: float) -> float:
    """Round price to nearest tick size."""
    if tick_size <= 0:
        return round(price, 2)
    return round(round(price / tick_size) * tick_size, 2)


def _kappa_decay_rr(
    initial_rr: float,
    days: int,
    kappa: float,
) -> float:
    """Project R:R ratio after κ-decay over ``days``.

    R:R(t) = R:R(0) × exp(-κ × t)
    """
    return initial_rr * math.exp(-kappa * days)


def estimate_execution_cost(
    bid: Optional[float],
    ask: Optional[float],
    current_price: float,
    realized_vol: float = 0.0,
    slippage_bps: float = 5.0,
    vol_slippage_scale: float = 0.5,
) -> float:
    """Estimate execution cost as a fraction of price.

    Combines half-spread (from bid/ask) + volatility-scaled slippage.
    Returns cost as a fraction (e.g., 0.0015 = 0.15%).
    """
    # Half-spread from order book
    if bid is not None and ask is not None and bid > 0 and ask > 0:
        half_spread = (ask - bid) / (2.0 * current_price)
    else:
        half_spread = 0.0005  # default 5 bps

    # Slippage: base + vol-scaled component
    slip = slippage_bps / 10_000 + vol_slippage_scale * realized_vol

    return half_spread + slip


# ── Main synthesiser ──────────────────────────────────────────────────

def synthesize_signal(
    stock_id: int,
    current_price: float,
    atr: float,
    lstm_mu: float,
    lstm_sigma: float,
    knn_median_return: float,
    knn_win_rate: float,
    sr_result: SRResult,
    fqs_score: float,
    regime_id: Optional[int] = None,
    bid: Optional[float] = None,
    ask: Optional[float] = None,
    realized_vol: float = 0.0,
    config: Optional[SynthesizerConfig] = None,
) -> TradeSignalCandidate:
    """Synthesize a trade signal from multiple model outputs.

    Parameters
    ----------
    stock_id : int
        Database stock ID.
    current_price : float
        Latest traded price (entry reference).
    atr : float
        14-day ATR.
    lstm_mu, lstm_sigma : float
        LSTM predicted mean return and uncertainty (from h=5 horizon).
    knn_median_return, knn_win_rate : float
        KNN neighbor return statistics.
    sr_result : SRResult
        Support/resistance analysis.
    fqs_score : float
        Fundamental Quality Score (0-1).
    regime_id : int | None
        Current regime classification (0-5).
    bid, ask : float | None
        Best bid/ask from order book (for cost estimation).
    realized_vol : float
        Recent realized volatility (for slippage estimation).
    config : SynthesizerConfig | None
        Tunable parameters.

    Returns
    -------
    TradeSignalCandidate
    """
    if config is None:
        config = SynthesizerConfig()

    # ── 1. Execution cost estimation ──────────────────────────────────
    exec_cost = estimate_execution_cost(
        bid=bid,
        ask=ask,
        current_price=current_price,
        realized_vol=realized_vol,
    )

    # ── 2. Blend LSTM and KNN predicted returns ──────────────────────
    # Weight LSTM by inverse-sigma (higher confidence = more weight)
    lstm_weight = 1.0 / (1.0 + lstm_sigma * 10)  # σ=0.05 → wt≈0.67
    knn_weight = 1.0 - lstm_weight

    blended_return = lstm_weight * lstm_mu + knn_weight * knn_median_return
    net_return = blended_return - exec_cost  # deduct cost

    # ── 3. Target price from blended return ───────────────────────────
    raw_target = current_price * (1.0 + net_return)

    # Clamp target to nearest resistance if it's nearby
    if sr_result.nearest_resistance is not None:
        res_mid = sr_result.nearest_resistance.midpoint
        # If resistance is between current and raw target, use resistance as target
        if current_price < res_mid < raw_target:
            # Use resistance minus a small buffer (so we sell just before resistance)
            raw_target = res_mid - 0.001 * res_mid

    target_price = _round_to_tick(raw_target, config.tick_size)

    # ── 4. Stoploss from ATR + nearest support ───────────────────────
    atr_sl = current_price - config.sl_atr_multiplier * atr

    # If nearest support is above ATR-SL, tighten SL to just below support
    if sr_result.nearest_support is not None:
        sup_low = sr_result.nearest_support.price_low
        if sup_low > atr_sl:
            atr_sl = sup_low - 0.002 * sup_low  # small buffer below support

    stoploss_price = _round_to_tick(atr_sl, config.tick_size)

    # ── 5. R:R ratio ─────────────────────────────────────────────────
    reward = target_price - current_price
    risk = current_price - stoploss_price
    if risk <= 0:
        risk = atr * config.sl_atr_multiplier  # fallback

    rr_ratio = reward / risk if risk > 0 else 0.0

    # ── 6. Confluence scoring ─────────────────────────────────────────
    scores = []

    # LSTM return significance: is μ meaningfully positive?
    if lstm_mu > 0.01:
        scores.append(min(1.0, lstm_mu / 0.05))  # 5% predicted return → score 1.0
    else:
        scores.append(0.0)

    # KNN win rate
    scores.append(knn_win_rate)

    # S/R positioning: closer to support = better
    sr_score = 0.5
    if sr_result.support_distance_pct > 0:
        # Closer to support = higher score
        sr_score = max(0.0, 1.0 - sr_result.support_distance_pct / 0.05)
    scores.append(sr_score * config.sr_weight / 0.25)  # normalise contribution

    # R:R ratio quality
    rr_score = min(1.0, rr_ratio / 4.0)
    scores.append(rr_score)

    # Fundamental quality
    scores.append(fqs_score)

    # LSTM-KNN agreement bonus
    agreement = 0.0
    if lstm_mu > 0 and knn_median_return > 0:
        agreement = config.agreement_bonus
    elif lstm_mu < 0 and knn_median_return < 0:
        agreement = config.agreement_bonus

    confluence = float(np.mean(scores)) + agreement

    # ── 7. BUY/NO_TRADE decision ─────────────────────────────────────
    is_buy = True
    reject_reason = ""

    if net_return < config.min_net_return_pct:
        is_buy = False
        reject_reason = f"net_return {net_return:.4f} < min {config.min_net_return_pct}"
    elif rr_ratio < config.min_rr_ratio:
        is_buy = False
        reject_reason = f"R:R {rr_ratio:.2f} < min {config.min_rr_ratio}"
    elif fqs_score < config.min_fqs:
        is_buy = False
        reject_reason = f"FQS {fqs_score:.3f} < min {config.min_fqs}"
    elif knn_win_rate < config.min_knn_win_rate:
        is_buy = False
        reject_reason = f"KNN win_rate {knn_win_rate:.3f} < min {config.min_knn_win_rate}"

    # Check κ-decay viability: R:R must still be > 1.0 at day 15
    if is_buy:
        rr_at_15 = _kappa_decay_rr(rr_ratio, 15, config.kappa)
        if rr_at_15 < 1.0:
            is_buy = False
            reject_reason = f"κ-decay: R:R falls to {rr_at_15:.2f} by day 15"

    return TradeSignalCandidate(
        stock_id=stock_id,
        entry_price=current_price,
        target_price=target_price,
        stoploss_price=stoploss_price,
        confluence_score=round(min(1.0, confluence), 4),
        fqs_score=round(fqs_score, 4),
        execution_cost_pct=round(exec_cost, 6),
        initial_rr_ratio=round(rr_ratio, 4),
        net_expected_return_pct=round(net_return, 6),
        lstm_mu=round(lstm_mu, 6),
        lstm_sigma=round(lstm_sigma, 6),
        knn_median_return=round(knn_median_return, 6),
        knn_win_rate=round(knn_win_rate, 4),
        regime_id=regime_id,
        is_buy=is_buy,
        reject_reason=reject_reason,
    )
