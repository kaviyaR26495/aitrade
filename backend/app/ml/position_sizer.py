"""Portfolio-level position sizing — separated entirely from ML models.

Two evidence-based sizing methods are provided:

Kelly Criterion
    Sizes each trade to maximise logarithmic equity growth.
    f* = win_rate - (1 - win_rate) / (avg_win / avg_loss)
    Half-Kelly (kelly_mult=0.5) is used by default because the theoretical
    full-Kelly requires perfectly known win-rate and return distributions —
    both highly uncertain in live markets.  Half-Kelly cuts position size in
    half, sacrificing some growth to drastically reduce peak drawdown.

Volatility Targeting
    Scales position size inversely with recent realised volatility:
        fraction = target_annual_vol / realised_annual_vol
    High-volatility stocks → smaller position.
    Low-volatility stocks  → larger position.
    This naturally de-risks during choppy/volatile regimes and scales up in
    calm trending markets — no manual regime switching required.

Risk Cap (always applied on top of both methods)
    No single trade may risk more than ``max_risk_pct`` of total equity.
    The risk per trade is:
        risk = position_fraction × stoploss_pct
    The position is clipped so that risk ≤ max_risk_pct.
    Example: stoploss=5%, max_risk=2%  →  max position = 40% of equity.
    Combined with max_positions=10, the true cap is typically much lower.

Usage example::

    from app.ml.position_sizer import size_trade

    # During a backtest or live order submission:
    frac = size_trade(
        method="volatility_target",
        realized_vol_daily=0.012,     # 1.2% daily std
        trades=[...],                 # accumulated closed trades for Kelly
        max_risk_pct=0.02,            # 2% equity at risk
        stoploss_pct=0.05,            # 5% stoploss
    )
    position_value = current_equity * frac
"""
from __future__ import annotations

from typing import Any

import numpy as np


# ─── pure math helpers ────────────────────────────────────────────────────────


def kelly_fraction(
    win_rate: float,
    avg_win: float,        # as a decimal fraction, e.g. 0.02 = 2 %
    avg_loss: float,       # as a decimal fraction (positive), e.g. 0.015 = 1.5 %
    kelly_mult: float = 0.5,
    max_fraction: float = 0.25,
) -> float:
    """Half-Kelly position fraction.

    Returns the fraction of equity to allocate, capped at ``max_fraction``.
    Returns 0.0 when the Kelly formula predicts a negative edge
    (expected value of the trade is negative — don't trade).

    Parameters
    ----------
    win_rate  : probability of a winning trade (0–1)
    avg_win   : average profit as a decimal fraction on winning trades
    avg_loss  : average loss as a decimal fraction (positive) on losing trades
    kelly_mult: safety multiplier — 0.5 = half-Kelly (recommended)
    max_fraction: hard cap regardless of Kelly output
    """
    if avg_loss <= 0 or avg_win <= 0:
        return 0.0
    b = avg_win / avg_loss          # win-to-loss ratio
    p = win_rate
    q = 1.0 - p
    full_kelly = (p * b - q) / b   # Kelly formula
    if full_kelly <= 0:
        return 0.0
    return min(full_kelly * kelly_mult, max_fraction)


def vol_target_fraction(
    realized_vol_daily: float,
    target_vol_annual: float = 0.15,
    max_fraction: float = 0.25,
) -> float:
    """Volatility-targeting position fraction.

    Parameters
    ----------
    realized_vol_daily  : standard deviation of daily log-returns (e.g. 0.012)
    target_vol_annual   : desired annualised portfolio volatility contribution
                          from this single position (e.g. 0.15 = 15 %)
    max_fraction        : hard cap
    """
    if realized_vol_daily <= 0:
        return max_fraction
    realized_vol_annual = realized_vol_daily * np.sqrt(252)
    frac = target_vol_annual / realized_vol_annual
    return min(float(frac), max_fraction)


# ─── main interface ───────────────────────────────────────────────────────────


def size_trade(
    method: str,
    *,
    realized_vol_daily: float,
    trades: list[Any] | None = None,
    max_risk_pct: float = 0.02,       # max 2 % of equity risked per trade
    stoploss_pct: float = 0.05,       # 5 % stoploss
    target_vol_annual: float = 0.15,
    kelly_mult: float = 0.5,
    fallback_pct: float = 0.10,       # 10 % of equity if sizing fails
    max_positions: int = 10,
    min_kelly_trades: int = 10,       # min closed trades before Kelly is used
) -> float:
    """Compute position size as a fraction of current equity.

    Parameters
    ----------
    method : "fixed" | "kelly" | "volatility_target"
    realized_vol_daily : recent daily price std-dev (from rolling window)
    trades : list of closed ``Trade`` objects with ``.pnl`` and ``.pnl_pct``
        fields.  Required for Kelly; ignored for other methods.
    max_risk_pct : hard cap — position × stoploss_pct ≤ max_risk_pct.
        E.g. 0.02 means no more than 2 % of equity can be lost at stoploss.
    stoploss_pct : stoploss as a fraction (0.05 = 5 %)
    fallback_pct : position fraction when Kelly has insufficient data
    max_positions : maximum concurrent positions (used to cap concentration)

    Returns
    -------
    float in (0, 1] — fraction of equity to allocate to this trade.
    """
    # ── risk cap: ensures loss-at-stoploss ≤ max_risk_pct × equity ──────────
    risk_cap = (max_risk_pct / stoploss_pct) if stoploss_pct > 0 else fallback_pct

    # ── concentration cap: never put too much into a single position ─────────
    pos_cap = 1.0 / max(max_positions, 1)

    # ── absolute hard cap: never exceed 50 % in one trade ────────────────────
    hard_cap = min(risk_cap, pos_cap, 0.50)

    # ── sizing method ─────────────────────────────────────────────────────────
    if method == "kelly":
        if trades and len(trades) >= min_kelly_trades:
            wins = [t for t in trades if t.pnl > 0]
            losses = [t for t in trades if t.pnl <= 0]
            if wins and losses:
                win_rate = len(wins) / len(trades)
                avg_win = float(np.mean([abs(t.pnl_pct) / 100 for t in wins]))
                avg_loss = float(np.mean([abs(t.pnl_pct) / 100 for t in losses]))
                frac = kelly_fraction(
                    win_rate, avg_win, avg_loss,
                    kelly_mult=kelly_mult, max_fraction=hard_cap,
                )
            else:
                # Can't compute Kelly (all wins or all losses): fall back to vol-target
                frac = vol_target_fraction(
                    realized_vol_daily, target_vol_annual, max_fraction=hard_cap
                )
        else:
            # Not enough trade history yet: fall back to vol-target
            frac = vol_target_fraction(
                realized_vol_daily, target_vol_annual, max_fraction=hard_cap
            )

    elif method == "volatility_target":
        frac = vol_target_fraction(
            realized_vol_daily, target_vol_annual, max_fraction=hard_cap
        )

    else:  # "fixed" or unknown
        frac = min(fallback_pct, hard_cap)

    # Ensure at least a minimal position (avoid zero sizing when vol spikes briefly)
    return float(max(frac, 0.01))


# ─── NIFTY 50 Market Breadth Kill-Switch ─────────────────────────────────────


def nifty_breadth_multiplier(
    nifty_close_series: "np.ndarray | list[float]",
    dma_period: int = 200,
    half_size_below_dma: bool = True,
) -> float:
    """Return a position-size multiplier based on NIFTY 50's trend health.

    Rules
    -----
    * NIFTY close > 200-DMA  → multiplier = 1.0  (full size, trend is healthy)
    * NIFTY close ≤ 200-DMA  → multiplier = 0.5  if ``half_size_below_dma``
                                            else 0.0  (BUYs fully disabled)

    The computation requires *at least* ``dma_period`` past closes.
    If fewer are available the function returns 1.0 (no signal = no restriction).

    Parameters
    ----------
    nifty_close_series : array-like of historical NIFTY 50 daily closes,
        ordered oldest → newest.  The last element is today's close.
    dma_period : int, default 200.  Rolling simple-average window.
    half_size_below_dma : bool, default True.
        True  → halve position size when below 200-DMA (conservative).
        False → block all new BUYs when below 200-DMA (strict kill-switch).

    Returns
    -------
    float in {0.0, 0.5, 1.0}.
    """
    arr = np.asarray(nifty_close_series, dtype=float)
    if len(arr) < dma_period:
        return 1.0  # not enough history — don't penalise

    dma = float(np.mean(arr[-dma_period:]))
    current = float(arr[-1])

    if current > dma:
        return 1.0
    return 0.5 if half_size_below_dma else 0.0


def size_trade_with_breadth(
    method: str,
    *,
    realized_vol_daily: float,
    nifty_close_series: "np.ndarray | list[float]",
    trades: "list[Any] | None" = None,
    max_risk_pct: float = 0.02,
    stoploss_pct: float = 0.05,
    target_vol_annual: float = 0.15,
    kelly_mult: float = 0.5,
    fallback_pct: float = 0.10,
    max_positions: int = 10,
    min_kelly_trades: int = 10,
    dma_period: int = 200,
    half_size_below_dma: bool = True,
) -> float:
    """``size_trade()`` + NIFTY 200-DMA breadth kill-switch, composed in one call.

    The base fraction from ``size_trade()`` is multiplied by
    ``nifty_breadth_multiplier()``.  A result of 0.0 means the caller
    should skip the BUY entirely.

    All parameters are identical to ``size_trade()`` except:

    nifty_close_series : historical NIFTY 50 closes (oldest → newest).
    dma_period         : 200-DMA window (default 200 trading days).
    half_size_below_dma: True = halve size, False = full block.

    Returns
    -------
    float — effective position fraction after breadth adjustment.
    """
    base_frac = size_trade(
        method=method,
        realized_vol_daily=realized_vol_daily,
        trades=trades,
        max_risk_pct=max_risk_pct,
        stoploss_pct=stoploss_pct,
        target_vol_annual=target_vol_annual,
        kelly_mult=kelly_mult,
        fallback_pct=fallback_pct,
        max_positions=max_positions,
        min_kelly_trades=min_kelly_trades,
    )
    multiplier = nifty_breadth_multiplier(
        nifty_close_series,
        dma_period=dma_period,
        half_size_below_dma=half_size_below_dma,
    )
    return float(base_frac * multiplier)


# ─── Sector Concentration Guard ───────────────────────────────────────────────


def sector_concentration_multiplier(
    candidate_sector: str | None,
    open_positions: "dict[str, dict]",
    total_equity: float,
    sector_cap: float = 0.25,
) -> float:
    """Return 0.0 if adding a new position would breach the sector concentration cap.

    The guard works entirely in terms of **mark-to-market position value** so
    it accounts for unrealised gains/losses already in the book.

    Algorithm
    ---------
    1. Sum the current market value of every open position whose ``sector``
       matches ``candidate_sector``.
    2. If that sum already equals or exceeds ``sector_cap × total_equity``,
       return 0.0 (block the trade).
    3. Otherwise return 1.0 (trade is allowed).

    The position dict must contain at least ``{"quantity": int, "price": float,
    "sector": str | None}`` per key.  Any position with a ``None`` or empty
    sector is treated as belonging to a unique unknown sector and never counted
    against any named sector.

    Parameters
    ----------
    candidate_sector : sector of the stock being evaluated for entry.
        Passing ``None`` or ``""`` means "unknown" — the guard always allows it.
    open_positions : mapping of position key → position dict, matching the
        structure used in backtester.py.
    total_equity : current portfolio equity (cash + MTM open positions).
        Used as the denominator for the sector weight calculation.
    sector_cap : maximum fraction of equity allowed in any single sector.
        Default 0.25 (25 %).

    Returns
    -------
    float — 1.0 (trade allowed) or 0.0 (sector cap breached).
    """
    if not candidate_sector or total_equity <= 0:
        return 1.0

    sector_exposure = sum(
        p["quantity"] * p.get("price", 0.0)
        for p in open_positions.values()
        if p.get("sector") == candidate_sector
    )

    if sector_exposure >= sector_cap * total_equity:
        return 0.0
    return 1.0

