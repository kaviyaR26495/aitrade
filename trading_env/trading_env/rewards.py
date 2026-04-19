"""Reward functions for the trading environment."""
from __future__ import annotations

import numpy as np


def risk_adjusted_pnl(
    net_worth_history: list[float],
    current_net_worth: float,
    initial_cash: float,
) -> float:
    """P&L penalized by drawdown (default reward)."""
    pnl = (current_net_worth - initial_cash) / initial_cash
    if len(net_worth_history) < 2:
        return pnl

    peak = max(net_worth_history)
    drawdown = (peak - current_net_worth) / peak if peak > 0 else 0
    return pnl - 0.5 * drawdown


def sharpe_reward(
    net_worth_history: list[float],
    risk_free_rate: float = 0.06 / 252,  # ~6% annual / 252 trading days
) -> float:
    """Rolling Sharpe ratio as reward."""
    if len(net_worth_history) < 3:
        return 0.0

    returns = np.diff(net_worth_history) / np.array(net_worth_history[:-1])
    excess = returns - risk_free_rate
    std = np.std(excess)
    if std < 1e-8:
        return 0.0
    return float(np.mean(excess) / std)


def sortino_reward(
    net_worth_history: list[float],
    risk_free_rate: float = 0.06 / 252,
) -> float:
    """Downside-risk adjusted reward (Sortino ratio)."""
    if len(net_worth_history) < 3:
        return 0.0

    returns = np.diff(net_worth_history) / np.array(net_worth_history[:-1])
    excess = returns - risk_free_rate
    downside = excess[excess < 0]
    downside_std = np.std(downside) if len(downside) > 0 else 1e-8
    if downside_std < 1e-8:
        return float(np.mean(excess)) * 10  # all positive → scale up
    return float(np.mean(excess) / downside_std)


def profit_reward(
    current_net_worth: float,
    previous_net_worth: float,
) -> float:
    """Simple step-wise raw P&L."""
    if previous_net_worth < 1e-8:
        return 0.0
    return (current_net_worth - previous_net_worth) / previous_net_worth


def dense_reward(
    net_worth_history: list[float],
    current_net_worth: float,
    initial_cash: float,
    holdings: int,
    avg_buy_price: float,
    current_price: float,
    action_history: list[int],
    churn_window: int = 5,
    churn_penalty: float = 0.002,
    drawdown_asymmetry: float = 2.0,
) -> float:
    """Dense step-wise reward combining three signals.

    1. Asymmetric step P&L — losses are penalised ``drawdown_asymmetry`` times
       harder than equivalent gains, matching the empirical fat-tail skew of
       equity returns.

    2. Continuous unrealized P&L — open positions emit a tiny reward/penalty
       every candle, converting sparse end-of-trade rewards into a dense signal
       that guides the agent earlier in each swing.

    3. Churn penalty — if the agent flip-flops between BUY and SELL within the
       last ``churn_window`` steps it pays ``churn_penalty`` per reversal,
       forcing higher-conviction, longer-duration trades.
    """
    # ── 1. Asymmetric step P&L ─────────────────────────────────────────
    if len(net_worth_history) >= 2:
        prev = net_worth_history[-2]
        step_ret = (current_net_worth - prev) / max(abs(prev), 1.0)
    else:
        step_ret = 0.0

    reward = step_ret if step_ret >= 0.0 else step_ret * drawdown_asymmetry

    # ── 2. Unrealized P&L for open positions ──────────────────────────
    if holdings > 0 and avg_buy_price > 0.0:
        unrealized_pct = (current_price - avg_buy_price) / avg_buy_price
        # Small weight — continuous signal, not the dominant component
        reward += unrealized_pct * 0.1

    # ── 3. Churn penalty ──────────────────────────────────────────────
    recent = action_history[-churn_window:] if action_history else []
    reversals = sum(
        1 for i in range(1, len(recent))
        if recent[i - 1] != 0 and recent[i] != 0 and recent[i - 1] * recent[i] < 0
    )
    reward -= churn_penalty * reversals

    return float(reward)


def target_hit_with_mae(
    entry_price: float,
    current_price: float,
    target_price: float,
    stoploss_price: float,
    max_adverse_excursion: float,
    atr: float,
) -> float:
    """Reward for target-price trades with MAE (Max Adverse Excursion) penalty.

    Rewards hitting the target proportionally to the R:R realised, while
    penalising trades that experienced deep drawdowns before recovering.
    This teaches the RL agent to prefer *clean* entries with low MAE.

    Parameters
    ----------
    entry_price : float
        Buy entry price.
    current_price : float
        Current or exit price.
    target_price : float
        Profit target set by the signal synthesiser.
    stoploss_price : float
        Stoploss level.
    max_adverse_excursion : float
        Worst intra-trade drawdown (lowest price seen after entry).
    atr : float
        Current ATR for normalisation.

    Returns
    -------
    float  — reward value (can be negative for SL hits).
    """
    if atr <= 0 or entry_price <= 0:
        return 0.0

    # Realised P&L as multiple of risk (ATR)
    pnl_pct = (current_price - entry_price) / entry_price
    risk = abs(entry_price - stoploss_price) / entry_price
    reward_pct = abs(target_price - entry_price) / entry_price

    # R-multiple: how much of the target was captured
    if reward_pct > 0:
        r_captured = pnl_pct / reward_pct
    else:
        r_captured = 0.0

    # MAE penalty: penalise trades that dipped deeply before recovering
    mae_depth = (entry_price - max_adverse_excursion) / entry_price
    mae_penalty = max(0.0, mae_depth / (risk + 1e-8) - 0.3)  # forgive up to 30% of risk

    # Base reward
    if current_price >= target_price:
        # Target hit — full reward minus MAE penalty
        reward = 1.0 + 0.5 * max(0, r_captured - 1.0)  # bonus for overshooting
        reward -= 0.3 * mae_penalty
    elif current_price <= stoploss_price:
        # Stoploss hit — negative reward
        reward = -1.0 - 0.2 * mae_penalty
    else:
        # Still open — small unrealised signal
        reward = 0.1 * r_captured - 0.1 * mae_penalty

    return float(reward)


REWARD_FUNCTIONS = {
    "risk_adjusted_pnl": risk_adjusted_pnl,
    "sharpe": sharpe_reward,
    "sortino": sortino_reward,
    "profit": profit_reward,
    "dense": dense_reward,
    "target_hit_mae": target_hit_with_mae,
}
