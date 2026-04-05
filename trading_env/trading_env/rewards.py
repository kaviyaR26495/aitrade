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


REWARD_FUNCTIONS = {
    "risk_adjusted_pnl": risk_adjusted_pnl,
    "sharpe": sharpe_reward,
    "sortino": sortino_reward,
    "profit": profit_reward,
    "dense": dense_reward,
}
