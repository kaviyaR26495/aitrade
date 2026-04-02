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


REWARD_FUNCTIONS = {
    "risk_adjusted_pnl": risk_adjusted_pnl,
    "sharpe": sharpe_reward,
    "sortino": sortino_reward,
    "profit": profit_reward,
}
