"""Unified Backtesting Engine.

Works for RL, KNN, LSTM, and Ensemble models.
Simulates trading with:
- Per-date prediction + order execution
- Configurable SL/target
- Commission costs
- Regime-aware analysis

Outputs: equity curve, trade log, performance metrics.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    initial_capital: float = 100_000.0
    commission_pct: float = 0.001  # 0.1% per trade (Zerodha CNC is ~0.03% but includes STT etc)
    stoploss_pct: float = 5.0
    target_pct: float | None = None  # None = hold until sell signal
    min_confidence: float = 0.65
    max_positions: int = 10
    position_size_pct: float = 10.0  # % of capital per trade


@dataclass
class Trade:
    entry_date: Any
    exit_date: Any | None = None
    action: str = "BUY"
    entry_price: float = 0.0
    exit_price: float = 0.0
    quantity: int = 0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    regime_id: int | None = None
    confidence: float = 0.0
    exit_reason: str = ""


@dataclass
class BacktestResult:
    total_return_pct: float = 0.0
    annual_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    equity_curve: list[float] = field(default_factory=list)
    trade_log: list[dict] = field(default_factory=list)
    buy_hold_return_pct: float = 0.0


def run_backtest(
    predictions: list[dict],
    close_prices: np.ndarray,
    dates: list,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """
    Run backtest on a sequence of predictions.

    predictions: list of dicts with keys: action (0=HOLD, 1=BUY, 2=SELL), confidence, regime_id
    close_prices: array of close prices aligned with predictions
    dates: list of dates aligned with predictions

    Returns BacktestResult with metrics and trade log.
    """
    if config is None:
        config = BacktestConfig()

    n = min(len(predictions), len(close_prices), len(dates))
    capital = config.initial_capital
    cash = capital
    positions: dict[str, dict] = {}  # stock_key → {entry_price, quantity, entry_date, regime_id}
    equity_curve = []
    trades: list[Trade] = []

    for i in range(n):
        price = float(close_prices[i])
        pred = predictions[i]
        action = pred.get("action", 0)
        confidence = pred.get("confidence", 0)
        regime_id = pred.get("regime_id")
        dt = dates[i]

        # Check stop losses on existing positions
        closed_keys = []
        for key, pos in positions.items():
            loss_pct = ((price - pos["entry_price"]) / pos["entry_price"]) * 100
            if loss_pct < -config.stoploss_pct:
                # Stop loss hit
                exit_value = pos["quantity"] * price * (1 - config.commission_pct)
                pnl = exit_value - (pos["quantity"] * pos["entry_price"])
                pnl_pct = (pnl / (pos["quantity"] * pos["entry_price"])) * 100
                cash += exit_value

                trades.append(Trade(
                    entry_date=pos["entry_date"],
                    exit_date=dt,
                    action="BUY",
                    entry_price=pos["entry_price"],
                    exit_price=price,
                    quantity=pos["quantity"],
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    regime_id=pos.get("regime_id"),
                    confidence=pos.get("confidence", 0),
                    exit_reason="stoploss",
                ))
                closed_keys.append(key)

            # Check target
            elif config.target_pct and loss_pct > config.target_pct:
                exit_value = pos["quantity"] * price * (1 - config.commission_pct)
                pnl = exit_value - (pos["quantity"] * pos["entry_price"])
                pnl_pct = (pnl / (pos["quantity"] * pos["entry_price"])) * 100
                cash += exit_value

                trades.append(Trade(
                    entry_date=pos["entry_date"],
                    exit_date=dt,
                    action="BUY",
                    entry_price=pos["entry_price"],
                    exit_price=price,
                    quantity=pos["quantity"],
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    regime_id=pos.get("regime_id"),
                    confidence=pos.get("confidence", 0),
                    exit_reason="target",
                ))
                closed_keys.append(key)

        for key in closed_keys:
            del positions[key]

        # Process new signals
        if action == 1 and confidence >= config.min_confidence:
            # BUY signal
            if len(positions) < config.max_positions and price > 0:
                position_value = capital * config.position_size_pct / 100
                cost = position_value * (1 + config.commission_pct)
                if cost <= cash:
                    qty = int(position_value / price)
                    if qty > 0:
                        actual_cost = qty * price * (1 + config.commission_pct)
                        cash -= actual_cost
                        key = f"pos_{i}"
                        positions[key] = {
                            "entry_price": price,
                            "quantity": qty,
                            "entry_date": dt,
                            "regime_id": regime_id,
                            "confidence": confidence,
                        }

        elif action == 2 and confidence >= config.min_confidence:
            # SELL signal — close all positions
            for key, pos in list(positions.items()):
                exit_value = pos["quantity"] * price * (1 - config.commission_pct)
                pnl = exit_value - (pos["quantity"] * pos["entry_price"])
                pnl_pct = (pnl / (pos["quantity"] * pos["entry_price"])) * 100
                cash += exit_value

                trades.append(Trade(
                    entry_date=pos["entry_date"],
                    exit_date=dt,
                    action="BUY",
                    entry_price=pos["entry_price"],
                    exit_price=price,
                    quantity=pos["quantity"],
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    regime_id=pos.get("regime_id"),
                    confidence=pos.get("confidence", 0),
                    exit_reason="sell_signal",
                ))
            positions.clear()

        # Calculate equity (cash + open positions value)
        open_value = sum(pos["quantity"] * price for pos in positions.values())
        equity = cash + open_value
        equity_curve.append(equity)

    # Close remaining positions at last price
    if positions and n > 0:
        last_price = float(close_prices[n - 1])
        for pos in positions.values():
            exit_value = pos["quantity"] * last_price * (1 - config.commission_pct)
            pnl = exit_value - (pos["quantity"] * pos["entry_price"])
            pnl_pct = (pnl / (pos["quantity"] * pos["entry_price"])) * 100
            cash += exit_value
            trades.append(Trade(
                entry_date=pos["entry_date"],
                exit_date=dates[-1] if dates else None,
                entry_price=pos["entry_price"],
                exit_price=last_price,
                quantity=pos["quantity"],
                pnl=pnl,
                pnl_pct=pnl_pct,
                regime_id=pos.get("regime_id"),
                exit_reason="end_of_backtest",
            ))

    # Compute metrics
    result = _compute_metrics(
        trades=trades,
        equity_curve=equity_curve,
        initial_capital=capital,
        close_prices=close_prices[:n],
        n_days=n,
    )

    return result


def _compute_metrics(
    trades: list[Trade],
    equity_curve: list[float],
    initial_capital: float,
    close_prices: np.ndarray,
    n_days: int,
) -> BacktestResult:
    """Compute backtest performance metrics."""
    result = BacktestResult()
    result.equity_curve = equity_curve
    result.total_trades = len(trades)

    if not equity_curve:
        return result

    final_equity = equity_curve[-1]
    result.total_return_pct = ((final_equity - initial_capital) / initial_capital) * 100

    # Annualized return (assume 252 trading days)
    if n_days > 0:
        years = n_days / 252
        if years > 0:
            result.annual_return_pct = ((final_equity / initial_capital) ** (1 / years) - 1) * 100

    # Sharpe ratio (daily returns)
    equity_arr = np.array(equity_curve)
    daily_returns = np.diff(equity_arr) / equity_arr[:-1]
    if len(daily_returns) > 1 and np.std(daily_returns) > 0:
        result.sharpe_ratio = round(
            float(np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)), 4
        )

    # Max drawdown
    peak = equity_arr[0]
    max_dd = 0
    for val in equity_arr:
        if val > peak:
            peak = val
        dd = (peak - val) / peak * 100
        if dd > max_dd:
            max_dd = dd
    result.max_drawdown_pct = round(max_dd, 4)

    # Trade statistics
    if trades:
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        result.winning_trades = len(wins)
        result.losing_trades = len(losses)
        result.win_rate = round(len(wins) / len(trades) * 100, 2)

        total_wins = sum(t.pnl for t in wins)
        total_losses = abs(sum(t.pnl for t in losses))
        result.profit_factor = round(total_wins / total_losses, 4) if total_losses > 0 else float("inf")

        if wins:
            result.avg_win_pct = round(float(np.mean([t.pnl_pct for t in wins])), 4)
        if losses:
            result.avg_loss_pct = round(float(np.mean([t.pnl_pct for t in losses])), 4)

    # Buy and hold return
    if len(close_prices) >= 2:
        result.buy_hold_return_pct = round(
            ((close_prices[-1] - close_prices[0]) / close_prices[0]) * 100, 4
        )

    # Trade log
    result.trade_log = [
        {
            "entry_date": str(t.entry_date),
            "exit_date": str(t.exit_date),
            "entry_price": round(t.entry_price, 2),
            "exit_price": round(t.exit_price, 2),
            "quantity": t.quantity,
            "pnl": round(t.pnl, 2),
            "pnl_pct": round(t.pnl_pct, 4),
            "regime_id": t.regime_id,
            "confidence": t.confidence,
            "exit_reason": t.exit_reason,
        }
        for t in trades
    ]

    return result
