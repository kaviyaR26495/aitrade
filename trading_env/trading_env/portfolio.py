"""Portfolio state tracking with trade logger for RL→KNN/LSTM distillation."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Trade:
    step: int
    action: int  # 1=BUY, -1=SELL, 0=HOLD
    price: float
    quantity: int
    cost: float  # brokerage + STT + DP charges
    regime_id: int | None = None


@dataclass
class TradeLogEntry:
    """Captured by TradeLogger for pattern extraction (Stage 1→2 bridge)."""
    step: int
    date_idx: int
    action: int
    feature_window: np.ndarray  # raw features (seq_len × num_features)
    price: float
    pnl_after_horizon: float | None = None  # filled after profit_horizon steps
    regime_id: int | None = None


class Portfolio:
    """Tracks cash, holdings, costs, and net worth for NSE CNC trading."""

    # NSE cost structure
    BROKERAGE_PCT = 0.0003       # 0.03%
    STT_SELL_PCT = 0.001         # 0.1% on sell-side
    DP_CHARGE = 15.93            # per sell transaction
    EXCHANGE_TXN_PCT = 0.0000345 # NSE exchange charges
    GST_PCT = 0.18               # 18% on brokerage

    def __init__(self, initial_cash: float = 100_000.0):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.holdings: int = 0  # number of shares held
        self.avg_buy_price: float = 0.0
        self.trades: list[Trade] = []
        self.net_worth_history: list[float] = []

    def buy(self, price: float, quantity: int, step: int, regime_id: int | None = None) -> bool:
        cost = self._buy_cost(price, quantity)
        total = price * quantity + cost
        if total > self.cash:
            return False

        self.cash -= total
        if self.holdings > 0:
            total_value = self.avg_buy_price * self.holdings + price * quantity
            self.holdings += quantity
            self.avg_buy_price = total_value / self.holdings
        else:
            self.holdings = quantity
            self.avg_buy_price = price

        self.trades.append(Trade(step=step, action=1, price=price, quantity=quantity, cost=cost, regime_id=regime_id))
        return True

    def sell(self, price: float, quantity: int, step: int, regime_id: int | None = None) -> bool:
        if quantity > self.holdings:
            return False

        cost = self._sell_cost(price, quantity)
        self.cash += price * quantity - cost
        self.holdings -= quantity
        if self.holdings == 0:
            self.avg_buy_price = 0.0

        self.trades.append(Trade(step=step, action=-1, price=price, quantity=quantity, cost=cost, regime_id=regime_id))
        return True

    def net_worth(self, current_price: float) -> float:
        return self.cash + self.holdings * current_price

    def unrealized_pnl(self, current_price: float) -> float:
        if self.holdings == 0:
            return 0.0
        return (current_price - self.avg_buy_price) * self.holdings

    def record_net_worth(self, current_price: float):
        self.net_worth_history.append(self.net_worth(current_price))

    def _buy_cost(self, price: float, qty: int) -> float:
        turnover = price * qty
        brokerage = turnover * self.BROKERAGE_PCT
        gst = brokerage * self.GST_PCT
        exchange = turnover * self.EXCHANGE_TXN_PCT
        return brokerage + gst + exchange

    def _sell_cost(self, price: float, qty: int) -> float:
        turnover = price * qty
        brokerage = turnover * self.BROKERAGE_PCT
        stt = turnover * self.STT_SELL_PCT
        gst = brokerage * self.GST_PCT
        exchange = turnover * self.EXCHANGE_TXN_PCT
        return brokerage + stt + gst + exchange + self.DP_CHARGE

    def reset(self):
        self.cash = self.initial_cash
        self.holdings = 0
        self.avg_buy_price = 0.0
        self.trades.clear()
        self.net_worth_history.clear()


class TradeLogger:
    """Records every action + raw feature window for distillation (Stage 1→2)."""

    def __init__(self, profit_horizon: int = 1):
        self.profit_horizon = profit_horizon
        self.entries: list[TradeLogEntry] = []
        self._pending_pnl: dict[int, int] = {}  # step → index in entries

    def log(
        self,
        step: int,
        date_idx: int,
        action: int,
        feature_window: np.ndarray,
        price: float,
        regime_id: int | None = None,
    ):
        entry = TradeLogEntry(
            step=step,
            date_idx=date_idx,
            action=action,
            feature_window=feature_window.copy(),
            price=price,
            regime_id=regime_id,
        )
        idx = len(self.entries)
        self.entries.append(entry)
        self._pending_pnl[step] = idx

    def update_pnl(self, step: int, future_price: float):
        """Called `profit_horizon` steps after the action to compute actual P&L."""
        target_step = step - self.profit_horizon
        if target_step in self._pending_pnl:
            idx = self._pending_pnl.pop(target_step)
            entry = self.entries[idx]
            if entry.price > 0:
                entry.pnl_after_horizon = (future_price - entry.price) / entry.price * 100
            else:
                entry.pnl_after_horizon = 0.0

    def reset(self):
        self.entries.clear()
        self._pending_pnl.clear()
