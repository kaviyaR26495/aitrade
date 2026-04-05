"""Swing Trading Environment — CNC-style, daily/weekly, configurable holding period.

- For daily interval: profit_horizon=1 (predict next trading day)
- For weekly interval: profit_horizon=1 (predict next trading week)
- SL default 5% (CNC)
- T+1 settlement simulation for daily
"""
from __future__ import annotations

from collections import deque
from typing import Literal

import numpy as np

from trading_env.envs.base_env import BaseTradingEnv


class SwingTradingEnv(BaseTradingEnv):
    """Swing/positional trading env for NSE CNC orders on daily/weekly data."""

    def __init__(
        self,
        data: np.ndarray,
        prices: np.ndarray,
        regime_features: np.ndarray | None = None,
        seq_len: int = 15,
        obs_mode: Literal["flat", "sequential"] = "flat",
        reward_type: str = "risk_adjusted_pnl",
        initial_cash: float = 100_000.0,
        profit_horizon: int = 1,
        stoploss_pct: float = 5.0,
        max_holding_days: int | None = None,
        render_mode: str | None = None,
        continuous: bool = False,
        churn_window: int = 5,
        churn_penalty: float = 0.002,
        drawdown_asymmetry: float = 2.0,
        slippage_bps: float = 5.0,
        vol_slippage_scale: float = 0.5,
    ):
        super().__init__(
            data=data,
            prices=prices,
            regime_features=regime_features,
            seq_len=seq_len,
            obs_mode=obs_mode,
            reward_type=reward_type,
            initial_cash=initial_cash,
            profit_horizon=profit_horizon,
            stoploss_pct=stoploss_pct,
            render_mode=render_mode,
            continuous=continuous,
            churn_window=churn_window,
            churn_penalty=churn_penalty,
            drawdown_asymmetry=drawdown_asymmetry,
            slippage_bps=slippage_bps,
            vol_slippage_scale=vol_slippage_scale,
        )
        self.max_holding_days = max_holding_days
        # FIFO queue of (quantity, buy_step) tuples.
        # One entry per buy lot; the oldest lot (index 0) governs T+1 eligibility.
        # Pyramiding is blocked at the _execute_action level (holdings == 0 guard),
        # so only one entry is ever present, but deque semantics make the T+1
        # logic correct even if that constraint is relaxed in the future.
        self._holding_lots: deque[tuple[int, int]] = deque()

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._holding_lots = deque()
        return obs, info

    def _execute_action(self, action: int, price: float, regime_id: int | None):
        if action == 1:  # BUY
            if self.portfolio.holdings == 0:
                fill = self._compute_fill_price(+1, price)
                qty = max(1, int(self.portfolio.cash * 0.95 / fill))
                if self.portfolio.buy(fill, qty, self.current_step, regime_id):
                    self._holding_lots.append((qty, self.current_step))
        elif action == -1:  # SELL
            if self.portfolio.holdings > 0:
                # T+1 settlement: can only sell if the oldest lot was bought before this step
                if self._holding_lots and self._holding_lots[0][1] < self.current_step:
                    fill = self._compute_fill_price(-1, price)
                    self.portfolio.sell(fill, self.portfolio.holdings, self.current_step, regime_id)
                    self._holding_lots.clear()

        # Auto-sell if max holding period exceeded (measured from the oldest lot)
        if (
            self.max_holding_days is not None
            and self._holding_lots
            and self.portfolio.holdings > 0
            and (self.current_step - self._holding_lots[0][1]) >= self.max_holding_days
        ):
            fill = self._compute_fill_price(-1, price)
            self.portfolio.sell(fill, self.portfolio.holdings, self.current_step)
            self._holding_lots.clear()
