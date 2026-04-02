"""Base trading environment — foundation for all trading envs.

Two observation modes:
- 'flat': flattened seq_len × features → 1D vector (for MLP policies: PPO, A2C, DDPG, TD3, SAC)
- 'sequential': single timestep features per step (for RecurrentPPO LSTM policy)
"""
from __future__ import annotations

from typing import Any, Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from trading_env.portfolio import Portfolio, TradeLogger
from trading_env.rewards import (
    risk_adjusted_pnl,
    sharpe_reward,
    sortino_reward,
    profit_reward,
)


class BaseTradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data: np.ndarray,                       # (num_candles, num_features)
        prices: np.ndarray,                      # (num_candles,) close prices
        regime_features: np.ndarray | None = None,  # (num_candles, regime_dim)
        seq_len: int = 15,
        obs_mode: Literal["flat", "sequential"] = "flat",
        reward_type: str = "risk_adjusted_pnl",
        initial_cash: float = 100_000.0,
        profit_horizon: int = 1,
        stoploss_pct: float = 5.0,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.data = data.astype(np.float32)
        self.prices = prices.astype(np.float32)
        self.regime_features = regime_features.astype(np.float32) if regime_features is not None else None
        self.seq_len = seq_len
        self.obs_mode = obs_mode
        self.reward_type = reward_type
        self.initial_cash = initial_cash
        self.profit_horizon = profit_horizon
        self.stoploss_pct = stoploss_pct
        self.render_mode = render_mode

        self.num_candles = len(data)
        self.num_features = data.shape[1]
        self.regime_dim = regime_features.shape[1] if regime_features is not None else 0
        self.total_features = self.num_features + self.regime_dim

        # Action space: 0=HOLD, 1=BUY, 2=SELL
        self.action_space = spaces.Discrete(3)

        # Observation space depends on mode
        if obs_mode == "flat":
            obs_dim = seq_len * self.total_features
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            )
        else:  # sequential — single timestep for RecurrentPPO
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.total_features,), dtype=np.float32
            )

        # State
        self.portfolio = Portfolio(initial_cash)
        self.trade_logger = TradeLogger(profit_horizon)
        self.current_step = 0
        self._start_step = seq_len  # need `seq_len` candles for initial window

    def reset(self, *, seed=None, options=None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.portfolio.reset()
        self.trade_logger.reset()
        self.current_step = self._start_step
        obs = self._get_observation()
        return obs, self._get_info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        price = self.prices[self.current_step]
        regime_id = self._current_regime_id()

        # Map action: 0=HOLD, 1=BUY, 2=SELL
        mapped_action = self._map_action(action)

        # Execute trade
        self._execute_action(mapped_action, price, regime_id)

        # Log for distillation
        feature_window = self._get_feature_window()
        self.trade_logger.log(
            step=self.current_step,
            date_idx=self.current_step,
            action=mapped_action,
            feature_window=feature_window,
            price=price,
            regime_id=regime_id,
        )

        # Update P&L for entries logged `profit_horizon` steps ago
        self.trade_logger.update_pnl(self.current_step, price)

        # Record net worth
        self.portfolio.record_net_worth(price)

        # Stoploss check
        self._check_stoploss(price)

        # Compute reward
        reward = self._compute_reward(price)

        # Advance step
        self.current_step += 1
        terminated = self.current_step >= self.num_candles - 1
        truncated = False

        obs = self._get_observation() if not terminated else np.zeros(
            self.observation_space.shape, dtype=np.float32
        )
        return obs, reward, terminated, truncated, self._get_info()

    def _map_action(self, action: int) -> int:
        """Map discrete action to trade signal: 0→0(HOLD), 1→1(BUY), 2→-1(SELL)."""
        return {0: 0, 1: 1, 2: -1}[action]

    def _execute_action(self, action: int, price: float, regime_id: int | None):
        if action == 1:  # BUY
            if self.portfolio.holdings == 0:
                qty = max(1, int(self.portfolio.cash * 0.95 / price))
                self.portfolio.buy(price, qty, self.current_step, regime_id)
        elif action == -1:  # SELL
            if self.portfolio.holdings > 0:
                self.portfolio.sell(price, self.portfolio.holdings, self.current_step, regime_id)

    def _check_stoploss(self, price: float):
        """Auto-sell if unrealized loss exceeds stoploss percentage."""
        if self.portfolio.holdings > 0 and self.portfolio.avg_buy_price > 0:
            loss_pct = (self.portfolio.avg_buy_price - price) / self.portfolio.avg_buy_price * 100
            if loss_pct >= self.stoploss_pct:
                self.portfolio.sell(price, self.portfolio.holdings, self.current_step)

    def _compute_reward(self, price: float) -> float:
        nw = self.portfolio.net_worth(price)
        history = self.portfolio.net_worth_history

        if self.reward_type == "risk_adjusted_pnl":
            return risk_adjusted_pnl(history, nw, self.initial_cash)
        elif self.reward_type == "sharpe":
            return sharpe_reward(history)
        elif self.reward_type == "sortino":
            return sortino_reward(history)
        elif self.reward_type == "profit":
            prev = history[-2] if len(history) >= 2 else self.initial_cash
            return profit_reward(nw, prev)
        return 0.0

    def _get_observation(self) -> np.ndarray:
        if self.obs_mode == "flat":
            window = self._get_feature_window()  # (seq_len, total_features)
            return window.flatten().astype(np.float32)
        else:
            # Sequential mode: single timestep
            row = self.data[self.current_step]
            if self.regime_features is not None:
                regime = self.regime_features[self.current_step]
                row = np.concatenate([row, regime])
            return row.astype(np.float32)

    def _get_feature_window(self) -> np.ndarray:
        """Get the feature window (seq_len × total_features) ending at current step."""
        start = max(0, self.current_step - self.seq_len + 1)
        end = self.current_step + 1
        window = self.data[start:end]

        if self.regime_features is not None:
            regime_window = self.regime_features[start:end]
            window = np.concatenate([window, regime_window], axis=1)

        # Pad if not enough history
        if len(window) < self.seq_len:
            pad = np.zeros((self.seq_len - len(window), window.shape[1]), dtype=np.float32)
            window = np.concatenate([pad, window], axis=0)

        return window.astype(np.float32)

    def _current_regime_id(self) -> int | None:
        if self.regime_features is None:
            return None
        # Regime ID is encoded in first column of regime_features (one-hot trend)
        # This is simplified — actual regime_id should be passed as metadata
        return None

    def _get_info(self) -> dict[str, Any]:
        if len(self.prices) == 0:
            return {
                "step": self.current_step,
                "cash": self.portfolio.cash,
                "holdings": self.portfolio.holdings,
                "net_worth": self.portfolio.cash,
                "num_trades": len(self.portfolio.trades),
            }
        price = self.prices[min(self.current_step, len(self.prices) - 1)]
        return {
            "step": self.current_step,
            "cash": self.portfolio.cash,
            "holdings": self.portfolio.holdings,
            "net_worth": self.portfolio.net_worth(price),
            "num_trades": len(self.portfolio.trades),
        }
