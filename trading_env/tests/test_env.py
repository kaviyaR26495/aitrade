"""Basic tests for the trading environment."""
import numpy as np
import pytest

from trading_env.envs.swing_trading_env import SwingTradingEnv


def _make_env(n=100, obs_mode="flat", seq_len=5):
    rng = np.random.default_rng(42)
    prices = 100 + np.cumsum(rng.standard_normal(n) * 0.5)
    prices = np.maximum(prices, 10)  # floor
    data = rng.standard_normal((n, 10)).astype(np.float32)
    return SwingTradingEnv(
        data=data,
        prices=prices.astype(np.float32),
        seq_len=seq_len,
        obs_mode=obs_mode,
        initial_cash=100_000.0,
    )


def test_env_reset():
    env = _make_env()
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape
    assert info["cash"] == 100_000.0
    assert info["holdings"] == 0


def test_env_step_hold():
    env = _make_env()
    env.reset()
    obs, reward, terminated, truncated, info = env.step(0)  # HOLD
    assert not terminated
    assert info["holdings"] == 0


def test_env_buy_sell_cycle():
    env = _make_env()
    env.reset()
    # BUY
    env.step(1)
    assert env.portfolio.holdings > 0
    # HOLD (can't sell same step due to T+1)
    env.step(0)
    # SELL
    env.step(2)
    assert env.portfolio.holdings == 0


def test_env_sequential_mode():
    env = _make_env(obs_mode="sequential", seq_len=5)
    obs, info = env.reset()
    assert obs.shape == (10,)  # single timestep, 10 features
    obs2, *_ = env.step(0)
    assert obs2.shape == (10,)


def test_env_flat_mode():
    env = _make_env(obs_mode="flat", seq_len=5)
    obs, _ = env.reset()
    assert obs.shape == (5 * 10,)  # seq_len * features


def test_env_runs_to_completion():
    env = _make_env(n=50, seq_len=5)
    obs, _ = env.reset()
    done = False
    steps = 0
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
    assert steps > 0


def test_trade_logger_records():
    env = _make_env(n=50, seq_len=5)
    env.reset()
    for _ in range(10):
        env.step(env.action_space.sample())
    assert len(env.trade_logger.entries) == 10


def test_portfolio_costs():
    """Verify NSE cost calculations are non-zero."""
    from trading_env.portfolio import Portfolio
    p = Portfolio(100_000)
    p.buy(100.0, 10, step=0)
    assert p.cash < 100_000
    assert p.holdings == 10
    p.sell(105.0, 10, step=1)
    assert p.holdings == 0
    assert p.cash > 0
