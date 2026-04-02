"""Pattern Extractor — replay RL model to extract golden patterns.

Core of the RL→KNN+LSTM distillation bridge (Step 5.1):
1. Load best RL model → replay on full historical quality data (deterministic)
2. TradeLogger captures: date, action, feature window, P&L
3. Label patterns: BUY (label=1), SELL (label=-1), HOLD (label=0)
4. Tag each pattern with regime_id from stock_regimes
5. Store in golden_patterns table
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def extract_patterns(
    model,
    feature_data: np.ndarray,
    close_prices: np.ndarray,
    dates: np.ndarray | list,
    regime_ids: np.ndarray | None = None,
    seq_len: int = 15,
    obs_mode: str = "flat",
    reward_function: str = "risk_adjusted_pnl",
    min_profit_threshold: float = 1.2,
    profit_horizon: int = 1,
) -> list[dict[str, Any]]:
    """
    Replay RL model on data and extract golden patterns.

    min_profit_threshold: minimum P&L % to qualify as golden pattern (default 1.2%)
    profit_horizon: number of candles forward to measure P&L (default 1)

    Returns list of pattern dicts ready for DB insertion.
    """
    from trading_env import SwingTradingEnv

    env = SwingTradingEnv(
        df=feature_data,
        close_prices=close_prices,
        seq_len=seq_len,
        obs_mode=obs_mode,
        reward_function=reward_function,
        regime_ids=regime_ids,
    )

    obs, info = env.reset()
    done = False
    actions = []
    step_idx = 0

    # Collect all actions
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        actions.append(int(action))
        step_idx += 1
        done = terminated or truncated

    logger.info("Replayed %d steps, collected %d actions", step_idx, len(actions))

    # Build patterns from actions + forward P&L
    patterns = []
    n = len(close_prices)

    for i, action in enumerate(actions):
        # Actual data index (considering seq_len warmup)
        data_idx = seq_len + i
        if data_idx >= n:
            break

        # Skip HOLD actions (action=0)
        if action == 0:
            continue

        entry_price = close_prices[data_idx]

        # Forward P&L
        exit_idx = min(data_idx + profit_horizon, n - 1)
        exit_price = close_prices[exit_idx]

        if action == 1:  # BUY
            pnl_pct = ((exit_price - entry_price) / entry_price) * 100
        elif action == 2:  # SELL
            pnl_pct = ((entry_price - exit_price) / entry_price) * 100
        else:
            continue

        # Only keep patterns that meet profit threshold
        if pnl_pct < min_profit_threshold:
            continue

        # Feature window: seq_len candles up to and including the action point
        start = data_idx - seq_len
        end = data_idx
        if start < 0:
            continue
        feature_window = feature_data[start:end]

        # Label: 1=BUY, -1=SELL
        label = 1 if action == 1 else -1

        # Regime at the action point
        rid = int(regime_ids[data_idx]) if regime_ids is not None and data_idx < len(regime_ids) else None

        # Confidence: normalized P&L (higher P&L = higher confidence)
        confidence = min(pnl_pct / 5.0, 1.0)  # Cap at 5% = confidence 1.0

        pattern = {
            "date": dates[data_idx] if data_idx < len(dates) else None,
            "label": label,
            "pnl_percent": round(float(pnl_pct), 4),
            "regime_id": rid,
            "confidence": round(float(confidence), 4),
            "feature_window": feature_window.tobytes(),
            "feature_shape": list(feature_window.shape),
        }
        patterns.append(pattern)

    logger.info(
        "Extracted %d golden patterns (BUY: %d, SELL: %d)",
        len(patterns),
        sum(1 for p in patterns if p["label"] == 1),
        sum(1 for p in patterns if p["label"] == -1),
    )

    return patterns


def patterns_to_training_data(
    patterns: list[dict],
    include_hold: bool = True,
    feature_data: np.ndarray | None = None,
    seq_len: int = 15,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert golden patterns to X, y arrays for KNN/LSTM training.

    If include_hold is True and feature_data is provided, adds HOLD samples
    from non-pattern timesteps to balance classes.

    Returns (X, y) where:
    - X: (n_samples, seq_len, n_features)
    - y: (n_samples,) with labels 0=HOLD, 1=BUY, 2=SELL (for 3-class)
    """
    if not patterns:
        return np.array([]), np.array([])

    X_list = []
    y_list = []

    for p in patterns:
        window = np.frombuffer(p["feature_window"], dtype=np.float32)
        shape = p["feature_shape"]
        window = window.reshape(shape)
        X_list.append(window)

        # Map label: 1 (BUY) → 1, -1 (SELL) → 2, 0 (HOLD) → 0
        if p["label"] == 1:
            y_list.append(1)
        elif p["label"] == -1:
            y_list.append(2)
        else:
            y_list.append(0)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    return X, y
