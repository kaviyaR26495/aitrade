"""Pattern Extractor — replay RL model and build supervised training data.

Two extraction modes are supported:

  mode="behavioral_cloning"  (default)
    Imitation learning — LSTM/KNN learn the RL agent's *complete* policy.
    Every RL action (BUY, SELL, and HOLD) is recorded as a labelled training
    sample.  HOLDs are downsampled to max_hold_ratio x (buy+sell count) so the
    supervised models also learn which setups the RL *correctly avoided*.
    This eliminates the survivorship bias of the golden-patterns approach where
    only profitable trades were passed downstream.

  mode="golden_patterns"
    Legacy behavior — only BUY/SELL actions that cleared min_profit_threshold
    are kept.  Useful as a high-precision signal source when labelled data is
    limited.
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
    mode: str = "behavioral_cloning",
    max_hold_ratio: float = 5.0,
) -> list[dict[str, Any]]:
    """
    Replay RL model on data and extract labelled training patterns.

    Parameters
    ----------
    mode : "behavioral_cloning" | "golden_patterns"
        behavioral_cloning (default) — records every RL action as a label,
        eliminating survivorship bias.  golden_patterns — legacy, keeps only
        BUY/SELL that cleared min_profit_threshold.
    max_hold_ratio : float
        behavioral_cloning only: HOLDs are downsampled so that
        #holds <= max_hold_ratio * (#buy + #sell).  Default 3.0.
    min_profit_threshold : float
        golden_patterns only: minimum forward P&L % to keep a pattern.
    profit_horizon : int
        Number of candles forward used to measure realised P&L.

    Returns list of pattern dicts ready for DB insertion.
    """
    from trading_env import SwingTradingEnv

    env = SwingTradingEnv(
        data=feature_data,
        prices=close_prices,
        seq_len=seq_len,
        obs_mode=obs_mode,
        reward_type=reward_function,
    )

    obs, info = env.reset()
    done = False
    actions: list[int] = []
    step_idx = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        actions.append(int(action))
        step_idx += 1
        done = terminated or truncated

    logger.info("Replayed %d steps, collected %d actions", step_idx, len(actions))

    n = len(close_prices)
    raw_patterns: list[dict[str, Any]] = []

    for i, action in enumerate(actions):
        # Map step index to data index (seq_len warmup shift)
        data_idx = seq_len + i
        if data_idx >= n:
            break

        start = data_idx - seq_len
        if start < 0:
            continue

        entry_price = close_prices[data_idx]

        # Forward P&L
        exit_idx = min(data_idx + profit_horizon, n - 1)
        exit_price = close_prices[exit_idx]

        if action == 1:    # BUY
            pnl_pct = ((exit_price - entry_price) / entry_price) * 100
        elif action == 2:  # SELL
            pnl_pct = ((entry_price - exit_price) / entry_price) * 100
        else:              # HOLD
            pnl_pct = 0.0

        # ── Golden patterns mode: filter by P&L, skip HOLDs ───────────
        if mode == "golden_patterns":
            if action == 0:
                continue
            if pnl_pct < min_profit_threshold:
                continue
            label = 1 if action == 1 else -1
            confidence = min(pnl_pct / 5.0, 1.0)
        else:
            # ── Behavioral cloning: all RL decisions become training labels
            if action == 1:
                label = 1
                confidence = min(max(pnl_pct / 5.0, 0.0), 1.0)
            elif action == 2:
                label = -1
                confidence = min(max(pnl_pct / 5.0, 0.0), 1.0)
            else:
                label = 0
                confidence = 0.5  # HOLD: neutral confidence

        # Feature window: seq_len candles up to and including the action point
        if start < 0:
            continue
        feature_window = feature_data[start:data_idx]

        # Regime at the action point
        rid = int(regime_ids[data_idx]) if regime_ids is not None and data_idx < len(regime_ids) else None

        pattern = {
            "date": dates[data_idx] if data_idx < len(dates) else None,
            "label": label,
            "pnl_percent": round(float(pnl_pct), 4),
            "regime_id": rid,
            "confidence": round(float(confidence), 4),
            "feature_window": feature_window.astype(np.float32).tobytes(),
            "_feature_shape": list(feature_window.shape),
        }
        raw_patterns.append(pattern)

    # ── Downsample HOLDs in behavioral_cloning mode ────────────────────
    if mode == "behavioral_cloning":
        trade_patterns = [p for p in raw_patterns if p["label"] != 0]
        hold_patterns  = [p for p in raw_patterns if p["label"] == 0]
        max_holds = int(len(trade_patterns) * max_hold_ratio)

        if len(hold_patterns) > max_holds and max_holds > 0:
            # Evenly-spaced subsample preserves chronological coverage
            step = max(len(hold_patterns) // max_holds, 1)
            hold_patterns = hold_patterns[::step][:max_holds]

        patterns = trade_patterns + hold_patterns
        patterns.sort(key=lambda p: str(p["date"]) if p["date"] is not None else "")
    else:
        patterns = raw_patterns

    n_buy  = sum(1 for p in patterns if p["label"] == 1)
    n_sell = sum(1 for p in patterns if p["label"] == -1)
    n_hold = sum(1 for p in patterns if p["label"] == 0)
    logger.info(
        "Extracted %d patterns in '%s' mode (BUY: %d, SELL: %d, HOLD: %d)",
        len(patterns), mode, n_buy, n_sell, n_hold,
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

    Jitter augmentation is intentionally NOT applied here — it must be applied
    only to the training split *inside* train_knn / train_lstm to prevent
    synthetic samples from leaking into the validation set.

    Returns (X, y) where:
    - X: (n_samples, seq_len, n_features)
    - y: (n_samples,) with labels 0=HOLD, 1=BUY, 2=SELL (for 3-class)
    """
    if not patterns:
        return np.array([]), np.array([])

    X_list = []
    y_list = []

    for p in patterns:
        raw = p["feature_window"]
        # _feature_shape is set in-memory by extract_patterns; DB rows don't have it.
        # Infer shape: n_features = total_floats / seq_len
        if "_feature_shape" in p:
            shape = p["_feature_shape"]
        else:
            total_floats = len(raw) // 4  # float32 = 4 bytes
            n_features = total_floats // seq_len
            shape = [seq_len, n_features]
        window = np.frombuffer(raw, dtype=np.float32).reshape(shape)
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

    # Optionally filter out HOLDs for golden-pattern style training
    if not include_hold:
        mask = y != 0
        X, y = X[mask], y[mask]

    return X, y


def jitter_augment(
    X: np.ndarray,
    y: np.ndarray,
    noise_std: float = 0.001,
    copies: int = 1,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Augment training data by injecting Gaussian noise into feature windows.

    Each sample in X receives `copies` noisy duplicates sharing the same label.
    Forces LSTM/KNN to learn the *shape* of patterns rather than memorising
    exact values, improving generalisation on small datasets.

    Parameters
    ----------
    noise_std : float
        Standard deviation of the noise added to each feature value.
        Default 0.001 ≈ 0.1% of a unit-normalised feature.
    copies : int
        Number of augmented copies produced per original sample.
        Total dataset size becomes X.shape[0] * (1 + copies).
    rng : np.random.Generator | None
        Optional seeded generator for reproducibility.

    Returns
    -------
    X_aug : (n * (1+copies), seq_len, n_features)
    y_aug : (n * (1+copies),)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    X_parts = [X]
    y_parts = [y]

    for _ in range(copies):
        noise = rng.normal(0.0, noise_std, size=X.shape).astype(np.float32)
        X_parts.append(X + noise)
        y_parts.append(y)

    X_aug = np.concatenate(X_parts, axis=0)
    y_aug = np.concatenate(y_parts, axis=0)

    logger.info(
        "Jitter augmentation: %d original + %d synthetic = %d total samples",
        len(X), len(X_aug) - len(X), len(X_aug),
    )
    return X_aug, y_aug
