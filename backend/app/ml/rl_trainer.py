"""RL Training Orchestrator.

Handles the full training flow:
1. Load quality-filtered, regime-tagged data
2. Append regime features to observation
3. Normalize features
4. Create trading environment
5. Train RL model with SB3
6. Save model artifacts
"""
from __future__ import annotations

import logging
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.config import settings
from app.ml.algorithms import (
    get_algorithm_class,
    get_default_hyperparams,
    get_obs_mode,
    is_continuous,
    ALGORITHM_CONFIGS,
)
from app.ml.callbacks import ProgressCallback, BestModelCallback
from app.core.indicators import compute_all_indicators
from app.core.normalizer import normalize_dataframe, get_feature_columns
from app.core.regime_classifier import classify_and_score

logger = logging.getLogger(__name__)


def _append_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append regime features to observation:
    - Trend one-hot (3): bullish, bearish, neutral
    - Volatility binary (1): high=1, low=0
    - Regime confidence (1)
    Total: 5 extra features per candle.
    """
    df = df.copy()

    # Trend one-hot
    df["regime_trend_bullish"] = (df["trend"] == "bullish").astype(float)
    df["regime_trend_bearish"] = (df["trend"] == "bearish").astype(float)
    df["regime_trend_neutral"] = (df["trend"] == "neutral").astype(float)

    # Volatility binary
    df["regime_vol_high"] = (df["volatility"] == "high").astype(float)

    # Confidence (already 0-1)
    # regime_confidence is already present

    return df


def prepare_training_data(
    ohlcv_df: pd.DataFrame,
    min_quality: float = 0.8,
    regime_ids: list[int] | None = None,
    exclude_transitions: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Full data preparation pipeline for RL training.

    Returns (normalized_df, feature_column_names).
    """
    # 1. Compute indicators
    df = compute_all_indicators(ohlcv_df)

    # 2. Classify regimes + quality score
    df = classify_and_score(df)

    # 3. Filter by quality + regime
    mask = df["quality_score"] >= min_quality
    if exclude_transitions:
        mask &= ~df["is_transition"]
    if regime_ids is not None:
        mask &= df["regime_id"].isin(regime_ids)
    df = df[mask].reset_index(drop=True)

    if len(df) < 100:
        raise ValueError(f"Insufficient quality data after filtering: {len(df)} rows (need >= 100)")

    # 4. Normalize first (before appending regime features)
    # Regime features are binary/pre-scaled and must NOT go through the log-return path
    df = normalize_dataframe(df)

    if len(df) < 100:
        raise ValueError(
            f"Insufficient data after normalization: {len(df)} rows (need >= 100). "
            "Try relaxing min_quality or regime filters."
        )

    # 5. Append regime features after normalization so they stay as-is (0/1 scaled)
    df = _append_regime_features(df)

    # 6. Get feature columns (includes both indicator + regime features)
    feature_cols = get_feature_columns(df)
    # Add regime features
    regime_feat_cols = [
        "regime_trend_bullish", "regime_trend_bearish", "regime_trend_neutral",
        "regime_vol_high", "regime_confidence",
    ]
    for col in regime_feat_cols:
        if col in df.columns and col not in feature_cols:
            feature_cols.append(col)

    return df, feature_cols


def _clear_gpu_memory() -> None:
    """Release unused CUDA tensors and free the GPU memory cache."""
    try:
        import torch
        if torch.cuda.is_available():
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


def train_rl_model(
    ohlcv_df: pd.DataFrame,
    algorithm: str = "PPO",
    hyperparams: dict | None = None,
    total_timesteps: int = 100_000,
    min_quality: float = 0.8,
    regime_ids: list[int] | None = None,
    reward_function: str = "risk_adjusted_pnl",
    seq_len: int = 15,
    save_dir: str | None = None,
    model_name: str | None = None,
    on_progress=None,
    device: str = "auto",
    stop_event=None,
    pause_event=None,
) -> dict[str, Any]:
    """
    Train an RL model on quality-filtered, regime-tagged data.

    Returns dict with model path, metrics, and training info.
    """
    # Validate algorithm
    if algorithm not in ALGORITHM_CONFIGS:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    obs_mode = get_obs_mode(algorithm)
    continuous = is_continuous(algorithm)

    # Prepare data
    df, feature_cols = prepare_training_data(
        ohlcv_df,
        min_quality=min_quality,
        regime_ids=regime_ids,
    )

    logger.info(
        "Prepared %d rows with %d features for %s training",
        len(df), len(feature_cols), algorithm,
    )

    # Extract feature matrix
    feature_data = df[feature_cols].values.astype(np.float32)

    # Raw price data for env reward calculation
    close_prices = df["close"].values.astype(np.float32)
    regime_ids_arr = df["regime_id"].values.astype(int) if "regime_id" in df.columns else None

    # ── CUDA optimisations ────────────────────────────────────────────
    if device == "cuda":
        import torch
        # Clear any leftover allocations from previous runs before starting
        _clear_gpu_memory()
        # cudnn autotuner finds the fastest conv kernels for the fixed input size
        torch.backends.cudnn.benchmark = True
        # Keep CPU threads lean — GPU handles the heavy lifting
        torch.set_num_threads(2)
        logger.info("GPU memory cleared. cudnn.benchmark enabled.")

    # ── Number of parallel envs ───────────────────────────────────────
    # More envs → more rollout data per step → GPU stays busy during policy updates.
    # Use SubprocVecEnv when we have CUDA (true parallelism) else DummyVecEnv.
    import multiprocessing
    if device == "cuda":
        # 4 or half of physical cores, whichever is less (cap at 8 to avoid memory pressure)
        n_envs = min(max(4, multiprocessing.cpu_count() // 2), 8)
    else:
        n_envs = 2

    logger.info("Creating %d parallel envs for %s training on %s", n_envs, algorithm, device)

    # ── Create vectorised environment ─────────────────────────────────
    from trading_env import SwingTradingEnv
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
    from stable_baselines3.common.monitor import Monitor

    def _make_env():
        def _init():
            env = SwingTradingEnv(
                data=feature_data,
                prices=close_prices,
                seq_len=seq_len,
                obs_mode=obs_mode,
                reward_type=reward_function,
            )
            return Monitor(env)
        return _init

    VecEnvClass = SubprocVecEnv if device == "cuda" else DummyVecEnv
    env = VecEnvClass([_make_env() for _ in range(n_envs)])

    # Merge hyperparams (defaults + user overrides)
    params = get_default_hyperparams(algorithm)
    if hyperparams:
        params.update(hyperparams)

    # Scale n_steps so total rollout buffer stays reasonable
    # PPO/A2C: collect n_steps * n_envs transitions per rollout
    if "n_steps" in params:
        # Keep the rollout buffer at ≥ 2048 transitions total
        params["n_steps"] = max(params["n_steps"] // n_envs, 512)

    # Ensure batch_size ≤ n_steps * n_envs (SB3 requirement for on-policy algos)
    if "batch_size" in params and "n_steps" in params:
        max_batch = params["n_steps"] * n_envs
        if params["batch_size"] > max_batch:
            params["batch_size"] = max_batch

    # Create SB3 model
    AlgorithmClass = get_algorithm_class(algorithm)
    policy = ALGORITHM_CONFIGS[algorithm]["policy"]

    # Extract policy_kwargs if present
    policy_kwargs = params.pop("policy_kwargs", None)

    model = AlgorithmClass(
        policy,
        env,
        verbose=0,
        device=device,
        **({"policy_kwargs": policy_kwargs} if policy_kwargs else {}),
        **params,
    )

    # Setup save directory
    if save_dir is None:
        save_dir = str(Path(settings.MODEL_DIR) / "rl")
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    if model_name is None:
        model_name = f"{algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Callbacks
    progress_cb = ProgressCallback(
        log_interval=1000,
        total_timesteps=total_timesteps,
        on_progress=on_progress,
        stop_event=stop_event,
        pause_event=pause_event,
    )
    best_model_cb = BestModelCallback(
        save_path=save_path / model_name,
        check_interval=max(total_timesteps // 20, 5000),
    )

    # Train
    logger.info("Starting %s training for %d timesteps on %d envs", algorithm, total_timesteps, n_envs)
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[progress_cb, best_model_cb],
        )
    finally:
        # Always close VecEnv workers and free GPU memory, even if training was stopped/failed
        try:
            env.close()
        except Exception:
            pass
        _clear_gpu_memory()
        logger.info("VecEnv closed and GPU memory cleared after training.")

    # Save final model
    final_path = save_path / model_name / "final_model"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(final_path))

    # Save training metadata
    meta = {
        "algorithm": algorithm,
        "model_name": model_name,
        "total_timesteps": total_timesteps,
        "hyperparams": {**params, **({"policy_kwargs": policy_kwargs} if policy_kwargs else {})},
        "feature_cols": feature_cols,
        "num_features": len(feature_cols),
        "seq_len": seq_len,
        "obs_mode": obs_mode,
        "min_quality": min_quality,
        "regime_ids": regime_ids,
        "reward_function": reward_function,
        "data_rows": len(df),
        "training_metrics": progress_cb.metrics_log,
        "best_reward": best_model_cb.best_reward,
    }

    meta_path = save_path / model_name / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    # Get final episode stats
    final_reward = None
    if len(model.ep_info_buffer) > 0:
        final_reward = sum(ep["r"] for ep in model.ep_info_buffer) / len(model.ep_info_buffer)

    return {
        "model_name": model_name,
        "algorithm": algorithm,
        "model_path": str(final_path) + ".zip",
        "metadata_path": str(meta_path),
        "total_timesteps": total_timesteps,
        "final_reward": final_reward,
        "best_reward": best_model_cb.best_reward,
        "data_rows": len(df),
        "num_features": len(feature_cols),
        "training_log": progress_cb.metrics_log,
    }


def load_rl_model(model_path: str, algorithm: str):
    """Load a saved RL model."""
    AlgorithmClass = get_algorithm_class(algorithm)
    return AlgorithmClass.load(model_path)


def evaluate_rl_model(
    model,
    ohlcv_df: pd.DataFrame,
    feature_cols: list[str],
    seq_len: int = 15,
    obs_mode: str = "flat",
    reward_function: str = "risk_adjusted_pnl",
    n_eval_episodes: int = 1,
) -> dict:
    """Evaluate a trained RL model on test data."""
    from trading_env import SwingTradingEnv

    df, _ = prepare_training_data(ohlcv_df, min_quality=0.0)

    feature_data = df[feature_cols].values.astype(np.float32)
    close_prices = df["close"].values.astype(np.float32)
    regime_ids_arr = df["regime_id"].values.astype(int) if "regime_id" in df.columns else None

    env = SwingTradingEnv(
        df=feature_data,
        close_prices=close_prices,
        seq_len=seq_len,
        obs_mode=obs_mode,
        reward_function=reward_function,
        regime_ids=regime_ids_arr,
    )

    total_rewards = []
    total_trades = []

    for _ in range(n_eval_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        total_rewards.append(episode_reward)
        total_trades.append(len(env.trade_logger.trades) if hasattr(env, "trade_logger") else 0)

    return {
        "mean_reward": float(np.mean(total_rewards)),
        "std_reward": float(np.std(total_rewards)),
        "mean_trades": float(np.mean(total_trades)),
        "episodes": n_eval_episodes,
    }
