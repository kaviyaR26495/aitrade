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
    ohlcv_df: pd.DataFrame | None = None,
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
    pretrained_model_path: str | None = None,
    multi_ohlcv_dfs: list[pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """
    Train an RL model on quality-filtered, regime-tagged data.

    Pass ``multi_ohlcv_dfs`` for multi-stock training: one environment is
    created per stock (cycling when there are more stocks than parallel envs).
    Pass ``ohlcv_df`` for traditional single-stock training.

    Returns dict with model path, metrics, and training info.
    """
    # Validate algorithm
    if algorithm not in ALGORITHM_CONFIGS:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    obs_mode = get_obs_mode(algorithm)
    continuous = is_continuous(algorithm)

    # ── CUDA optimisations ────────────────────────────────────────────
    if device == "cuda":
        import torch
        torch.backends.cudnn.benchmark = True
        torch.set_num_threads(2)
        logger.info("cudnn.benchmark enabled.")

    # ── Number of parallel envs ───────────────────────────────────────
    import multiprocessing
    if device == "cuda":
        n_envs = min(max(4, multiprocessing.cpu_count() // 2), 8)
    else:
        n_envs = 2

    # ── Prepare per-stock data ────────────────────────────────────────
    # multi_ohlcv_dfs: each df is prepared independently so indicators
    # don't bleed across stock boundaries.
    if multi_ohlcv_dfs:
        stock_datasets: list[tuple[np.ndarray, np.ndarray]] = []
        for sdf in multi_ohlcv_dfs:
            try:
                prep_df, feat_cols = prepare_training_data(
                    sdf, min_quality=min_quality, regime_ids=regime_ids,
                )
                fd = prep_df[feat_cols].values.astype(np.float32)
                cp = prep_df["adj_close"].fillna(prep_df["close"]).values.astype(np.float32)
                stock_datasets.append((fd, cp))
            except Exception as exc:
                logger.warning("Multi-stock prep: skipping a stock — %s", exc)
        if not stock_datasets:
            raise ValueError("All stocks failed data preparation — cannot train")
        # Use the first stock's feature_cols for shape reference
        prep_df, feature_cols = prepare_training_data(
            multi_ohlcv_dfs[0], min_quality=min_quality, regime_ids=regime_ids,
        )
        # feature_data / close_prices not used directly — envs use stock_datasets
        feature_data = stock_datasets[0][0]
        close_prices = stock_datasets[0][1]
        logger.info(
            "Multi-stock training: %d stocks prepared, %d features each, %d envs",
            len(stock_datasets), feature_data.shape[1], n_envs,
        )
    else:
        if ohlcv_df is None:
            raise ValueError("Either ohlcv_df or multi_ohlcv_dfs must be provided")
        prep_df, feature_cols = prepare_training_data(
            ohlcv_df, min_quality=min_quality, regime_ids=regime_ids,
        )
        feature_data = prep_df[feature_cols].values.astype(np.float32)
        close_prices = prep_df["adj_close"].fillna(prep_df["close"]).values.astype(np.float32)
        stock_datasets = [(feature_data, close_prices)]
        logger.info("Prepared %d rows with %d features for %s training", len(prep_df), len(feature_cols), algorithm)

    regime_ids_arr = prep_df["regime_id"].values.astype(int) if "regime_id" in prep_df.columns else None

    logger.info("Creating %d parallel envs for %s training on %s", n_envs, algorithm, device)

    # ── Create vectorised environment ─────────────────────────────────
    from trading_env import SwingTradingEnv
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
    from stable_baselines3.common.monitor import Monitor

    def _make_env(idx: int):
        fd, cp = stock_datasets[idx % len(stock_datasets)]
        def _init():
            env = SwingTradingEnv(
                data=fd,
                prices=cp,
                seq_len=seq_len,
                obs_mode=obs_mode,
                reward_type=reward_function,
                continuous=continuous,
            )
            return Monitor(env)
        return _init

    VecEnvClass = SubprocVecEnv if device == "cuda" else DummyVecEnv
    env = VecEnvClass([_make_env(i) for i in range(n_envs)])

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

    # Inject AttentionFeaturesExtractor for AttentionPPO (requires live class)
    if algorithm == "AttentionPPO":
        from app.ml.attention_extractor import AttentionFeaturesExtractor
        if policy_kwargs is None:
            policy_kwargs = {}
        policy_kwargs["features_extractor_class"] = AttentionFeaturesExtractor
        policy_kwargs["features_extractor_kwargs"] = {
            "features_dim": 256,
            "seq_len": seq_len,
            "num_heads": 4,
        }
        logger.info("AttentionPPO: injected AttentionFeaturesExtractor (seq_len=%d)", seq_len)

    # Create or load SB3 model
    if pretrained_model_path:
        logger.info("Loading pretrained model from %s", pretrained_model_path)
        model = AlgorithmClass.load(
            pretrained_model_path,
            env=env,
            device=device,
        )
    else:
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

    # Save final model — timestamped so old artifacts are preserved on disk
    _ts = datetime.now().strftime("%Y%m%d_%H%M")
    final_path = save_path / model_name / f"final_model_{_ts}"
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
        "data_rows": len(prep_df),
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
        "data_rows": len(prep_df),
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
    close_prices = df["adj_close"].fillna(df["close"]).values.astype(np.float32)
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


def train_curriculum(
    ohlcv_df: pd.DataFrame,
    algorithm: str = "RecurrentPPO",
    total_timesteps: int = 300_000,
    reward_function: str = "dense",
    seq_len: int = 15,
    save_dir: str | None = None,
    model_name: str | None = None,
    on_progress=None,
    device: str = "auto",
    stop_event=None,
    pause_event=None,
    hyperparams: dict | None = None,
) -> dict[str, Any]:
    """Three-phase curriculum training — progressively harder regime filters.

    Phase 1 — Clean trends (30% of timesteps)
        Regimes 0 (Bullish+LowVol) and 4 (Bearish+LowVol).
        Trends are obvious; the agent learns that MACD crossovers and SMA
        alignments predict direction reliably.

    Phase 2 — Volatile trends (30%)
        Adds Regimes 1 (Bullish+HighVol) and 5 (Bearish+HighVol).
        The agent encounters drawdowns mid-trend and must learn position
        management and wider stop-loss tolerance while retaining Phase 1 knowledge.

    Phase 3 — All regimes (40%)
        Includes sideways/choppy regimes (2, 3).
        Full generalisation; the agent must learn to HOLD in lateral markets
        without over-trading, combining all prior lessons.

    Between phases, the final model is used as the pre-trained starting point so
    knowledge is transferred rather than discarded.

    Returns
    -------
    dict with keys: ``phases`` (list of per-phase result dicts) + all keys from
    the final phase result (model_path, best_reward, etc.).
    """
    if save_dir is None:
        save_dir = str(Path(settings.MODEL_DIR) / "rl")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = model_name or f"{algorithm}_curriculum_{ts}"

    phases = [
        {
            "name": f"{base_name}_phase1",
            "regime_ids": [0, 4],
            "fraction": 0.30,
            "description": "Phase 1: clean trends (Regime 0+4)",
        },
        {
            "name": f"{base_name}_phase2",
            "regime_ids": [0, 1, 4, 5],
            "fraction": 0.30,
            "description": "Phase 2: volatile trends (Regime 0+1+4+5)",
        },
        {
            "name": f"{base_name}_phase3",
            "regime_ids": None,
            "fraction": 0.40,
            "description": "Phase 3: all regimes (full generalisation)",
        },
    ]

    phase_results: list[dict] = []
    prev_model_path: str | None = None

    for phase in phases:
        phase_steps = int(total_timesteps * phase["fraction"])
        logger.info(
            "Curriculum %s: %s — %d timesteps",
            base_name, phase["description"], phase_steps,
        )

        try:
            result = train_rl_model(
                ohlcv_df=ohlcv_df,
                algorithm=algorithm,
                hyperparams=hyperparams,
                total_timesteps=phase_steps,
                min_quality=0.7,
                regime_ids=phase["regime_ids"],
                reward_function=reward_function,
                seq_len=seq_len,
                save_dir=save_dir,
                model_name=phase["name"],
                on_progress=on_progress,
                device=device,
                stop_event=stop_event,
                pause_event=pause_event,
                pretrained_model_path=prev_model_path,
            )
            result["phase"] = phase["description"]
            phase_results.append(result)
            prev_model_path = result["model_path"]
            logger.info(
                "Curriculum phase complete: best_reward=%.4f, model=%s",
                result.get("best_reward") or 0.0,
                result["model_path"],
            )
        except ValueError as exc:
            logger.warning(
                "Curriculum phase skipped (%s): %s — continuing with unchanged weights.",
                phase["description"], exc,
            )
            phase_results.append({
                "phase": phase["description"],
                "skipped": True,
                "reason": str(exc),
                "model_path": prev_model_path,
            })

    if not phase_results:
        raise RuntimeError("All curriculum phases failed — no training data available.")

    # Return the last successful phase result, augmented with all phases log
    final = next((r for r in reversed(phase_results) if not r.get("skipped")), phase_results[-1])
    final["phases"] = phase_results
    final["model_name"] = base_name
    return final


def hybrid_train(
    ohlcv_df: pd.DataFrame,
    total_timesteps: int = 300_000,
    seq_len: int = 15,
    reward_function: str = "dense",
    save_dir: str | None = None,
    model_name: str | None = None,
    min_quality: float = 0.8,
    on_progress=None,
    device: str = "auto",
    stop_event=None,
    pause_event=None,
    # ── Phase 1: Offline CQL ─────────────────────────────
    cql_n_steps: int = 50_000,
    cql_alpha: float = 4.0,
    cql_n_episodes: int = 3,
    source_model=None,
    # ── Phase 2: BC warm-up ────────────────────────────────────────────
    bc_warmup_steps: int = 2_000,
    # ── Phase 3: Online AttentionPPO ───────────────────────────────────
    online_fraction: float = 0.7,
) -> dict[str, Any]:
    """Hybrid Offline→Online training pipeline.
    Phase 1 — Offline CQL Pre-training
        DiscreteCQL learns a conservative baseline policy purely from a static
        historical dataset.  No simulator interaction.  The CQL conservative
        regularisation (``alpha``) prevents the agent from assigning high
        Q-values to actions that never appear in the data, mitigating the
        over-estimation that kills standard RL on financial replay data.

    Phase 2 — Behavioral Cloning Warm-up
        The AttentionPPO actor is warm-started to produce the same greedy
        actions as the CQL policy via a short supervised pass (cross-entropy
        loss on the PPO actor's log-probs against CQL labels).
        No environment interaction — pure offline optimisation.

    Phase 3 — Online AttentionPPO Fine-tuning
        AttentionPPO explores the SwingTradingEnv starting from the
        CQL-warmed weights.  Because the policy already knows which actions are
        reasonable, it needs far fewer timesteps to converge.  The dense_reward
        sharpens execution timing.

    Parameters
    ----------
    total_timesteps : int
        Total online timesteps for Phase 3.  CQL offline steps are in addition
        to this budget — they run entirely without the simulator.
    source_model : optional SB3 model
        If provided, used to collect the offline dataset (higher-quality
        transitions near the behaviour policy distribution).  Pass ``None``
        (default) to use a uniform-random policy instead.
    cql_alpha : float
        CQL conservative regularisation strength.  Higher → more conservative.
        Range 2.0–8.0; 4.0 is a good default for financial data.
    bc_warmup_steps : int
        Number of BC gradient steps to align the PPO actor with CQL before
        online fine-tuning.  2 000 is usually enough.
    online_fraction : float
        Fraction of ``total_timesteps`` used for Phase 3 online PPO fine-tuning.

    Returns
    -------
    dict with keys:
        model_path   — final AttentionPPO model (.zip).
        algorithm    — "HybridCQL+AttentionPPO".
        cql_result   — Phase 1 CQL metadata and offline metrics.
        bc_warmup    — Phase 2 BC loss info.
        ppo_result   — Phase 3 train_rl_model() result dict.
        total_timesteps — as passed in.
    """
    from app.ml.cql_trainer import (
        collect_offline_transitions,
        build_offline_dataset,
        train_cql,
        evaluate_offline,
        bc_warmup_ppo,
    )

    if save_dir is None:
        save_dir = str(Path(settings.MODEL_DIR) / "rl")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = model_name or f"HybridCQL_AttentionPPO_{ts}"

    # ── Phase 1: Offline CQL ──────────────────────────────────────────
    logger.info("=== Hybrid Phase 1: Offline CQL Pre-training ===")

    transitions = collect_offline_transitions(
        ohlcv_df=ohlcv_df,
        source_model=source_model,
        seq_len=seq_len,
        obs_mode="flat",
        reward_function=reward_function,
        min_quality=min_quality,
        n_episodes=cql_n_episodes,
    )

    dataset = build_offline_dataset(transitions)

    cql_save_dir = str(Path(save_dir) / "cql")
    cql_algo, cql_path = train_cql(
        dataset=dataset,
        n_steps=cql_n_steps,
        alpha=cql_alpha,
        save_dir=cql_save_dir,
        model_name=f"{base_name}_cql",
        device=device,
    )

    cql_metrics = evaluate_offline(cql_algo, transitions)
    logger.info("CQL offline metrics: %s", cql_metrics)

    cql_result = {
        "model_path": cql_path,
        "n_steps": cql_n_steps,
        "alpha": cql_alpha,
        "metrics": cql_metrics,
    }

    # ── Phase 2: Build AttentionPPO + BC warm-up ──────────────────────
    logger.info("=== Hybrid Phase 2: AttentionPPO BC Warm-up from CQL ===")

    df, feature_cols = prepare_training_data(ohlcv_df, min_quality=min_quality)
    feature_data = df[feature_cols].values.astype(np.float32)
    close_prices = df["adj_close"].fillna(df["close"]).values.astype(np.float32)

    from trading_env import SwingTradingEnv
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    from app.ml.attention_extractor import AttentionFeaturesExtractor

    def _make_warmup_env():
        env = SwingTradingEnv(
            data=feature_data,
            prices=close_prices,
            seq_len=seq_len,
            obs_mode="flat",
            reward_type=reward_function,
        )
        return Monitor(env)

    warmup_vec_env = DummyVecEnv([_make_warmup_env])

    # Build AttentionPPO on a single env for the BC warm-up
    attn_params = get_default_hyperparams("AttentionPPO")
    attn_params.pop("policy_kwargs", None)  # injected manually below
    if "n_steps" in attn_params:
        attn_params["n_steps"] = max(attn_params["n_steps"], 512)

    ppo_model = PPO(
        "MlpPolicy",
        warmup_vec_env,
        verbose=0,
        device=device,
        policy_kwargs={
            "net_arch": {"pi": [256, 128], "vf": [512, 512, 256]},
            "features_extractor_class": AttentionFeaturesExtractor,
            "features_extractor_kwargs": {
                "features_dim": 256,
                "seq_len": seq_len,
                "num_heads": 4,
            },
        },
        **attn_params,
    )

    bc_result = bc_warmup_ppo(
        sb3_model=ppo_model,
        cql_algo=cql_algo,
        observations=transitions["observations"],
        device=device,
        n_steps=bc_warmup_steps,
    )
    logger.info("BC warm-up: %s", bc_result)

    # Save warm-started weights so train_rl_model can load them
    warmup_dir = str(Path(save_dir) / "warmup")
    Path(warmup_dir).mkdir(parents=True, exist_ok=True)
    warmup_path = str(Path(warmup_dir) / f"{base_name}_warmup.zip")
    ppo_model.save(warmup_path)
    warmup_vec_env.close()
    _clear_gpu_memory()

    # ── Phase 3: Online AttentionPPO fine-tuning ──────────────────────
    logger.info("=== Hybrid Phase 3: Online AttentionPPO Fine-tuning ===")

    online_steps = max(int(total_timesteps * online_fraction), 10_000)
    ppo_result = train_rl_model(
        ohlcv_df=ohlcv_df,
        algorithm="AttentionPPO",
        total_timesteps=online_steps,
        min_quality=min_quality,
        reward_function=reward_function,
        seq_len=seq_len,
        save_dir=save_dir,
        model_name=f"{base_name}_final",
        on_progress=on_progress,
        device=device,
        stop_event=stop_event,
        pause_event=pause_event,
        pretrained_model_path=warmup_path,
    )

    logger.info("Hybrid training complete — final model: %s", ppo_result["model_path"])

    return {
        "model_name": base_name,
        "model_path": ppo_result["model_path"],
        "algorithm": "HybridCQL+AttentionPPO",
        "cql_result": cql_result,
        "bc_warmup": bc_result,
        "ppo_result": ppo_result,
        "total_timesteps": total_timesteps,
    }
def run_bc_warmup(
    cql_path: str,
    ohlcv_df: pd.DataFrame,
    algorithm: str = "AttentionPPO",
    seq_len: int = 15,
    reward_function: str = "dense",
    min_quality: float = 0.8,
    bc_steps: int = 2000,
    device: str = "auto",
    save_dir: str | None = None,
    model_name: str | None = None,
) -> str:
    """Run Behavioral Cloning to align a PPO model with a pretrained CQL policy.
    
    Returns path to the warmed-up model (.zip).
    """
    from app.ml.cql_trainer import load_cql, bc_warmup_ppo, collect_offline_transitions
    from trading_env import SwingTradingEnv
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    from app.ml.attention_extractor import AttentionFeaturesExtractor

    # 1. Load CQL
    cql_algo = load_cql(cql_path)

    # 2. Prepare data for BC supervision
    transitions = collect_offline_transitions(
        ohlcv_df=ohlcv_df,
        source_model=None,
        seq_len=seq_len,
        obs_mode="flat",
        reward_function=reward_function,
        min_quality=min_quality,
        n_episodes=1,
    )
    
    # 3. Create dummy env for PPO init
    df, feature_cols = prepare_training_data(ohlcv_df, min_quality=min_quality)
    fd = df[feature_cols].values.astype(np.float32)
    cp = df["adj_close"].fillna(df["close"]).values.astype(np.float32)
    
    def _make_env():
        return Monitor(SwingTradingEnv(fd, cp, seq_len=seq_len, obs_mode="flat", reward_type=reward_function))
    
    vec_env = DummyVecEnv([_make_env])

    # 4. Create PPO model
    attn_params = get_default_hyperparams("AttentionPPO")
    attn_params.pop("policy_kwargs", None)
    if "n_steps" in attn_params:
        attn_params["n_steps"] = max(attn_params["n_steps"], 512)

    ppo_model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=0,
        device=device,
        policy_kwargs={
            "net_arch": {"pi": [256, 128], "vf": [512, 512, 256]},
            "features_extractor_class": AttentionFeaturesExtractor,
            "features_extractor_kwargs": {
                "features_dim": 256,
                "seq_len": seq_len,
                "num_heads": 4,
            },
        },
        **attn_params,
    )

    # 5. Run BC
    bc_warmup_ppo(
        sb3_model=ppo_model,
        cql_algo=cql_algo,
        observations=transitions["observations"],
        device=device,
        n_steps=bc_steps,
    )

    # 6. Save
    if save_dir is None:
        save_dir = str(Path(settings.MODEL_DIR) / "rl" / "warmup")
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    if model_name is None:
        model_name = f"BC_Warmup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    warmup_path = str(Path(save_dir) / model_name)
    ppo_model.save(warmup_path)
    vec_env.close()
    
    return warmup_path + ".zip"
