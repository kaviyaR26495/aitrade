"""CQL (Conservative Q-Learning) Offline RL Trainer.

Uses the d3rlpy library to train a DiscreteCQL agent from a static dataset of
historical (obs, action, reward, next_obs, terminal) transitions.  The trained
CQL policy is conservative by design — it penalises out-of-distribution actions,
making it ideal for financial time-series where live exploration is too costly.

Workflow
--------
1. collect_offline_transitions()  — rolls out a source model (or random policy)
                                    on SwingTradingEnv and records every step.
2. build_offline_dataset()        — packages the transition arrays into a
                                    d3rlpy MDPDataset.
3. train_cql()                    — DiscreteCQL offline training.
4. evaluate_offline()             — action-distribution and reward metrics.

Hybrid Warm-Start
-----------------
After CQL training, call bc_warmup_ppo() to run a short Behavioral Cloning
pass on an SB3 PPO model using CQL's greedy actions as supervision labels.
This aligns the PPO actor with the conservative CQL baseline *before* online
PPO exploration begins — no direct weight-copying required (the two models can
differ in architecture).

Backtest Comparison
-------------------
compare_agents() runs any mix of SB3 models and CQLAgentWrapper instances on
the same held-out dataset and returns side-by-side metrics.  Use this to
"race" the offline CQL policy against the hybrid AttentionPPO and pick the
better generaliser before deploying to live markets.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.config import settings
from app.ml.rl_trainer import prepare_training_data

logger = logging.getLogger(__name__)


# ─── helpers ─────────────────────────────────────────────────────────────────

def _resolve_device(device: str) -> str:
    """Resolve 'auto' to 'cuda:0' or 'cpu'."""
    if device == "auto":
        try:
            import torch
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    if device == "cuda":
        return "cuda:0"
    return device


def _check_d3rlpy() -> None:
    try:
        import d3rlpy  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "d3rlpy is required for offline CQL training.  "
            "Install it with:  pip install 'd3rlpy>=2.4.0'"
        ) from exc


# ─── dataset collection ──────────────────────────────────────────────────────

def collect_offline_transitions(
    ohlcv_df: pd.DataFrame,
    source_model=None,
    seq_len: int = 15,
    obs_mode: str = "flat",
    reward_function: str = "dense",
    min_quality: float = 0.8,
    regime_ids: list[int] | None = None,
    n_episodes: int = 3,
) -> dict[str, np.ndarray]:
    """Collect offline (obs, action, reward, next_obs, terminal) tuples.

    Parameters
    ----------
    source_model : SB3 model with `.predict()` interface, or ``None``.
        When provided, uses its greedy policy to populate the dataset (gives
        higher-quality transitions near the behaviour policy distribution).
        When ``None``, samples uniformly random actions — useful for maximising
        state-space coverage when no pretrained model is available yet.
    n_episodes : int
        Number of independent env rollouts.  Each starts from the beginning of
        the same historical window (env resets the data pointer each time).

    Returns
    -------
    dict with keys: observations, actions, rewards, next_observations, terminals
        All values are numpy arrays.  ``terminals`` is float32 with 1.0 at
        episode-end states (the true end-of-data boundary) and 0.0 elsewhere.
    """
    from trading_env import SwingTradingEnv

    df, feature_cols = prepare_training_data(
        ohlcv_df,
        min_quality=min_quality,
        regime_ids=regime_ids,
    )
    feature_data = df[feature_cols].values.astype(np.float32)
    close_prices = df["close"].values.astype(np.float32)

    obs_list: list[np.ndarray] = []
    act_list: list[int] = []
    rew_list: list[float] = []
    next_obs_list: list[np.ndarray] = []
    term_list: list[float] = []

    for ep in range(n_episodes):
        env = SwingTradingEnv(
            data=feature_data,
            prices=close_prices,
            seq_len=seq_len,
            obs_mode=obs_mode,
            reward_type=reward_function,
        )
        obs, _ = env.reset()
        done = False

        while not done:
            if source_model is not None:
                action_arr, _ = source_model.predict(obs, deterministic=True)
                action = int(np.squeeze(action_arr))
            else:
                action = env.action_space.sample()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            obs_list.append(obs.copy())
            act_list.append(action)
            rew_list.append(float(reward))
            next_obs_list.append(next_obs.copy())
            # d3rlpy distinguishes terminal (true episode end) from timeout
            term_list.append(1.0 if terminated else 0.0)

            obs = next_obs

        logger.info(
            "Offline collection — episode %d/%d  steps_so_far=%d",
            ep + 1, n_episodes, len(obs_list),
        )

    transitions = {
        "observations": np.array(obs_list, dtype=np.float32),
        "actions": np.array(act_list, dtype=np.int32),
        "rewards": np.array(rew_list, dtype=np.float32),
        "next_observations": np.array(next_obs_list, dtype=np.float32),
        "terminals": np.array(term_list, dtype=np.float32),
    }
    logger.info(
        "Offline dataset collected: %d transitions, obs_dim=%d",
        len(transitions["observations"]),
        transitions["observations"].shape[1],
    )
    return transitions


# ─── MDPDataset builder ───────────────────────────────────────────────────────

def build_offline_dataset(transitions: dict[str, np.ndarray]):
    """Wrap transition arrays into a d3rlpy MDPDataset.

    Supports d3rlpy ≥ 2.0.  The dataset is used as the fixed replay buffer for
    ``train_cql()``.
    """
    _check_d3rlpy()
    from d3rlpy.dataset import MDPDataset

    dataset = MDPDataset(
        observations=transitions["observations"],
        actions=transitions["actions"].astype(np.int32),
        rewards=transitions["rewards"],
        terminals=transitions["terminals"],
    )
    logger.info(
        "MDPDataset created: %d episodes  %d total transitions",
        len(dataset.episodes),
        sum(len(e) for e in dataset.episodes),
    )
    return dataset


# ─── CQL training ─────────────────────────────────────────────────────────────

def train_cql(
    dataset,
    n_steps: int = 50_000,
    n_steps_per_epoch: int = 5_000,
    learning_rate: float = 1e-4,
    batch_size: int = 256,
    gamma: float = 0.99,
    alpha: float = 4.0,
    save_dir: str | None = None,
    model_name: str | None = None,
    device: str = "auto",
) -> tuple[Any, str]:
    """Train a DiscreteCQL agent on an offline dataset.

    Parameters
    ----------
    alpha : float
        CQL conservative regularisation strength.  Higher → the Q-values of
        out-of-distribution actions are penalised more aggressively, keeping
        the learnt policy close to the behaviour data.  Typical range for
        financial data: 2.0–8.0.  Default 4.0 is a reasonable starting point.

    Returns
    -------
    (cql_algo, save_path_str) — the trained algo object and path to the saved
    model directory.
    """
    _check_d3rlpy()
    import d3rlpy
    from d3rlpy.algos import DiscreteCQLConfig

    device_str = _resolve_device(device)
    logger.info(
        "Starting DiscreteCQL offline training — n_steps=%d  alpha=%.1f  device=%s",
        n_steps, alpha, device_str,
    )

    cql = DiscreteCQLConfig(
        learning_rate=learning_rate,
        batch_size=batch_size,
        gamma=gamma,
        alpha=alpha,
    ).create(device=device_str)

    # build_with_dataset infers obs/action dims without running .fit()
    cql.build_with_dataset(dataset)

    from d3rlpy.logging import NoopAdapterFactory
    results = cql.fit(
        dataset,
        n_steps=n_steps,
        n_steps_per_epoch=n_steps_per_epoch,
        show_progress=False,
        logger_adapter=NoopAdapterFactory(),
        evaluators={},
    )

    if results:
        last_epoch, last_metrics = results[-1]
        logger.info("DiscreteCQL training complete — epoch=%d  metrics=%s", last_epoch, last_metrics)
    else:
        logger.info("DiscreteCQL training complete.")

    # Save
    if save_dir is None:
        save_dir = str(Path(settings.MODEL_DIR) / "rl" / "cql")
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = model_name or f"DiscreteCQL_{ts}"
    save_path = str(Path(save_dir) / name)
    cql.save(save_path)
    logger.info("DiscreteCQL model saved to %s", save_path)

    # Also save metadata alongside the model
    meta = {
        "algorithm": "DiscreteCQL",
        "model_name": name,
        "n_steps": n_steps,
        "alpha": alpha,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "gamma": gamma,
        "device": device_str,
        "saved_at": ts,
    }
    meta_path = str(Path(save_dir) / f"{name}_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return cql, save_path


def load_cql(model_path: str):
    """Load a saved d3rlpy CQL model.

    Parameters
    ----------
    model_path : str
        Path passed to ``cql.save()`` during training (no extension needed).
    """
    _check_d3rlpy()
    import d3rlpy
    return d3rlpy.load_learnable(model_path)


# ─── offline evaluation ───────────────────────────────────────────────────────

def evaluate_offline(
    cql_algo,
    transitions: dict[str, np.ndarray],
) -> dict[str, float]:
    """Compute offline action-distribution and reward metrics.

    Returns
    -------
    dict with keys:
        avg_observed_reward  : Mean per-step reward in the collected dataset.
        predicted_buy_pct    : Fraction of steps CQL greedy-predicts BUY (1).
        predicted_sell_pct   : Fraction of steps CQL greedy-predicts SELL (2).
        predicted_hold_pct   : Fraction of steps CQL greedy-predicts HOLD (0).
        dataset_size         : Total transition count.
    """
    observations = transitions["observations"]
    rewards = transitions["rewards"]

    predicted_actions = cql_algo.predict(observations)

    return {
        "avg_observed_reward": round(float(rewards.mean()), 6),
        "predicted_buy_pct": round(float((predicted_actions == 1).mean()), 4),
        "predicted_sell_pct": round(float((predicted_actions == 2).mean()), 4),
        "predicted_hold_pct": round(float((predicted_actions == 0).mean()), 4),
        "dataset_size": int(len(observations)),
    }


# ─── CQL → SB3 behavioral-cloning warm-up ───────────────────────────────────

def bc_warmup_ppo(
    sb3_model,
    cql_algo,
    observations: np.ndarray,
    device: str = "auto",
    n_steps: int = 2_000,
    lr: float = 3e-4,
    batch_size: int = 256,
) -> dict[str, float]:
    """Warm-start an SB3 PPO policy with CQL's greedy actions via Behavioral Cloning.

    Runs a supervised learning pass on the PPO actor using CQL's predicted
    actions as ground-truth labels (cross-entropy / log-probability loss).
    No environment interaction takes place — this is pure offline optimisation.

    After this call the PPO policy reproduces the conservative CQL behaviour,
    giving the subsequent online PPO stage a much better starting point:
    the agent won't waste early timesteps discovering obviously-wrong actions.

    Parameters
    ----------
    sb3_model : SB3 PPO (including AttentionPPO) — **modified in-place**.
    cql_algo  : Trained d3rlpy DiscreteCQL model.
    observations : np.ndarray, shape (N, obs_dim) — collected transition obs.
    n_steps   : Number of BC gradient steps.
    lr        : Learning rate for the BC Adam optimiser.

    Returns
    -------
    dict with 'final_bc_loss' and 'n_steps_completed'.
    """
    import torch
    from torch.optim import Adam

    _device = _resolve_device(device)

    # CQL greedy labels for the whole buffer (computed once, stored on CPU)
    logger.info("BC warm-up: computing CQL greedy labels for %d observations…", len(observations))
    cql_actions = cql_algo.predict(observations.astype(np.float32))  # (N,) int

    policy = sb3_model.policy
    policy.set_training_mode(True)
    optimizer = Adam(policy.parameters(), lr=lr)

    n = len(observations)
    losses: list[float] = []

    for step in range(n_steps):
        idx = np.random.randint(0, n, size=batch_size)

        # obs_to_tensor handles any normalisation the policy applies
        obs_batch, _ = policy.obs_to_tensor(observations[idx])
        act_batch = torch.as_tensor(cql_actions[idx], dtype=torch.long, device=obs_batch.device)

        distribution = policy.get_distribution(obs_batch)
        log_probs = distribution.log_prob(act_batch)
        bc_loss = -log_probs.mean()

        optimizer.zero_grad()
        bc_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

        losses.append(bc_loss.item())

        if (step + 1) % 500 == 0:
            window = losses[-100:]
            logger.info(
                "BC warm-up step %d/%d — avg_loss=%.4f",
                step + 1, n_steps, float(np.mean(window)),
            )

    policy.set_training_mode(False)
    final_loss = float(np.mean(losses[-200:] if len(losses) >= 200 else losses))
    logger.info("BC warm-up complete — final_loss=%.4f  steps=%d", final_loss, n_steps)

    return {"final_bc_loss": round(final_loss, 6), "n_steps_completed": n_steps}


# ─── CQL env-wrapper (predict interface) ─────────────────────────────────────

class CQLAgentWrapper:
    """Thin wrapper giving a d3rlpy CQL algo the same `.predict()` interface
    as SB3 models.  Used by ``compare_agents()`` and ``evaluate_rl_model()``.

    Example::

        wrapper = CQLAgentWrapper(cql_algo)
        action, _ = wrapper.predict(obs, deterministic=True)
    """

    def __init__(self, cql_algo) -> None:
        self._algo = cql_algo

    def predict(
        self,
        obs: np.ndarray,
        deterministic: bool = True,
        state=None,
        episode_start=None,
    ) -> tuple[np.ndarray, None]:
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        action = self._algo.predict(obs.astype(np.float32))
        return action, None


# ─── backtest comparison ─────────────────────────────────────────────────────

def compare_agents(
    eval_df: pd.DataFrame,
    agents: dict[str, Any],
    seq_len: int = 15,
    reward_function: str = "dense",
    n_eval_episodes: int = 3,
    min_quality: float = 0.0,
) -> dict[str, dict]:
    """Run all agents on the same eval dataset and return side-by-side metrics.

    Parameters
    ----------
    agents : dict mapping a display name to a model.  Each model must expose a
        ``.predict(obs, deterministic=True)`` method — both SB3 models and
        ``CQLAgentWrapper`` instances qualify.

    Returns
    -------
    dict mapping agent name → {mean_reward, std_reward, mean_trades, episodes}.

    Example::

        from app.ml.cql_trainer import CQLAgentWrapper, compare_agents
        results = compare_agents(
            eval_df,
            {
                "DiscreteCQL": CQLAgentWrapper(cql_algo),
                "AttentionPPO": ppo_model,
            },
        )
    """
    from trading_env import SwingTradingEnv
    from app.ml.rl_trainer import prepare_training_data

    df, feature_cols = prepare_training_data(eval_df, min_quality=min_quality)
    feature_data = df[feature_cols].values.astype(np.float32)
    close_prices = df["close"].values.astype(np.float32)

    results: dict[str, dict] = {}

    for name, model in agents.items():
        ep_rewards: list[float] = []
        ep_trades: list[int] = []

        for _ in range(n_eval_episodes):
            env = SwingTradingEnv(
                data=feature_data,
                prices=close_prices,
                seq_len=seq_len,
                obs_mode="flat",
                reward_type=reward_function,
            )
            obs, _ = env.reset()
            done = False
            ep_reward = 0.0

            while not done:
                action_arr, _ = model.predict(obs, deterministic=True)
                action = int(np.squeeze(action_arr))
                obs, reward, terminated, truncated, _ = env.step(action)
                ep_reward += reward
                done = terminated or truncated

            ep_rewards.append(ep_reward)
            n_trades = len(env.trade_logger.trades) if hasattr(env, "trade_logger") else 0
            ep_trades.append(n_trades)

        results[name] = {
            "mean_reward": round(float(np.mean(ep_rewards)), 4),
            "std_reward": round(float(np.std(ep_rewards)), 4),
            "mean_trades": round(float(np.mean(ep_trades)), 1),
            "episodes": n_eval_episodes,
        }
        logger.info(
            "Agent %-20s — mean_reward=%+.4f  std=%.4f  trades=%.1f",
            name,
            results[name]["mean_reward"],
            results[name]["std_reward"],
            results[name]["mean_trades"],
        )

    return results
