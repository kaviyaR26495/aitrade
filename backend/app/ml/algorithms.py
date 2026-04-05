"""RL algorithm registry — maps algorithm names to SB3 classes and default hyperparams.

Supports: PPO, RecurrentPPO, AttentionPPO, QRDQN, ContinuousPPO.
A2C, DDPG, TD3, and SAC have been removed — they do not fit the sequential
observation / offline-hybrid architecture used by this project.
"""
from __future__ import annotations

from typing import Any


# Algorithm configurations
ALGORITHM_CONFIGS: dict[str, dict[str, Any]] = {
    "PPO": {
        "class_path": "stable_baselines3.PPO",
        "policy": "MlpPolicy",
        "obs_mode": "flat",
        "defaults": {
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "n_epochs": 10,
            # Larger batch → more GPU work per update; keep divisible into n_steps * n_envs
            "batch_size": 512,
            "policy_kwargs": {
                "net_arch": [256, 256],   # wider network → more GPU compute per step
            },
        },
    },
    "RecurrentPPO": {
        "class_path": "sb3_contrib.RecurrentPPO",
        "policy": "MlpLstmPolicy",
        "obs_mode": "sequential",
        "defaults": {
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "n_epochs": 10,
            "batch_size": 512,
            "policy_kwargs": {
                "lstm_hidden_size": 256,
                "n_lstm_layers": 1,
                "enable_critic_lstm": True,
                "net_arch": [256, 256],
            },
        },
    },
    # ── Attention PPO ─────────────────────────────────────────────────
    # PPO with a Transformer self-attention feature extractor + separate
    # actor/critic network sizes.  The AttentionFeaturesExtractor is injected
    # by rl_trainer.py at runtime (requires the actual class, not a string).
    "AttentionPPO": {
        "class_path": "stable_baselines3.PPO",
        "policy": "MlpPolicy",
        "obs_mode": "flat",
        "defaults": {
            "learning_rate": 2e-4,
            "n_steps": 2048,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "n_epochs": 10,
            "batch_size": 256,
            "policy_kwargs": {
                # Separate actor/critic: Critic gets a larger network so the
                # value-function can converge independently of the policy.
                "net_arch": {"pi": [256, 128], "vf": [512, 512, 256]},
                # features_extractor_class and features_extractor_kwargs are
                # injected by rl_trainer.py (need the live Python class).
            },
        },
    },
    # ── QR-DQN (Distributional RL) ────────────────────────────────────
    # Predicts the *full return distribution* via quantile regression instead
    # of a scalar expected return.  Fat-tail events (crashes/rallies) that
    # standard PPO averages away are explicitly represented in each quantile,
    # so the agent can refuse a trade when the lower quantiles look dangerous.
    "QRDQN": {
        "class_path": "sb3_contrib.QRDQN",
        "policy": "MlpPolicy",
        "obs_mode": "flat",
        "defaults": {
            "learning_rate": 5e-4,
            "buffer_size": 100_000,
            "learning_starts": 1_000,
            "batch_size": 256,
            "policy_kwargs": {
                "n_quantiles": 200,
                "net_arch": [256, 256],
            },
        },
    },
    # ── Continuous PPO ────────────────────────────────────────────────
    # Action space Box(-1, 1): sign = direction, magnitude = conviction.
    # > 0.33 → BUY, < -0.33 → SELL, else → HOLD.  The continuous magnitude
    # feeds directly into the confidence score during pattern extraction.
    "ContinuousPPO": {
        "class_path": "stable_baselines3.PPO",
        "policy": "MlpPolicy",
        "obs_mode": "flat",
        "continuous": True,
        "defaults": {
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "clip_range": 0.2,
            "ent_coef": 0.005,
            "n_epochs": 10,
            "batch_size": 512,
            "policy_kwargs": {
                "net_arch": [256, 256],
            },
        },
    },
}


def get_algorithm_class(name: str):
    """Dynamically import and return the SB3 algorithm class."""
    config = ALGORITHM_CONFIGS.get(name)
    if not config:
        raise ValueError(f"Unknown algorithm: {name}. Available: {list(ALGORITHM_CONFIGS.keys())}")

    class_path = config["class_path"]
    module_path, class_name = class_path.rsplit(".", 1)

    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def get_default_hyperparams(name: str) -> dict:
    """Get default hyperparameters for an algorithm."""
    config = ALGORITHM_CONFIGS.get(name)
    if not config:
        raise ValueError(f"Unknown algorithm: {name}")
    return config["defaults"].copy()


def get_obs_mode(name: str) -> str:
    """Get observation mode for an algorithm."""
    config = ALGORITHM_CONFIGS.get(name)
    if not config:
        raise ValueError(f"Unknown algorithm: {name}")
    return config["obs_mode"]


def is_continuous(name: str) -> bool:
    """Check if algorithm uses continuous action space."""
    config = ALGORITHM_CONFIGS.get(name)
    if not config:
        return False
    return config.get("continuous", False)


def list_algorithms() -> list[dict]:
    """Return list of available algorithms with their info."""
    result = []
    for name, config in ALGORITHM_CONFIGS.items():
        result.append({
            "name": name,
            "policy": config["policy"],
            "obs_mode": config["obs_mode"],
            "continuous": config.get("continuous", False),
            "defaults": config["defaults"],
        })
    return result
