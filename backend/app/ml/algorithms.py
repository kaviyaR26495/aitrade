"""RL algorithm registry — maps algorithm names to SB3 classes and default hyperparams.

Supports: PPO, RecurrentPPO, A2C, DDPG, TD3, SAC.
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
    "A2C": {
        "class_path": "stable_baselines3.A2C",
        "policy": "MlpPolicy",
        "obs_mode": "flat",
        "defaults": {
            "learning_rate": 7e-4,
            "n_steps": 1024,
            "ent_coef": 0.01,
            "policy_kwargs": {
                "net_arch": [256, 256],
            },
        },
    },
    "DDPG": {
        "class_path": "stable_baselines3.DDPG",
        "policy": "MlpPolicy",
        "obs_mode": "flat",
        "continuous": True,
        "defaults": {
            "learning_rate": 1e-3,
            "buffer_size": 1_000_000,
            "tau": 0.005,
            "batch_size": 1024,
            "policy_kwargs": {
                "net_arch": [256, 256],
            },
        },
    },
    "TD3": {
        "class_path": "stable_baselines3.TD3",
        "policy": "MlpPolicy",
        "obs_mode": "flat",
        "continuous": True,
        "defaults": {
            "learning_rate": 1e-3,
            "buffer_size": 1_000_000,
            "tau": 0.005,
            "batch_size": 1024,
            "policy_delay": 2,
            "target_policy_noise": 0.2,
            "policy_kwargs": {
                "net_arch": [256, 256],
            },
        },
    },
    "SAC": {
        "class_path": "stable_baselines3.SAC",
        "policy": "MlpPolicy",
        "obs_mode": "flat",
        "continuous": True,
        "defaults": {
            "learning_rate": 3e-4,
            "buffer_size": 1_000_000,
            "ent_coef": "auto",
            "batch_size": 1024,
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
