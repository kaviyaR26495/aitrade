"""SB3 training callbacks — progress tracking, checkpointing."""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)


class ProgressCallback(BaseCallback):
    """Logs training progress to logger and accumulates metrics for DB storage.

    Emits two kinds of updates:
      - _on_step: stop/pause checks + lightweight heartbeat every log_interval steps
        (timestep, fps, progress, current net_worth)
      - _on_rollout_end: rich metrics after each rollout
        (ep_rew_mean, ep_len_mean, loss, plus net_worth / profit_pct from the env)

    Supports cooperative stop and pause via threading.Event objects:
      - stop_event: set → callback returns False, ending training immediately
      - pause_event: set → callback blocks until cleared (resume)
    """

    def __init__(
        self,
        log_interval: int = 1000,
        total_timesteps: int = 0,
        verbose: int = 0,
        on_progress=None,
        stop_event: threading.Event | None = None,
        pause_event: threading.Event | None = None,
    ):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.total_timesteps = total_timesteps
        self.metrics_log: list[dict[str, Any]] = []
        self._start_time = 0.0
        self.on_progress = on_progress
        self.stop_event = stop_event
        self.pause_event = pause_event
        self._last_rollout_timestep: int = -1

    def _on_training_start(self):
        self._start_time = time.time()

    # ── helpers ──────────────────────────────────────────────────────

    def _base_info(self) -> dict[str, Any]:
        elapsed = time.time() - self._start_time
        fps = self.num_timesteps / elapsed if elapsed > 0 else 0
        return {
            "timestep": self.num_timesteps,
            "fps": round(fps, 1),
            "progress": round(self.num_timesteps / self.total_timesteps, 4) if self.total_timesteps > 0 else 0,
        }

    def _get_net_worth(self) -> dict | None:
        """Safely extract portfolio net_worth from the wrapped env."""
        try:
            env = self.model.env.envs[0]
            prices = env.prices
            step = min(env.current_step, len(prices) - 1)
            nw = float(env.portfolio.net_worth(prices[step]))
            return {
                "net_worth": round(nw, 2),
                "profit_pct": round((nw - env.initial_cash) / env.initial_cash * 100, 3),
            }
        except Exception:
            return None

    def _get_sb3_metrics(self) -> dict:
        """Extract rollout/train metrics from SB3 logger."""
        metrics: dict[str, Any] = {}
        try:
            if self.model.logger is not None:
                nv = self.model.logger.name_to_value
                if "rollout/ep_rew_mean" in nv:
                    metrics["ep_rew_mean"] = round(float(nv["rollout/ep_rew_mean"]), 4)
                if "rollout/ep_len_mean" in nv:
                    metrics["ep_len_mean"] = round(float(nv["rollout/ep_len_mean"]), 1)
                if "train/loss" in nv:
                    metrics["loss"] = round(float(nv["train/loss"]), 6)
                if "train/value_loss" in nv:
                    metrics["value_loss"] = round(float(nv["train/value_loss"]), 6)
        except Exception:
            pass
        return metrics

    def _emit(self, info: dict) -> None:
        self.metrics_log.append(info)
        if self.on_progress is not None:
            try:
                self.on_progress(info)
            except Exception:
                pass
        logger.debug(
            "step=%d (%.1f%%) fps=%.0f rew=%s loss=%s profit=%s%%",
            info.get("timestep", 0),
            info.get("progress", 0) * 100,
            info.get("fps", 0),
            info.get("ep_rew_mean", "—"),
            info.get("loss", "—"),
            info.get("profit_pct", "—"),
        )

    # ── SB3 hooks ──────────────────────────────────────────────────────

    def _on_step(self) -> bool:
        # ── Stop check ────────────────────────────────────────────────
        if self.stop_event is not None and self.stop_event.is_set():
            logger.info("Training stopped by user at step %d", self.num_timesteps)
            if self.on_progress is not None:
                try:
                    self.on_progress({"info": "Training stopped by user.", "stopped": True})
                except Exception:
                    pass
            return False  # signals SB3 to abort

        # ── Pause check ───────────────────────────────────────────────
        if self.pause_event is not None and self.pause_event.is_set():
            logger.info("Training paused at step %d", self.num_timesteps)
            if self.on_progress is not None:
                try:
                    self.on_progress({"info": "Training paused.", "paused": True})
                except Exception:
                    pass
            # Block until pause_event is cleared (resumed or stopped)
            while self.pause_event.is_set():
                if self.stop_event is not None and self.stop_event.is_set():
                    return False
                time.sleep(0.2)
            if self.on_progress is not None:
                try:
                    self.on_progress({"info": "Training resumed."})
                except Exception:
                    pass

        # ── Heartbeat at log_interval (skip if rollout_end just fired) ─
        if (
            self.num_timesteps % self.log_interval == 0
            and self.num_timesteps != self._last_rollout_timestep
        ):
            info = self._base_info()
            nw = self._get_net_worth()
            if nw:
                info.update(nw)
            self._emit(info)

        return True

    def _on_rollout_end(self) -> None:
        """Emit rich metrics after each rollout collection — this is where SB3 computes ep_rew_mean."""
        self._last_rollout_timestep = self.num_timesteps
        info = self._base_info()
        info.update(self._get_sb3_metrics())
        nw = self._get_net_worth()
        if nw:
            info.update(nw)
        self._emit(info)

    def _on_training_end(self) -> None:
        """Emit a final completion event when training finishes naturally (all timesteps done)."""
        # Only emit if training was NOT stopped by the user
        if self.stop_event is not None and self.stop_event.is_set():
            return
        if self.on_progress is not None:
            try:
                self.on_progress({"info": "Training completed successfully.", "completed": True})
            except Exception:
                pass


class BestModelCallback(BaseCallback):
    """Save model when evaluation reward improves."""

    def __init__(
        self,
        save_path: str | Path,
        check_interval: int = 10000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.save_path = Path(save_path)
        self.check_interval = check_interval
        self.best_reward = float("-inf")

    def _on_step(self) -> bool:
        if self.num_timesteps % self.check_interval == 0:
            # Check recent episode rewards
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = sum(
                    ep["r"] for ep in self.model.ep_info_buffer
                ) / len(self.model.ep_info_buffer)

                if mean_reward > self.best_reward:
                    self.best_reward = mean_reward
                    self.save_path.parent.mkdir(parents=True, exist_ok=True)
                    self.model.save(str(self.save_path / "best_model"))
                    logger.info(
                        "New best model at step %d: reward=%.4f",
                        self.num_timesteps,
                        mean_reward,
                    )

        return True
