"""Gymnasium adapter over Episode, distinct from the legacy teaching environment."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .episode import Episode, _positive


class ControlledPWREnv(gym.Env):
    """Three rod-velocity actions: insert / hold / withdraw, in 1/s.

    Reward is negative absolute thermal-power tracking error / target power.
    SCRAM terminates; decision budget truncates. No auto-reset or error masking.
    Checkpoints belong to Episode; this wrapper must be reset after restoring one.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        duration_s=0.1,
        max_steps=100,
        tier="training",
        target_power_w=3e9,
        rod_speed_per_s=0.001,
    ):
        self.target_power_w = _positive(target_power_w, 1e12)
        self.rod_speed_per_s = _positive(rod_speed_per_s, 1)
        # Validate configuration without initializing the core twice on reset.
        self.config = dict(duration_s=duration_s, max_steps=max_steps, tier=tier)
        self.episode = Episode(**self.config)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(len(self.episode.observe()["state"]),), dtype=np.float64
        )
        self._ready = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options:
            raise ValueError("No reset options supported")
        self.episode = Episode(**self.config, seed=0 if seed is None else seed)
        self._ready = True
        obs = self.episode.observe()
        return np.asarray(obs["state"], dtype=np.float64), {"observation": obs}

    def step(self, action):
        if not self._ready:
            raise ValueError("Call reset before step")
        if not self.action_space.contains(action):
            raise ValueError("Action must be 0, 1 or 2")
        obs = self.episode.observe()
        result = self.episode.step(
            {"rod_command": (int(action) - 1) * self.rod_speed_per_s},
            observation_id=obs["observation_id"],
        )
        obs = result["observation"]
        reward = -abs(obs["thermal_power_w"] - self.target_power_w) / self.target_power_w
        return (
            np.asarray(obs["state"], dtype=np.float64),
            float(reward),
            obs["terminated"],
            obs["truncated"],
            result,
        )
