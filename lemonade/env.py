"""Gymnasium adapter for the pure simulation core."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from lemonade.core import GameConfig, LemonadeGame


class LemonadeEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(self, config: GameConfig | None = None, render_mode: str | None = None) -> None:
        self.game = LemonadeGame(config)
        self.config = self.game.config
        self.render_mode = render_mode
        self.action_space = gym.spaces.MultiDiscrete(
            [self.config.max_price + 1, self.config.max_production + 1]
        )
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, np.finfo(np.float32).max, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def _observation(self) -> np.ndarray:
        if self.game.terminated:
            return np.array(
                [1.0, self.game.capital / self.config.starting_capital, 0.0], dtype=np.float32
            )
        return np.array(
            [
                self.game.day / self.config.episode_days,
                self.game.capital / self.config.starting_capital,
                int(self.game.current_weather) / 4,
            ],
            dtype=np.float32,
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        scenarios = None if options is None else options.get("scenarios")
        self.game.reset(seed=seed, scenarios=scenarios)
        return self._observation(), self.game.public_state()

    def step(
        self, action: np.ndarray | list[int] | tuple[int, int]
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action_array = np.asarray(action)
        if action_array.shape != (2,):
            raise ValueError("action must contain [price, requested_production]")
        result = self.game.step(int(action_array[0]), int(action_array[1]))
        info = self.game.public_state() | {"day_result": result}
        return self._observation(), float(result.profit), result.terminated, False, info

    def render(self) -> str | None:
        if self.render_mode != "ansi":
            return None
        return str(self.game.public_state())
