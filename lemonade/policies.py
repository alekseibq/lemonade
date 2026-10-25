"""Simple policies used as baselines, deliberately independent from Gym."""

from __future__ import annotations

from random import Random

from lemonade.core import GameConfig, Weather


class RandomPolicy:
    def __init__(self, config: GameConfig, seed: int = 0) -> None:
        self.config, self.rng = config, Random(seed)

    def act(self, observation: dict[str, object]) -> tuple[int, int]:
        return (
            self.rng.randint(0, self.config.max_price),
            self.rng.randint(0, self.config.max_production),
        )


class FixedPolicy:
    def __init__(self, price: int = 20, production: int = 100) -> None:
        self.price, self.production = price, production

    def act(self, observation: dict[str, object]) -> tuple[int, int]:
        return self.price, self.production


class RuleBasedPolicy:
    """Known-demand heuristic: price 20 and produce weather-adjusted baseline demand."""

    _production = {
        Weather.DOWNPOUR: 5,
        Weather.RAIN: 50,
        Weather.NORMAL: 100,
        Weather.WARM: 120,
        Weather.HOT: 150,
    }

    def __init__(self, config: GameConfig) -> None:
        self.config = config

    def act(self, observation: dict[str, object]) -> tuple[int, int]:
        weather = Weather[str(observation["weather_forecast"]).upper()]
        affordable = int(observation["capital"]) // self.config.unit_cost
        return 20, min(self._production[weather], affordable, self.config.max_production)
