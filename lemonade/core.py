"""Pure, deterministic Lemonade Stand simulation; it has no Gym or HTTP dependency."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntEnum
from random import Random
from typing import Any


class Weather(IntEnum):
    DOWNPOUR = 0
    RAIN = 1
    NORMAL = 2
    WARM = 3
    HOT = 4


WEATHER_MULTIPLIERS = {
    Weather.DOWNPOUR: 0.05,
    Weather.RAIN: 0.5,
    Weather.NORMAL: 1.0,
    Weather.WARM: 1.2,
    Weather.HOT: 1.5,
}


@dataclass(frozen=True, slots=True)
class GameConfig:
    """Versioned parameters for ruleset lemonade-stand/v1."""

    rules_version: str = "lemonade-stand/v1"
    episode_days: int = 10
    starting_capital: int = 2_000
    unit_cost: int = 10
    max_price: int = 99
    max_production: int = 200
    baseline_demand: int = 100
    demand_noise_fraction: float = 0.05

    def __post_init__(self) -> None:
        if self.episode_days < 1 or self.starting_capital < 0 or self.unit_cost <= 0:
            raise ValueError("episode_days >= 1, starting_capital >= 0, and unit_cost > 0 required")
        if self.max_price < 0 or self.max_production < 0:
            raise ValueError("max_price and max_production must be non-negative")
        if not 0 <= self.demand_noise_fraction <= 1:
            raise ValueError("demand_noise_fraction must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class DayScenario:
    weather: Weather
    demand_noise: float


@dataclass(frozen=True, slots=True)
class DayResult:
    day: int
    weather: Weather
    price: int
    requested_production: int
    produced: int
    unaffordable_units: int
    demand: int
    sold: int
    unsold: int
    revenue: int
    production_cost: int
    profit: int
    capital_after: int
    terminated: bool


def price_multiplier(price: int) -> float:
    """Demand response: 1.2 at <=15, 1.0 at 20, and 0 at >=40."""
    if price <= 15:
        return 1.2
    if price < 20:
        return 1.2 - (price - 15) * 0.04
    if price < 40:
        return 1.0 - (price - 20) * 0.05
    return 0.0


class LemonadeGame:
    """A full-information simulator internally, exposing only current-day forecast publicly."""

    def __init__(self, config: GameConfig | None = None) -> None:
        self.config = config or GameConfig()
        self._scenarios: tuple[DayScenario, ...] = ()
        self.day = 0
        self.capital = self.config.starting_capital
        self.terminated = False
        self.history: list[DayResult] = []

    def reset(
        self, seed: int | None = None, scenarios: tuple[DayScenario, ...] | None = None
    ) -> None:
        """Reset with exogenous scenarios generated upfront, independent of policy RNG usage."""
        if scenarios is not None:
            if len(scenarios) != self.config.episode_days:
                raise ValueError("scenario count must equal episode_days")
            if any(abs(s.demand_noise) > self.config.demand_noise_fraction for s in scenarios):
                raise ValueError("scenario noise outside configured range")
            self._scenarios = scenarios
        else:
            rng = Random(seed)
            self._scenarios = tuple(
                DayScenario(
                    weather=Weather(rng.randrange(len(Weather))),
                    demand_noise=rng.uniform(
                        -self.config.demand_noise_fraction, self.config.demand_noise_fraction
                    ),
                )
                for _ in range(self.config.episode_days)
            )
        self.day = 0
        self.capital = self.config.starting_capital
        self.terminated = False
        self.history = []

    @property
    def current_weather(self) -> Weather:
        if self.terminated or not self._scenarios:
            raise RuntimeError("no current-day forecast after episode termination or before reset")
        return self._scenarios[self.day].weather

    def step(self, price: int, requested_production: int) -> DayResult:
        if not self._scenarios:
            raise RuntimeError("reset must be called before step")
        if self.terminated:
            raise RuntimeError("episode already terminated; call reset")
        if not 0 <= price <= self.config.max_price:
            raise ValueError(f"price must be in [0, {self.config.max_price}]")
        if not 0 <= requested_production <= self.config.max_production:
            raise ValueError(f"production must be in [0, {self.config.max_production}]")

        scenario = self._scenarios[self.day]
        affordable_units = self.capital // self.config.unit_cost
        produced = min(requested_production, affordable_units)
        production_cost = produced * self.config.unit_cost
        expected = (
            self.config.baseline_demand
            * WEATHER_MULTIPLIERS[scenario.weather]
            * price_multiplier(price)
            * (1 + scenario.demand_noise)
        )
        demand = max(0, round(expected))
        sold = min(produced, demand)
        revenue = sold * price
        profit = revenue - production_cost
        self.capital += profit
        self.day += 1
        self.terminated = self.day >= self.config.episode_days
        result = DayResult(
            day=self.day,
            weather=scenario.weather,
            price=price,
            requested_production=requested_production,
            produced=produced,
            unaffordable_units=requested_production - produced,
            demand=demand,
            sold=sold,
            unsold=produced - sold,
            revenue=revenue,
            production_cost=production_cost,
            profit=profit,
            capital_after=self.capital,
            terminated=self.terminated,
        )
        self.history.append(result)
        return result

    def public_state(self) -> dict[str, Any]:
        """Only information legitimately available before the current decision."""
        if self.terminated:
            return {"day": self.day, "capital": self.capital, "terminated": True}
        return {
            "day": self.day,
            "days_remaining": self.config.episode_days - self.day,
            "capital": self.capital,
            "weather_forecast": self.current_weather.name.lower(),
            "terminated": False,
        }

    def reproducibility_metadata(self, seed: int | None) -> dict[str, Any]:
        return {"seed": seed, "config": asdict(self.config)}
