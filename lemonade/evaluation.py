"""Comparable policy evaluation using reset seeds as exogenous scenario identifiers."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median, pstdev
from typing import Protocol

from lemonade.core import GameConfig, LemonadeGame


class Policy(Protocol):
    def act(self, observation: dict[str, object]) -> tuple[int, int]: ...


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    episodes: int
    mean_profit: float
    median_profit: float
    profit_stddev: float
    profitable_share: float


def evaluate(
    policy: Policy, seeds: list[int], config: GameConfig | None = None
) -> EvaluationSummary:
    game = LemonadeGame(config)
    profits: list[int] = []
    for seed in seeds:
        game.reset(seed=seed)
        while not game.terminated:
            game.step(*policy.act(game.public_state()))
        profits.append(game.capital - game.config.starting_capital)
    return EvaluationSummary(
        episodes=len(profits),
        mean_profit=mean(profits),
        median_profit=median(profits),
        profit_stddev=pstdev(profits),
        profitable_share=sum(p > 0 for p in profits) / len(profits),
    )
