"""Bounded PPO training and held-out evaluation for the Lemonade environment."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

from stable_baselines3 import PPO

from lemonade.core import GameConfig
from lemonade.env import LemonadeEnv
from lemonade.evaluation import EvaluationSummary, evaluate
from lemonade.policies import RandomPolicy, RuleBasedPolicy


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    seed: int = 123
    total_timesteps: int = 10_000
    n_steps: int = 128
    batch_size: int = 64
    learning_rate: float = 3e-4


def evaluate_model(model: PPO, seeds: list[int], config: GameConfig) -> EvaluationSummary:
    profits: list[float] = []
    env = LemonadeEnv(config)
    for seed in seeds:
        observation, _ = env.reset(seed=seed)
        terminated = False
        total_profit = 0.0
        while not terminated:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, _ = env.step(action)
            assert not truncated
            total_profit += reward
        profits.append(total_profit)
    return EvaluationSummary(
        episodes=len(profits),
        mean_profit=mean(profits),
        median_profit=median(profits),
        profit_stddev=pstdev(profits),
        profitable_share=sum(profit > 0 for profit in profits) / len(profits),
    )


def load_and_evaluate(
    model_path: Path, evaluation_seeds: list[int], game: GameConfig | None = None
) -> EvaluationSummary:
    """Load a saved PPO artifact and evaluate it against an explicit scenario set."""
    model = PPO.load(model_path, device="cpu")
    return evaluate_model(model, evaluation_seeds, game or GameConfig())


def train_and_evaluate(
    output_dir: Path,
    training: TrainingConfig | None = None,
    game: GameConfig | None = None,
    evaluation_seeds: list[int] | None = None,
) -> dict[str, Any]:
    """Train on reset-generated scenarios and persist all reproduction metadata."""
    training = training or TrainingConfig()
    game = game or GameConfig()
    if training.total_timesteps < training.n_steps:
        raise ValueError("total_timesteps must be at least n_steps")
    if training.n_steps % training.batch_size:
        raise ValueError("n_steps must be divisible by batch_size")
    seeds = evaluation_seeds or list(range(10_000, 10_200))
    output_dir.mkdir(parents=True, exist_ok=True)
    env = LemonadeEnv(game)
    model = PPO(
        "MlpPolicy",
        env,
        device="cpu",
        seed=training.seed,
        n_steps=training.n_steps,
        batch_size=training.batch_size,
        learning_rate=training.learning_rate,
        policy_kwargs={"net_arch": [64, 64]},
        verbose=0,
    )
    model.learn(total_timesteps=training.total_timesteps, progress_bar=False)
    model_path = output_dir / "model"
    model.save(model_path)
    trained = evaluate_model(model, seeds, game)
    random = evaluate(RandomPolicy(game, seed=training.seed), seeds, game)
    rule = evaluate(RuleBasedPolicy(game), seeds, game)
    report: dict[str, Any] = {
        "experiment_id": output_dir.name,
        "rules_version": game.rules_version,
        "algorithm": "PPO",
        "training": asdict(training),
        "game": asdict(game),
        "evaluation_seeds": seeds,
        "packages": {
            "gymnasium": version("gymnasium"),
            "stable-baselines3": version("stable-baselines3"),
        },
        "model": str(model_path.with_suffix(".zip")),
        "evaluation": {
            "trained": asdict(trained),
            "random": asdict(random),
            "rule_based": asdict(rule),
            "improvement_over_random_mean_profit": trained.mean_profit - random.mean_profit,
        },
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
