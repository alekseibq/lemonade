"""Train PPO with an explicit budget and write a reproducible artifact directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lemonade.training import TrainingConfig, train_and_evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--timesteps", type=int, default=10_000)
    parser.add_argument("--evaluation-episodes", type=int, default=200)
    args = parser.parse_args()
    report = train_and_evaluate(
        args.output,
        training=TrainingConfig(seed=args.seed, total_timesteps=args.timesteps),
        evaluation_seeds=list(range(10_000, 10_000 + args.evaluation_episodes)),
    )
    print(json.dumps(report["evaluation"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
