"""Run comparable bounded PPO experiments over multiple training seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lemonade.training import train_multiple_seeds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303])
    parser.add_argument("--timesteps", type=int, default=10_000)
    parser.add_argument("--evaluation-episodes", type=int, default=200)
    args = parser.parse_args()
    report = train_multiple_seeds(
        args.output,
        args.seeds,
        args.timesteps,
        list(range(20_000, 20_000 + args.evaluation_episodes)),
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
