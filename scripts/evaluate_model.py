"""Evaluate a saved PPO artifact on an explicit, reproducible seed range."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from lemonade.training import load_and_evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--seed-start", type=int, default=10_000)
    parser.add_argument("--episodes", type=int, default=200)
    args = parser.parse_args()
    summary = load_and_evaluate(
        args.model, list(range(args.seed_start, args.seed_start + args.episodes))
    )
    print(json.dumps(asdict(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
