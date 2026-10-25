"""Measure pure simulation throughput; no RL library is involved."""

from __future__ import annotations

import argparse
from time import perf_counter

from lemonade.core import LemonadeGame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10_000)
    args = parser.parse_args()
    game = LemonadeGame()
    start = perf_counter()
    for seed in range(args.episodes):
        game.reset(seed=seed)
        while not game.terminated:
            game.step(20, 100)
    elapsed = perf_counter() - start
    print(
        f"episodes={args.episodes} seconds={elapsed:.4f} "
        f"episodes_per_second={args.episodes / elapsed:.0f}"
    )


if __name__ == "__main__":
    main()
