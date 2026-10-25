"""Small headless runner for smoke-testing a complete episode."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from lemonade.core import GameConfig, LemonadeGame
from lemonade.policies import FixedPolicy, RandomPolicy, RuleBasedPolicy


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Lemonade Stand episode")
    parser.add_argument("--policy", choices=("random", "fixed", "rule"), default="rule")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config = GameConfig()
    policies = {
        "random": RandomPolicy(config, seed=args.seed),
        "fixed": FixedPolicy(),
        "rule": RuleBasedPolicy(config),
    }
    game = LemonadeGame(config)
    game.reset(seed=args.seed)
    while not game.terminated:
        result = game.step(*policies[args.policy].act(game.public_state()))
        payload = asdict(result)
        payload["weather"] = result.weather.name.lower()
        print(json.dumps(payload))
    print(
        json.dumps(
            {"total_profit": game.capital - config.starting_capital, "capital": game.capital}
        )
    )


if __name__ == "__main__":
    main()
