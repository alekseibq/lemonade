from lemonade.core import GameConfig
from lemonade.evaluation import evaluate
from lemonade.policies import FixedPolicy, RandomPolicy, RuleBasedPolicy


def test_policies_complete_comparable_evaluations() -> None:
    config = GameConfig()
    seeds = list(range(20))
    rule = evaluate(RuleBasedPolicy(config), seeds, config)
    fixed = evaluate(FixedPolicy(), seeds, config)
    random = evaluate(RandomPolicy(config, seed=9), seeds, config)
    assert rule.episodes == fixed.episodes == random.episodes == len(seeds)
    assert rule.mean_profit > random.mean_profit
