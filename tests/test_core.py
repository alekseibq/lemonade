from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from lemonade.core import DayScenario, GameConfig, LemonadeGame, Weather, price_multiplier


def test_same_seed_and_actions_have_identical_episode() -> None:
    actions = [(20, 100), (15, 110), (40, 200)] * 4
    first, second = LemonadeGame(), LemonadeGame()
    first.reset(seed=123)
    second.reset(seed=123)
    left = [first.step(*action) for action in actions[:10]]
    right = [second.step(*action) for action in actions[:10]]
    assert left == right


def test_exogenous_scenarios_are_independent_of_actions() -> None:
    first, second = LemonadeGame(), LemonadeGame()
    first.reset(seed=8)
    second.reset(seed=8)
    first.step(0, 200)
    second.step(20, 100)
    assert first.current_weather == second.current_weather
    assert first.step(20, 100).demand == second.step(20, 100).demand


def test_insufficient_capital_reduces_production_consistently() -> None:
    game = LemonadeGame(GameConfig(starting_capital=25))
    game.reset(scenarios=(DayScenario(Weather.NORMAL, 0),) * 10)
    result = game.step(20, 10)
    assert (result.produced, result.unaffordable_units, result.production_cost) == (2, 8, 20)
    assert result.capital_after == 45


def test_no_sales_above_stock_and_reward_matches_capital() -> None:
    game = LemonadeGame()
    game.reset(scenarios=(DayScenario(Weather.HOT, 0),) * 10)
    before = game.capital
    result = game.step(20, 3)
    assert result.sold == 3
    assert result.unsold == 0
    assert result.profit == game.capital - before


def test_price_curve_endpoints_and_clamping() -> None:
    assert price_multiplier(0) == price_multiplier(15) == 1.2
    assert price_multiplier(20) == 1.0
    assert price_multiplier(40) == price_multiplier(99) == 0.0


@given(
    price=st.integers(min_value=0, max_value=99),
    production=st.integers(min_value=0, max_value=200),
    capital=st.integers(min_value=0, max_value=5000),
    weather=st.sampled_from(list(Weather)),
    noise=st.floats(min_value=-0.05, max_value=0.05, allow_nan=False),
)
def test_economic_invariants(
    price: int, production: int, capital: int, weather: Weather, noise: float
) -> None:
    config = GameConfig(starting_capital=capital)
    game = LemonadeGame(config)
    game.reset(scenarios=(DayScenario(weather, noise),) * config.episode_days)
    before = game.capital
    result = game.step(price, production)
    assert 0 <= result.produced <= result.requested_production
    assert 0 <= result.sold <= result.produced
    assert result.unsold == result.produced - result.sold
    assert result.production_cost == result.produced * config.unit_cost
    assert result.profit == result.revenue - result.production_cost
    assert result.capital_after == before + result.profit
    assert result.capital_after >= 0
