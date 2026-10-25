from __future__ import annotations

import numpy as np
from gymnasium.utils.env_checker import check_env

from lemonade.env import LemonadeEnv


def test_gymnasium_checker() -> None:
    check_env(LemonadeEnv(), skip_render_check=True)


def test_complete_episode_has_natural_termination_not_truncation() -> None:
    env = LemonadeEnv()
    observation, _ = env.reset(seed=10)
    assert env.observation_space.contains(observation)
    for index in range(10):
        observation, reward, terminated, truncated, info = env.step(np.array([20, 100]))
        assert env.observation_space.contains(observation)
        assert isinstance(reward, float)
        assert not truncated
        assert terminated is (index == 9)
    assert info["day_result"].profit == reward
