import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))


import pytest
from Environment import Environment


@pytest.fixture
def small_env():
    env = Environment(width=3, height=3)
    walls = [(1, 1)]
    hazards = [(0, 2)]
    goal = (2, 2)
    start = (0, 0)
    env.initialize_grid(walls, hazards, goal, start)
    return env


def test_is_valid(small_env):
    assert small_env.is_valid((0, 0))
    assert not small_env.is_valid((1, 1))
    assert not small_env.is_valid((3, 3))


def test_step_and_reset(small_env):
    env = small_env
    env.reset_agent_pos()
    pos, reward, done = env.step(3)  # Left from (0,0) is invalid → stay (0,0)
    assert pos == (0, 0)
    # move Down (action=2)
    pos, reward, done = env.step(2)
    assert pos == (0, 1) and reward == 0 and not done
