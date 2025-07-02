import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))


import numpy as np
import pytest
from Environment import Environment
from QLearningAgent import QLearningAgent


def test_choose_action_exploits():
    env = Environment(2, 2)
    env.initialize_grid([], [], (1, 1), (0, 0))
    agent = QLearningAgent(env, 0.5, 0.9, 0.0, 0.9, 0.01)
    # force q_table so action 2 has highest value
    agent.q_table[0, 0, :] = np.array([0.1, 0.2, 1.0, 0.0])
    assert agent.choose_action((0, 0)) == 2


def test_update_q_value_moves_toward_target():
    env = Environment(2, 2)
    env.initialize_grid([], [], (1, 1), (0, 0))
    agent = QLearningAgent(
        env,
        learning_rate=1.0,
        discount_factor=0.0,
        exploration_rate=0.0,
        exploration_rate_decay=1.0,
        min_exploration_rate=0.0,
    )
    # Q=0 initially
    agent.update_q_value((0, 0), action=0, reward=5.0, next_state=(1, 1))
    # With γ=0, target=5.0 so Q should become exactly 5.0
    assert pytest.approx(agent.q_table[0, 0, 0]) == 5.0
