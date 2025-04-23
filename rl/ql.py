# qlearner.py

import numpy as np
import random


class QLearner:
    def __init__(
        self,
        rows,
        cols,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        min_epsilon=0.1,
    ):
        self.rows = rows
        self.cols = cols
        self.n_states = rows * cols
        self.n_actions = 4

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.Q = np.zeros((self.n_states, self.n_actions))

    def state_from_pos(self, pos):
        return pos[0] * self.cols + pos[1]

    def pos_from_state(self, s):
        return (s // self.cols, s % self.cols)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        return int(np.argmax(self.Q[state]))

    def update(self, s, a, r, s_next):
        best_next = np.max(self.Q[s_next])
        td = r + self.gamma * best_next - self.Q[s, a]
        self.Q[s, a] += self.alpha * td

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
