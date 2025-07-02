from collections import deque
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import Model, layers, optimizers
from Environment import Environment


class DQNAgent:
    type = "DQN"

    def __init__(
        self,
        env: Environment,
        learning_rate: float = 1e-4,
        discount_factor: float = 0.99,
        exploration_rate: float = 1.0,
        exploration_rate_decay: float = 0.9995,
        min_exploration_rate: float = 0.01,
        replay_buffer_size: int = 5000,
        batch_size: int = 32,
        target_update_freq: int = 500,
    ):
        self.env = env
        self.state_dim = 2
        self.action_dim = 4
        self.lr = learning_rate
        self.gamma = discount_factor
        self.exploration_rate = exploration_rate
        self.eps_decay = exploration_rate_decay
        self.eps_min = min_exploration_rate

        self.memory = deque(maxlen=replay_buffer_size)
        self.batch_size = batch_size

        self.model = self._build_model()
        self.target = self._build_model()
        self.target.set_weights(self.model.get_weights())

        self.model.compile(
            optimizer=optimizers.Adam(self.lr), loss=tf.keras.losses.Huber()
        )

        self.train_steps = 0
        self.target_freq = target_update_freq

    def _build_model(self) -> Model:
        inp = layers.Input((self.state_dim,))
        x = layers.Dense(64, activation="relu")(inp)
        x = layers.Dense(64, activation="relu")(x)
        out = layers.Dense(self.action_dim, activation="linear")(x)
        return Model(inp, out)

    def remember(self, s, a, r, s2, done):
        self.memory.append((s, a, r, s2, done))

    def choose_action(self, state):
        if np.random.rand() < self.exploration_rate:
            return np.random.randint(self.action_dim)
        x, y = state
        norm = np.array([[x / (self.env.width - 1), y / (self.env.height - 1)]])
        qs = self.model.predict(norm, verbose=0)[0]
        return int(np.argmax(qs))

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        S = np.array(
            [
                [s[0] / (self.env.width - 1), s[1] / (self.env.height - 1)]
                for s, *_ in batch
            ]
        )
        A = np.array([a for _, a, _, _, _ in batch])
        R = np.array([r for *_, r, _, _ in batch])
        S2 = np.array(
            [
                [s2[0] / (self.env.width - 1), s2[1] / (self.env.height - 1)]
                for *_, _, s2, _ in batch
            ]
        )
        D = np.array([done for *_, done in batch], dtype=bool)

        q = self.model.predict(S, verbose=0)
        q_next = self.target.predict(S2, verbose=0)

        target = q.copy()
        for i in range(self.batch_size):
            if D[i]:
                target[i, A[i]] = R[i]
            else:
                target[i, A[i]] = R[i] + self.gamma * np.max(q_next[i])

        self.model.train_on_batch(S, target)

        # decay
        self.exploration_rate = max(
            self.eps_min, self.exploration_rate * self.eps_decay
        )

        # update target
        self.train_steps += 1
        if self.train_steps % self.target_freq == 0:
            self.target.set_weights(self.model.get_weights())
