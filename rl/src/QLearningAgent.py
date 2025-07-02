import numpy as np
from Environment import Environment


class QLearningAgent:
    type = "QLearning"

    def __init__(
        self,
        env: Environment,
        learning_rate: float,
        discount_factor: float,
        exploration_rate: float,
        exploration_rate_decay: float,
        min_exploration_rate: float,
    ):
        super().__init__()
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_rate_decay = exploration_rate_decay
        self.min_exploration_rate = min_exploration_rate
        self.q_table = np.zeros((env.width, env.height, 4), dtype=np.float32)

    # Chooses the next action based on the current state using an epsilon-greedy strategy
    def choose_action(self, state: tuple[int, int]) -> int:
        x, y = state
        # Exploration strategy: with probability exploration_rate, choose a random action
        if np.random.rand() < self.exploration_rate:
            return np.random.randint(4)
        # Exploitation strategy: choose the action with the highest Q-value for the current state
        return int(np.argmax(self.q_table[x, y]))

    #
    def update_q_value(
        self,
        state: tuple[int, int],
        action: int,
        reward: float,
        next_state: tuple[int, int],
        done: bool = False,
    ):
        x, y = state
        next_x, next_y = next_state

        old_q = self.q_table[x, y, action]

        # Bellman target: the immediate reward plus the discounted maximum future reward
        future = 0.0 if done else np.max(self.q_table[next_x, next_y])
        target = reward + self.discount_factor * future

        # Q-learning update rule
        self.q_table[x, y, action] = old_q + self.learning_rate * (target - old_q)

    def train(self, episodes: int, max_steps: int):

        for ep in range(episodes):
            state = self.env.reset_agent_pos()
            total_reward = 0.0

            for step in range(max_steps):
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)

                self.update_q_value(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward

                if done:
                    break

                self.exploration_rate = max(
                    self.min_exploration_rate,
                    self.exploration_rate * self.exploration_rate_decay,
                )
