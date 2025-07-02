import numpy as np
import pygame


class Environment:
    cell_size = 100
    stats_width = 200

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        # Set display to have width + stats_width and height
        self.screen = pygame.display.set_mode(
            ((width * self.cell_size + self.stats_width), height * self.cell_size)
        )
        self.walls = set()
        self.hazards = set()
        self.bonuses = set()
        self.goal = None
        self.start = None
        self.agent_pos = None
        self.rewards = np.zeros((width, height))

    # Initialize the grid with walls, hazards, goal, start position, and reward map
    def initialize_grid(
        self,
        walls: list[tuple[int, int]],
        hazards: list[tuple[int, int]],
        bonuses: list[tuple[int, int]],
        goal: tuple[int, int],
        start: tuple[int, int],
        reward_map: dict[tuple[int, int], float] | None = None,
    ):
        self.walls = set(walls)
        self.hazards = set(hazards)
        self.bonuses = set(bonuses)
        self.goal = goal
        self.start = start
        self.agent_pos = start

        self.rewards.fill(-0.01)
        for hx, hy in self.hazards:
            self.rewards[hx, hy] = -1.0

        for bx, by in self.bonuses:
            self.rewards[bx, by] = 0.1

        gx, gy = self.goal
        self.rewards[gx, gy] = 50.0

        if reward_map:
            for (x, y), reward in reward_map.items():
                self.rewards[x, y] = reward

    # Reset the agent position to the start position
    def reset_agent_pos(self) -> tuple[int, int]:
        if self.start is None:
            raise ValueError("Environment not initialized with a start position.")

        self.agent_pos = self.start

        return self.agent_pos

    # A position is valid if it is within the grid bounds and not a wall
    def is_valid(self, pos: tuple[int, int]) -> bool:
        x, y = pos
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False
        if pos in self.walls:
            return False
        return True

    # Take a valid step with a given direction
    # Returns the new position, reward, and if the goal is reached
    def step(self, action: int) -> tuple[tuple[int, int], float, bool]:
        if self.agent_pos is None:
            raise ValueError("Environment not initialized with an agent position.")

        x, y = self.agent_pos
        moves = {
            0: (x, y - 1),  # Up
            1: (x, y + 1),  # Down
            2: (x - 1, y),  # Left
            3: (x + 1, y),  # Right
        }

        candidate = moves[action]
        if self.is_valid(candidate):
            self.agent_pos = candidate

        rx, ry = self.agent_pos
        reward = self.rewards[rx, ry]
        done = self.agent_pos == self.goal

        return self.agent_pos, reward, done

    # Render the environment with Pygame
    def render(self, agent, episode: int, step: int, mode: str):
        if self.agent_pos is None:
            raise ValueError("Environment not initialized with an agent position.")

        # Clear the screen
        self.screen.fill((255, 255, 255))

        panel_x = self.width * self.cell_size
        panel_rect = pygame.Rect(
            panel_x, 0, self.stats_width, self.height * self.cell_size
        )
        pygame.draw.rect(self.screen, (240, 240, 240), panel_rect)

        stat_font = pygame.font.Font(None, 24)

        stats = [
            f"Mode: {mode}",
            f"Episode: {episode}",
            f"Step: {step}",
            f"" f"Exploration Rate: {agent.exploration_rate:.3f}",
        ]
        y = 20
        for line in stats:
            txt = stat_font.render(line, True, (0, 0, 0))
            self.screen.blit(txt, (panel_x + 10, y))
            y += 30

        pygame.font.init()
        font = pygame.font.Font(None, 20)
        arrow_font = pygame.font.Font("assets/fonts/DejaVuSansMono.ttf", 48)

        # Draw the grid cells
        for x in range(self.width):
            for y in range(self.height):
                rect = pygame.Rect(
                    x * self.cell_size,  # x position
                    y * self.cell_size,  # y position
                    self.cell_size,  # width
                    self.cell_size,  # height
                )
                if (x, y) in self.walls:
                    pygame.draw.rect(self.screen, (0, 0, 0), rect)
                elif (x, y) in self.hazards:
                    pygame.draw.rect(self.screen, (200, 0, 0), rect)
                elif (x, y) in self.bonuses:
                    pygame.draw.rect(self.screen, (0, 200, 200), rect)
                elif (x, y) == self.goal:
                    pygame.draw.rect(self.screen, (0, 200, 0), rect)
                else:
                    pygame.draw.rect(self.screen, (200, 200, 200), rect)
                pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)

                arrow_map = {
                    0: ("↑", (self.cell_size / 2 - 12, -12)),
                    1: ("↓", (self.cell_size / 2 - 12, self.cell_size - 48)),
                    2: ("←", (5, self.cell_size // 2 - 28)),
                    3: ("→", (self.cell_size - 32, self.cell_size // 2 - 28)),
                }

                q_value_map = {
                    0: (arrow_map[0][1][0] + 24, arrow_map[0][1][1] - 80),
                    1: (arrow_map[1][1][0] + 24, arrow_map[1][1][1] + 32),
                    2: (arrow_map[2][1][0], arrow_map[2][1][1] + 40),
                    3: (arrow_map[3][1][0], arrow_map[3][1][1] + 40),
                }

                for a in range(4):
                    arrow, (dx, dy) = arrow_map[a]
                    qv = agent.q_table[x, y, a]
                    color = self.q_color(qv)

                    arrow_text = arrow_font.render(arrow, True, color)
                    self.screen.blit(
                        arrow_text, (x * self.cell_size + dx, y * self.cell_size + dy)
                    )

                    q_value_text = font.render(f"{qv:.2f}", True, color)
                    self.screen.blit(
                        q_value_text,
                        (
                            x * self.cell_size + q_value_map[a][0],
                            y * self.cell_size + q_value_map[a][1],
                        ),
                    )

        # Draw the agent
        ax, ay = self.agent_pos
        center = (
            ax * self.cell_size + self.cell_size // 2,
            ay * self.cell_size + self.cell_size // 2,
        )
        pygame.draw.circle(self.screen, (0, 0, 200), center, self.cell_size // 4)

    def q_color(self, val: float) -> tuple[int, int, int]:
        v = max(-1, min(1, val))

        if v >= 0:
            # green gradient
            g = int(v * 255)
            return (0, g, 0)
        else:
            # red gradient
            r = int(-v * 255)
            return (r, 0, 0)
