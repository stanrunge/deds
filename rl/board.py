import pygame


class Board:
    def __init__(self, maze, cell_size=60, stats_width=250):
        """
        maze: 2D list of ints (0=empty, -1=wall, +1=goal)
        stats_width: horizontal pixels reserved for stats panel
        """
        self.maze = maze
        self.rows = len(maze)
        self.cols = len(maze[0])
        self.cell_size = cell_size
        self.stats_width = stats_width

        # precompute rects
        self.grid_rects = [
            [
                pygame.Rect(c * cell_size, r * cell_size, cell_size, cell_size)
                for c in range(self.cols)
            ]
            for r in range(self.rows)
        ]

        pygame.font.init()
        self.font = pygame.font.SysFont("Consolas", 16)

    def draw(self, screen, agent_pos, qlearner):
        # draw grid & coords
        for r in range(self.rows):
            for c in range(self.cols):
                rect = self.grid_rects[r][c]
                val = self.maze[r][c]
                if val == -1:
                    color = (50, 50, 50)  # wall
                elif val == +1:
                    color = (0, 200, 0)  # goal
                else:
                    color = (200, 200, 200)  # floor
                pygame.draw.rect(screen, color, rect)

                # draw tile coords in top-left
                coord_surf = self.font.render(f"{r},{c}", True, (100, 100, 100))
                screen.blit(coord_surf, (rect.x + 2, rect.y + 2))

                # draw best-action arrow & q-value for non-wall/non-goal
                if val == 0:
                    state = r * self.cols + c
                    qvals = qlearner.Q[state]
                    best = int(qvals.argmax())
                    cx, cy = rect.center
                    sz = self.cell_size // 4

                    # arrow triangle points
                    if best == 0:  # up
                        pts = [(cx, cy - sz), (cx - sz, cy + sz), (cx + sz, cy + sz)]
                    elif best == 1:  # right
                        pts = [(cx + sz, cy), (cx - sz, cy - sz), (cx - sz, cy + sz)]
                    elif best == 2:  # down
                        pts = [(cx, cy + sz), (cx - sz, cy - sz), (cx + sz, cy - sz)]
                    else:  # left
                        pts = [(cx - sz, cy), (cx + sz, cy - sz), (cx + sz, cy + sz)]

                    pygame.draw.polygon(screen, (0, 0, 255), pts)

                    # draw that best q-value in bottom-right of cell
                    q_surf = self.font.render(f"{qvals[best]:.1f}", True, (0, 0, 0))
                    screen.blit(
                        q_surf,
                        (
                            rect.x + self.cell_size - q_surf.get_width() - 2,
                            rect.y + self.cell_size - q_surf.get_height() - 2,
                        ),
                    )

        # highlight agent
        ar, ac = agent_pos
        center = (
            ac * self.cell_size + self.cell_size // 2,
            ar * self.cell_size + self.cell_size // 2,
        )
        pygame.draw.circle(screen, (200, 0, 0), center, self.cell_size // 3)

    def draw_stats(self, screen, qlearner, episode, step, reward):
        x0 = self.cols * self.cell_size + 10
        y = 10

        # Episode, Epsilon, Last reward
        for line in [
            f"Episode:  {episode}",
            f"Step:     {step}",
            f"Epsilon:  {qlearner.epsilon:.3f}",
            f"Reward:   {reward}",
            "",
            "Q‑table:",
        ]:
            surf = self.font.render(line, True, (0, 0, 0))
            screen.blit(surf, (x0, y))
            y += surf.get_height() + 2

        # full Q‑table rows
        for s in range(qlearner.n_states):
            qvals = qlearner.Q[s]
            r, c = qlearner.pos_from_state(s)
            txt = f"{(r, c)}: " + " ".join(f"{q:6.2f}" for q in qvals)
            surf = self.font.render(txt, True, (0, 0, 0))
            screen.blit(surf, (x0, y))
            y += surf.get_height() + 1
