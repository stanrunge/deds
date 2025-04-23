import pygame
from board import Board
from ql import QLearner

ALPHA = 0.1
GAMMA = 0.99
EPS_START = 1.0
EPS_DECAY = 0.995
EPS_MIN = 0.1
EPISODES = 500
MAX_STEPS = 200
CELL_SIZE = 60
STATS_WIDTH = 300

MAZE = [
    [0, 0, 0, 0, -1, 0, -1, 0, 1],
    [0, 0, -1, 0, -1, 0, -1, 0, -1],
    [0, 0, -1, 0, 0, 0, -1, 0, 0],
    [-1, 0, -1, -1, -1, 0, -1, -1, 0],
    [0, 0, 0, 0, -1, 0, 0, 0, 0],
    [0, -1, -1, 0, -1, -1, -1, -1, 0],
]
START_POS = (4, 0)


def get_next(pos, action, maze):
    moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    dr, dc = moves[action]
    nr, nc = pos[0] + dr, pos[1] + dc

    if not (0 <= nr < len(maze) and 0 <= nc < len(maze[0])) or maze[nr][nc] == -1:
        return pos, -10
    if maze[nr][nc] == 1:
        return (nr, nc), +100
    return (nr, nc), -1


def draw_buttons(screen, font, run_mode):
    buttons = {}
    padding = 10
    x, y = padding, screen.get_height() - 50

    # Step button
    step_rect = pygame.Rect(x, y, 80, 40)
    pygame.draw.rect(screen, (180, 180, 180), step_rect)
    screen.blit(font.render("Step", True, (0, 0, 0)), (x + 20, y + 10))
    buttons["step"] = step_rect

    # Run / Pause
    x += 90
    toggle_rect = pygame.Rect(x, y, 80, 40)
    pygame.draw.rect(screen, (180, 180, 180), toggle_rect)
    label = "Pause" if run_mode else "Run"
    screen.blit(font.render(label, True, (0, 0, 0)), (x + 10, y + 10))
    buttons["toggle"] = toggle_rect

    return buttons


def run():
    pygame.init()
    rows, cols = len(MAZE), len(MAZE[0])

    board = Board(MAZE, cell_size=CELL_SIZE, stats_width=STATS_WIDTH)
    fh = board.font.get_height() + 2
    n_states = rows * cols
    total_lines = 6 + n_states
    stats_h = total_lines * fh + 80
    board_h = rows * CELL_SIZE
    height = max(board_h, stats_h)
    width = cols * CELL_SIZE + STATS_WIDTH

    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Q‑Learning with Arrows & Coords")

    clock = pygame.time.Clock()
    agent = QLearner(
        rows,
        cols,
        alpha=ALPHA,
        gamma=GAMMA,
        epsilon=EPS_START,
        epsilon_decay=EPS_DECAY,
        min_epsilon=EPS_MIN,
    )

    run_mode = False
    step_req = False
    pos = START_POS
    state = agent.state_from_pos(pos)
    total_reward = 0
    step = 0
    ep = 1

    while ep <= EPISODES:
        for evt in pygame.event.get():
            if evt.type == pygame.QUIT:
                pygame.quit()
                return
            if evt.type == pygame.MOUSEBUTTONDOWN and evt.button == 1:
                btns = draw_buttons(screen, board.font, run_mode)
                if btns["step"].collidepoint(evt.pos):
                    step_req = True
                elif btns["toggle"].collidepoint(evt.pos):
                    run_mode = not run_mode

        if run_mode or step_req:
            step_req = False
            a = agent.choose_action(state)
            nxt, rew = get_next(pos, a, MAZE)
            ns = agent.state_from_pos(nxt)
            agent.update(state, a, rew, ns)
            pos, state = nxt, ns
            total_reward += rew
            step += 1

            if MAZE[pos[0]][pos[1]] == 1 or step >= MAX_STEPS:
                print(
                    f"Episode {ep}/{EPISODES}, reward={total_reward}, eps={agent.epsilon:.3f}"
                )
                agent.decay_epsilon()
                ep += 1
                pos = START_POS
                state = agent.state_from_pos(pos)
                total_reward = 0
                step = 0

        # render
        screen.fill((240, 240, 240))
        board.draw(screen, pos, agent)
        board.draw_stats(screen, agent, ep, step, rew if step > 0 else 0)
        draw_buttons(screen, board.font, run_mode)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    run()
