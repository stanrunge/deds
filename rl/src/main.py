import pygame
from DQNAgent import DQNAgent
from Environment import Environment
from QLearningAgent import QLearningAgent


def main():
    pygame.init()
    env = Environment(width=10, height=10)
    walls = [
        (1, 1),
        (1, 2),
        (1, 3),
        (1, 4),
        (1, 5),
        (1, 6),
        (1, 8),
        (2, 6),
        (2, 8),
        (3, 2),
        (3, 3),
        (3, 4),
        (3, 6),
        (3, 8),
        (4, 2),
        (4, 6),
        (4, 8),
        (5, 2),
        (5, 4),
        (5, 5),
        (5, 6),
        (5, 8),
        (6, 2),
        (6, 4),
        (6, 8),
        (7, 0),
        (7, 1),
        (7, 2),
        (7, 4),
        (7, 6),
        (7, 7),
        (7, 8),
        (8, 4),
        (9, 2),
        (9, 3),
        (9, 4),
    ]
    hazards = [(2, 1), (3, 1), (4, 1), (5, 1)]
    bonuses = []
    goal = (9, 9)
    start = (0, 0)
    env.initialize_grid(walls, hazards, bonuses, goal, start)

    if not env.agent_pos:
        raise ValueError("Environment not initialized with an agent position.")

    agent = QLearningAgent(env, 0.5, 0.99, 1.0, 0.9999, 0.01)
    # agent = DQNAgent(
    #     env,
    #     learning_rate=0.001,
    #     discount_factor=0.99,
    #     exploration_rate=1.0,
    #     exploration_rate_decay=0.995,
    #     min_exploration_rate=0.01,
    # )

    training = False

    button_rects = {
        "train": pygame.Rect(env.width * env.cell_size + 10, 10, 100, 30),
        "reset": pygame.Rect(env.width * env.cell_size + 10, 50, 100, 30),
    }

    font = pygame.font.Font(None, 24)

    episode = 1
    step = 0
    max_steps = 100
    total_reward = 0.0
    clock = pygame.time.Clock()
    while True:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if button_rects["train"].collidepoint(event.pos):
                    training = not training
                if button_rects["reset"].collidepoint(event.pos):
                    env.reset_agent_pos()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_t:
                    training = not training

                if event.key == pygame.K_r:
                    env.reset_agent_pos()
                if not training and event.key in (
                    pygame.K_UP,
                    pygame.K_DOWN,
                    pygame.K_LEFT,
                    pygame.K_RIGHT,
                ):
                    mapping = {
                        pygame.K_UP: 0,  # Up
                        pygame.K_DOWN: 1,  # Down
                        pygame.K_LEFT: 2,  # Left
                        pygame.K_RIGHT: 3,  # Right
                    }

                    state = env.agent_pos
                    action = mapping[event.key]
                    next_state, reward, done = env.step(action)
                    agent.update_q_value(state, action, reward, next_state, done)
                    agent.exploration_rate = max(
                        agent.min_exploration_rate,
                        agent.exploration_rate * agent.exploration_rate_decay,
                    )
                    if done or step > max_steps - 1:
                        episode += 1
                        step = 0
                        env.reset_agent_pos()
                    else:
                        step += 1

        for label, rect in button_rects.items():
            pygame.draw.rect(env.screen, (180, 180, 180), rect)
            text = font.render(label.capitalize(), True, (0, 0, 0))
            env.screen.blit(text, (rect.x + 10, rect.y + 5))

        if training:
            state = env.agent_pos
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)

            agent.update_q_value(state, action, reward, next_state, done)

            agent.exploration_rate = max(
                agent.min_exploration_rate,
                agent.exploration_rate * agent.exploration_rate_decay,
            )

            step += 1
            total_reward += reward

            if done or step > max_steps:
                episode += 1
                step = 0
                total_reward = 0.0
                env.reset_agent_pos()

        print(
            f"Episode {episode}, "
            f"Steps: {step+1}, "
            f"Exploration Rate={agent.exploration_rate:.3f}"
        )

        env.render(agent, episode, step, "Training" if training else "Manual")
        pygame.display.flip()
        clock.tick(120)


if __name__ == "__main__":
    main()
