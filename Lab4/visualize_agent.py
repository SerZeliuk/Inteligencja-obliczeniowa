import pygame
from stable_baselines3 import PPO
from basketball_env import BasketballEnv, SCREEN_WIDTH, SCREEN_HEIGHT

# Parameters for playback
episodes = 5
frame_delay_ms = 50


def play(model, env, episodes=episodes, delay=frame_delay_ms):
    """
    Run playback using Pygame graphics.
    """
    pygame.display.init()
    # Use environment's configured screen size
    env.window = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render(mode='human')
            done = terminated or truncated
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                    break
            pygame.time.delay(delay)
        pygame.time.delay(500)

    env.close()
    pygame.display.quit()


def main():
    # Load model using SB3
    model = PPO.load('best_model', env=BasketballEnv())

    env = BasketballEnv()
    print("Playing loaded agent...")
    for i in range(1, 15):
        play(model, env)

if __name__ == '__main__':
    main()