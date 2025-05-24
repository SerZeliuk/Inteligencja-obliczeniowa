# File: train_agent.py

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import matplotlib.pyplot as plt

from basketball_env import BasketballEnv

class RewardLogger(BaseCallback):
    """Callback for logging episode rewards."""
    def __init__(self):
        super().__init__()
        self.rewards = []

    def _on_step(self) -> bool:
        dones = self.locals.get('dones')
        rewards = self.locals.get('rewards')
        if dones is not None:
            for idx, done in enumerate(dones):
                if done:
                    self.rewards.append(rewards[idx])
        return True

def train(env: gym.Env, timesteps: int = 100_000):
    # Instantiate the PPO agent with custom hyperparameters
    model = PPO(
        policy='MlpPolicy',
        env=env,
        learning_rate=5e-3,
        gamma=0.99,
        n_steps=2048,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
    )

    # Create a separate eval environment
    eval_env = BasketballEnv()
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='.',
        log_path='./logs',
        eval_freq=10_000,          # evaluate every 10k steps
        n_eval_episodes=5,
        deterministic=True,
    )

    # Train with both reward logging and evaluation callbacks
    logger = RewardLogger()
    model.learn(
        total_timesteps=timesteps,
        callback=[logger, eval_callback]
    )

    # eval_callback.best_model_path holds path to best checkpoint
    return logger, eval_callback.best_model_save_path

def plot_learning_curve(logger: RewardLogger):
    plt.figure()
    plt.plot(logger.rewards)
    plt.title('Episode Reward over Time')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()

def main():
    env = BasketballEnv()
    logger, best_model_path = train(env, timesteps=100_000)
    plot_learning_curve(logger)
    print(f"Best model saved to: {best_model_path}")

if __name__ == '__main__':
    main()
