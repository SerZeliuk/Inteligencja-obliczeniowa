import gymnasium as gym
import numpy as np
from gymnasium import spaces
import numpy as np
import math
import pygame

# Physics constants
grav = -9.8  # gravity (m/s^2)

# Graphics constants (adjustable)
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
SCALE = 50  # pixels per meter
BALL_RADIUS_PIX = int(0.12 * SCALE)
HOOP_RADIUS_PIX = int(0.23 * SCALE)
MARGIN_BOTTOM = 50  # pixels from bottom for ground line

# Physics simulation for a 2D basketball shot
def simulate_shot(x0, y0, v0, theta, basket_x, basket_y,
                  dt=0.02, ball_radius=0.12, hoop_radius=0.23):
    """
    Simulate projectile motion until the ball hits the ground or goes through the hoop.
    Returns:
      traj: list of (x, y) positions (meters)
      success: bool indicating if the shot went in
    """
    theta_rad = math.radians(theta)
    vx = v0 * math.cos(theta_rad)
    vy = v0 * math.sin(theta_rad)
    x, y = x0, y0
    traj = [(x, y)]
    success = False

    while y >= 0:
        x += vx * dt
        y += vy * dt
        vy += grav * dt
        traj.append((x, y))
        # Check if ball intersects hoop circle
        dist = math.hypot(x - basket_x, y - basket_y)
        if dist <= (hoop_radius + ball_radius):
            success = True
            break

    return traj, success

class BasketballEnv(gym.Env):
    """
    Gym environment for single-shot 2D basketball with Pygame graphics.
    Observation:
      [dx, dy]: vector from shooter to hoop (m)
    Action:
      [force, angle]: continuous force (m/s) and launch angle (deg)
    Reward:
      +1 for basket, else negative miss distance plus height and range penalties.
    Episode:
      One shot per episode.
    """
    metadata = {"render.modes": ["human", "rgb_array"]}

    # Penalty weights (tune as needed)
    HEIGHT_PENALTY_WEIGHT = 10
    X_PENALTY_WEIGHT = 2

    def __init__(self,
                 shoot_x=0.0, shoot_y=1.8,
                 basket_x=10.0, basket_y=3.05):
        super().__init__()
        # Action: [force, angle]
        self.MAX_FORCE = 60.0  #in m/s
        self.MAX_ANGLE = 89.0  # in degrees

        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([self.MAX_FORCE, self.MAX_ANGLE], dtype=np.float32),
            dtype=np.float32
        )
        # Observation: [dx, dy]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
        )

        # Fixed physics positions (meters)
        self.shoot_x = shoot_x
        self.shoot_y = shoot_y
        self.basket_x = basket_x
        self.basket_y = basket_y
        self.trajectory = []

        # Init Pygame for graphics
        pygame.init()
        self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.window = None

    def reset(self, seed=None, options=None):
        # Handle seeding
        if seed is not None:
            np.random.seed(seed)
        # Compute relative vector to hoop
        dx = self.basket_x - self.shoot_x
        dy = self.basket_y - self.shoot_y
        self.trajectory = []
        obs = np.array([dx, dy], dtype=np.float32)
        return obs, {}

    def step(self, action):
        force, angle = action
        traj, success = simulate_shot(
            self.shoot_x, self.shoot_y,
            float(force), float(angle),
            self.basket_x, self.basket_y
        )
        self.trajectory = traj

        if success:
            reward = 1.0
        else:
            # Base penalty: distance from hoop at landing
            final_x, final_y = traj[-1]
            dist = math.hypot(final_x - self.basket_x,
                            final_y - self.basket_y)

            # Height penalty: if max height below hoop height
            max_height = max(y for (_, y) in traj)
            height_deficit = max(0.0, self.basket_y - max_height)
            height_penalty = height_deficit * self.HEIGHT_PENALTY_WEIGHT

            # Range penalty: if falls short of hoop
            range_deficit = max(0.0, self.basket_x - final_x)
            range_penalty = range_deficit * self.X_PENALTY_WEIGHT

            # Bonus for clearing the rim
            height_bonus = max(0.0, max_height - self.basket_y) * 0.05
            # Bonus for force usage
            force_bonus = (force / self.MAX_FORCE) * 0.1

            # Total reward combining penalties and bonuses
            reward = (
                -dist
                - height_penalty
                - range_penalty
                + height_bonus
                + force_bonus
            )

        terminated = True
        truncated = False
        obs = np.array([0.0, 0.0], dtype=np.float32)
        info = {"success": success, "trajectory": traj}
        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        # Draw court, trajectory, and hoop
        # Clear screen
        self.screen.fill((30, 30, 30))

        # Ground line
        floor_y = SCREEN_HEIGHT - MARGIN_BOTTOM
        pygame.draw.line(self.screen, (200, 200, 200),
                         (0, floor_y), (SCREEN_WIDTH, floor_y), 2)

        # Hoop
        hoop_x = int(self.basket_x * SCALE)
        hoop_y = floor_y - int(self.basket_y * SCALE)
        pygame.draw.circle(self.screen, (255, 165, 0),
                           (hoop_x, hoop_y), HOOP_RADIUS_PIX, 3)

        # Shooter
        shooter_x = int(self.shoot_x * SCALE)
        shooter_y = floor_y - int(self.shoot_y * SCALE)
        pygame.draw.circle(self.screen, (0, 0, 255), (shooter_x, shooter_y), 8)

        # Ball trajectory
        for x, y in self.trajectory:
            px = int(x * SCALE)
            py = floor_y - int(y * SCALE)
            pygame.draw.circle(self.screen, (255, 255, 255), (px, py), BALL_RADIUS_PIX)

        if mode == "human":
            if self.window is None:
                pygame.display.init()
                self.window = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            self.window.blit(self.screen, (0, 0))
            pygame.display.flip()
        elif mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)),
                (1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
        pygame.quit()