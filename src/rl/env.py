import gymnasium as gym
import numpy as np
from gymnasium import spaces


class SimpleNavEnv(gym.Env):
    """Minimal navigation environment compatible with Gymnasium.

    Observation: 8-dim float vector (position/velocity placeholder)
    Action space: Discrete(4) representing up/down/left/right.
    Goal: Move position toward target; reward is negative distance.
    """

    metadata = {"render_modes": []}

    def __init__(self, max_steps: int = 200, seed: int | None = None):
        super().__init__()
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        # Simple 2D position + velocity embedded in 8-dim vector
        self.pos = np.zeros(2, dtype=np.float32)
        self.vel = np.zeros(2, dtype=np.float32)
        self.goal = np.array([0.8, 0.8], dtype=np.float32)
        self.step_count = 0

    def seed(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        return [seed]

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.seed(seed)
        self.pos = self.rng.uniform(low=-0.5, high=0.5, size=2).astype(np.float32)
        self.vel = np.zeros(2, dtype=np.float32)
        self.step_count = 0
        obs = self._build_obs()
        info = {}
        return obs, info

    def step(self, action: int):
        self.step_count += 1
        # Simple velocity update
        if action == 0:   # up
            self.vel += np.array([0.0, 0.05], dtype=np.float32)
        elif action == 1:  # down
            self.vel += np.array([0.0, -0.05], dtype=np.float32)
        elif action == 2:  # left
            self.vel += np.array([-0.05, 0.0], dtype=np.float32)
        elif action == 3:  # right
            self.vel += np.array([0.05, 0.0], dtype=np.float32)

        # Clip velocity and update position
        self.vel = np.clip(self.vel, -0.2, 0.2)
        self.pos = np.clip(self.pos + self.vel, -1.0, 1.0)

        # Reward: negative L2 distance to goal
        dist = np.linalg.norm(self.goal - self.pos)
        reward = -float(dist)

        terminated = dist < 0.05
        truncated = self.step_count >= self.max_steps
        obs = self._build_obs()
        info = {"distance": dist}
        return obs, reward, terminated, truncated, info

    def _build_obs(self) -> np.ndarray:
        # Pack position, velocity, goal and padding to length 8
        obs = np.zeros(8, dtype=np.float32)
        obs[0:2] = self.pos
        obs[2:4] = self.vel
        obs[4:6] = self.goal
        # obs[6:8] remain zeros
        return obs

    def render(self):
        # No-op render; could be extended with pygame if needed
        pass

    def close(self):
        pass
