"""
Robot Navigation Environment with Pygame Rendering
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from typing import Tuple, Optional, Dict, Any


class NavigationEnv(gym.Env):
    """2D Robot Navigation Environment"""
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}
    
    def __init__(
        self,
        width: int = 800,
        height: int = 600,
        robot_radius: float = 15.0,
        goal_radius: float = 20.0,
        max_speed: float = 5.0,
        obstacles: list = None,
        render_mode: Optional[str] = None
    ):
        super().__init__()
        
        self.width = width
        self.height = height
        self.robot_radius = robot_radius
        self.goal_radius = goal_radius
        self.max_speed = max_speed
        self.render_mode = render_mode
        
        # Define action space: [forward, rotate_left, rotate_right, backward]
        self.action_space = spaces.Discrete(4)
        
        # Define observation space: [x, y, vx, vy, goal_x, goal_y, distance, angle]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(8,),
            dtype=np.float32
        )
        
        # Static obstacles
        self.static_obstacles = obstacles if obstacles else []
        
        # State variables
        self.robot_pos = np.array([0.0, 0.0])
        self.robot_vel = np.array([0.0, 0.0])
        self.robot_angle = 0.0
        self.goal_pos = np.array([0.0, 0.0])
        
        self.steps = 0
        self.max_steps = 1000
        
        # Rendering
        self.window = None
        self.clock = None
        
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        """Reset the environment"""
        super().reset(seed=seed)
        
        # Random robot position (away from edges)
        margin = 50
        self.robot_pos = np.array([
            np.random.uniform(margin, self.width - margin),
            np.random.uniform(margin, self.height - margin)
        ])
        
        # Random goal position (far from robot)
        min_distance = 200
        while True:
            self.goal_pos = np.array([
                np.random.uniform(margin, self.width - margin),
                np.random.uniform(margin, self.height - margin)
            ])
            if np.linalg.norm(self.goal_pos - self.robot_pos) > min_distance:
                break
        
        self.robot_vel = np.array([0.0, 0.0])
        self.robot_angle = np.random.uniform(0, 2 * np.pi)
        self.steps = 0
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step"""
        self.steps += 1
        
        # Apply action
        acceleration = 0.5
        rotation_speed = 0.1
        
        if action == 0:  # Forward
            self.robot_vel[0] += acceleration * np.cos(self.robot_angle)
            self.robot_vel[1] += acceleration * np.sin(self.robot_angle)
        elif action == 1:  # Rotate left
            self.robot_angle -= rotation_speed
        elif action == 2:  # Rotate right
            self.robot_angle += rotation_speed
        elif action == 3:  # Backward
            self.robot_vel[0] -= acceleration * np.cos(self.robot_angle)
            self.robot_vel[1] -= acceleration * np.sin(self.robot_angle)
        
        # Apply friction
        self.robot_vel *= 0.95
        
        # Limit speed
        speed = np.linalg.norm(self.robot_vel)
        if speed > self.max_speed:
            self.robot_vel = self.robot_vel / speed * self.max_speed
        
        # Update position
        self.robot_pos += self.robot_vel
        
        # Check boundaries
        collision = False
        if self.robot_pos[0] < self.robot_radius or self.robot_pos[0] > self.width - self.robot_radius:
            self.robot_vel[0] *= -0.5
            self.robot_pos[0] = np.clip(self.robot_pos[0], self.robot_radius, self.width - self.robot_radius)
            collision = True
        
        if self.robot_pos[1] < self.robot_radius or self.robot_pos[1] > self.height - self.robot_radius:
            self.robot_vel[1] *= -0.5
            self.robot_pos[1] = np.clip(self.robot_pos[1], self.robot_radius, self.height - self.robot_radius)
            collision = True
        
        # Check goal reached
        distance_to_goal = np.linalg.norm(self.goal_pos - self.robot_pos)
        goal_reached = distance_to_goal < self.goal_radius + self.robot_radius
        
        # Calculate reward
        reward = self._calculate_reward(distance_to_goal, collision, goal_reached)
        
        # Check termination
        terminated = goal_reached
        truncated = self.steps >= self.max_steps
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _calculate_reward(
        self,
        distance: float,
        collision: bool,
        goal_reached: bool
    ) -> float:
        """Calculate reward"""
        if goal_reached:
            return 100.0
        
        reward = -0.1  # Time penalty
        
        # Distance reward (encourage getting closer)
        reward += -distance / 100.0
        
        # Collision penalty
        if collision:
            reward -= 5.0
        
        return reward
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        distance = np.linalg.norm(self.goal_pos - self.robot_pos)
        angle_to_goal = np.arctan2(
            self.goal_pos[1] - self.robot_pos[1],
            self.goal_pos[0] - self.robot_pos[0]
        )
        
        return np.array([
            self.robot_pos[0] / self.width,
            self.robot_pos[1] / self.height,
            self.robot_vel[0] / self.max_speed,
            self.robot_vel[1] / self.max_speed,
            self.goal_pos[0] / self.width,
            self.goal_pos[1] / self.height,
            distance / np.sqrt(self.width**2 + self.height**2),
            angle_to_goal / np.pi
        ], dtype=np.float32)
    
    def _get_info(self) -> dict:
        """Get additional info"""
        return {
            'distance_to_goal': np.linalg.norm(self.goal_pos - self.robot_pos),
            'robot_pos': self.robot_pos.copy(),
            'goal_pos': self.goal_pos.copy(),
            'steps': self.steps
        }
    
    def render(self):
        """Render the environment"""
        if self.render_mode is None:
            return
        
        if self.window is None:
            pygame.init()
            if self.render_mode == 'human':
                self.window = pygame.display.set_mode((self.width, self.height))
                pygame.display.set_caption("Robot Navigation")
            else:
                self.window = pygame.Surface((self.width, self.height))
        
        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        # Clear screen
        self.window.fill((240, 240, 240))
        
        # Draw goal
        pygame.draw.circle(
            self.window,
            (0, 200, 0),
            self.goal_pos.astype(int),
            int(self.goal_radius)
        )
        
        # Draw robot
        pygame.draw.circle(
            self.window,
            (0, 0, 255),
            self.robot_pos.astype(int),
            int(self.robot_radius)
        )
        
        # Draw direction indicator
        end_pos = self.robot_pos + 25 * np.array([
            np.cos(self.robot_angle),
            np.sin(self.robot_angle)
        ])
        pygame.draw.line(
            self.window,
            (255, 0, 0),
            self.robot_pos.astype(int),
            end_pos.astype(int),
            3
        )
        
        if self.render_mode == 'human':
            pygame.display.flip()
            self.clock.tick(self.metadata['render_fps'])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window)),
                axes=(1, 0, 2)
            )
    
    def close(self):
        """Close rendering window"""
        if self.window is not None:
            pygame.quit()
            self.window = None
            self.clock = None
