"""
Dynamic Obstacles for Navigation Environment
"""
import numpy as np
from typing import List, Tuple


class Obstacle:
    """Base obstacle class"""
    
    def __init__(self, x: float, y: float, radius: float):
        self.pos = np.array([x, y], dtype=np.float32)
        self.radius = radius
    
    def update(self, dt: float):
        """Update obstacle position"""
        pass
    
    def check_collision(self, point: np.ndarray, point_radius: float = 0) -> bool:
        """Check if point collides with obstacle"""
        distance = np.linalg.norm(point - self.pos)
        return distance < (self.radius + point_radius)


class StaticObstacle(Obstacle):
    """Static obstacle that doesn't move"""
    pass


class DynamicObstacle(Obstacle):
    """Moving obstacle"""
    
    def __init__(
        self,
        x: float,
        y: float,
        radius: float,
        velocity: np.ndarray,
        bounds: Tuple[float, float, float, float]
    ):
        super().__init__(x, y, radius)
        self.velocity = np.array(velocity, dtype=np.float32)
        self.bounds = bounds  # (min_x, max_x, min_y, max_y)
    
    def update(self, dt: float):
        """Update position with boundary bouncing"""
        self.pos += self.velocity * dt
        
        # Bounce off boundaries
        if self.pos[0] - self.radius < self.bounds[0] or \
           self.pos[0] + self.radius > self.bounds[1]:
            self.velocity[0] *= -1
            self.pos[0] = np.clip(
                self.pos[0],
                self.bounds[0] + self.radius,
                self.bounds[1] - self.radius
            )
        
        if self.pos[1] - self.radius < self.bounds[2] or \
           self.pos[1] + self.radius > self.bounds[3]:
            self.velocity[1] *= -1
            self.pos[1] = np.clip(
                self.pos[1],
                self.bounds[2] + self.radius,
                self.bounds[3] - self.radius
            )


class CircularObstacle(DynamicObstacle):
    """Obstacle moving in circular pattern"""
    
    def __init__(
        self,
        center_x: float,
        center_y: float,
        orbit_radius: float,
        obstacle_radius: float,
        angular_velocity: float,
        initial_angle: float = 0
    ):
        x = center_x + orbit_radius * np.cos(initial_angle)
        y = center_y + orbit_radius * np.sin(initial_angle)
        
        super().__init__(x, y, obstacle_radius, np.array([0.0, 0.0]), (0, 0, 0, 0))
        
        self.center = np.array([center_x, center_y], dtype=np.float32)
        self.orbit_radius = orbit_radius
        self.angular_velocity = angular_velocity
        self.angle = initial_angle
    
    def update(self, dt: float):
        """Update circular motion"""
        self.angle += self.angular_velocity * dt
        self.pos[0] = self.center[0] + self.orbit_radius * np.cos(self.angle)
        self.pos[1] = self.center[1] + self.orbit_radius * np.sin(self.angle)


class ObstacleManager:
    """Manages multiple obstacles"""
    
    def __init__(self):
        self.obstacles: List[Obstacle] = []
    
    def add_obstacle(self, obstacle: Obstacle):
        """Add obstacle to manager"""
        self.obstacles.append(obstacle)
    
    def add_static(self, x: float, y: float, radius: float):
        """Add static obstacle"""
        self.obstacles.append(StaticObstacle(x, y, radius))
    
    def add_dynamic(
        self,
        x: float,
        y: float,
        radius: float,
        velocity: np.ndarray,
        bounds: Tuple[float, float, float, float]
    ):
        """Add dynamic obstacle"""
        self.obstacles.append(
            DynamicObstacle(x, y, radius, velocity, bounds)
        )
    
    def add_circular(
        self,
        center_x: float,
        center_y: float,
        orbit_radius: float,
        obstacle_radius: float,
        angular_velocity: float,
        initial_angle: float = 0
    ):
        """Add circular obstacle"""
        self.obstacles.append(
            CircularObstacle(
                center_x, center_y, orbit_radius,
                obstacle_radius, angular_velocity, initial_angle
            )
        )
    
    def update(self, dt: float = 1.0):
        """Update all obstacles"""
        for obstacle in self.obstacles:
            obstacle.update(dt)
    
    def check_collision(self, point: np.ndarray, radius: float = 0) -> bool:
        """Check if point collides with any obstacle"""
        return any(
            obs.check_collision(point, radius)
            for obs in self.obstacles
        )
    
    def get_closest_obstacle(self, point: np.ndarray) -> Tuple[float, np.ndarray]:
        """Get distance and position of closest obstacle"""
        if not self.obstacles:
            return float('inf'), np.array([0, 0])
        
        distances = [
            np.linalg.norm(obs.pos - point)
            for obs in self.obstacles
        ]
        
        closest_idx = np.argmin(distances)
        return distances[closest_idx], self.obstacles[closest_idx].pos
    
    def clear(self):
        """Remove all obstacles"""
        self.obstacles.clear()
