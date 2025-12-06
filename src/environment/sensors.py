"""
LIDAR Sensor Simulation
"""
import numpy as np
from typing import List, Tuple


class LIDARSensor:
    """Simulated LIDAR sensor for obstacle detection"""
    
    def __init__(
        self,
        num_beams: int = 16,
        max_range: float = 200.0,
        field_of_view: float = 2 * np.pi,
        noise_std: float = 0.01
    ):
        """
        Args:
            num_beams: Number of laser beams
            max_range: Maximum detection range
            field_of_view: Field of view in radians
            noise_std: Standard deviation of measurement noise
        """
        self.num_beams = num_beams
        self.max_range = max_range
        self.field_of_view = field_of_view
        self.noise_std = noise_std
        
        # Precompute beam angles
        start_angle = -field_of_view / 2
        end_angle = field_of_view / 2
        self.beam_angles = np.linspace(start_angle, end_angle, num_beams)
    
    def scan(
        self,
        robot_pos: np.ndarray,
        robot_angle: float,
        obstacles: list,
        bounds: Tuple[float, float, float, float]
    ) -> np.ndarray:
        """
        Perform LIDAR scan
        
        Args:
            robot_pos: Robot position [x, y]
            robot_angle: Robot orientation in radians
            obstacles: List of obstacle objects
            bounds: Environment bounds (min_x, max_x, min_y, max_y)
        
        Returns:
            distances: Array of distances for each beam
        """
        distances = np.full(self.num_beams, self.max_range)
        
        for i, beam_angle in enumerate(self.beam_angles):
            # Calculate absolute beam angle
            absolute_angle = robot_angle + beam_angle
            
            # Beam direction
            direction = np.array([
                np.cos(absolute_angle),
                np.sin(absolute_angle)
            ])
            
            # Check intersection with obstacles
            min_distance = self.max_range
            
            # Check circular obstacles
            for obstacle in obstacles:
                dist = self._ray_circle_intersection(
                    robot_pos,
                    direction,
                    obstacle.pos,
                    obstacle.radius
                )
                if dist is not None and dist < min_distance:
                    min_distance = dist
            
            # Check boundaries
            boundary_dist = self._ray_boundary_intersection(
                robot_pos, direction, bounds
            )
            if boundary_dist < min_distance:
                min_distance = boundary_dist
            
            # Add noise
            if min_distance < self.max_range:
                noise = np.random.normal(0, self.noise_std * min_distance)
                min_distance = max(0, min_distance + noise)
            
            distances[i] = min_distance
        
        return distances
    
    def _ray_circle_intersection(
        self,
        ray_origin: np.ndarray,
        ray_direction: np.ndarray,
        circle_center: np.ndarray,
        circle_radius: float
    ) -> float:
        """
        Calculate ray-circle intersection distance
        
        Returns:
            Distance to intersection or None if no intersection
        """
        # Vector from ray origin to circle center
        oc = ray_origin - circle_center
        
        # Quadratic equation coefficients
        a = np.dot(ray_direction, ray_direction)
        b = 2.0 * np.dot(oc, ray_direction)
        c = np.dot(oc, oc) - circle_radius**2
        
        discriminant = b**2 - 4*a*c
        
        if discriminant < 0:
            return None
        
        # Calculate distances
        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2*a)
        t2 = (-b + sqrt_disc) / (2*a)
        
        # Return closest positive intersection
        if t1 > 0:
            return t1
        elif t2 > 0:
            return t2
        else:
            return None
    
    def _ray_boundary_intersection(
        self,
        ray_origin: np.ndarray,
        ray_direction: np.ndarray,
        bounds: Tuple[float, float, float, float]
    ) -> float:
        """Calculate distance to boundary intersection"""
        min_x, max_x, min_y, max_y = bounds
        min_dist = self.max_range
        
        # Check each boundary
        # Left boundary
        if ray_direction[0] < 0:
            t = (min_x - ray_origin[0]) / ray_direction[0]
            if t > 0:
                y = ray_origin[1] + t * ray_direction[1]
                if min_y <= y <= max_y:
                    min_dist = min(min_dist, t)
        
        # Right boundary
        if ray_direction[0] > 0:
            t = (max_x - ray_origin[0]) / ray_direction[0]
            if t > 0:
                y = ray_origin[1] + t * ray_direction[1]
                if min_y <= y <= max_y:
                    min_dist = min(min_dist, t)
        
        # Bottom boundary
        if ray_direction[1] < 0:
            t = (min_y - ray_origin[1]) / ray_direction[1]
            if t > 0:
                x = ray_origin[0] + t * ray_direction[0]
                if min_x <= x <= max_x:
                    min_dist = min(min_dist, t)
        
        # Top boundary
        if ray_direction[1] > 0:
            t = (max_y - ray_origin[1]) / ray_direction[1]
            if t > 0:
                x = ray_origin[0] + t * ray_direction[0]
                if min_x <= x <= max_x:
                    min_dist = min(min_dist, t)
        
        return min_dist
    
    def get_beam_endpoints(
        self,
        robot_pos: np.ndarray,
        robot_angle: float,
        distances: np.ndarray
    ) -> np.ndarray:
        """
        Get endpoint positions of LIDAR beams for visualization
        
        Returns:
            Array of shape (num_beams, 2) with endpoint positions
        """
        endpoints = np.zeros((self.num_beams, 2))
        
        for i, (beam_angle, distance) in enumerate(zip(self.beam_angles, distances)):
            absolute_angle = robot_angle + beam_angle
            endpoints[i] = robot_pos + distance * np.array([
                np.cos(absolute_angle),
                np.sin(absolute_angle)
            ])
        
        return endpoints
