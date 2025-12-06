"""
Advanced Renderer for Navigation Environment
"""
import pygame
import numpy as np
from typing import Optional, Tuple


class Renderer:
    """Advanced graphics renderer"""
    
    def __init__(
        self,
        width: int = 800,
        height: int = 600,
        fps: int = 30
    ):
        self.width = width
        self.height = height
        self.fps = fps
        
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Robot Navigation RL")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        # Colors
        self.colors = {
            'background': (240, 240, 245),
            'robot': (64, 128, 255),
            'robot_direction': (255, 64, 64),
            'goal': (64, 255, 128),
            'obstacle': (128, 128, 128),
            'dynamic_obstacle': (255, 128, 64),
            'lidar': (255, 200, 0, 100),
            'path': (200, 200, 255),
            'text': (50, 50, 50),
            'grid': (220, 220, 225)
        }
        
        # Trajectory tracking
        self.trajectory = []
        self.max_trajectory_length = 200
        
    def clear(self):
        """Clear screen"""
        self.screen.fill(self.colors['background'])
    
    def draw_grid(self, cell_size: int = 50):
        """Draw background grid"""
        for x in range(0, self.width, cell_size):
            pygame.draw.line(
                self.screen,
                self.colors['grid'],
                (x, 0),
                (x, self.height),
                1
            )
        
        for y in range(0, self.height, cell_size):
            pygame.draw.line(
                self.screen,
                self.colors['grid'],
                (0, y),
                (self.width, y),
                1
            )
    
    def draw_robot(
        self,
        pos: np.ndarray,
        angle: float,
        radius: float = 15.0,
        show_direction: bool = True
    ):
        """Draw robot"""
        # Robot body
        pygame.draw.circle(
            self.screen,
            self.colors['robot'],
            pos.astype(int),
            int(radius)
        )
        
        # Robot outline
        pygame.draw.circle(
            self.screen,
            (0, 0, 0),
            pos.astype(int),
            int(radius),
            2
        )
        
        # Direction indicator
        if show_direction:
            end_pos = pos + (radius + 15) * np.array([
                np.cos(angle),
                np.sin(angle)
            ])
            pygame.draw.line(
                self.screen,
                self.colors['robot_direction'],
                pos.astype(int),
                end_pos.astype(int),
                4
            )
    
    def draw_goal(self, pos: np.ndarray, radius: float = 20.0):
        """Draw goal with pulsing effect"""
        # Outer glow
        pygame.draw.circle(
            self.screen,
            (100, 255, 150),
            pos.astype(int),
            int(radius * 1.3),
            3
        )
        
        # Goal body
        pygame.draw.circle(
            self.screen,
            self.colors['goal'],
            pos.astype(int),
            int(radius)
        )
        
        # Center point
        pygame.draw.circle(
            self.screen,
            (0, 200, 100),
            pos.astype(int),
            int(radius // 2)
        )
    
    def draw_obstacle(
        self,
        pos: np.ndarray,
        radius: float,
        is_dynamic: bool = False
    ):
        """Draw obstacle"""
        color = self.colors['dynamic_obstacle'] if is_dynamic else self.colors['obstacle']
        
        pygame.draw.circle(
            self.screen,
            color,
            pos.astype(int),
            int(radius)
        )
        
        pygame.draw.circle(
            self.screen,
            (0, 0, 0),
            pos.astype(int),
            int(radius),
            2
        )
    
    def draw_lidar(
        self,
        robot_pos: np.ndarray,
        robot_angle: float,
        distances: np.ndarray,
        beam_angles: np.ndarray,
        max_range: float
    ):
        """Draw LIDAR beams"""
        surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        
        for beam_angle, distance in zip(beam_angles, distances):
            absolute_angle = robot_angle + beam_angle
            end_pos = robot_pos + distance * np.array([
                np.cos(absolute_angle),
                np.sin(absolute_angle)
            ])
            
            # Beam line
            alpha = int(255 * (1 - distance / max_range))
            color = (*self.colors['lidar'][:3], min(alpha, 150))
            
            pygame.draw.line(
                surface,
                color,
                robot_pos.astype(int),
                end_pos.astype(int),
                1
            )
            
            # Hit point
            if distance < max_range:
                pygame.draw.circle(
                    surface,
                    (255, 200, 0),
                    end_pos.astype(int),
                    3
                )
        
        self.screen.blit(surface, (0, 0))
    
    def draw_trajectory(self, pos: np.ndarray):
        """Draw robot trajectory"""
        self.trajectory.append(pos.copy())
        
        if len(self.trajectory) > self.max_trajectory_length:
            self.trajectory.pop(0)
        
        if len(self.trajectory) > 1:
            points = [p.astype(int) for p in self.trajectory]
            for i in range(1, len(points)):
                alpha = int(255 * i / len(points))
                color = (*self.colors['path'][:3], alpha)
                
                surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
                pygame.draw.line(surface, color, points[i-1], points[i], 2)
                self.screen.blit(surface, (0, 0))
    
    def draw_text(
        self,
        text: str,
        pos: Tuple[int, int],
        color: Optional[Tuple] = None,
        small: bool = False
    ):
        """Draw text"""
        if color is None:
            color = self.colors['text']
        
        font = self.small_font if small else self.font
        text_surface = font.render(text, True, color)
        self.screen.blit(text_surface, pos)
    
    def draw_stats(
        self,
        episode: int,
        reward: float,
        distance: float,
        epsilon: Optional[float] = None
    ):
        """Draw training statistics"""
        y_offset = 10
        line_height = 25
        
        self.draw_text(f"Episode: {episode}", (10, y_offset))
        y_offset += line_height
        
        self.draw_text(f"Reward: {reward:.2f}", (10, y_offset))
        y_offset += line_height
        
        self.draw_text(f"Distance: {distance:.1f}", (10, y_offset))
        y_offset += line_height
        
        if epsilon is not None:
            self.draw_text(f"Epsilon: {epsilon:.3f}", (10, y_offset))
    
    def update(self):
        """Update display"""
        pygame.display.flip()
        self.clock.tick(self.fps)
    
    def clear_trajectory(self):
        """Clear trajectory history"""
        self.trajectory.clear()
    
    def close(self):
        """Close renderer"""
        pygame.quit()
