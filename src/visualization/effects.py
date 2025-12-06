"""
Particle Effects System
"""
import pygame
import numpy as np
from typing import List, Tuple


class Particle:
    """Single particle"""
    
    def __init__(
        self,
        pos: np.ndarray,
        velocity: np.ndarray,
        color: Tuple[int, int, int],
        lifetime: float,
        size: float = 3.0
    ):
        self.pos = pos.copy()
        self.velocity = velocity.copy()
        self.color = color
        self.lifetime = lifetime
        self.max_lifetime = lifetime
        self.size = size
    
    def update(self, dt: float):
        """Update particle"""
        self.pos += self.velocity * dt
        self.velocity *= 0.98  # Friction
        self.lifetime -= dt
    
    def is_alive(self) -> bool:
        """Check if particle is still alive"""
        return self.lifetime > 0
    
    def get_alpha(self) -> int:
        """Get alpha based on lifetime"""
        return int(255 * (self.lifetime / self.max_lifetime))


class ParticleSystem:
    """Particle system manager"""
    
    def __init__(self):
        self.particles: List[Particle] = []
    
    def emit(
        self,
        pos: np.ndarray,
        count: int,
        color: Tuple[int, int, int],
        speed_range: Tuple[float, float] = (1, 5),
        lifetime_range: Tuple[float, float] = (0.5, 1.5),
        size: float = 3.0
    ):
        """Emit particles"""
        for _ in range(count):
            angle = np.random.uniform(0, 2 * np.pi)
            speed = np.random.uniform(*speed_range)
            velocity = speed * np.array([np.cos(angle), np.sin(angle)])
            lifetime = np.random.uniform(*lifetime_range)
            
            particle = Particle(pos, velocity, color, lifetime, size)
            self.particles.append(particle)
    
    def update(self, dt: float = 1.0):
        """Update all particles"""
        for particle in self.particles[:]:
            particle.update(dt)
            if not particle.is_alive():
                self.particles.remove(particle)
    
    def render(self, screen: pygame.Surface):
        """Render all particles"""
        surface = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
        
        for particle in self.particles:
            alpha = particle.get_alpha()
            color = (*particle.color, alpha)
            
            pygame.draw.circle(
                surface,
                color,
                particle.pos.astype(int),
                int(particle.size)
            )
        
        screen.blit(surface, (0, 0))
    
    def clear(self):
        """Remove all particles"""
        self.particles.clear()


class EffectManager:
    """Manage various visual effects"""
    
    def __init__(self):
        self.particle_system = ParticleSystem()
    
    def goal_reached_effect(self, pos: np.ndarray):
        """Goal reached celebration effect"""
        # Green explosion
        self.particle_system.emit(
            pos,
            count=50,
            color=(64, 255, 128),
            speed_range=(5, 15),
            lifetime_range=(1.0, 2.0),
            size=5.0
        )
        
        # Yellow sparkles
        self.particle_system.emit(
            pos,
            count=30,
            color=(255, 255, 64),
            speed_range=(3, 10),
            lifetime_range=(0.5, 1.5),
            size=3.0
        )
    
    def collision_effect(self, pos: np.ndarray, velocity: np.ndarray):
        """Collision effect"""
        # Red particles in opposite direction
        direction = -velocity / (np.linalg.norm(velocity) + 1e-6)
        
        for _ in range(10):
            angle_offset = np.random.uniform(-np.pi/4, np.pi/4)
            angle = np.arctan2(direction[1], direction[0]) + angle_offset
            speed = np.random.uniform(3, 8)
            vel = speed * np.array([np.cos(angle), np.sin(angle)])
            
            particle = Particle(
                pos.copy(),
                vel,
                (255, 64, 64),
                lifetime=0.5,
                size=4.0
            )
            self.particle_system.particles.append(particle)
    
    def motion_trail(self, pos: np.ndarray, velocity: np.ndarray):
        """Motion trail effect"""
        if np.linalg.norm(velocity) > 1.0:
            # Blue trail particles
            direction = -velocity / (np.linalg.norm(velocity) + 1e-6)
            trail_pos = pos + direction * 10
            
            self.particle_system.emit(
                trail_pos,
                count=2,
                color=(100, 150, 255),
                speed_range=(0.5, 1.5),
                lifetime_range=(0.3, 0.6),
                size=2.0
            )
    
    def update(self, dt: float = 1.0):
        """Update all effects"""
        self.particle_system.update(dt)
    
    def render(self, screen: pygame.Surface):
        """Render all effects"""
        self.particle_system.render(screen)
    
    def clear(self):
        """Clear all effects"""
        self.particle_system.clear()
