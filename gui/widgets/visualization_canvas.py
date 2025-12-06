"""
🎨 Visualization Canvas - Center panel with real-time environment rendering
"""

import numpy as np
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QTimer, QPointF
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QPixmap, QPainterPath
import pygame


class VisualizationCanvas(QWidget):
    """Real-time visualization of the environment."""
    
    def __init__(self):
        super().__init__()
        self.setMinimumSize(600, 600)
        
        # State
        self.robot_pos = [400, 300]
        self.robot_angle = 0
        self.goal_pos = [700, 500]
        self.obstacles = []
        self.path_history = []
        self.lidar_rays = []
        
        # Animation
        self.animation_speed = 1.0
        self.show_lidar = True
        self.show_path = True
        self.show_grid = False
        
        # Timer for animation
        self.anim_timer = QTimer()
        self.anim_timer.timeout.connect(self.update)
        self.anim_timer.start(33)  # ~30 FPS
        
        # Colors
        self.bg_color = QColor(30, 30, 30)
        self.robot_color = QColor(13, 115, 119)
        self.goal_color = QColor(255, 215, 0)
        self.obstacle_color = QColor(100, 100, 100)
        self.path_color = QColor(13, 115, 119, 100)
        self.grid_color = QColor(50, 50, 50)
    
    def paintEvent(self, event):
        """Paint the canvas."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Background
        painter.fillRect(self.rect(), self.bg_color)
        
        # Grid
        if self.show_grid:
            self._draw_grid(painter)
        
        # Path history
        if self.show_path and len(self.path_history) > 1:
            self._draw_path(painter)
        
        # LIDAR rays
        if self.show_lidar:
            self._draw_lidar(painter)
        
        # Obstacles
        self._draw_obstacles(painter)
        
        # Goal
        self._draw_goal(painter)
        
        # Robot
        self._draw_robot(painter)
        
        # HUD overlay
        self._draw_hud(painter)
    
    def _draw_grid(self, painter):
        """Draw background grid."""
        pen = QPen(self.grid_color, 1)
        painter.setPen(pen)
        
        # Vertical lines
        for x in range(0, self.width(), 50):
            painter.drawLine(x, 0, x, self.height())
        
        # Horizontal lines
        for y in range(0, self.height(), 50):
            painter.drawLine(0, y, self.width(), y)
    
    def _draw_path(self, painter):
        """Draw robot's path history."""
        if len(self.path_history) < 2:
            return
        
        pen = QPen(self.path_color, 2)
        painter.setPen(pen)
        
        path = QPainterPath()
        path.moveTo(self.path_history[0][0], self.path_history[0][1])
        
        for pos in self.path_history[1:]:
            path.lineTo(pos[0], pos[1])
        
        painter.drawPath(path)
    
    def _draw_lidar(self, painter):
        """Draw LIDAR rays."""
        pen = QPen(QColor(0, 255, 0, 50), 1)
        painter.setPen(pen)
        
        for ray in self.lidar_rays:
            painter.drawLine(
                int(self.robot_pos[0]), int(self.robot_pos[1]),
                int(ray[0]), int(ray[1])
            )
    
    def _draw_obstacles(self, painter):
        """Draw obstacles."""
        brush = QBrush(self.obstacle_color)
        painter.setBrush(brush)
        painter.setPen(Qt.PenStyle.NoPen)
        
        for obs in self.obstacles:
            x, y, w, h = obs
            painter.drawRect(int(x), int(y), int(w), int(h))
    
    def _draw_goal(self, painter):
        """Draw goal with sparkle effect."""
        # Outer glow
        for i in range(3, 0, -1):
            alpha = int(50 / i)
            color = QColor(255, 215, 0, alpha)
            painter.setBrush(QBrush(color))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(
                QPointF(self.goal_pos[0], self.goal_pos[1]),
                20 * i, 20 * i
            )
        
        # Main goal
        painter.setBrush(QBrush(self.goal_color))
        painter.setPen(QPen(Qt.GlobalColor.white, 2))
        painter.drawEllipse(
            QPointF(self.goal_pos[0], self.goal_pos[1]),
            15, 15
        )
        
        # Star sparkle
        painter.setPen(QPen(Qt.GlobalColor.white, 3))
        painter.drawLine(
            int(self.goal_pos[0]), int(self.goal_pos[1] - 10),
            int(self.goal_pos[0]), int(self.goal_pos[1] + 10)
        )
        painter.drawLine(
            int(self.goal_pos[0] - 10), int(self.goal_pos[1]),
            int(self.goal_pos[0] + 10), int(self.goal_pos[1])
        )
    
    def _draw_robot(self, painter):
        """Draw robot with direction indicator."""
        # Robot body
        painter.setBrush(QBrush(self.robot_color))
        painter.setPen(QPen(Qt.GlobalColor.white, 2))
        painter.drawEllipse(
            QPointF(self.robot_pos[0], self.robot_pos[1]),
            20, 20
        )
        
        # Direction indicator
        angle_rad = np.deg2rad(self.robot_angle)
        dx = np.cos(angle_rad) * 25
        dy = np.sin(angle_rad) * 25
        
        painter.setPen(QPen(Qt.GlobalColor.white, 3))
        painter.drawLine(
            int(self.robot_pos[0]), int(self.robot_pos[1]),
            int(self.robot_pos[0] + dx), int(self.robot_pos[1] + dy)
        )
    
    def _draw_hud(self, painter):
        """Draw HUD overlay."""
        painter.setPen(QPen(Qt.GlobalColor.white))
        
        # FPS counter
        painter.drawText(10, 20, "FPS: 60")
        
        # Position
        painter.drawText(
            10, 40,
            f"Position: ({int(self.robot_pos[0])}, {int(self.robot_pos[1])})"
        )
        
        # Distance to goal
        dist = np.linalg.norm(
            np.array(self.robot_pos) - np.array(self.goal_pos)
        )
        painter.drawText(10, 60, f"Distance to Goal: {int(dist)}px")
    
    def update_state(self, state):
        """Update visualization state."""
        if state is None:
            return
        
        # Extract state information
        # TODO: Parse state based on actual structure
        pass
    
    def reset(self):
        """Reset visualization."""
        self.path_history = []
        self.lidar_rays = []
        self.update()
    
    def save_screenshot(self, filename):
        """Save screenshot to file."""
        pixmap = self.grab()
        pixmap.save(filename)
    
    def set_animation_speed(self, speed):
        """Set animation speed (0.5x, 1x, 2x, etc.)."""
        self.animation_speed = speed
    
    def toggle_lidar(self, show):
        """Toggle LIDAR visualization."""
        self.show_lidar = show
        self.update()
    
    def toggle_path(self, show):
        """Toggle path history."""
        self.show_path = show
        self.update()
    
    def toggle_grid(self, show):
        """Toggle background grid."""
        self.show_grid = show
        self.update()
