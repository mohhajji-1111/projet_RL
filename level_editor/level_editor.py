"""
🗺️ Level Editor - Create custom navigation scenarios
"""

import json
from pathlib import Path
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QToolBar, QStatusBar, QPushButton, QFileDialog,
    QMessageBox, QLabel, QSpinBox, QComboBox
)
from PyQt6.QtCore import Qt, QPoint, QRect
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QMouseEvent


class LevelEditorCanvas(QWidget):
    """Canvas for level editing."""
    
    def __init__(self):
        super().__init__()
        self.setMinimumSize(800, 600)
        
        # Grid settings
        self.grid_size = 20
        self.show_grid = True
        self.snap_to_grid = True
        
        # Tools
        self.current_tool = "obstacle"  # obstacle, goal, robot, eraser
        self.drawing = False
        
        # Level data
        self.obstacles = []
        self.goals = []
        self.robot_start = [100, 100]
        self.level_size = [800, 600]
        
        # Drawing state
        self.drag_start = None
        self.temp_rect = None
    
    def paintEvent(self, event):
        """Paint the canvas."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Background
        painter.fillRect(self.rect(), QColor(40, 40, 40))
        
        # Grid
        if self.show_grid:
            self._draw_grid(painter)
        
        # Obstacles
        painter.setBrush(QBrush(QColor(100, 100, 100)))
        painter.setPen(QPen(QColor(150, 150, 150), 2))
        for obs in self.obstacles:
            painter.drawRect(obs['x'], obs['y'], obs['width'], obs['height'])
        
        # Goals
        painter.setBrush(QBrush(QColor(255, 215, 0)))
        painter.setPen(QPen(QColor(255, 255, 0), 2))
        for goal in self.goals:
            painter.drawEllipse(goal['x'] - 15, goal['y'] - 15, 30, 30)
        
        # Robot start position
        painter.setBrush(QBrush(QColor(13, 115, 119)))
        painter.setPen(QPen(QColor(255, 255, 255), 2))
        painter.drawEllipse(
            self.robot_start[0] - 20, self.robot_start[1] - 20, 40, 40
        )
        
        # Temporary drawing
        if self.temp_rect and self.current_tool == "obstacle":
            painter.setBrush(QBrush(QColor(100, 100, 100, 100)))
            painter.setPen(QPen(QColor(150, 150, 150), 2, Qt.PenStyle.DashLine))
            painter.drawRect(self.temp_rect)
    
    def _draw_grid(self, painter):
        """Draw grid."""
        pen = QPen(QColor(60, 60, 60), 1)
        painter.setPen(pen)
        
        # Vertical lines
        for x in range(0, self.width(), self.grid_size):
            painter.drawLine(x, 0, x, self.height())
        
        # Horizontal lines
        for y in range(0, self.height(), self.grid_size):
            painter.drawLine(0, y, self.width(), y)
    
    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press."""
        pos = event.pos()
        
        if self.snap_to_grid:
            pos = QPoint(
                (pos.x() // self.grid_size) * self.grid_size,
                (pos.y() // self.grid_size) * self.grid_size
            )
        
        if self.current_tool == "obstacle":
            self.drawing = True
            self.drag_start = pos
        elif self.current_tool == "goal":
            self.goals.append({'x': pos.x(), 'y': pos.y()})
            self.update()
        elif self.current_tool == "robot":
            self.robot_start = [pos.x(), pos.y()]
            self.update()
        elif self.current_tool == "eraser":
            self._erase_at(pos)
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move."""
        if self.drawing and self.drag_start:
            pos = event.pos()
            
            if self.snap_to_grid:
                pos = QPoint(
                    (pos.x() // self.grid_size) * self.grid_size,
                    (pos.y() // self.grid_size) * self.grid_size
                )
            
            x = min(self.drag_start.x(), pos.x())
            y = min(self.drag_start.y(), pos.y())
            w = abs(pos.x() - self.drag_start.x())
            h = abs(pos.y() - self.drag_start.y())
            
            self.temp_rect = QRect(x, y, w, h)
            self.update()
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release."""
        if self.drawing and self.temp_rect:
            if self.temp_rect.width() > 10 and self.temp_rect.height() > 10:
                self.obstacles.append({
                    'x': self.temp_rect.x(),
                    'y': self.temp_rect.y(),
                    'width': self.temp_rect.width(),
                    'height': self.temp_rect.height()
                })
            
            self.drawing = False
            self.drag_start = None
            self.temp_rect = None
            self.update()
    
    def _erase_at(self, pos):
        """Erase element at position."""
        # Check obstacles
        self.obstacles = [
            obs for obs in self.obstacles
            if not (obs['x'] <= pos.x() <= obs['x'] + obs['width'] and
                   obs['y'] <= pos.y() <= obs['y'] + obs['height'])
        ]
        
        # Check goals
        self.goals = [
            goal for goal in self.goals
            if ((goal['x'] - pos.x())**2 + (goal['y'] - pos.y())**2) > 225  # 15^2
        ]
        
        self.update()
    
    def clear_all(self):
        """Clear all elements."""
        self.obstacles = []
        self.goals = []
        self.robot_start = [100, 100]
        self.update()
    
    def get_level_data(self):
        """Get level data as dict."""
        return {
            'size': self.level_size,
            'robot_start': self.robot_start,
            'obstacles': self.obstacles,
            'goals': self.goals
        }
    
    def load_level_data(self, data):
        """Load level data."""
        self.level_size = data.get('size', [800, 600])
        self.robot_start = data.get('robot_start', [100, 100])
        self.obstacles = data.get('obstacles', [])
        self.goals = data.get('goals', [])
        self.update()


class LevelEditor(QMainWindow):
    """Main level editor window."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🗺️ Level Editor")
        self.setGeometry(100, 100, 1000, 700)
        
        self.canvas = LevelEditorCanvas()
        self.setCentralWidget(self.canvas)
        
        self._setup_toolbar()
        self._setup_statusbar()
    
    def _setup_toolbar(self):
        """Setup toolbar."""
        toolbar = QToolBar()
        self.addToolBar(toolbar)
        
        # Tool selection
        tools_label = QLabel("Tool: ")
        toolbar.addWidget(tools_label)
        
        self.tool_combo = QComboBox()
        self.tool_combo.addItems(["Obstacle", "Goal", "Robot", "Eraser"])
        self.tool_combo.currentTextChanged.connect(self._on_tool_changed)
        toolbar.addWidget(self.tool_combo)
        
        toolbar.addSeparator()
        
        # Grid toggle
        grid_btn = QPushButton("📐 Toggle Grid")
        grid_btn.clicked.connect(self._toggle_grid)
        toolbar.addWidget(grid_btn)
        
        # Snap toggle
        snap_btn = QPushButton("🧲 Toggle Snap")
        snap_btn.clicked.connect(self._toggle_snap)
        toolbar.addWidget(snap_btn)
        
        toolbar.addSeparator()
        
        # Clear
        clear_btn = QPushButton("🗑️ Clear All")
        clear_btn.clicked.connect(self._clear_all)
        toolbar.addWidget(clear_btn)
        
        toolbar.addSeparator()
        
        # Save/Load
        save_btn = QPushButton("💾 Save")
        save_btn.clicked.connect(self._save_level)
        toolbar.addWidget(save_btn)
        
        load_btn = QPushButton("📂 Load")
        load_btn.clicked.connect(self._load_level)
        toolbar.addWidget(load_btn)
        
        toolbar.addSeparator()
        
        # Test
        test_btn = QPushButton("▶️ Test Level")
        test_btn.clicked.connect(self._test_level)
        toolbar.addWidget(test_btn)
    
    def _setup_statusbar(self):
        """Setup status bar."""
        self.statusBar().showMessage("Ready - Select a tool to start editing")
    
    def _on_tool_changed(self, tool):
        """Handle tool change."""
        self.canvas.current_tool = tool.lower()
        self.statusBar().showMessage(f"Tool: {tool}")
    
    def _toggle_grid(self):
        """Toggle grid visibility."""
        self.canvas.show_grid = not self.canvas.show_grid
        self.canvas.update()
    
    def _toggle_snap(self):
        """Toggle snap to grid."""
        self.canvas.snap_to_grid = not self.canvas.snap_to_grid
        self.statusBar().showMessage(
            f"Snap to grid: {'ON' if self.canvas.snap_to_grid else 'OFF'}"
        )
    
    def _clear_all(self):
        """Clear all."""
        reply = QMessageBox.question(
            self, "Clear All",
            "Are you sure you want to clear all elements?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.canvas.clear_all()
    
    def _save_level(self):
        """Save level to file."""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Level", "", "JSON Files (*.json)"
        )
        if filename:
            data = self.canvas.get_level_data()
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            self.statusBar().showMessage(f"Saved: {filename}")
    
    def _load_level(self):
        """Load level from file."""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Level", "", "JSON Files (*.json)"
        )
        if filename:
            with open(filename, 'r') as f:
                data = json.load(f)
            self.canvas.load_level_data(data)
            self.statusBar().showMessage(f"Loaded: {filename}")
    
    def _test_level(self):
        """Test current level."""
        QMessageBox.information(
            self, "Test Level",
            "Testing level... (This would launch the environment)"
        )


if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    editor = LevelEditor()
    editor.show()
    sys.exit(app.exec())
