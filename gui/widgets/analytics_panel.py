"""
📊 Analytics Panel - Right sidebar with real-time metrics and charts
"""

import numpy as np
from collections import deque
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QTabWidget, QLabel,
    QTextEdit, QPushButton
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QColor, QPen
import pyqtgraph as pg


class AnalyticsPanel(QWidget):
    """Analytics panel showing metrics and charts."""
    
    def __init__(self):
        super().__init__()
        self.setMaximumWidth(500)
        
        # Data storage
        self.rewards = deque(maxlen=1000)
        self.steps = deque(maxlen=1000)
        self.successes = deque(maxlen=1000)
        self.losses = deque(maxlen=1000)
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup UI layout."""
        layout = QVBoxLayout(self)
        
        # Real-time metrics section
        metrics_widget = self._create_metrics_widget()
        layout.addWidget(metrics_widget)
        
        # Tabs for different charts
        tabs = QTabWidget()
        
        # Rewards tab
        rewards_tab = self._create_rewards_tab()
        tabs.addTab(rewards_tab, "📈 Rewards")
        
        # Steps tab
        steps_tab = self._create_steps_tab()
        tabs.addTab(steps_tab, "👣 Steps")
        
        # Success rate tab
        success_tab = self._create_success_tab()
        tabs.addTab(success_tab, "✅ Success")
        
        # Logs tab
        logs_tab = self._create_logs_tab()
        tabs.addTab(logs_tab, "📝 Logs")
        
        layout.addWidget(tabs, stretch=1)
    
    def _create_metrics_widget(self):
        """Create real-time metrics display."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Title
        title = QLabel("📊 Real-Time Metrics")
        title.setStyleSheet("font-size: 14pt; font-weight: bold;")
        layout.addWidget(title)
        
        # Current reward (large display)
        self.reward_label = QLabel("0.00")
        self.reward_label.setStyleSheet("""
            QLabel {
                font-size: 36pt;
                font-weight: bold;
                color: #0d7377;
                background-color: #2d2d2d;
                border-radius: 10px;
                padding: 20px;
            }
        """)
        self.reward_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.reward_label)
        
        # Episode number
        self.episode_label = QLabel("Episode: 0 / 0")
        self.episode_label.setStyleSheet("font-size: 12pt;")
        layout.addWidget(self.episode_label)
        
        # Steps taken
        self.steps_label = QLabel("Steps: 0")
        self.steps_label.setStyleSheet("font-size: 12pt;")
        layout.addWidget(self.steps_label)
        
        # Success rate
        self.success_label = QLabel("Success Rate: 0.0%")
        self.success_label.setStyleSheet("font-size: 12pt;")
        layout.addWidget(self.success_label)
        
        # Average reward (last 100)
        self.avg_reward_label = QLabel("Avg Reward (100): 0.00")
        self.avg_reward_label.setStyleSheet("font-size: 12pt;")
        layout.addWidget(self.avg_reward_label)
        
        return widget
    
    def _create_rewards_tab(self):
        """Create rewards chart tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # PyQtGraph plot widget
        self.rewards_plot = pg.PlotWidget()
        self.rewards_plot.setBackground('#1e1e1e')
        self.rewards_plot.setLabel('left', 'Reward')
        self.rewards_plot.setLabel('bottom', 'Episode')
        self.rewards_plot.showGrid(x=True, y=True, alpha=0.3)
        
        # Plot curves
        self.rewards_curve = self.rewards_plot.plot(
            pen=pg.mkPen(color=(13, 115, 119), width=2)
        )
        self.rewards_avg_curve = self.rewards_plot.plot(
            pen=pg.mkPen(color=(255, 0, 0), width=2, style=Qt.PenStyle.DashLine)
        )
        
        layout.addWidget(self.rewards_plot)
        
        return widget
    
    def _create_steps_tab(self):
        """Create steps chart tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.steps_plot = pg.PlotWidget()
        self.steps_plot.setBackground('#1e1e1e')
        self.steps_plot.setLabel('left', 'Steps')
        self.steps_plot.setLabel('bottom', 'Episode')
        self.steps_plot.showGrid(x=True, y=True, alpha=0.3)
        
        self.steps_curve = self.steps_plot.plot(
            pen=pg.mkPen(color=(255, 165, 0), width=2)
        )
        
        layout.addWidget(self.steps_plot)
        
        return widget
    
    def _create_success_tab(self):
        """Create success rate chart tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.success_plot = pg.PlotWidget()
        self.success_plot.setBackground('#1e1e1e')
        self.success_plot.setLabel('left', 'Success Rate (%)')
        self.success_plot.setLabel('bottom', 'Episode')
        self.success_plot.showGrid(x=True, y=True, alpha=0.3)
        self.success_plot.setYRange(0, 100)
        
        self.success_curve = self.success_plot.plot(
            pen=pg.mkPen(color=(0, 255, 0), width=2),
            fillLevel=0,
            brush=(0, 255, 0, 50)
        )
        
        layout.addWidget(self.success_plot)
        
        return widget
    
    def _create_logs_tab(self):
        """Create logs tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Log text area
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #ffffff;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 10pt;
            }
        """)
        layout.addWidget(self.log_text)
        
        # Export button
        export_btn = QPushButton("📤 Export Logs")
        export_btn.clicked.connect(self._export_logs)
        layout.addWidget(export_btn)
        
        return widget
    
    def add_data_point(self, episode, reward, data):
        """Add new data point."""
        self.rewards.append(reward)
        self.steps.append(data.get('steps', 0))
        self.successes.append(data.get('success', 0))
        
        # Update metrics
        self.reward_label.setText(f"{reward:.2f}")
        self.episode_label.setText(f"Episode: {episode} / {data.get('total_episodes', 0)}")
        self.steps_label.setText(f"Steps: {data.get('steps', 0)}")
        
        # Calculate success rate (last 100)
        if len(self.successes) > 0:
            recent_successes = list(self.successes)[-100:]
            success_rate = np.mean(recent_successes) * 100
            self.success_label.setText(f"Success Rate: {success_rate:.1f}%")
        
        # Calculate average reward (last 100)
        if len(self.rewards) > 0:
            recent_rewards = list(self.rewards)[-100:]
            avg_reward = np.mean(recent_rewards)
            self.avg_reward_label.setText(f"Avg Reward (100): {avg_reward:.2f}")
        
        # Update plots
        self._update_plots()
        
        # Add log entry
        self.log(f"Episode {episode}: Reward={reward:.2f}, Steps={data.get('steps', 0)}")
    
    def _update_plots(self):
        """Update all plots."""
        if len(self.rewards) == 0:
            return
        
        episodes = list(range(len(self.rewards)))
        
        # Update rewards plot
        self.rewards_curve.setData(episodes, list(self.rewards))
        
        # Moving average
        if len(self.rewards) >= 50:
            ma = np.convolve(self.rewards, np.ones(50)/50, mode='valid')
            self.rewards_avg_curve.setData(range(49, len(self.rewards)), ma)
        
        # Update steps plot
        self.steps_curve.setData(episodes, list(self.steps))
        
        # Update success plot (moving average)
        if len(self.successes) >= 50:
            success_ma = np.convolve(self.successes, np.ones(50)/50, mode='valid') * 100
            self.success_curve.setData(range(49, len(self.successes)), success_ma)
    
    def log(self, message):
        """Add log message."""
        self.log_text.append(message)
        # Auto-scroll to bottom
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def clear(self):
        """Clear all data."""
        self.rewards.clear()
        self.steps.clear()
        self.successes.clear()
        self.losses.clear()
        
        self.reward_label.setText("0.00")
        self.episode_label.setText("Episode: 0 / 0")
        self.steps_label.setText("Steps: 0")
        self.success_label.setText("Success Rate: 0.0%")
        self.avg_reward_label.setText("Avg Reward (100): 0.00")
        
        self.log_text.clear()
        self._update_plots()
    
    def _export_logs(self):
        """Export logs to file."""
        logs = self.log_text.toPlainText()
        # TODO: Save to file
        print("Exporting logs...")
