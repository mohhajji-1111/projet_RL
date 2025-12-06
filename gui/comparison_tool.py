"""
🎯 Model Comparison Tool - Compare multiple trained models
"""

import numpy as np
from pathlib import Path
import torch
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QListWidget, QLabel, QTableWidget,
    QTableWidgetItem
)
from PyQt6.QtCore import Qt
import pyqtgraph as pg


class ModelComparisonTool(QMainWindow):
    """Tool for comparing multiple models side-by-side."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("⚖️ Model Comparison Tool")
        self.setGeometry(100, 100, 1400, 800)
        
        self.selected_models = []
        self.comparison_data = {}
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup UI layout."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QHBoxLayout(central_widget)
        
        # Left: Model selection
        left_panel = self._create_selection_panel()
        layout.addWidget(left_panel, stretch=1)
        
        # Right: Comparison display
        right_panel = self._create_comparison_panel()
        layout.addWidget(right_panel, stretch=3)
    
    def _create_selection_panel(self):
        """Create model selection panel."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        title = QLabel("📂 Select Models (2-4)")
        title.setStyleSheet("font-size: 14pt; font-weight: bold;")
        layout.addWidget(title)
        
        # Model list
        self.model_list = QListWidget()
        self.model_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        
        # Load available models
        models_dir = Path("checkpoints")
        if models_dir.exists():
            for model_file in models_dir.glob("*.pt"):
                self.model_list.addItem(model_file.name)
        
        layout.addWidget(self.model_list)
        
        # Compare button
        compare_btn = QPushButton("🔍 Compare Selected")
        compare_btn.clicked.connect(self._compare_models)
        compare_btn.setStyleSheet("""
            QPushButton {
                background-color: #0d7377;
                color: white;
                padding: 10px;
                font-size: 14pt;
                border-radius: 5px;
            }
        """)
        layout.addWidget(compare_btn)
        
        # Clear button
        clear_btn = QPushButton("🗑️ Clear Selection")
        clear_btn.clicked.connect(self.model_list.clearSelection)
        layout.addWidget(clear_btn)
        
        return widget
    
    def _create_comparison_panel(self):
        """Create comparison display panel."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        title = QLabel("📊 Comparison Results")
        title.setStyleSheet("font-size: 14pt; font-weight: bold;")
        layout.addWidget(title)
        
        # Metrics table
        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(5)
        self.metrics_table.setHorizontalHeaderLabels([
            "Metric", "Model 1", "Model 2", "Model 3", "Model 4"
        ])
        layout.addWidget(self.metrics_table)
        
        # Charts
        self.rewards_plot = pg.PlotWidget()
        self.rewards_plot.setTitle("Rewards Comparison")
        self.rewards_plot.setBackground('#1e1e1e')
        layout.addWidget(self.rewards_plot)
        
        # Winner display
        self.winner_label = QLabel("🏆 Winner: -")
        self.winner_label.setStyleSheet("font-size: 16pt; font-weight: bold; color: gold;")
        self.winner_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.winner_label)
        
        return widget
    
    def _compare_models(self):
        """Compare selected models."""
        selected_items = self.model_list.selectedItems()
        
        if len(selected_items) < 2:
            return
        
        if len(selected_items) > 4:
            selected_items = selected_items[:4]
        
        self.selected_models = [item.text() for item in selected_items]
        
        # Load and evaluate models
        self._load_comparison_data()
        self._display_comparison()
    
    def _load_comparison_data(self):
        """Load data for selected models."""
        self.comparison_data = {}
        
        for model_name in self.selected_models:
            # TODO: Load actual model and evaluate
            # For now, generate fake data
            self.comparison_data[model_name] = {
                'avg_reward': np.random.uniform(-100, 200),
                'success_rate': np.random.uniform(0, 1),
                'avg_steps': np.random.randint(100, 500),
                'training_time': np.random.uniform(1, 10),
                'model_size': np.random.uniform(1, 50),
                'inference_speed': np.random.uniform(10, 100),
                'robustness': np.random.uniform(0, 1),
                'rewards_history': np.random.randn(100).cumsum()
            }
    
    def _display_comparison(self):
        """Display comparison results."""
        # Update table
        metrics = [
            'Average Reward',
            'Success Rate (%)',
            'Average Steps',
            'Training Time (h)',
            'Model Size (MB)',
            'Inference Speed (FPS)',
            'Robustness Score'
        ]
        
        self.metrics_table.setRowCount(len(metrics))
        
        for i, metric in enumerate(metrics):
            self.metrics_table.setItem(i, 0, QTableWidgetItem(metric))
            
            for j, model_name in enumerate(self.selected_models):
                data = self.comparison_data[model_name]
                
                if metric == 'Average Reward':
                    value = f"{data['avg_reward']:.2f}"
                elif metric == 'Success Rate (%)':
                    value = f"{data['success_rate']*100:.1f}%"
                elif metric == 'Average Steps':
                    value = f"{data['avg_steps']}"
                elif metric == 'Training Time (h)':
                    value = f"{data['training_time']:.2f}"
                elif metric == 'Model Size (MB)':
                    value = f"{data['model_size']:.2f}"
                elif metric == 'Inference Speed (FPS)':
                    value = f"{data['inference_speed']:.1f}"
                else:  # Robustness
                    value = f"{data['robustness']:.2f}"
                
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.metrics_table.setItem(i, j + 1, item)
        
        # Update charts
        self.rewards_plot.clear()
        colors = ['#0d7377', '#ff9500', '#ff0000', '#00ff00']
        
        for i, model_name in enumerate(self.selected_models):
            rewards = self.comparison_data[model_name]['rewards_history']
            self.rewards_plot.plot(
                rewards,
                pen=pg.mkPen(color=colors[i], width=2),
                name=model_name
            )
        
        self.rewards_plot.addLegend()
        
        # Determine winner
        winner = max(
            self.selected_models,
            key=lambda m: self.comparison_data[m]['avg_reward']
        )
        self.winner_label.setText(f"🏆 Winner: {winner}")


if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    tool = ModelComparisonTool()
    tool.show()
    sys.exit(app.exec())
