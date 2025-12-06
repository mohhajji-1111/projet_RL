"""
🎛️ Control Panel - Left sidebar with all training controls
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QFormLayout,
    QComboBox, QSlider, QSpinBox, QDoubleSpinBox,
    QPushButton, QLabel, QLineEdit, QCheckBox
)
from PyQt6.QtCore import Qt, pyqtSignal


class ControlPanel(QWidget):
    """Control panel for training configuration."""
    
    training_started = pyqtSignal()
    settings_changed = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.setMaximumWidth(350)
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup UI layout."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        # Mode Selection
        mode_group = self._create_mode_group()
        layout.addWidget(mode_group)
        
        # Agent Selection
        agent_group = self._create_agent_group()
        layout.addWidget(agent_group)
        
        # Environment Settings
        env_group = self._create_environment_group()
        layout.addWidget(env_group)
        
        # Training Settings
        training_group = self._create_training_group()
        layout.addWidget(training_group)
        
        # Action Buttons
        button_layout = self._create_action_buttons()
        layout.addLayout(button_layout)
        
        layout.addStretch()
    
    def _create_mode_group(self):
        """Create mode selection group."""
        group = QGroupBox("🎮 Mode Selection")
        layout = QVBoxLayout()
        
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([
            "⭐ Basic Navigation",
            "⭐⭐ Dynamic Obstacles",
            "⭐⭐⭐ Multi-Goal Planning",
            "⭐⭐⭐⭐ Full Challenge"
        ])
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        
        layout.addWidget(QLabel("Difficulty:"))
        layout.addWidget(self.mode_combo)
        
        # Description
        self.mode_description = QLabel("Navigate to a single goal")
        self.mode_description.setWordWrap(True)
        self.mode_description.setStyleSheet("color: #888; font-size: 10pt;")
        layout.addWidget(self.mode_description)
        
        group.setLayout(layout)
        return group
    
    def _create_agent_group(self):
        """Create agent selection group."""
        group = QGroupBox("🤖 Agent Selection")
        layout = QVBoxLayout()
        
        self.agent_combo = QComboBox()
        self.agent_combo.addItems([
            "DQN (Deep Q-Network)",
            "Rainbow DQN",
            "PPO (Proximal Policy Optimization)",
            "A3C (Asynchronous Actor-Critic)",
            "Custom Agent"
        ])
        
        layout.addWidget(QLabel("Algorithm:"))
        layout.addWidget(self.agent_combo)
        
        # Load model button
        load_btn = QPushButton("📂 Load Pretrained Model")
        load_btn.clicked.connect(self._load_model)
        layout.addWidget(load_btn)
        
        group.setLayout(layout)
        return group
    
    def _create_environment_group(self):
        """Create environment settings group."""
        group = QGroupBox("🗺️ Environment Settings")
        layout = QFormLayout()
        
        # Grid size
        self.grid_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.grid_size_slider.setRange(400, 1200)
        self.grid_size_slider.setValue(800)
        self.grid_size_slider.setTickInterval(100)
        self.grid_size_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.grid_size_label = QLabel("800x800")
        self.grid_size_slider.valueChanged.connect(
            lambda v: self.grid_size_label.setText(f"{v}x{v}")
        )
        
        grid_layout = QVBoxLayout()
        grid_layout.addWidget(self.grid_size_slider)
        grid_layout.addWidget(self.grid_size_label)
        layout.addRow("Grid Size:", grid_layout)
        
        # Number of obstacles
        self.obstacles_spin = QSpinBox()
        self.obstacles_spin.setRange(0, 20)
        self.obstacles_spin.setValue(5)
        layout.addRow("Obstacles:", self.obstacles_spin)
        
        # Number of goals
        self.goals_spin = QSpinBox()
        self.goals_spin.setRange(1, 10)
        self.goals_spin.setValue(1)
        layout.addRow("Goals:", self.goals_spin)
        
        # Random seed
        self.seed_edit = QLineEdit("42")
        layout.addRow("Random Seed:", self.seed_edit)
        
        group.setLayout(layout)
        return group
    
    def _create_training_group(self):
        """Create training settings group."""
        group = QGroupBox("🚀 Training Settings")
        layout = QFormLayout()
        
        # Episodes
        self.episodes_spin = QSpinBox()
        self.episodes_spin.setRange(10, 10000)
        self.episodes_spin.setValue(1000)
        self.episodes_spin.setSingleStep(100)
        layout.addRow("Episodes:", self.episodes_spin)
        
        # Learning rate
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.00001, 0.01)
        self.lr_spin.setValue(0.0005)
        self.lr_spin.setDecimals(5)
        self.lr_spin.setSingleStep(0.0001)
        layout.addRow("Learning Rate:", self.lr_spin)
        
        # Epsilon
        self.epsilon_slider = QSlider(Qt.Orientation.Horizontal)
        self.epsilon_slider.setRange(0, 100)
        self.epsilon_slider.setValue(100)
        self.epsilon_label = QLabel("1.00")
        self.epsilon_slider.valueChanged.connect(
            lambda v: self.epsilon_label.setText(f"{v/100:.2f}")
        )
        
        eps_layout = QVBoxLayout()
        eps_layout.addWidget(self.epsilon_slider)
        eps_layout.addWidget(self.epsilon_label)
        layout.addRow("Epsilon Start:", eps_layout)
        
        # Batch size
        self.batch_combo = QComboBox()
        self.batch_combo.addItems(["32", "64", "128", "256", "512"])
        self.batch_combo.setCurrentText("128")
        layout.addRow("Batch Size:", self.batch_combo)
        
        # Save frequency
        self.save_freq_spin = QSpinBox()
        self.save_freq_spin.setRange(10, 1000)
        self.save_freq_spin.setValue(100)
        self.save_freq_spin.setSingleStep(10)
        layout.addRow("Save Every:", self.save_freq_spin)
        
        # Use GPU
        self.gpu_check = QCheckBox("Use GPU (if available)")
        self.gpu_check.setChecked(True)
        layout.addRow("", self.gpu_check)
        
        group.setLayout(layout)
        return group
    
    def _create_action_buttons(self):
        """Create action buttons."""
        layout = QVBoxLayout()
        
        # Start Training
        self.start_btn = QPushButton("🚀 Start Training")
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #0d7377;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #14a085;
            }
        """)
        self.start_btn.clicked.connect(self.training_started.emit)
        layout.addWidget(self.start_btn)
        
        # Load Model
        load_btn = QPushButton("📂 Load Model")
        load_btn.clicked.connect(self._load_model)
        layout.addWidget(load_btn)
        
        # Save Model
        save_btn = QPushButton("💾 Save Model")
        save_btn.clicked.connect(self._save_model)
        layout.addWidget(save_btn)
        
        # Run Evaluation
        eval_btn = QPushButton("🧪 Run Evaluation")
        eval_btn.clicked.connect(self._run_evaluation)
        layout.addWidget(eval_btn)
        
        return layout
    
    def _on_mode_changed(self, index):
        """Handle mode change."""
        descriptions = [
            "Navigate to a single goal without obstacles",
            "Avoid dynamic moving obstacles",
            "Visit multiple goals in sequence",
            "Complete mission with all challenges"
        ]
        self.mode_description.setText(descriptions[index])
        self.settings_changed.emit(self.get_config())
    
    def _load_model(self):
        """Load model."""
        print("Load model clicked")
    
    def _save_model(self):
        """Save model."""
        print("Save model clicked")
    
    def _run_evaluation(self):
        """Run evaluation."""
        print("Run evaluation clicked")
    
    def get_config(self):
        """Get current configuration."""
        return {
            'mode': self.mode_combo.currentIndex(),
            'agent': self.agent_combo.currentText(),
            'grid_size': self.grid_size_slider.value(),
            'num_obstacles': self.obstacles_spin.value(),
            'num_goals': self.goals_spin.value(),
            'random_seed': int(self.seed_edit.text() or 0),
            'episodes': self.episodes_spin.value(),
            'learning_rate': self.lr_spin.value(),
            'epsilon_start': self.epsilon_slider.value() / 100,
            'batch_size': int(self.batch_combo.currentText()),
            'save_frequency': self.save_freq_spin.value(),
            'use_gpu': self.gpu_check.isChecked()
        }
