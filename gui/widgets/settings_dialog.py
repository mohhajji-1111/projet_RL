"""
⚙️ Settings Dialog
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QTabWidget, QWidget,
    QFormLayout, QLineEdit, QCheckBox, QComboBox,
    QDialogButtonBox, QLabel
)


class SettingsDialog(QDialog):
    """Settings dialog."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("⚙️ Settings")
        self.setMinimumSize(500, 400)
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup UI."""
        layout = QVBoxLayout(self)
        
        tabs = QTabWidget()
        
        # General tab
        general_tab = self._create_general_tab()
        tabs.addTab(general_tab, "General")
        
        # Appearance tab
        appearance_tab = self._create_appearance_tab()
        tabs.addTab(appearance_tab, "Appearance")
        
        # Advanced tab
        advanced_tab = self._create_advanced_tab()
        tabs.addTab(advanced_tab, "Advanced")
        
        layout.addWidget(tabs)
        
        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def _create_general_tab(self):
        """Create general settings tab."""
        widget = QWidget()
        layout = QFormLayout(widget)
        
        # Auto-save
        auto_save = QCheckBox("Enable auto-save")
        auto_save.setChecked(True)
        layout.addRow("Auto-save:", auto_save)
        
        # Default episodes
        episodes = QLineEdit("1000")
        layout.addRow("Default episodes:", episodes)
        
        return widget
    
    def _create_appearance_tab(self):
        """Create appearance settings tab."""
        widget = QWidget()
        layout = QFormLayout(widget)
        
        # Theme
        theme = QComboBox()
        theme.addItems(["Dark", "Light", "Auto"])
        layout.addRow("Theme:", theme)
        
        # FPS
        fps = QComboBox()
        fps.addItems(["30", "60", "120"])
        fps.setCurrentText("60")
        layout.addRow("FPS:", fps)
        
        return widget
    
    def _create_advanced_tab(self):
        """Create advanced settings tab."""
        widget = QWidget()
        layout = QFormLayout(widget)
        
        # GPU
        gpu = QCheckBox("Use GPU")
        gpu.setChecked(True)
        layout.addRow("GPU:", gpu)
        
        # Threads
        threads = QLineEdit("4")
        layout.addRow("CPU Threads:", threads)
        
        return widget
