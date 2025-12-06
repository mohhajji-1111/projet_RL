"""
🚀 Launcher - Main entry point for all GUI applications
"""

import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QPushButton, QLabel, QMessageBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont


class LauncherWindow(QMainWindow):
    """Main launcher window."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🤖 Robot Navigation - Launcher")
        self.setGeometry(400, 200, 600, 500)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QPushButton {
                background-color: #0d7377;
                color: white;
                border: none;
                padding: 20px;
                font-size: 16pt;
                border-radius: 10px;
                margin: 10px;
            }
            QPushButton:hover {
                background-color: #14a085;
            }
            QLabel {
                color: white;
            }
        """)
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup UI."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(50, 50, 50, 50)
        layout.setSpacing(20)
        
        # Title
        title = QLabel("🤖 Robot Navigation AI\nTraining Studio")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(24)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)
        
        # Subtitle
        subtitle = QLabel("Choose an application to launch:")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle_font = QFont()
        subtitle_font.setPointSize(12)
        subtitle.setFont(subtitle_font)
        subtitle.setStyleSheet("color: #888;")
        layout.addWidget(subtitle)
        
        layout.addSpacing(20)
        
        # Main GUI button
        main_btn = QPushButton("🖥️ Main Training Interface")
        main_btn.clicked.connect(self._launch_main_gui)
        layout.addWidget(main_btn)
        
        # Level Editor button
        editor_btn = QPushButton("🗺️ Level Editor")
        editor_btn.clicked.connect(self._launch_level_editor)
        layout.addWidget(editor_btn)
        
        # Comparison Tool button
        compare_btn = QPushButton("⚖️ Model Comparison Tool")
        compare_btn.clicked.connect(self._launch_comparison)
        layout.addWidget(compare_btn)
        
        # Web Dashboard button
        web_btn = QPushButton("🌐 Web Dashboard")
        web_btn.clicked.connect(self._launch_web_dashboard)
        layout.addWidget(web_btn)
        
        layout.addStretch()
        
        # Version info
        version_label = QLabel("Version 1.0.0 • © 2025")
        version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        version_label.setStyleSheet("color: #666; font-size: 10pt;")
        layout.addWidget(version_label)
    
    def _launch_main_gui(self):
        """Launch main GUI."""
        try:
            from gui.main_window import MainWindow
            self.main_window = MainWindow()
            self.main_window.show()
            self.hide()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to launch main GUI:\n{e}")
    
    def _launch_level_editor(self):
        """Launch level editor."""
        try:
            from level_editor.level_editor import LevelEditor
            self.level_editor = LevelEditor()
            self.level_editor.show()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to launch level editor:\n{e}")
    
    def _launch_comparison(self):
        """Launch comparison tool."""
        try:
            from gui.comparison_tool import ModelComparisonTool
            self.comparison_tool = ModelComparisonTool()
            self.comparison_tool.show()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to launch comparison tool:\n{e}")
    
    def _launch_web_dashboard(self):
        """Launch web dashboard."""
        import webbrowser
        import subprocess
        from pathlib import Path
        
        # Start backend
        backend_path = Path("web_dashboard/backend.py")
        if backend_path.exists():
            try:
                subprocess.Popen([sys.executable, str(backend_path)])
                QMessageBox.information(
                    self, "Web Dashboard",
                    "Backend started!\n\n"
                    "Opening dashboard in browser...\n"
                    "URL: http://localhost:3000"
                )
                webbrowser.open("http://localhost:3000")
            except Exception as e:
                QMessageBox.warning(
                    self, "Error",
                    f"Failed to start backend:\n{e}\n\n"
                    "Please start manually:\n"
                    "python web_dashboard/backend.py"
                )
        else:
            QMessageBox.warning(
                self, "Not Found",
                "Web dashboard backend not found.\n\n"
                "Please run from project root."
            )


def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    app.setApplicationName("Robot Navigation AI")
    app.setOrganizationName("RL Studio")
    
    launcher = LauncherWindow()
    launcher.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
