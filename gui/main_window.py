"""
🎮 Main GUI Window - Professional Robot Navigation Interface
Features: Real-time visualization, training control, analytics dashboard
"""

import sys
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QMenuBar, QMenu, QToolBar, QStatusBar, QMessageBox,
    QFileDialog, QDockWidget
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt6.QtGui import QAction, QIcon, QKeySequence

from gui.widgets.control_panel import ControlPanel
from gui.widgets.visualization_canvas import VisualizationCanvas
from gui.widgets.analytics_panel import AnalyticsPanel
from gui.widgets.training_thread import TrainingThread
from gui.widgets.settings_dialog import SettingsDialog


class MainWindow(QMainWindow):
    """Main application window with full GUI layout."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🤖 Robot Navigation - AI Training Studio")
        self.setGeometry(100, 100, 1600, 900)
        
        # State
        self.training_thread = None
        self.is_training = False
        self.is_paused = False
        self.current_model_path = None
        self.dark_mode = True
        
        # Setup UI
        self._setup_menus()
        self._setup_toolbar()
        self._setup_main_layout()
        self._setup_statusbar()
        self._setup_shortcuts()
        self._apply_theme()
        
        # Timers
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_ui)
        self.update_timer.start(100)  # 10 FPS UI updates
        
        print("✅ GUI initialized successfully!")
    
    def _setup_menus(self):
        """Create menu bar."""
        menubar = self.menuBar()
        
        # File Menu
        file_menu = menubar.addMenu("📁 &File")
        
        new_action = QAction("🆕 New Project", self)
        new_action.setShortcut(QKeySequence.StandardKey.New)
        new_action.triggered.connect(self._new_project)
        file_menu.addAction(new_action)
        
        open_action = QAction("📂 Open Model...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._open_model)
        file_menu.addAction(open_action)
        
        save_action = QAction("💾 Save Model", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self._save_model)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        export_action = QAction("📤 Export Results...", self)
        export_action.triggered.connect(self._export_results)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("🚪 Exit", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Training Menu
        training_menu = menubar.addMenu("🚀 &Training")
        
        start_action = QAction("▶️ Start Training", self)
        start_action.setShortcut("F5")
        start_action.triggered.connect(self._start_training)
        training_menu.addAction(start_action)
        
        pause_action = QAction("⏸️ Pause Training", self)
        pause_action.setShortcut("F6")
        pause_action.triggered.connect(self._pause_training)
        training_menu.addAction(pause_action)
        
        stop_action = QAction("⏹️ Stop Training", self)
        stop_action.setShortcut("F7")
        stop_action.triggered.connect(self._stop_training)
        training_menu.addAction(stop_action)
        
        training_menu.addSeparator()
        
        resume_action = QAction("🔄 Resume Training", self)
        resume_action.triggered.connect(self._resume_training)
        training_menu.addAction(resume_action)
        
        # Evaluation Menu
        eval_menu = menubar.addMenu("🧪 &Evaluation")
        
        test_action = QAction("🎯 Test Agent", self)
        test_action.setShortcut("F8")
        test_action.triggered.connect(self._test_agent)
        eval_menu.addAction(test_action)
        
        benchmark_action = QAction("📊 Run Benchmark", self)
        benchmark_action.triggered.connect(self._run_benchmark)
        eval_menu.addAction(benchmark_action)
        
        compare_action = QAction("⚖️ Compare Models", self)
        compare_action.triggered.connect(self._compare_models)
        eval_menu.addAction(compare_action)
        
        # Tools Menu
        tools_menu = menubar.addMenu("🛠️ &Tools")
        
        hyperparameter_action = QAction("🎛️ Hyperparameter Tuner", self)
        hyperparameter_action.triggered.connect(self._open_hyperparameter_tuner)
        tools_menu.addAction(hyperparameter_action)
        
        visualizer_action = QAction("📈 Advanced Visualizer", self)
        visualizer_action.triggered.connect(self._open_visualizer)
        tools_menu.addAction(visualizer_action)
        
        level_editor_action = QAction("🗺️ Level Editor", self)
        level_editor_action.triggered.connect(self._open_level_editor)
        tools_menu.addAction(level_editor_action)
        
        replay_action = QAction("🎬 Replay Viewer", self)
        replay_action.triggered.connect(self._open_replay_viewer)
        tools_menu.addAction(replay_action)
        
        tools_menu.addSeparator()
        
        settings_action = QAction("⚙️ Settings...", self)
        settings_action.setShortcut("Ctrl+,")
        settings_action.triggered.connect(self._open_settings)
        tools_menu.addAction(settings_action)
        
        # View Menu
        view_menu = menubar.addMenu("👁️ &View")
        
        theme_action = QAction("🌓 Toggle Dark/Light", self)
        theme_action.setShortcut("Ctrl+T")
        theme_action.triggered.connect(self._toggle_theme)
        view_menu.addAction(theme_action)
        
        fullscreen_action = QAction("🖥️ Fullscreen", self)
        fullscreen_action.setShortcut("F11")
        fullscreen_action.triggered.connect(self._toggle_fullscreen)
        view_menu.addAction(fullscreen_action)
        
        # Help Menu
        help_menu = menubar.addMenu("❓ &Help")
        
        tutorial_action = QAction("📚 Tutorial", self)
        tutorial_action.setShortcut("F1")
        tutorial_action.triggered.connect(self._show_tutorial)
        help_menu.addAction(tutorial_action)
        
        docs_action = QAction("📖 Documentation", self)
        docs_action.triggered.connect(self._show_documentation)
        help_menu.addAction(docs_action)
        
        help_menu.addSeparator()
        
        about_action = QAction("ℹ️ About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
    
    def _setup_toolbar(self):
        """Create toolbar with quick actions."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        
        # Play/Pause
        self.play_action = QAction("▶️ Play", self)
        self.play_action.triggered.connect(self._start_training)
        toolbar.addAction(self.play_action)
        
        # Stop
        stop_action = QAction("⏹️ Stop", self)
        stop_action.triggered.connect(self._stop_training)
        toolbar.addAction(stop_action)
        
        toolbar.addSeparator()
        
        # Reset
        reset_action = QAction("🔄 Reset", self)
        reset_action.triggered.connect(self._reset_environment)
        toolbar.addAction(reset_action)
        
        # Screenshot
        screenshot_action = QAction("📸 Screenshot", self)
        screenshot_action.triggered.connect(self._take_screenshot)
        toolbar.addAction(screenshot_action)
        
        toolbar.addSeparator()
        
        # Load Model
        load_action = QAction("📂 Load", self)
        load_action.triggered.connect(self._open_model)
        toolbar.addAction(load_action)
        
        # Save Model
        save_action = QAction("💾 Save", self)
        save_action.triggered.connect(self._save_model)
        toolbar.addAction(save_action)
    
    def _setup_main_layout(self):
        """Setup main window layout with panels."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # Main splitter (3-way split)
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left Panel: Control Center
        self.control_panel = ControlPanel()
        self.control_panel.training_started.connect(self._start_training)
        self.control_panel.settings_changed.connect(self._on_settings_changed)
        main_splitter.addWidget(self.control_panel)
        
        # Center Panel: Visualization
        self.visualization_canvas = VisualizationCanvas()
        main_splitter.addWidget(self.visualization_canvas)
        
        # Right Panel: Analytics
        self.analytics_panel = AnalyticsPanel()
        main_splitter.addWidget(self.analytics_panel)
        
        # Set initial sizes (20%, 50%, 30%)
        main_splitter.setSizes([300, 800, 500])
        
        main_layout.addWidget(main_splitter)
    
    def _setup_statusbar(self):
        """Create status bar with indicators."""
        statusbar = self.statusBar()
        statusbar.showMessage("Ready")
        
        # Add permanent widgets
        self.status_label = statusbar
    
    def _setup_shortcuts(self):
        """Setup keyboard shortcuts."""
        # Additional shortcuts not in menus
        pass
    
    def _apply_theme(self):
        """Apply dark/light theme."""
        if self.dark_mode:
            self.setStyleSheet("""
                QMainWindow {
                    background-color: #1e1e1e;
                    color: #ffffff;
                }
                QMenuBar {
                    background-color: #2d2d2d;
                    color: #ffffff;
                }
                QMenuBar::item:selected {
                    background-color: #3d3d3d;
                }
                QMenu {
                    background-color: #2d2d2d;
                    color: #ffffff;
                }
                QMenu::item:selected {
                    background-color: #0d7377;
                }
                QToolBar {
                    background-color: #2d2d2d;
                    border: none;
                }
                QStatusBar {
                    background-color: #2d2d2d;
                    color: #ffffff;
                }
            """)
        else:
            self.setStyleSheet("")
    
    # ==================== MENU ACTIONS ====================
    
    def _new_project(self):
        """Create new project."""
        reply = QMessageBox.question(
            self, "New Project",
            "Create a new project? Current progress will be lost.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._reset_environment()
            self.statusBar().showMessage("New project created")
    
    def _open_model(self):
        """Open model file."""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open Model", "", "Model Files (*.pt *.pth);;All Files (*)"
        )
        if filename:
            self.current_model_path = filename
            self.statusBar().showMessage(f"Loaded: {Path(filename).name}")
            # TODO: Actually load the model
    
    def _save_model(self):
        """Save current model."""
        if not self.current_model_path:
            filename, _ = QFileDialog.getSaveFileName(
                self, "Save Model", "", "Model Files (*.pt)"
            )
            if filename:
                self.current_model_path = filename
        
        if self.current_model_path:
            # TODO: Actually save the model
            self.statusBar().showMessage(f"Saved: {Path(self.current_model_path).name}")
    
    def _export_results(self):
        """Export results."""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "", "JSON Files (*.json);;CSV Files (*.csv)"
        )
        if filename:
            # TODO: Export results
            self.statusBar().showMessage(f"Exported to: {filename}")
    
    def _start_training(self):
        """Start training."""
        if self.is_training:
            return
        
        self.is_training = True
        self.is_paused = False
        self.play_action.setText("⏸️ Pause")
        self.statusBar().showMessage("Training started...")
        
        # Create and start training thread
        config = self.control_panel.get_config()
        self.training_thread = TrainingThread(config)
        self.training_thread.progress_update.connect(self._on_training_progress)
        self.training_thread.finished.connect(self._on_training_finished)
        self.training_thread.start()
    
    def _pause_training(self):
        """Pause training."""
        if self.is_training and not self.is_paused:
            self.is_paused = True
            self.play_action.setText("▶️ Resume")
            self.statusBar().showMessage("Training paused")
            if self.training_thread:
                self.training_thread.pause()
    
    def _stop_training(self):
        """Stop training."""
        if self.is_training:
            self.is_training = False
            self.is_paused = False
            self.play_action.setText("▶️ Play")
            self.statusBar().showMessage("Training stopped")
            if self.training_thread:
                self.training_thread.stop()
    
    def _resume_training(self):
        """Resume training."""
        if self.is_paused:
            self.is_paused = False
            self.play_action.setText("⏸️ Pause")
            self.statusBar().showMessage("Training resumed")
            if self.training_thread:
                self.training_thread.resume()
    
    def _test_agent(self):
        """Test current agent."""
        QMessageBox.information(self, "Test Agent", "Testing agent...")
    
    def _run_benchmark(self):
        """Run benchmark."""
        QMessageBox.information(self, "Benchmark", "Running benchmark...")
    
    def _compare_models(self):
        """Compare models."""
        QMessageBox.information(self, "Compare", "Opening comparison tool...")
    
    def _open_hyperparameter_tuner(self):
        """Open hyperparameter tuner."""
        QMessageBox.information(self, "Tuner", "Opening hyperparameter tuner...")
    
    def _open_visualizer(self):
        """Open advanced visualizer."""
        QMessageBox.information(self, "Visualizer", "Opening visualizer...")
    
    def _open_level_editor(self):
        """Open level editor."""
        QMessageBox.information(self, "Level Editor", "Opening level editor...")
    
    def _open_replay_viewer(self):
        """Open replay viewer."""
        QMessageBox.information(self, "Replay", "Opening replay viewer...")
    
    def _open_settings(self):
        """Open settings dialog."""
        dialog = SettingsDialog(self)
        dialog.exec()
    
    def _toggle_theme(self):
        """Toggle dark/light theme."""
        self.dark_mode = not self.dark_mode
        self._apply_theme()
    
    def _toggle_fullscreen(self):
        """Toggle fullscreen."""
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()
    
    def _show_tutorial(self):
        """Show tutorial."""
        QMessageBox.information(
            self, "Tutorial",
            "📚 Welcome to Robot Navigation AI!\n\n"
            "1. Select a mode in the left panel\n"
            "2. Configure training settings\n"
            "3. Click 'Start Training'\n"
            "4. Monitor progress in real-time\n"
            "5. Evaluate your trained model\n\n"
            "Press F1 anytime for help!"
        )
    
    def _show_documentation(self):
        """Show documentation."""
        QMessageBox.information(self, "Docs", "Opening documentation...")
    
    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self, "About",
            "🤖 Robot Navigation - AI Training Studio\n\n"
            "Version: 1.0.0\n"
            "Developed with PyQt6 & PyTorch\n\n"
            "A professional environment for training\n"
            "reinforcement learning agents.\n\n"
            "© 2025 - Open Source Project"
        )
    
    def _reset_environment(self):
        """Reset environment."""
        self.visualization_canvas.reset()
        self.analytics_panel.clear()
        self.statusBar().showMessage("Environment reset")
    
    def _take_screenshot(self):
        """Take screenshot of visualization."""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Screenshot", "", "PNG Files (*.png)"
        )
        if filename:
            self.visualization_canvas.save_screenshot(filename)
            self.statusBar().showMessage(f"Screenshot saved: {filename}")
    
    # ==================== EVENT HANDLERS ====================
    
    def _on_settings_changed(self, config):
        """Handle settings changes."""
        print(f"Settings changed: {config}")
    
    def _on_training_progress(self, data):
        """Handle training progress update."""
        episode = data.get('episode', 0)
        reward = data.get('reward', 0)
        
        # Update visualization
        self.visualization_canvas.update_state(data.get('state'))
        
        # Update analytics
        self.analytics_panel.add_data_point(episode, reward, data)
        
        # Update status
        self.statusBar().showMessage(
            f"Episode {episode}: Reward={reward:.2f}"
        )
    
    def _on_training_finished(self):
        """Handle training finished."""
        self.is_training = False
        self.play_action.setText("▶️ Play")
        self.statusBar().showMessage("Training completed!")
        
        QMessageBox.information(
            self, "Training Complete",
            "Training has finished successfully!\n\n"
            "Check the analytics panel for results."
        )
    
    def _update_ui(self):
        """Update UI periodically."""
        # Update any time-based displays
        pass
    
    def closeEvent(self, event):
        """Handle window close."""
        if self.is_training:
            reply = QMessageBox.question(
                self, "Quit",
                "Training is in progress. Are you sure you want to quit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return
            else:
                self._stop_training()
        
        event.accept()


def main():
    """Run the application."""
    app = QApplication(sys.argv)
    app.setApplicationName("Robot Navigation AI")
    app.setOrganizationName("RL Studio")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
