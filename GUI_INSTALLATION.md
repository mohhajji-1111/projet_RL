# 🎮 GUI & Interactive Features - Installation Guide

## 📦 Complete Installation

### 1. Desktop GUI (PyQt6)

```bash
# Install GUI dependencies
pip install -r requirements-gui.txt

# Verify installation
python -c "import PyQt6; print('PyQt6 installed successfully!')"
```

### 2. Web Dashboard

#### Backend (Flask)
```bash
cd web_dashboard
pip install flask flask-cors flask-socketio python-socketio
```

#### Frontend (React)
```bash
cd web_dashboard/frontend
npm install
# or
yarn install
```

### 3. Optional Dependencies

```bash
# For advanced visualizations
pip install plotly kaleido

# For video export
pip install opencv-python moviepy

# For PDF reports
pip install reportlab
```

---

## 🚀 Quick Start

### Option 1: Launcher (Recommended)

```bash
python launcher.py
```

This opens a menu with all available tools:
- 🖥️ Main Training Interface
- 🗺️ Level Editor
- ⚖️ Model Comparison Tool
- 🌐 Web Dashboard

### Option 2: Direct Launch

```bash
# Main GUI
python gui/main_window.py

# Level Editor
python level_editor/level_editor.py

# Comparison Tool
python gui/comparison_tool.py

# Web Dashboard Backend
python web_dashboard/backend.py
```

---

## 📚 Project Structure

```
projet_RL/
├── gui/                          # Desktop GUI application
│   ├── main_window.py           # Main window with all features
│   ├── comparison_tool.py       # Model comparison tool
│   ├── gamification.py          # Achievements & XP system
│   ├── replay_system.py         # Record & playback episodes
│   └── widgets/                 # UI components
│       ├── control_panel.py     # Training controls
│       ├── visualization_canvas.py  # Real-time rendering
│       ├── analytics_panel.py   # Metrics & charts
│       ├── training_thread.py   # Background training
│       └── settings_dialog.py   # Settings UI
│
├── level_editor/                 # Level editor application
│   └── level_editor.py          # Map creation tool
│
├── web_dashboard/                # Web-based dashboard
│   ├── backend.py               # Flask API server
│   └── frontend/                # React application
│       ├── App.jsx              # Main dashboard
│       └── package.json         # NPM dependencies
│
├── launcher.py                   # Main launcher
├── requirements-gui.txt          # GUI dependencies
└── docs/
    └── GUI_USER_MANUAL.md       # Complete user manual
```

---

## 🎯 Features Overview

### Desktop GUI Application

**Main Window:**
- ✅ Real-time training visualization
- ✅ Interactive control panel
- ✅ Live metrics & charts
- ✅ Model management
- ✅ Dark/Light themes
- ✅ Keyboard shortcuts

**Visualization Canvas:**
- ✅ Robot animation with direction indicator
- ✅ Obstacle rendering
- ✅ Goal with sparkle effects
- ✅ LIDAR ray visualization
- ✅ Path history trail
- ✅ HUD overlay (FPS, position, distance)

**Analytics Panel:**
- ✅ Real-time metrics (LCD-style displays)
- ✅ Interactive charts (Rewards, Steps, Success Rate)
- ✅ Training logs console
- ✅ Export functionality

### Level Editor

- ✅ Drag & drop obstacle creation
- ✅ Place goals and robot start position
- ✅ Snap to grid
- ✅ Save/Load maps (JSON format)
- ✅ Instant testing
- ✅ Undo/Redo support

### Model Comparison Tool

- ✅ Compare 2-4 models side-by-side
- ✅ Metrics comparison table
- ✅ Synchronized visualization
- ✅ Performance charts
- ✅ Winner determination

### Replay System

- ✅ Record episodes with compression
- ✅ Frame-by-frame playback
- ✅ Speed control (0.5x - 5x)
- ✅ Statistics calculation
- ✅ Export to video

### Gamification System

- ✅ Achievement system (9 achievements)
- ✅ XP & leveling system
- ✅ Progress tracking
- ✅ Daily challenges
- ✅ Leaderboards

### Web Dashboard

- ✅ Real-time updates via WebSocket
- ✅ Responsive design (mobile-friendly)
- ✅ Live metrics streaming
- ✅ Model management
- ✅ Remote training control
- ✅ RESTful API

---

## 🎨 Usage Examples

### 1. Basic Training Session

```bash
# Launch GUI
python launcher.py

# Or directly
python gui/main_window.py

# In GUI:
1. Select "Basic Navigation" mode
2. Configure settings (episodes, learning rate, etc.)
3. Click "Start Training"
4. Monitor real-time progress
5. Save model when complete
```

### 2. Create Custom Level

```bash
# Launch level editor
python level_editor/level_editor.py

# Create level:
1. Select "Obstacle" tool
2. Draw obstacles by clicking & dragging
3. Select "Goal" tool and place goals
4. Select "Robot" tool and set start position
5. Test level with ▶️ button
6. Save to JSON file
```

### 3. Compare Models

```bash
# Launch comparison tool
python gui/comparison_tool.py

# Compare:
1. Select 2-4 models from list
2. Click "Compare Selected"
3. View metrics table and charts
4. Export comparison report
```

### 4. Record & Replay

```python
from gui.replay_system import ReplayRecorder, ReplayPlayer

# Record
recorder = ReplayRecorder()
recorder.start_recording(metadata={'episode': 1})

# During training
for step in episode:
    recorder.record_frame(state, action, reward)

replay_file = recorder.stop_recording()

# Playback
player = ReplayPlayer()
player.load_replay(replay_file)

while True:
    frame = player.next_frame()
    if frame:
        visualize(frame)
```

### 5. Web Dashboard API

```python
import requests

# Start training
response = requests.post('http://localhost:5000/api/training/start', json={
    'episodes': 1000,
    'learning_rate': 0.0005,
    'batch_size': 128
})

# Get status
status = requests.get('http://localhost:5000/api/training/status').json()
print(f"Episode: {status['current_episode']}/{status['total_episodes']}")

# List models
models = requests.get('http://localhost:5000/api/models').json()
for model in models['models']:
    print(f"{model['name']} - {model['size']} bytes")
```

---

## ⚙️ Configuration

### Desktop GUI Settings

Edit `gui/settings.json`:

```json
{
  "theme": "dark",
  "fps": 60,
  "auto_save": true,
  "save_interval": 100,
  "show_lidar": true,
  "show_path": true,
  "animation_speed": 1.0
}
```

### Web Dashboard

Edit `web_dashboard/config.py`:

```python
# Server configuration
HOST = '0.0.0.0'
PORT = 5000
DEBUG = True

# Frontend URL
FRONTEND_URL = 'http://localhost:3000'

# Database (optional)
DATABASE_URL = 'sqlite:///training.db'
```

---

## 🔧 Troubleshooting

### GUI won't start

```bash
# Check PyQt6
pip show PyQt6

# Reinstall
pip uninstall PyQt6
pip install PyQt6
```

### Web dashboard connection failed

```bash
# Check if backend is running
curl http://localhost:5000/api/health

# Check ports
netstat -an | findstr "5000"

# Restart backend
python web_dashboard/backend.py
```

### Import errors

```bash
# Ensure you're in project root
cd /path/to/projet_RL

# Install all dependencies
pip install -r requirements.txt
pip install -r requirements-gui.txt
```

### Charts not displaying

```bash
# Install pyqtgraph
pip install pyqtgraph

# For web dashboard
cd web_dashboard/frontend
npm install recharts
```

---

## 📖 Documentation

- [Complete User Manual](docs/GUI_USER_MANUAL.md)
- [API Documentation](web_dashboard/API_DOCS.md)
- [Keyboard Shortcuts](docs/SHORTCUTS.md)
- [Architecture Guide](docs/ARCHITECTURE.md)

---

## 🎓 Tutorials

### Tutorial 1: First Training Session
1. Launch `launcher.py`
2. Click "Main Training Interface"
3. Select "Basic Navigation" mode
4. Set episodes to 100
5. Click "Start Training"
6. Watch real-time progress
7. Save your model

### Tutorial 2: Creating Custom Scenarios
1. Launch level editor
2. Create obstacles and goals
3. Test your level
4. Save as JSON
5. Load in main GUI

### Tutorial 3: Analyzing Performance
1. Train multiple models
2. Launch comparison tool
3. Select models to compare
4. Review metrics and charts
5. Export report

---

## 🚀 Performance Tips

1. **Reduce visualization FPS** (30 instead of 60) for slower machines
2. **Disable LIDAR rays** if experiencing lag
3. **Use smaller grid sizes** for faster rendering
4. **Enable GPU** for training (not visualization)
5. **Close unnecessary panels** to save resources

---

## 🌟 Advanced Features

### Custom Themes

Create custom color scheme:

```python
# In main_window.py
CUSTOM_THEME = {
    'background': '#1e1e1e',
    'text': '#ffffff',
    'primary': '#0d7377',
    'success': '#00ff00',
    'danger': '#ff0000'
}
```

### Keyboard Macros

Record action sequences for automated testing.

### Remote Monitoring

Access training from any device via web dashboard.

---

## 📞 Support

- 📧 Email: support@example.com
- 💬 Discord: [Join Server](https://discord.gg/example)
- 🐛 Issues: [GitHub Issues](https://github.com/example/issues)
- 📚 Docs: [Full Documentation](https://docs.example.com)

---

## 📝 License

MIT License - See LICENSE file for details

---

## 🙏 Credits

Built with:
- PyQt6 - GUI framework
- Flask - Web backend
- React - Web frontend
- PyTorch - Deep learning
- PyQtGraph - Plotting

---

**Version**: 1.0.0  
**Last Updated**: December 6, 2025

**Happy Training! 🚀**
