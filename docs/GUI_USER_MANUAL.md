# 🎮 GUI & Interactive Features - User Manual

## 📋 Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Desktop GUI Application](#desktop-gui)
4. [Web Dashboard](#web-dashboard)
5. [Level Editor](#level-editor)
6. [Replay System](#replay-system)
7. [Features Guide](#features-guide)
8. [Troubleshooting](#troubleshooting)

---

## 🚀 Installation

### Desktop GUI (PyQt6)

```bash
# Install GUI dependencies
pip install -r requirements-gui.txt

# Or install individually
pip install PyQt6 pyqtgraph pygame
```

### Web Dashboard (Flask + React)

```bash
# Backend
cd web_dashboard
pip install flask flask-cors flask-socketio

# Frontend
cd frontend
npm install
```

---

## 🎯 Quick Start

### 1. Launch Desktop GUI

```bash
python gui/main_window.py
```

**Features:**
- ✅ Real-time training visualization
- ✅ Interactive control panel
- ✅ Live metrics & charts
- ✅ Model management
- ✅ Dark/Light themes

### 2. Launch Web Dashboard

```bash
# Terminal 1: Start backend
cd web_dashboard
python backend.py

# Terminal 2: Start frontend
cd frontend
npm start
```

Access dashboard at: `http://localhost:3000`

### 3. Launch Level Editor

```bash
python level_editor/level_editor.py
```

---

## 🖥️ Desktop GUI Application

### Main Window Layout

```
┌─────────────────────────────────────────────────────┐
│  📁 File  🚀 Training  🧪 Evaluation  🛠️ Tools  ❓ Help │
├─────────────────────────────────────────────────────┤
│  ▶️  ⏹️  🔄  📸  💾                                    │  Toolbar
├──────────┬──────────────────────────┬───────────────┤
│          │                          │               │
│ Control  │    Visualization         │   Analytics   │
│  Panel   │        Canvas            │     Panel     │
│          │                          │               │
│  🎮 Mode │   🤖 [Robot Animation]   │ 📊 Metrics   │
│  🤖 Agent│   🗺️  [Environment]      │ 📈 Charts    │
│  🗺️ Env  │   📍 [Path History]      │ 📝 Logs      │
│  🚀 Train│                          │               │
│          │                          │               │
└──────────┴──────────────────────────┴───────────────┘
```

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `F5` | Start Training |
| `F6` | Pause Training |
| `F7` | Stop Training |
| `F8` | Test Agent |
| `F11` | Fullscreen |
| `Ctrl+N` | New Project |
| `Ctrl+O` | Open Model |
| `Ctrl+S` | Save Model |
| `Ctrl+T` | Toggle Theme |
| `Ctrl+,` | Settings |
| `F1` | Help/Tutorial |

### Control Panel Features

#### 🎮 Mode Selection
- **⭐ Basic Navigation**: Simple goal reaching
- **⭐⭐ Dynamic Obstacles**: Moving obstacles
- **⭐⭐⭐ Multi-Goal**: Multiple goals in sequence
- **⭐⭐⭐⭐ Full Challenge**: All features combined

#### 🤖 Agent Selection
- DQN (Deep Q-Network)
- Rainbow DQN
- PPO (Proximal Policy Optimization)
- A3C (Asynchronous Actor-Critic)
- Custom Agent

#### 🗺️ Environment Settings
- Grid Size: 400-1200px
- Number of Obstacles: 0-20
- Number of Goals: 1-10
- Random Seed: For reproducibility

#### 🚀 Training Settings
- Episodes: 10-10,000
- Learning Rate: 0.00001-0.01
- Epsilon Start: 0.0-1.0
- Batch Size: 32/64/128/256/512
- Save Frequency: 10-1000 episodes
- Use GPU: Checkbox

### Visualization Canvas

**Features:**
- Real-time robot movement
- Obstacle rendering
- Goal with sparkle effects
- LIDAR ray visualization
- Path history trail
- HUD overlay (FPS, position, distance)

**Controls:**
- Toggle LIDAR: Show/hide sensor rays
- Toggle Path: Show/hide trail
- Toggle Grid: Show/hide background grid
- Animation Speed: 0.5x, 1x, 2x, 5x
- Screenshot: Save current view

### Analytics Panel

**Real-Time Metrics:**
- Current Reward (large LCD display)
- Episode Number
- Steps Taken
- Success Rate (gauge)
- Average Reward (last 100)

**Charts:**
- 📈 Rewards: Line chart with moving average
- 👣 Steps: Area chart
- ✅ Success Rate: Percentage over time
- 📝 Logs: Console output

**Export:**
- Export logs to text file
- Export charts as PNG
- Export data as CSV/JSON

---

## 🌐 Web Dashboard

### Features

1. **Real-Time Updates** via WebSocket
2. **Responsive Design** (mobile-friendly)
3. **Dark/Light Mode**
4. **Live Metrics** streaming
5. **Model Management**
6. **Remote Training Control**

### API Endpoints

```python
# Training Control
POST   /api/training/start
POST   /api/training/stop
GET    /api/training/status
GET    /api/training/logs

# Model Management
GET    /api/models
POST   /api/models/upload
GET    /api/models/{id}
DELETE /api/models/{id}

# Evaluation
POST   /api/evaluation/run
GET    /api/evaluation/results

# Data & Metrics
GET    /api/metrics/history
GET    /api/visualization/trajectory
GET    /api/export/report
```

### Usage Example

```javascript
// Start training via API
fetch('http://localhost:5000/api/training/start', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    episodes: 1000,
    learning_rate: 0.0005,
    batch_size: 128
  })
});

// Subscribe to live metrics
socket.on('metrics_update', (data) => {
  console.log(`Episode ${data.episode}: ${data.reward}`);
});
```

---

## 🗺️ Level Editor

### Tools

| Tool | Description | Shortcut |
|------|-------------|----------|
| 🟦 Obstacle | Draw rectangular obstacles | Click & drag |
| 🎯 Goal | Place goal points | Single click |
| 🤖 Robot | Set robot start position | Single click |
| 🗑️ Eraser | Remove elements | Click on element |

### Workflow

1. **Select Tool** from toolbar
2. **Draw/Place** elements on canvas
3. **Configure** properties (optional)
4. **Test Level** ▶️ button
5. **Save** 💾 to JSON file
6. **Load** 📂 existing levels

### Features

- ✅ Snap to Grid (toggleable)
- ✅ Grid Visibility (toggleable)
- ✅ Undo/Redo (Ctrl+Z / Ctrl+Y)
- ✅ Copy/Paste (Ctrl+C / Ctrl+V)
- ✅ Export/Import JSON
- ✅ Instant Testing

### File Format

```json
{
  "size": [800, 600],
  "robot_start": [100, 100],
  "obstacles": [
    {"x": 300, "y": 200, "width": 100, "height": 80}
  ],
  "goals": [
    {"x": 700, "y": 500}
  ]
}
```

---

## 🎬 Replay System

### Recording

```python
from gui.replay_system import ReplayRecorder

# Start recording
recorder = ReplayRecorder()
recorder.start_recording(metadata={
    'agent': 'DQN',
    'episode': 1,
    'date': '2025-12-06'
})

# Record frames during episode
for step in episode:
    recorder.record_frame(state, action, reward, info)

# Stop and save
filepath = recorder.stop_recording()
print(f"Saved to: {filepath}")
```

### Playback

```python
from gui.replay_system import ReplayPlayer

# Load replay
player = ReplayPlayer()
player.load_replay('replays/replay_20251206_143022.replay')

# Get metadata
print(player.get_metadata())

# Get statistics
stats = player.get_statistics()
print(f"Total reward: {stats['total_reward']}")
print(f"Average reward: {stats['average_reward']}")

# Play frames
while True:
    frame = player.next_frame()
    if frame is None:
        break
    # Render frame
    visualize(frame)
```

### Features

- ✅ Compressed storage (gzip)
- ✅ Metadata tagging
- ✅ Frame-by-frame playback
- ✅ Speed control (0.5x - 5x)
- ✅ Scrubbing/seeking
- ✅ Statistics calculation
- ✅ Export to video (MP4)
- ✅ Side-by-side comparison

---

## 🎯 Features Guide

### 1. Training Workflow

```
1. Configure environment settings
2. Select agent and hyperparameters
3. Click "Start Training" ▶️
4. Monitor real-time metrics
5. Observe visualization
6. Wait for completion or stop manually
7. Evaluate trained model 🧪
8. Export results 📤
```

### 2. Model Management

- **Save**: Automatic every N episodes
- **Load**: Drag & drop or file browser
- **Compare**: Side-by-side visualization
- **Export**: Share with team
- **Delete**: Remove old models

### 3. Visualization Options

**Canvas Settings:**
- Show/Hide LIDAR rays
- Show/Hide path history
- Show/Hide background grid
- Animation speed control
- Screenshot capture

**HUD Display:**
- FPS counter
- Robot position
- Distance to goal
- Current reward
- Episode number

### 4. Analytics & Metrics

**Real-Time:**
- Current reward (large display)
- Episode progress bar
- Steps counter
- Success indicator

**Historical:**
- Reward curve (with moving average)
- Steps per episode
- Success rate over time
- Loss curve
- Q-value distribution

### 5. Export & Sharing

**Export Options:**
- 📊 Charts as PNG
- 📝 Logs as TXT
- 📈 Data as CSV
- 📄 Report as JSON/PDF
- 🎬 Replay files
- 💾 Model checkpoints

---

## 🔧 Troubleshooting

### Common Issues

#### 1. GUI won't start
```bash
# Check PyQt6 installation
pip show PyQt6

# Reinstall if needed
pip uninstall PyQt6
pip install PyQt6
```

#### 2. No GPU detected
- Check CUDA installation
- Verify PyTorch GPU support: `torch.cuda.is_available()`
- Update GPU drivers

#### 3. Web dashboard connection failed
```bash
# Check if backend is running
curl http://localhost:5000/api/health

# Check port availability
netstat -an | grep 5000

# Restart backend
python web_dashboard/backend.py
```

#### 4. Charts not updating
- Check WebSocket connection
- Verify firewall settings
- Check browser console for errors

#### 5. Replay files corrupted
- Ensure training completed properly
- Check disk space
- Try re-recording

### Performance Tips

1. **Reduce Visualization FPS** for slower machines
2. **Disable LIDAR rays** if lagging
3. **Use smaller grid sizes** for testing
4. **Enable GPU** for faster training
5. **Close unnecessary applications**

### Getting Help

- 📚 Check documentation
- 🐛 Report bugs on GitHub
- 💬 Join Discord community
- 📧 Email support

---

## 🎉 Advanced Features

### Custom Themes

Create custom color schemes in settings:
```python
# Dark theme example
THEME = {
    'background': '#1e1e1e',
    'text': '#ffffff',
    'primary': '#0d7377',
    'success': '#00ff00',
    'danger': '#ff0000'
}
```

### Keyboard Macros

Record and playback action sequences for testing.

### Remote Monitoring

Access training from any device via web dashboard.

### Achievement System

Unlock badges and track progress:
- 🏅 First Training
- 🏆 100 Episodes
- 💯 90% Success Rate
- ⚡ Speed Demon
- 🎯 Perfect Episode

---

## 📝 Tips & Best Practices

1. **Save Frequently**: Enable auto-save
2. **Use Checkpoints**: Save every 100 episodes
3. **Monitor GPU Usage**: Check temperature
4. **Test Levels**: Before long training runs
5. **Record Replays**: For interesting episodes
6. **Export Results**: Backup your data
7. **Clean Up Models**: Delete old checkpoints

---

**Version**: 1.0.0  
**Last Updated**: December 6, 2025  
**License**: MIT

**Enjoy your AI training journey! 🚀**
