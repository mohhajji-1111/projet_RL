# 🎨 VISUALIZATION QUICK REFERENCE

## 🚀 One-Command Visualizations

### Launch Real-Time Dashboard
```bash
python scripts/visualize.py dashboard --log-dir results/logs --port 8050
```
Open: `http://localhost:8050`

### Generate All Visualizations
```bash
python scripts/visualize.py all --config configs/visualization_config.json
```

### Compare Algorithms
```bash
python scripts/visualize.py compare --config comparison_config.json
```

### Animate Trajectory
```bash
python scripts/visualize.py trajectory --log trajectory.json --output video.mp4
```

### Visualize Network
```bash
python scripts/visualize.py network --model trained_models/dqn.pt
```

### Analyze State-Space
```bash
python scripts/visualize.py statespace --agent agent.pt --trajectories traj.json --rewards rewards.txt
```

### Performance Dashboard
```bash
python scripts/visualize.py performance --config performance_config.json
```

---

## 📊 Python API Quick Examples

### Dashboard
```python
from src.visualization.dashboard_realtime import TrainingDashboard

dashboard = TrainingDashboard(log_dir="results/logs")
dashboard.update_data({'episode': 1, 'reward': 10.5, ...})
dashboard.run(port=8050)
```

### Comparisons
```python
from src.visualization.compare_runs import ComparisonVisualizer

viz = ComparisonVisualizer()
viz.compare_algorithms(runs={'DQN': 'dqn.json', 'Rainbow': 'rainbow.json'})
viz.ablation_study('baseline.json', {'No PER': 'no_per.json'})
```

### Trajectory
```python
from src.visualization.trajectory_animator import TrajectoryAnimator

animator = TrajectoryAnimator(fps=30)
animator.create_matplotlib_animation(trajectory, 'video.mp4', format='mp4')
animator.create_plotly_interactive(trajectory, 'interactive.html')
```

### Network
```python
from src.visualization.network_visualizer import NetworkVisualizer

viz = NetworkVisualizer(model)
viz.visualize_architecture('arch.png')
viz.visualize_weight_distributions('weights.png')
viz.visualize_gradient_flow('gradients.png')
```

### State-Space
```python
from src.visualization.state_space_analyzer import StateSpaceAnalyzer

analyzer = StateSpaceAnalyzer(state_bounds=((0, 10), (0, 10)))
analyzer.update_visitation(positions)
analyzer.visualize_state_visitation('heatmap.png')
analyzer.visualize_q_value_landscape(agent, action=0, goal=(9, 9))
analyzer.visualize_policy_arrows(agent, goal=(9, 9))
```

### Performance
```python
from src.visualization.performance_dashboard import PerformanceMetricsDashboard

dashboard = PerformanceMetricsDashboard()
dashboard.create_box_plots(rewards, 'Reward', 'box.png')
dashboard.create_learning_curves_with_ci(runs, 'Reward', 0.95, 'curves.png')
dashboard.create_convergence_analysis(data, window=50, threshold=0.01)
```

---

## 📦 What Gets Created

### Dashboard (Real-Time)
- 10+ live charts
- Multiple tabs
- Save/load sessions
- Export capability

### Comparisons
- `01_algorithm_comparison.png`
- `02_learning_curves.png`
- `03_performance_matrix.png`
- `04_distribution_comparison.png`

### Trajectory
- `navigation.mp4` (video)
- `navigation.gif` (animation)
- `navigation.html` (interactive)

### Network
- `01_architecture.png`
- `02_weight_distributions.png`
- `03_activations.png`
- `04_gradient_flow.png`
- `05_weights_*.png`

### State-Space
- `01_state_visitation.png`
- `02_q_landscape_action_*.png`
- `03_policy_arrows.png`
- `04_reward_distribution.png`
- `05_exploration_coverage.png`

### Performance
- `01_rewards_boxplot.png`
- `02_success_boxplot.png`
- `03_rewards_violin.png`
- `05_correlation_matrix.png`
- `06_learning_curves_ci.png`
- `07_convergence_analysis.png`
- `08_summary_table.png`

---

## 🎯 Common Workflows

### Workflow 1: Monitor Training
```bash
# Terminal 1
python scripts/train.py --config configs/base_config.yaml

# Terminal 2
python scripts/visualize.py dashboard --log-dir results/logs
```

### Workflow 2: Compare Runs
```bash
# Run experiments
python scripts/train.py --config configs/dqn_config.yaml
python scripts/train.py --config configs/rainbow_config.yaml

# Compare results
cat > comparison.json << EOF
{
  "runs": {
    "DQN": "results/logs/dqn_training.json",
    "Rainbow": "results/logs/rainbow_training.json"
  }
}
EOF

python scripts/visualize.py compare --config comparison.json
```

### Workflow 3: Paper Figures
```bash
# Generate high-res figures
python scripts/visualize.py all --config paper_config.json --output paper_figures/

# Outputs in 300 DPI, ready for publication
```

---

## 🔧 Configuration Files

### comparison_config.json
```json
{
  "runs": {
    "DQN": "results/logs/dqn_training.json",
    "Rainbow DQN": "results/logs/rainbow_training.json"
  }
}
```

### performance_config.json
```json
{
  "runs": {
    "Run 1": "results/logs/run1.json",
    "Run 2": "results/logs/run2.json",
    "Run 3": "results/logs/run3.json"
  }
}
```

### visualization_config.json (Master)
```json
{
  "comparison_runs": {...},
  "performance_runs": {...},
  "model_path": "trained_models/best_model.pt",
  "agent_path": "trained_models/dqn_agent.pt",
  "trajectory_log": "results/logs/trajectories.json",
  "rewards_file": "results/logs/rewards.txt",
  "trajectory_animations": {...}
}
```

---

## 📊 Customization

### Change DPI
```python
viz = ComparisonVisualizer(dpi=600)  # Ultra high-res
```

### Custom Colors
```python
viz.colors = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'success': '#06A77D'
}
```

### Export Formats
```python
plt.savefig('figure.pdf', dpi=300)  # LaTeX
plt.savefig('figure.svg')           # Inkscape
plt.savefig('figure.png', dpi=300)  # Presentation
```

---

## ⚡ Performance Tips

### Reduce Memory Usage
```python
# Lower grid resolution
analyzer = StateSpaceAnalyzer(grid_resolution=25)  # Default: 50

# Subsample trajectories
positions = positions[::5]  # Every 5th point
```

### Speed Up Animations
```python
# Lower FPS
animator = TrajectoryAnimator(fps=15)  # Default: 30

# Use GIF instead of MP4
animator.create_matplotlib_animation(..., format='gif')
```

---

## 🐛 Troubleshooting

### Dashboard Port In Use
```bash
python scripts/visualize.py dashboard --port 8051
```

### FFmpeg Not Found
```bash
# Windows
choco install ffmpeg

# Linux
sudo apt install ffmpeg

# Or use GIF
--format gif
```

### Out of Memory
```python
# Reduce grid resolution
StateSpaceAnalyzer(grid_resolution=25)

# Subsample data
data = data[::10]
```

---

## 📚 Documentation

- **Full Guide**: `docs/VISUALIZATION_GUIDE.md`
- **Summary**: `docs/VISUALIZATION_SUMMARY.md`
- **Quick Ref**: `docs/VISUALIZATION_QUICKREF.md` (this file)

### Get Help
```bash
python scripts/visualize.py --help
python scripts/visualize.py dashboard --help
python scripts/visualize.py compare --help
```

---

## 🎓 Key Features

- ✅ 300 DPI publication quality
- ✅ Color-blind friendly palettes
- ✅ Statistical significance testing
- ✅ Multiple export formats (PNG, PDF, MP4, GIF, HTML)
- ✅ Real-time monitoring
- ✅ Interactive visualizations
- ✅ Modular and extensible

---

## 📦 Installation

```bash
# Install dependencies
pip install -r requirements.txt
pip install -r requirements_viz.txt

# Test installation
python -c "from src.visualization.dashboard_realtime import TrainingDashboard; print('✓ OK')"
```

---

**Quick, Easy, Beautiful! 🎨📊✨**
