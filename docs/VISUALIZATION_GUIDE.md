# 🎨 VISUALIZATION GUIDE

## Overview

This project includes a comprehensive suite of **publication-quality** visualization tools for reinforcement learning training analysis. All visualizations are designed to be:

- **300 DPI** publication-ready
- **Color-blind friendly** palettes
- **Professional** styling with clear labels
- **Interactive** where appropriate
- **Statistically rigorous** with significance testing

---

## 📊 Available Visualization Tools

### 1. Real-Time Training Dashboard 🚀

**Interactive Plotly/Dash dashboard** for monitoring training in real-time.

**Features:**
- 10+ real-time charts updating during training
- Multiple tabs (Overview, Learning Dynamics, State-Action, Performance)
- Dark theme optimized for long viewing sessions
- Save/load training sessions
- Export charts as PNG/PDF
- Compare multiple runs
- Automatic outlier detection

**Launch:**
```bash
python scripts/visualize.py dashboard --log-dir results/logs --port 8050
```

**Access:** Open browser at `http://localhost:8050`

**Metrics Displayed:**
- Episode Reward (with moving average)
- Success Rate (percentage)
- Steps per Episode (with min/max bands)
- Loss Evolution (Q-loss)
- Exploration Rate (epsilon decay)
- Action Distribution (pie chart)
- Q-Value Heatmap
- Learning Rate Schedule
- Replay Buffer Statistics
- Episode Duration

---

### 2. Comparison Visualizations 📈

**Side-by-side comparisons** with statistical significance testing.

**Features:**
- Compare multiple algorithms (DQN vs Rainbow)
- Statistical significance markers (*, **, ***)
- Error bars with confidence intervals
- Color-blind friendly palettes
- Grid layouts (2x2, 2x3)
- Ablation study support

**Generate:**
```bash
# Create comparison config
cat > comparison_config.json << EOF
{
  "runs": {
    "DQN": "results/logs/dqn_training.json",
    "Rainbow DQN": "results/logs/rainbow_training.json"
  }
}
EOF

python scripts/visualize.py compare --config comparison_config.json --output results/comparisons
```

**Comparisons Included:**
- Algorithm comparison (4 metrics)
- Environment comparison (static vs dynamic obstacles)
- Hyperparameter comparison
- Learning curves with confidence intervals
- Performance matrix heatmap
- Box plots with significance tests

---

### 3. Trajectory Animations 🎬

**Animated visualizations** of robot navigation paths.

**Output Formats:**
- **MP4** video (30 FPS, high quality)
- **GIF** animation (optimized size)
- **HTML** interactive (Plotly)

**Features:**
- Robot path colored by time
- LIDAR rays animated
- Speed indicator
- Collision points marked in red
- Successful paths in green
- State visitation heatmap
- 3D trajectories with Q-values

**Create Animation:**
```bash
python scripts/visualize.py trajectory \
  --log results/logs/trajectory.json \
  --output animations/navigation.mp4 \
  --format mp4
```

**Interactive HTML:**
```python
from src.visualization.trajectory_animator import create_interactive_trajectory
create_interactive_trajectory('trajectory.json', 'output.html')
```

---

### 4. Neural Network Visualization 🧠

**Comprehensive network analysis** and visualization.

**Features:**
- Architecture diagram (layers and connections)
- Weight distributions (histograms per layer)
- Activation patterns (for sample inputs)
- Gradient flow analysis (detect vanishing/exploding)
- Weight matrix heatmaps
- SVD analysis (rank analysis)

**Generate:**
```bash
python scripts/visualize.py network \
  --model trained_models/dqn_agent.pt \
  --output results/network_vis
```

**Outputs:**
- `01_architecture.png` - Network diagram
- `02_weight_distributions.png` - Weight histograms + statistics
- `03_activations.png` - Activation patterns
- `04_gradient_flow.png` - Gradient health check
- `05_weights_*.png` - Individual layer weight matrices

---

### 5. State-Space Analysis 🗺️

**Advanced analytics** for state exploration and policy behavior.

**Features:**
- State visitation heatmap (with smoothing)
- Q-value landscape (3D surface plot)
- Policy visualization (arrow field)
- Reward distribution (histogram + KDE)
- Exploration coverage (% states visited)
- Policy entropy over time

**Generate:**
```bash
python scripts/visualize.py statespace \
  --agent trained_models/dqn_agent.pt \
  --trajectories results/logs/trajectories.json \
  --rewards results/logs/rewards.txt \
  --output results/state_space
```

**Outputs:**
- `01_state_visitation.png` - Heatmap of visited states
- `02_q_landscape_action_*.png` - Q-value landscapes (per action)
- `03_policy_arrows.png` - Policy as arrow field
- `04_reward_distribution.png` - Reward statistics
- `05_exploration_coverage.png` - Coverage analysis

---

### 6. Performance Metrics Dashboard 📊

**Statistical analysis** with publication-quality plots.

**Features:**
- Box plots (distribution comparison)
- Violin plots (detailed distributions)
- Swarm plots (all data points)
- Correlation matrix (feature relationships)
- Learning curves with confidence intervals
- Convergence analysis
- Summary statistics table

**Generate:**
```bash
python scripts/visualize.py performance \
  --config performance_config.json \
  --output results/performance
```

**Example Config:**
```json
{
  "runs": {
    "Run 1": "results/logs/run1.json",
    "Run 2": "results/logs/run2.json",
    "Run 3": "results/logs/run3.json"
  }
}
```

**Outputs:**
- Box plots with significance tests
- Violin plots with quartiles
- Correlation heatmap
- Learning curves with 95% CI
- Convergence analysis
- Summary table

---

## 🚀 Quick Start

### Installation

```bash
# Install base requirements first
pip install -r requirements.txt

# Install visualization requirements
pip install -r requirements_viz.txt
```

### Generate Everything

```bash
# Create master config
python scripts/visualize.py all --config configs/visualization_config.json --output results
```

This generates:
- All comparisons
- Performance dashboard
- Network visualizations
- State-space analysis
- Trajectory animations

---

## 📝 Usage Examples

### Example 1: Monitor Training in Real-Time

```bash
# Terminal 1: Start training
python scripts/train.py --config configs/base_config.yaml

# Terminal 2: Launch dashboard
python scripts/visualize.py dashboard --log-dir results/logs
```

### Example 2: Compare DQN vs Rainbow

```python
from src.visualization.compare_runs import compare_dqn_vs_rainbow

compare_dqn_vs_rainbow(
    'results/logs/dqn.json',
    'results/logs/rainbow.json',
    'results/comparisons'
)
```

### Example 3: Create Trajectory Video

```python
from src.visualization.trajectory_animator import TrajectoryAnimator

animator = TrajectoryAnimator(fps=30)
animator.create_matplotlib_animation(
    trajectory_data,
    'output.mp4',
    format='mp4',
    show_lidar=True,
    show_speed=True
)
```

### Example 4: Analyze Network

```python
from src.visualization.network_visualizer import NetworkVisualizer
import torch

model = torch.load('trained_models/dqn_agent.pt')
viz = NetworkVisualizer(model)
viz.create_complete_report('results/network_vis')
```

### Example 5: State-Space Analysis

```python
from src.visualization.state_space_analyzer import StateSpaceAnalyzer

analyzer = StateSpaceAnalyzer(state_bounds=((0, 10), (0, 10)))
analyzer.update_visitation(trajectory_positions)
analyzer.visualize_state_visitation('heatmap.png')
analyzer.visualize_q_value_landscape(agent, action=0, goal=(9, 9))
analyzer.visualize_policy_arrows(agent, goal=(9, 9))
```

---

## 🎨 Customization

### Change Color Scheme

All visualizers use color-blind friendly palettes. To customize:

```python
from src.visualization.compare_runs import ComparisonVisualizer

viz = ComparisonVisualizer()
viz.colors['blue'] = '#YOUR_COLOR'
viz.palette = ['#COLOR1', '#COLOR2', '#COLOR3']
```

### Adjust DPI

```python
viz = ComparisonVisualizer(dpi=600)  # Ultra high-res
```

### Custom Export

```python
fig = viz.compare_algorithms(runs)
fig.savefig('output.pdf', dpi=300, format='pdf')  # PDF for LaTeX
fig.savefig('output.svg', format='svg')  # SVG for editing
```

---

## 📊 Publication Tips

### For Papers

1. **Use PDF format** for vector graphics:
   ```python
   plt.savefig('figure.pdf', dpi=300, bbox_inches='tight')
   ```

2. **Include statistical tests**:
   - All comparison tools automatically include p-values
   - Significance markers: * (p<0.05), ** (p<0.01), *** (p<0.001)

3. **Consistent styling**:
   - All tools use the same font (Arial/Helvetica)
   - Color-blind friendly palettes
   - Clear labels and legends

4. **High resolution**:
   - Default: 300 DPI (publication quality)
   - Increase for posters: 600 DPI

### For Presentations

1. **Use PNG format** with transparency:
   ```python
   plt.savefig('figure.png', dpi=150, transparent=True)
   ```

2. **Interactive dashboards**:
   - Launch real-time dashboard for live demos
   - Use HTML trajectory animations

3. **Animations**:
   - MP4 for embedded videos
   - GIF for simple loops

---

## 🔧 Troubleshooting

### Dashboard won't start

```bash
# Check port availability
netstat -ano | findstr :8050

# Use different port
python scripts/visualize.py dashboard --port 8051
```

### MP4 export fails

```bash
# Install ffmpeg
# Windows: choco install ffmpeg
# Linux: sudo apt install ffmpeg

# Or use GIF instead
python scripts/visualize.py trajectory --format gif
```

### Out of memory

```bash
# Reduce grid resolution
analyzer = StateSpaceAnalyzer(grid_resolution=25)  # Default: 50

# Subsample trajectories
positions = positions[::10]  # Every 10th point
```

---

## 📚 API Reference

### Dashboard
- `TrainingDashboard(log_dir, update_interval)` - Main dashboard class
- `update_data(metrics)` - Update with new metrics
- `load_from_log(log_file)` - Load historical data
- `run(host, port, debug)` - Start server

### Comparisons
- `ComparisonVisualizer(style, dpi)` - Main visualizer
- `compare_algorithms(runs, metrics, save_path)` - Compare algorithms
- `compare_environments(runs, save_path)` - Compare environments
- `ablation_study(baseline, ablations, save_path)` - Ablation analysis
- `export_comparison_report(runs, output_dir)` - Generate full report

### Trajectory
- `TrajectoryAnimator(environment_size, fps, dpi)` - Animator
- `create_matplotlib_animation(trajectory, save_path, format)` - Create animation
- `create_plotly_interactive(trajectory, save_path)` - Interactive HTML
- `create_3d_trajectory(trajectory, save_path, z_metric)` - 3D plot

### Network
- `NetworkVisualizer(model, dpi)` - Network visualizer
- `visualize_architecture(save_path)` - Architecture diagram
- `visualize_weight_distributions(save_path)` - Weight histograms
- `visualize_activations(input_sample, save_path)` - Activation patterns
- `visualize_gradient_flow(save_path)` - Gradient analysis

### State-Space
- `StateSpaceAnalyzer(state_bounds, grid_resolution, dpi)` - Analyzer
- `update_visitation(states)` - Update heatmap
- `visualize_state_visitation(save_path)` - Heatmap
- `visualize_q_value_landscape(agent, action, goal, save_path)` - Q-landscape
- `visualize_policy_arrows(agent, goal, save_path)` - Policy field

### Performance
- `PerformanceMetricsDashboard(dpi)` - Dashboard
- `create_box_plots(data_dict, metric_name, save_path)` - Box plots
- `create_violin_plots(data_dict, metric_name, save_path)` - Violin plots
- `create_learning_curves_with_ci(runs_dict, metric_name, confidence, save_path)` - Learning curves

---

## 🎓 Learn More

- **Matplotlib**: https://matplotlib.org/stable/gallery/index.html
- **Seaborn**: https://seaborn.pydata.org/examples/index.html
- **Plotly**: https://plotly.com/python/
- **Dash**: https://dash.plotly.com/

---

## 📄 License

All visualization tools are MIT licensed. Feel free to use in your research!

---

**Happy Visualizing! 🎨📊🚀**
