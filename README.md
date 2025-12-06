# 🤖 Robot Navigation with Reinforcement Learning

A comprehensive reinforcement learning framework for training autonomous robot navigation using DQN and Rainbow DQN algorithms.

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Configuration](#configuration)
- [Training](#training)
- [Evaluation](#evaluation)
- [Visualization](#visualization)
- [Architecture](#architecture)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

### Agents
- **DQN (Deep Q-Network)**: Classic deep reinforcement learning algorithm
- **Rainbow DQN**: Advanced agent with:
  - Double DQN
  - Dueling Networks
  - Noisy Networks for exploration
  - Prioritized Experience Replay support

### Environment
- **2D Navigation Environment**: Gymnasium-compatible environment
- **Dynamic Obstacles**: Moving obstacles with configurable patterns
- **LIDAR Sensor**: Simulated laser range finder
- **Configurable Physics**: Adjustable robot dynamics and environment parameters

### Training Features
- **Basic Training**: Standard DQN training loop
- **Adaptive Training**: Curriculum learning with automatic difficulty adjustment
- **Prioritized Replay Buffer**: More efficient learning from important experiences
- **Comprehensive Logging**: TensorBoard integration and custom metrics tracking

### Visualization
- **Advanced Renderer**: Professional pygame-based visualization
- **Particle Effects**: Visual feedback for events (collisions, goal reached)
- **Training GUI**: Real-time training monitoring with live plots
- **Trajectory Tracking**: Visual path history

## 📁 Project Structure

```
robot-navigation-rl/
├── 📁 src/
│   ├── agents/              # RL agent implementations
│   │   ├── base_agent.py    # Abstract base class
│   │   ├── dqn_agent.py     # Basic DQN
│   │   └── rainbow_agent.py # Rainbow DQN
│   ├── environment/         # Environment components
│   │   ├── navigation_env.py
│   │   ├── obstacles.py
│   │   └── sensors.py
│   ├── training/            # Training utilities
│   │   ├── trainer_base.py
│   │   ├── train_basic.py
│   │   └── train_adaptive.py
│   ├── visualization/       # Rendering and GUI
│   │   ├── renderer.py
│   │   ├── effects.py
│   │   └── gui.py
│   └── utils/               # Utility modules
│       ├── replay_buffer.py
│       ├── logger.py
│       └── metrics.py
│
├── 📁 notebooks/            # Jupyter notebooks
│   ├── 01_environment_test.ipynb
│   ├── 02_dqn_training.ipynb
│   └── ...
│
├── 📁 configs/              # Configuration files
│   ├── base_config.yaml
│   ├── rainbow_config.yaml
│   └── adaptive_config.yaml
│
├── 📁 scripts/              # Executable scripts
│   ├── train.py             # Main training
│   ├── evaluate.py          # Model evaluation
│   ├── generate_plots.py    # Visualization
│   └── demo.py              # Live demo
│
├── 📁 trained_models/       # Saved models
├── 📁 results/              # Training results
│   ├── figures/
│   ├── videos/
│   └── logs/
│
├── 📁 tests/                # Unit tests
├── requirements.txt
├── setup.py
├── README.md
└── LICENSE
```

## 🚀 Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (optional, for faster training)

### Install from source

```bash
# Clone the repository
git clone https://github.com/yourusername/robot-navigation-rl.git
cd robot-navigation-rl

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## 🎯 Quick Start

### 1. Test Environment

```python
from src.environment import NavigationEnv

env = NavigationEnv(render_mode='human')
state, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    state, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

env.close()
```

### 2. Train DQN Agent

```bash
python scripts/train.py --config configs/base_config.yaml
```

### 3. Evaluate Trained Model

```bash
python scripts/evaluate.py --model trained_models/basic/final.pt --render
```

### 4. Run Live Demo

```bash
python scripts/demo.py --model trained_models/basic/final.pt --episodes 5
```

## 📖 Usage

### Training

#### Basic Training
```bash
python scripts/train.py --config configs/base_config.yaml --seed 42
```

#### Rainbow Agent
```bash
python scripts/train.py --config configs/rainbow_config.yaml --device cuda
```

#### Adaptive Curriculum Learning
```bash
python scripts/train.py --config configs/adaptive_config.yaml
```

### Evaluation

```bash
# Evaluate with rendering
python scripts/evaluate.py --model trained_models/basic/final.pt \
    --agent-type dqn --episodes 20 --render

# Evaluate without rendering (faster)
python scripts/evaluate.py --model trained_models/rainbow/final.pt \
    --agent-type rainbow --episodes 100
```

### Generate Plots

```bash
python scripts/generate_plots.py \
    --log results/logs/experiment_metrics.json \
    --output results/figures
```

## ⚙️ Configuration

Configuration files use YAML format. Example:

```yaml
# configs/base_config.yaml
environment:
  width: 800
  height: 600
  robot_radius: 15.0
  max_speed: 5.0

agent:
  type: "dqn"
  learning_rate: 0.0001
  gamma: 0.99
  hidden_dims: [256, 256]

training:
  num_episodes: 1000
  batch_size: 64
  epsilon_start: 1.0
  epsilon_decay: 0.995
```

## 🎓 Training Details

### DQN Algorithm
- Experience replay buffer
- Target network with soft updates
- Epsilon-greedy exploration
- Huber loss for stability

### Rainbow DQN Improvements
- **Double DQN**: Reduces overestimation bias
- **Dueling Networks**: Separate value and advantage streams
- **Noisy Networks**: Learnable exploration
- **Prioritized Replay**: Focus on important transitions

### Curriculum Learning
The adaptive trainer implements curriculum learning:
1. **Easy Stage**: No obstacles, close goal
2. **Medium Stage**: Static obstacles
3. **Hard Stage**: Dynamic obstacles, complex scenarios

## 📊 Evaluation Metrics

The framework tracks:
- Episode rewards
- Episode lengths
- Success rate
- Training loss
- Q-value estimates
- TD errors
- Exploration rate

## 🎨 Visualization

### Training Curves
Automatically generated plots include:
- Reward progression
- Episode lengths
- Loss curves
- Q-value evolution

### Live Demo
Interactive visualization with:
- Robot trajectory
- Particle effects
- Real-time statistics
- Goal indicators

## 🏗️ Architecture

### Agent Architecture
```
Input (State)
    ↓
Feature Extraction (MLP)
    ↓
[Optional: Dueling Networks]
    ├─→ Value Stream
    └─→ Advantage Stream
    ↓
Q-Values (Actions)
```

### Training Loop
```
1. Initialize agent, environment, replay buffer
2. For each episode:
   a. Reset environment
   b. Select action (ε-greedy or noisy)
   c. Execute action, observe reward
   d. Store transition in buffer
   e. Sample batch and train
   f. Update target network
3. Save model and metrics
```

## 🧪 Testing

Run unit tests:
```bash
pytest tests/
```

## 📝 Examples

### Custom Agent

```python
from src.agents import BaseAgent
import torch.nn as nn

class CustomAgent(BaseAgent):
    def __init__(self, state_dim, action_dim):
        super().__init__(state_dim, action_dim)
        # Your custom implementation
    
    def select_action(self, state, epsilon=0.0):
        # Your action selection logic
        pass
    
    def train_step(self, batch):
        # Your training logic
        pass
```

### Custom Environment

```python
from src.environment import NavigationEnv

class CustomEnv(NavigationEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Add your customizations
    
    def _calculate_reward(self, distance, collision, goal_reached):
        # Custom reward function
        return reward
```

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI Gymnasium for the environment interface
- DeepMind for the DQN and Rainbow algorithms
- PyTorch team for the deep learning framework

## 📧 Contact

For questions or suggestions:
- Open an issue on GitHub
- Email: your.email@example.com

## 🔗 References

1. Mnih et al. (2015). "Human-level control through deep reinforcement learning"
2. Van Hasselt et al. (2016). "Deep Reinforcement Learning with Double Q-learning"
3. Wang et al. (2016). "Dueling Network Architectures for Deep Reinforcement Learning"
4. Hessel et al. (2018). "Rainbow: Combining Improvements in Deep Reinforcement Learning"

---

⭐ **Star this repository if you find it helpful!**
