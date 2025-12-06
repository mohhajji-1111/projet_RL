# 🧠 Intrinsic Curiosity Module (ICM) - Complete Guide

A comprehensive guide for understanding and using the ICM implementation in this project.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Theory](#2-theory)
3. [Architecture](#3-architecture)
4. [Hyperparameters](#4-hyperparameters)
5. [Installation](#5-installation)
6. [Usage](#6-usage)
7. [Configuration](#7-configuration)
8. [Results Interpretation](#8-results-interpretation)
9. [Troubleshooting](#9-troubleshooting)
10. [Advanced Usage](#10-advanced-usage)
11. [Performance Tips](#11-performance-tips)
12. [References](#12-references)

---

## 1. Introduction

### What is Intrinsic Curiosity?

**Intrinsic Curiosity** is a mechanism that enables reinforcement learning agents to explore environments more effectively by generating internal "curiosity-driven" rewards. Instead of relying solely on sparse external rewards, curious agents are motivated to visit novel states where they can learn something new.

### Why use ICM?

Traditional RL agents struggle in environments with:
- **Sparse rewards**: Little feedback about what's good/bad
- **Long horizons**: Delayed gratification
- **Deceptive reward structures**: Local optima

ICM addresses these by:
- ✅ Encouraging exploration of unknown states
- ✅ Providing dense intrinsic reward signals
- ✅ Learning useful state representations
- ✅ Improving sample efficiency

### When to use vs when not to use

**✅ Use ICM when:**
- Environment has sparse rewards
- Agent gets stuck in local optima
- Exploration is critical for success
- State space is large and underexplored
- Task requires discovering distant goals

**❌ Don't use ICM when:**
- Environment already has dense rewards
- Optimal policy is straightforward
- Stochastic environments (curiosity can be noisy)
- Computational resources are limited
- Task is fully supervised

---

## 2. Theory

### Mathematical Formulation

ICM consists of three neural networks:

#### 1. Feature Network (φ)

Encodes raw states into learned features:

```
φ: S → F
where S = state space, F = feature space
```

**Purpose**: Remove task-irrelevant information (e.g., background noise, lighting).

#### 2. Inverse Dynamics Model (g)

Predicts action from state transition:

```
g: (φ(s_t), φ(s_{t+1})) → â_t
```

**Loss**:
```
L_inverse = CrossEntropy(g(φ(s_t), φ(s_{t+1})), a_t)
```

**Purpose**: Forces features to capture aspects of the environment that the agent can control.

#### 3. Forward Dynamics Model (f)

Predicts next feature state:

```
f: (φ(s_t), a_t) → φ̂(s_{t+1})
```

**Loss**:
```
L_forward = ||f(φ(s_t), a_t) - φ(s_{t+1})||²
```

**Purpose**: Prediction error serves as intrinsic reward (curiosity signal).

### Intrinsic Reward

```
r_intrinsic(s_t, a_t, s_{t+1}) = η/2 × ||f(φ(s_t), a_t) - φ(s_{t+1})||²
```

Where:
- `η` = forward loss scale (default: 1.0)
- High prediction error → Novel state → High curiosity!

### Total Reward

```
r_total = r_extrinsic + β × r_intrinsic
```

Where:
- `r_extrinsic` = environment reward
- `β` = curiosity weight (default: 0.2)

### ICM Training

Combined loss:
```
L_ICM = λ × L_inverse + η × L_forward
```

Where:
- `λ` = inverse loss weight (default: 0.1)
- `η` = forward loss weight (default: 1.0)

### Algorithm (Step-by-Step)

```
1. Initialize DQN and ICM networks (φ, g, f)
2. For each episode:
   3. For each step:
      4. Select action: a_t = π(s_t)
      5. Take action: s_{t+1}, r_ext = env.step(a_t)
      
      6. Compute intrinsic reward:
         φ_t = φ(s_t)
         φ_{t+1} = φ(s_{t+1})
         φ̂_{t+1} = f(φ_t, a_t)
         r_int = ||φ̂_{t+1} - φ_{t+1}||²
      
      7. Total reward: r = r_ext + β × r_int
      
      8. Store transition: (s_t, a_t, r, s_{t+1})
      
      9. Train ICM:
         L_inv = CrossEntropy(g(φ_t, φ_{t+1}), a_t)
         L_fwd = ||φ̂_{t+1} - φ_{t+1}||²
         L_ICM = λ × L_inv + η × L_fwd
         Update φ, g, f
      
      10. Train DQN:
          L_DQN = (Q(s_t, a_t) - target)²
          Update Q-network
```

### Intuition

**Why does this work?**

1. **Novel states** → Agent can't predict what happens next → **High forward error**
2. **High forward error** → **High intrinsic reward** → Agent visits more
3. **Repeated visits** → Agent learns to predict → **Low forward error** → Moves on
4. **Result**: Agent systematically explores the environment

**The inverse model** ensures features capture controllable aspects (not random noise).

---

## 3. Architecture

### Network Diagrams

```
┌─────────────────────────────────────────────────────────┐
│                    ICM Architecture                      │
└─────────────────────────────────────────────────────────┘

Input: s_t, a_t, s_{t+1}

         s_t                            s_{t+1}
          │                                │
          ▼                                ▼
    ┌──────────┐                    ┌──────────┐
    │ Feature  │                    │ Feature  │
    │ Network  │                    │ Network  │
    │    φ     │                    │    φ     │
    └──────────┘                    └──────────┘
          │                                │
          ▼                                ▼
       φ(s_t)                          φ(s_{t+1})
          │                                │
          ├────────────────────────────────┤
          │                                │
          ▼                                ▼
    ┌──────────────────────────────────────┐
    │        Inverse Model (g)             │
    │  Predicts: â_t from φ_t, φ_{t+1}    │
    └──────────────────────────────────────┘
          │
          ▼
     â_t (predicted action)
     
     Loss: CrossEntropy(â_t, a_t)


       φ(s_t)         a_t (one-hot)
          │               │
          └───────┬───────┘
                  ▼
    ┌──────────────────────────────┐
    │   Forward Model (f)          │
    │  Predicts: φ̂_{t+1}          │
    └──────────────────────────────┘
          │
          ▼
     φ̂(s_{t+1})
     
     Intrinsic Reward: ||φ̂_{t+1} - φ_{t+1}||²
```

### Data Flow

```
Training Episode:
1. Environment → state (10D vector)
2. Feature Network → features (32D vector)
3. Forward Model → predicted next features
4. Compute intrinsic reward (prediction error)
5. Add to replay buffer (with augmented reward)
6. Sample batch → Train DQN
7. Sample batch → Train ICM (forward + inverse)
8. Repeat
```

### Integration with DQN

```python
class CuriosityAgent(DQNAgent):
    # Inherits DQN functionality
    # Adds ICM on top
    
    def train_step():
        # 1. Train DQN (as usual)
        dqn_loss = self._train_dqn_step(batch)
        
        # 2. Train ICM (new)
        icm_losses = self.train_icm(batch)
        
        return {**dqn_loss, **icm_losses}
```

---

## 4. Hyperparameters

### ICM Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `curiosity_beta` | 0.2 | 0.0 - 1.0 | Weight of intrinsic reward |
| `curiosity_eta` | 1.0 | 0.1 - 10.0 | Forward loss scale |
| `curiosity_lambda` | 0.1 | 0.01 - 1.0 | Inverse loss scale |
| `feature_dim` | 32 | 16 - 128 | Feature space dimension |
| `icm_lr` | 0.001 | 0.0001 - 0.01 | ICM learning rate |
| `normalize_intrinsic` | True | True/False | Normalize intrinsic rewards |

### Detailed Explanations

#### curiosity_beta (β)

**What it does**: Balances extrinsic vs intrinsic rewards.

```
r_total = r_extrinsic + β × r_intrinsic
```

**Tuning**:
- `β = 0.0`: No curiosity (baseline DQN)
- `β = 0.1`: Mild curiosity
- `β = 0.2`: **Recommended for navigation**
- `β = 0.5`: Strong curiosity (sparse rewards)
- `β = 1.0`: Equal weight (very sparse rewards)

**Too low**: Agent doesn't explore enough
**Too high**: Agent ignores task rewards, explores aimlessly

#### curiosity_eta (η)

**What it does**: Scales forward dynamics loss.

**Tuning**:
- Increase if forward model learns too slowly
- Decrease if forward model overfits
- Usually keep at 1.0

#### curiosity_lambda (λ)

**What it does**: Scales inverse dynamics loss.

**Tuning**:
- Lower than η (typically 10:1 ratio)
- Inverse model is easier to train
- Adjust if inverse model accuracy is poor

#### feature_dim

**What it does**: Size of learned feature representation.

**Tuning**:
- **16-32**: Simple environments (grid worlds)
- **32-64**: **Recommended for most tasks**
- **64-128**: Complex visual environments
- Monitor forward/inverse losses to validate

**Too small**: Can't capture state complexity
**Too large**: Overfitting, slow training

#### icm_lr

**What it does**: Learning rate for ICM networks.

**Tuning**:
- Should be similar to DQN learning rate
- Start with 0.001
- Increase if ICM trains too slowly
- Decrease if losses oscillate

#### normalize_intrinsic

**What it does**: Normalizes intrinsic rewards using running statistics.

```python
r_int_normalized = (r_int - mean) / (std + ε)
```

**Recommended**: Keep True for stable training.

### Recommended Values by Scenario

#### Navigation (Grid World)
```yaml
curiosity_beta: 0.2
feature_dim: 32
icm_lr: 0.001
```

#### Sparse Rewards (e.g., Montezuma's Revenge)
```yaml
curiosity_beta: 0.5
feature_dim: 64
icm_lr: 0.0005
```

#### Dense Rewards (e.g., Cart-Pole)
```yaml
# Don't use ICM - not needed!
```

---

## 5. Installation

### Dependencies

```bash
# Core dependencies
torch >= 1.9.0
numpy >= 1.19.0
gymnasium >= 0.26.0

# Visualization
matplotlib >= 3.3.0
seaborn >= 0.11.0

# Testing
pytest >= 6.2.0
```

### Setup Instructions

1. **Clone repository**:
```bash
git clone https://github.com/your-repo/robot-navigation-rl.git
cd robot-navigation-rl
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Verify installation**:
```bash
python -c "from src.agents.curiosity_agent import CuriosityAgent; print('✅ ICM installed')"
```

4. **Run tests**:
```bash
pytest tests/test_curiosity_agent.py -v
```

Expected output:
```
tests/test_curiosity_agent.py::test_curiosity_agent_init PASSED
tests/test_curiosity_agent.py::test_compute_intrinsic_reward PASSED
tests/test_curiosity_agent.py::test_train_icm PASSED
...
======================== 27 passed in 15.2s ========================
```

---

## 6. Usage

### Training a Curiosity Agent

#### Basic Example

```python
from src.environment.navigation_env import NavigationEnv
from src.agents.curiosity_agent import CuriosityAgent

# Create environment
env = NavigationEnv(width=800, height=600)

# Create curiosity agent
config = {
    'curiosity_beta': 0.2,
    'feature_dim': 32,
    'icm_lr': 0.001
}
agent = CuriosityAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n,
    config=config
)

# Training loop
for episode in range(1000):
    state = env.reset()
    episode_reward = 0
    done = False
    
    while not done:
        # Select action
        action = agent.select_action(state, epsilon=0.1)
        
        # Take step
        next_state, reward, done, info = env.step(action)
        
        # Compute intrinsic reward
        intrinsic_reward = agent.compute_intrinsic_reward(
            state, action, next_state
        )
        
        # Total reward
        total_reward = reward + agent.curiosity_beta * intrinsic_reward
        episode_reward += total_reward
        
        # Store and train
        agent.replay_buffer.push(state, action, total_reward, next_state, done)
        if len(agent.replay_buffer) > 1000:
            agent.train_step(batch_size=64)
        
        state = next_state
    
    print(f"Episode {episode}: Reward = {episode_reward:.2f}")
```

#### Using Training Script

```bash
# Train with default config
python scripts/train_curiosity.py --config configs/curiosity_config.yaml

# Train for 2000 episodes
python scripts/train_curiosity.py --episodes 2000

# Train on GPU
python scripts/train_curiosity.py --device cuda

# Resume from checkpoint
python scripts/train_curiosity.py --resume
```

### Evaluation

#### Compare with Baseline

```bash
python scripts/evaluate_curiosity.py \
  --curiosity-model results/models/curiosity/best.pth \
  --baseline-model results/models/dqn/best.pth \
  --episodes 100
```

#### Evaluation Code

```python
from src.agents.curiosity_agent import CuriosityAgent

# Load trained agent
agent = CuriosityAgent(state_dim=10, action_dim=4, config={})
agent.load_checkpoint('results/models/curiosity/best.pth')

# Evaluate
env = NavigationEnv(width=800, height=600)
total_rewards = []

for _ in range(100):
    state = env.reset()
    episode_reward = 0
    done = False
    
    while not done:
        action = agent.select_action(state, epsilon=0.0)  # Greedy
        next_state, reward, done, info = env.step(action)
        episode_reward += reward
        state = next_state
    
    total_rewards.append(episode_reward)

print(f"Average Reward: {np.mean(total_rewards):.2f}")
print(f"Success Rate: {sum(r > 0 for r in total_rewards) / 100:.2%}")
```

### Visualization

```python
from src.visualization.curiosity_plots import (
    plot_intrinsic_rewards,
    plot_exploration_coverage,
    plot_curiosity_heatmap
)

# Plot intrinsic rewards
plot_intrinsic_rewards(
    intrinsic_rewards=training_data['intrinsic_rewards'],
    save_path='results/figures/intrinsic_rewards.png'
)

# Plot exploration coverage
plot_exploration_coverage(
    visited_states=training_data['visited_states'],
    grid_size=10,
    save_path='results/figures/coverage.png',
    robot_start=(0, 0),
    goals=[(9, 9)]
)

# Plot curiosity heatmap
plot_curiosity_heatmap(
    episode_data={
        'positions': [(1, 1), (2, 2), ...],
        'intrinsic_rewards': [0.5, 0.3, ...]
    },
    grid_size=10,
    save_path='results/figures/curiosity_heatmap.png'
)
```

---

## 7. Configuration

### Understanding curiosity_config.yaml

```yaml
agent:
  type: "curiosity"
  
  # DQN parameters
  learning_rate: 0.0001
  gamma: 0.99
  epsilon_start: 1.0
  epsilon_end: 0.01
  
  # ICM parameters
  curiosity_beta: 0.2      # Intrinsic reward weight
  curiosity_eta: 1.0        # Forward loss scale
  curiosity_lambda: 0.1     # Inverse loss scale
  feature_dim: 32           # Feature space size
  icm_lr: 0.001            # ICM learning rate
  normalize_intrinsic: true # Normalize rewards
```

### Modifying Settings

#### Increase Exploration

```yaml
curiosity_beta: 0.5  # Up from 0.2
```

#### Adjust Feature Complexity

```yaml
feature_dim: 64  # Up from 32 for complex environments
```

#### Faster ICM Learning

```yaml
icm_lr: 0.002  # Up from 0.001
```

### Example Configurations

#### Configuration 1: Sparse Rewards

```yaml
agent:
  curiosity_beta: 0.5
  feature_dim: 64
  icm_lr: 0.001
  epsilon_decay: 0.99  # Slower decay
```

#### Configuration 2: Dense Rewards

```yaml
agent:
  curiosity_beta: 0.1
  feature_dim: 32
  icm_lr: 0.0005
```

#### Configuration 3: Large State Space

```yaml
agent:
  curiosity_beta: 0.3
  feature_dim: 128
  icm_lr: 0.0005
  hidden_sizes: [256, 256, 128]
```

---

## 8. Results Interpretation

### Reading Training Logs

```
Episode 100/1500 | Avg Reward: 45.23 | Success Rate: 35% | 
ICM Forward Loss: 0.0234 | ICM Inverse Loss: 0.1456
```

**What to look for**:
- ✅ Reward increasing over time
- ✅ Success rate improving
- ✅ ICM losses decreasing (learning)
- ❌ Reward stuck → Adjust β
- ❌ Losses not decreasing → Adjust icm_lr

### Understanding Intrinsic Rewards

**High intrinsic rewards** (early training):
- Agent exploring new states
- Good sign! Exploration is working

**Decreasing intrinsic rewards** (late training):
- Agent has seen most states
- Normal behavior
- Should still maintain some curiosity for new scenarios

**Consistently low intrinsic rewards**:
- Agent might be stuck
- Increase β or check feature network

### Diagnosing Issues

#### Issue 1: Agent Not Exploring

**Symptoms**:
- Low intrinsic rewards
- Repeating same trajectories
- Poor coverage

**Solutions**:
1. Increase `curiosity_beta` (0.2 → 0.5)
2. Check feature network is training
3. Verify intrinsic rewards are being added

#### Issue 2: Agent Ignoring Task

**Symptoms**:
- High intrinsic rewards
- Good coverage but low success rate
- Exploring aimlessly

**Solutions**:
1. Decrease `curiosity_beta` (0.5 → 0.2)
2. Increase extrinsic rewards
3. Add shaped rewards

#### Issue 3: ICM Not Learning

**Symptoms**:
- Forward/inverse losses not decreasing
- Constant high intrinsic rewards

**Solutions**:
1. Increase `icm_lr` (0.001 → 0.002)
2. Check ICM optimizer is updating
3. Verify gradients are flowing
4. Increase `feature_dim`

#### Issue 4: Training Unstable

**Symptoms**:
- Losses oscillating
- Reward variance high
- NaN values

**Solutions**:
1. Decrease `icm_lr`
2. Enable `normalize_intrinsic`
3. Clip intrinsic rewards
4. Add gradient clipping

---

## 9. Troubleshooting

### Common Issues

#### Q1: ImportError: No module named 'src.agents.curiosity_agent'

**Solution**:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
# Or
python -m scripts.train_curiosity
```

#### Q2: CUDA out of memory

**Solution**:
```yaml
# Reduce batch size
batch_size: 32  # Down from 64

# Or use CPU
device: "cpu"
```

#### Q3: Intrinsic rewards are NaN

**Solution**:
```yaml
normalize_intrinsic: true
clip_intrinsic_reward: true
intrinsic_reward_max: 1.0
```

#### Q4: Agent performs worse than baseline

**Possible causes**:
1. β too high → Decrease to 0.1
2. Environment has dense rewards → Don't use ICM
3. Need more training → Increase episodes
4. ICM not learning → Check losses

### FAQ

**Q: Should I use ICM for all RL tasks?**

A: No. ICM is beneficial for sparse reward environments. If your task already has dense rewards, ICM may hurt performance.

**Q: How long does ICM take to train?**

A: ~20-30% slower than baseline DQN due to additional network updates.

**Q: Can I combine ICM with other algorithms?**

A: Yes! ICM can be integrated with PPO, A3C, Rainbow DQN, etc.

**Q: What if my state space is continuous?**

A: ICM works well with continuous states. The feature network learns to encode them.

**Q: How do I know if ICM is working?**

A: Check if:
1. Intrinsic rewards are non-zero
2. Exploration coverage increases
3. Agent discovers distant goals
4. Performance improves over baseline

### Debugging Tips

1. **Enable debug logging**:
```bash
python scripts/train_curiosity.py --debug
```

2. **Visualize features** (t-SNE):
```python
from sklearn.manifold import TSNE

# Extract features
features = [agent.feature_network(state) for state in states]
features_np = torch.stack(features).detach().cpu().numpy()

# Visualize
tsne = TSNE(n_components=2)
features_2d = tsne.fit_transform(features_np)
plt.scatter(features_2d[:, 0], features_2d[:, 1])
```

3. **Monitor ICM statistics**:
```python
icm_stats = agent.get_icm_stats()
print(f"Forward Loss: {icm_stats['forward_loss']:.4f}")
print(f"Inverse Loss: {icm_stats['inverse_loss']:.4f}")
```

---

## 10. Advanced Usage

### Combining with Rainbow DQN

```python
from src.agents.rainbow_dqn_agent import RainbowDQNAgent

class CuriosityRainbowAgent(RainbowDQNAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add ICM components
        self.feature_network = ...
        self.inverse_model = ...
        self.forward_model = ...
```

### Multi-Goal Environments

```python
# Compute curiosity for each goal separately
for goal in goals:
    goal_intrinsic_reward = agent.compute_intrinsic_reward(
        state, action, next_state, goal=goal
    )
```

### Custom Reward Shaping

```python
# Combine curiosity with potential-based shaping
def shaped_reward(state, next_state, reward, intrinsic):
    potential = compute_potential(next_state) - compute_potential(state)
    return reward + beta * intrinsic + gamma * potential
```

### Transfer Learning

```python
# Pre-train feature network on source task
source_agent = CuriosityAgent(...)
source_agent.train(source_env)

# Transfer to target task
target_agent = CuriosityAgent(...)
target_agent.feature_network.load_state_dict(
    source_agent.feature_network.state_dict()
)
```

---

## 11. Performance Tips

### GPU Optimization

```python
# Use mixed precision
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    losses = agent.train_step(batch_size=64)
```

### Batch Processing

```python
# Process multiple environments in parallel
from multiprocessing import Pool

def train_env(env_id):
    env = create_env(env_id)
    return run_episode(env, agent)

with Pool(processes=4) as pool:
    results = pool.map(train_env, range(4))
```

### Memory Management

```python
# Limit replay buffer size
replay_buffer = ReplayBuffer(capacity=50000)  # Not 1M

# Use float32 instead of float64
torch.set_default_dtype(torch.float32)

# Delete unused tensors
del old_batch
torch.cuda.empty_cache()
```

### Training Speed

1. **Reduce batch size** initially, increase later
2. **Use target network updates** less frequently
3. **Profile code** to find bottlenecks:
```python
import cProfile
cProfile.run('agent.train_step()')
```

---

## 12. References

### Papers

1. **Original ICM Paper** (Pathak et al., 2017):
   - Title: "Curiosity-driven Exploration by Self-supervised Prediction"
   - Link: https://arxiv.org/abs/1705.05363
   - Key contribution: Intrinsic motivation via prediction error

2. **RND** (Burda et al., 2018):
   - Title: "Exploration by Random Network Distillation"
   - Link: https://arxiv.org/abs/1810.12894
   - Alternative curiosity method

3. **NGU** (Badia et al., 2020):
   - Title: "Never Give Up: Learning Directed Exploration Strategies"
   - Link: https://arxiv.org/abs/2002.06038
   - State-of-the-art exploration

4. **Empowerment** (Mohamed & Rezende, 2015):
   - Title: "Variational Information Maximisation for Intrinsically Motivated RL"
   - Link: https://arxiv.org/abs/1509.08731

### Code Implementations

- **OpenAI Baselines**: https://github.com/openai/baselines
- **Stable-Baselines3**: https://github.com/DLR-RM/stable-baselines3
- **CleanRL**: https://github.com/vwxyzjn/cleanrl

### Further Reading

- **Sutton & Barto** - "Reinforcement Learning: An Introduction"
- **Schmidhuber** - "Formal Theory of Creativity, Fun, and Intrinsic Motivation"
- **Oudeyer & Kaplan** - "What is Intrinsic Motivation?"

### Tutorials

- Lilian Weng's Blog: https://lilianweng.github.io/posts/2020-06-07-exploration-drl/
- Spinning Up (OpenAI): https://spinningup.openai.com/

---

## Appendix: Quick Reference

### Key Equations

```
Intrinsic Reward:
r_int = η/2 × ||f(φ(s_t), a_t) - φ(s_{t+1})||²

Total Reward:
r_total = r_ext + β × r_int

ICM Loss:
L_ICM = λ × L_inverse + η × L_forward

Forward Loss:
L_forward = MSE(f(φ(s_t), a_t), φ(s_{t+1}))

Inverse Loss:
L_inverse = CrossEntropy(g(φ(s_t), φ(s_{t+1})), a_t)
```

### Default Hyperparameters

```yaml
curiosity_beta: 0.2
curiosity_eta: 1.0
curiosity_lambda: 0.1
feature_dim: 32
icm_lr: 0.001
normalize_intrinsic: true
```

### Command Cheat Sheet

```bash
# Train
python scripts/train_curiosity.py --config configs/curiosity_config.yaml

# Evaluate
python scripts/evaluate_curiosity.py --curiosity-model results/models/curiosity/best.pth

# Test
pytest tests/test_curiosity_agent.py -v

# Visualize
python -c "from src.visualization.curiosity_plots import *; ..."
```

---

**Version**: 1.0  
**Last Updated**: December 6, 2025  
**Maintainer**: Your Name  
**License**: MIT

For questions or issues, please open an issue on GitHub or contact [email].

---

Happy Exploring! 🚀🧠
