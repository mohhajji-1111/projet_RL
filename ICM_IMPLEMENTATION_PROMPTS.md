# 🤖 ICM IMPLEMENTATION - COPILOT PROMPTS

Copy each prompt below and paste into GitHub Copilot Chat one by one.

---

## 📝 PROMPT 1: Main ICM Agent File

```
@workspace Create src/agents/curiosity_agent.py

Implement a complete DQN agent with Intrinsic Curiosity Module (ICM) for robot navigation.

REQUIREMENTS:

1. Inherit from DQNAgent class (from src/agents/dqn_agent.py)

2. ICM consists of 3 neural networks:
   - Feature Network φ: state(10D) → features(32D)
     Architecture: Dense(64, ReLU) → Dense(32, ReLU)
   
   - Inverse Model: [φ(s_t), φ(s_{t+1})] → action
     Architecture: concat(64D) → Dense(64, ReLU) → Dense(4, Softmax)
     Loss: CrossEntropyLoss
   
   - Forward Model: [φ(s_t), action] → φ(s_{t+1})
     Architecture: concat(36D) → Dense(64, ReLU) → Dense(32)
     Loss: MSE

3. Intrinsic Reward Computation:
   r_intrinsic = ||forward_pred - actual_features||²
   r_total = r_extrinsic + β * r_intrinsic
   where β = 0.2 (hyperparameter)

4. Class structure:
```python
class CuriosityAgent(DQNAgent):
    def __init__(self, state_dim, action_dim, config):
        super().__init__(state_dim, action_dim, config)
        self.curiosity_beta = config.get('curiosity_beta', 0.2)
        self.feature_dim = 32
        
        # Build ICM networks
        self.feature_network = self._build_feature_network()
        self.inverse_model = self._build_inverse_model()
        self.forward_model = self._build_forward_model()
        
        # ICM optimizer
        self.icm_optimizer = optim.Adam(...)
        
    def _build_feature_network(self) -> nn.Module
    def _build_inverse_model(self) -> nn.Module
    def _build_forward_model(self) -> nn.Module
    def compute_intrinsic_reward(self, state, action, next_state) -> float
    def train_icm(self, batch) -> Dict[str, float]
    def train_step(self) -> float  # Override to include ICM training
```

5. Key features:
   - Compute intrinsic reward for each transition
   - Train ICM networks alongside DQN
   - Log intrinsic rewards and ICM losses
   - Save/load ICM networks with checkpoints
   - Type hints for all methods
   - Comprehensive docstrings
   - Error handling
   - Input validation

6. Include test function at the end to verify functionality.

Generate complete, production-ready code (400-500 lines) with detailed comments explaining the ICM algorithm.
```

---

## 📝 PROMPT 2: Configuration File

```
@workspace Create configs/curiosity_config.yaml

Configuration file for training the Curiosity Agent (DQN + ICM).

Include:

1. Environment settings:
   - grid_size: 10
   - num_obstacles: 5
   - num_goals: 3
   - obstacle_speed: 0.5
   - max_steps: 200

2. Agent settings:
   - type: "curiosity"
   - Base DQN parameters (learning_rate, gamma, epsilon, etc.)
   - ICM specific parameters:
     * curiosity_beta: 0.2 (intrinsic reward weight)
     * curiosity_eta: 1.0 (forward loss scale)
     * curiosity_lambda: 0.1 (inverse loss scale)
     * feature_dim: 32
     * icm_lr: 0.001
     * normalize_intrinsic: true
   - Network architecture: hidden_sizes [128, 128, 64]

3. Training settings:
   - total_episodes: 1500
   - Curriculum learning stages (3 stages)
   - eval_frequency: 100
   - save_frequency: 100

4. Visualization settings:
   - plot_intrinsic_rewards: true
   - plot_exploration_coverage: true
   - plot_curiosity_heatmap: true

5. Paths for saving models, logs, figures

Format as valid YAML with comments explaining each parameter.
```

---

## 📝 PROMPT 3: Training Script

```
@workspace Create scripts/train_curiosity.py

Command-line script to train the Curiosity Agent.

REQUIREMENTS:

1. Parse command-line arguments:
   - --config (path to config file, default: configs/curiosity_config.yaml)
   - --episodes (number of episodes, overrides config)
   - --save-dir (directory for saving models)
   - --device (cuda/cpu)
   - --seed (random seed for reproducibility)

2. Main training loop:
   - Load configuration
   - Create environment (NavigationEnv)
   - Create CuriosityAgent
   - Initialize logger
   - Set random seeds
   
   For each episode:
     - Reset environment
     - Collect transitions
     - Store in replay buffer (with intrinsic rewards)
     - Train agent if buffer has enough samples
     - Log metrics (episode reward, intrinsic reward, success)
     - Evaluate every N episodes
     - Save checkpoint every N episodes
     - Print progress with tqdm
   
3. After training:
   - Save final model
   - Generate training plots
   - Save metrics to CSV
   - Print summary statistics

4. Include:
   - Error handling (try-except blocks)
   - Graceful shutdown (Ctrl+C handling)
   - Resume from checkpoint (if exists)
   - Logging to both console and file

5. Usage example in docstring:
```bash
python scripts/train_curiosity.py --config configs/curiosity_config.yaml --episodes 1500
```

Generate clean, well-documented code (~200 lines).
```

---

## 📝 PROMPT 4: Evaluation Script

```
@workspace Create scripts/evaluate_curiosity.py

Script to evaluate a trained Curiosity Agent and compare with baseline DQN.

REQUIREMENTS:

1. Arguments:
   - --curiosity-model (path to curiosity agent checkpoint)
   - --baseline-model (path to baseline DQN checkpoint)
   - --episodes (number of evaluation episodes, default: 100)
   - --render (whether to render episodes)
   - --save-video (save videos of best episodes)
   - --output-dir (where to save results)

2. Evaluation metrics:
   - Success rate
   - Average reward
   - Average steps per episode
   - Collision rate
   - Goals reached
   - State space coverage (% of grid visited)
   - Path efficiency (vs optimal A* path)

3. Comparison:
   - Load both models
   - Evaluate each on same scenarios
   - Statistical comparison (t-test)
   - Generate comparison plots:
     * Box plots (reward distribution)
     * Success rate bars
     * Exploration heatmaps (side-by-side)
     * Trajectory visualizations

4. Output:
   - Print summary table
   - Save plots to output directory
   - Export metrics to CSV
   - Save comparison report (markdown)

5. Example usage:
```bash
python scripts/evaluate_curiosity.py \
  --curiosity-model results/models/curiosity/best.pth \
  --baseline-model results/models/dqn/best.pth \
  --episodes 100 \
  --save-video
```

Generate complete code (~250 lines) with visualization functions.
```

---

## 📝 PROMPT 5: Visualization for ICM

```
@workspace Create src/visualization/curiosity_plots.py

Module for generating ICM-specific visualizations.

Implement these plotting functions:

1. plot_intrinsic_rewards(intrinsic_rewards, save_path):
   - Line plot of intrinsic rewards over training
   - Show moving average (window=100)
   - Highlight peaks (high curiosity moments)
   - Save as PNG

2. plot_exploration_coverage(visited_states, grid_size, save_path):
   - Heatmap showing which states were visited
   - Color intensity = visit frequency
   - Overlay robot start and goal positions
   - Save as PNG

3. plot_curiosity_heatmap(episode_data, save_path):
   - 2D heatmap of intrinsic rewards per grid cell
   - Shows where agent was most curious
   - Overlay obstacles and goals
   - Save as PNG

4. plot_icm_losses(forward_losses, inverse_losses, save_path):
   - Dual-axis plot: forward loss + inverse loss
   - Moving averages
   - Save as PNG

5. plot_reward_comparison(extrinsic_rewards, intrinsic_rewards, save_path):
   - Stacked area chart showing contribution of each reward type
   - Total reward line on top
   - Save as PNG

6. plot_exploration_comparison(baseline_coverage, curiosity_coverage, save_path):
   - Side-by-side heatmaps
   - Baseline DQN vs Curiosity Agent
   - Show improvement in exploration
   - Save as PNG

7. animate_curiosity_episode(episode_data, save_path):
   - Create MP4 video of episode
   - Show intrinsic reward as color intensity around robot
   - Display metrics overlay (current intrinsic reward)
   - Save as MP4

Requirements:
- Use matplotlib and seaborn for plots
- Publication-quality (300 DPI)
- Consistent color scheme (use colorblind-friendly palette)
- Clear labels and legends
- Grid lines where appropriate
- Type hints for all functions
- Comprehensive docstrings

Generate complete module (~300 lines).
```

---

## 📝 PROMPT 6: Update Training Loop

```
@workspace Update src/training/trainer.py to support CuriosityAgent

Modify the existing trainer to work with both DQN and CuriosityAgent.

CHANGES NEEDED:

1. In __init__ method:
   - Detect agent type (check if isinstance(agent, CuriosityAgent))
   - Initialize ICM-specific metrics storage if curiosity agent

2. In train_episode method:
   - If curiosity agent, log intrinsic rewards
   - Store both extrinsic and intrinsic rewards separately

3. In _log_metrics method:
   - If curiosity agent, include:
     * Average intrinsic reward
     * ICM forward loss
     * ICM inverse loss
     * Exploration coverage

4. In save_checkpoint method:
   - If curiosity agent, save ICM networks too

5. Add new method: _compute_exploration_coverage()
   - Track which grid cells were visited
   - Return coverage percentage

6. Keep backward compatibility:
   - Should work with regular DQN without changes
   - Only activate ICM features if CuriosityAgent detected

IMPORTANT:
- Don't break existing DQN training
- Add ICM features conditionally
- Type hints and docstrings for new code
- Minimal changes to existing code

Generate the modified trainer.py with clear comments showing what was changed.
```

---

## 📝 PROMPT 7: Colab Notebook

```
@workspace Create notebooks/train_curiosity_colab.ipynb

Google Colab notebook for training Curiosity Agent with GPU.

NOTEBOOK STRUCTURE:

## 1. Setup
- Check GPU availability
- Mount Google Drive
- Install dependencies (if needed)
- Clone/sync code repository

## 2. Configuration
- Load curiosity_config.yaml
- Show configuration (pretty print)
- Allow user to modify key parameters (widgets)

## 3. Environment Test
- Create NavigationEnv
- Visualize environment
- Show LIDAR sensor
- Test reset/step functions

## 4. Agent Initialization
- Create CuriosityAgent
- Print network architectures
- Show parameter count
- Test forward pass

## 5. Training
- Initialize trainer
- Training loop with progress bar (tqdm)
- Live plotting (update every 50 episodes):
  * Episode rewards (extrinsic + intrinsic)
  * Success rate
  * ICM losses
- Save checkpoints to Drive every 100 episodes

## 6. Visualization
- Load best checkpoint
- Run evaluation (10 episodes)
- Generate all plots:
  * Training curves
  * Intrinsic rewards
  * Exploration heatmap
  * Sample trajectories
- Display in notebook

## 7. Comparison with Baseline
- Load baseline DQN (if available)
- Side-by-side comparison
- Statistical tests
- Visualize differences

## 8. Export Results
- Save all plots to Drive
- Export metrics to CSV
- Generate summary markdown
- Download best model

FEATURES:
- Clear markdown explanations for each section
- Code cells with comments
- Output cells showing results
- Interactive widgets where useful
- Error handling (graceful failures)
- Automatic session management (save before timeout)

Generate complete notebook with all cells ready to run.
```

---

## 📝 PROMPT 8: Analysis Notebook

```
@workspace Create notebooks/analyze_curiosity_results.ipynb

Jupyter notebook for analyzing ICM training results.

SECTIONS:

## 1. Load Data
- Load training logs (CSV)
- Load checkpoints
- Load baseline results for comparison

## 2. Training Analysis
- Plot learning curves (with confidence intervals)
- Identify convergence point
- Analyze training stability
- Compare convergence speed (curiosity vs baseline)

## 3. Intrinsic Rewards Analysis
- Distribution of intrinsic rewards
- Correlation with episode success
- Evolution over training (early vs late)
- Identify curiosity peaks (what caused them?)

## 4. Exploration Analysis
- State space coverage over time
- Heatmaps (early, middle, late training)
- Compare with baseline DQN coverage
- Quantify exploration improvement

## 5. ICM Network Analysis
- Feature space visualization (t-SNE)
- Inverse model accuracy over training
- Forward model prediction error trends
- Feature importance analysis

## 6. Performance Metrics
- Success rate comparison
- Average reward comparison
- Steps per episode comparison
- Statistical significance tests (t-test, effect size)

## 7. Trajectory Analysis
- Compare typical trajectories (curiosity vs baseline)
- Path efficiency analysis
- Collision patterns
- Goal reaching strategies

## 8. Ablation Study (if data available)
- Effect of β (curiosity weight)
- Effect of feature dimension
- Effect of ICM learning rate

## 9. Visualizations
- Generate all publication-quality plots
- Export high-resolution figures (300 DPI)
- Create summary infographic

## 10. Report Generation
- Markdown summary of findings
- Key insights and recommendations
- Suggestions for improvement

REQUIREMENTS:
- Interactive plots (Plotly where useful)
- Clean, well-commented code
- Markdown explanations between cells
- Statistical rigor (proper tests, confidence intervals)
- Professional visualizations

Generate complete notebook with example outputs.
```

---

## 📝 PROMPT 9: Documentation

```
@workspace Create docs/ICM_GUIDE.md

Comprehensive guide for understanding and using the ICM implementation.

CONTENTS:

## 1. Introduction
- What is Intrinsic Curiosity?
- Why use ICM?
- When to use vs when not to use

## 2. Theory
- Mathematical formulation
- Algorithm explanation (step-by-step)
- Intuition behind each component
- Diagrams (ASCII art or describe what to draw)

## 3. Architecture
- Network diagrams
- Data flow
- Training process
- Integration with DQN

## 4. Hyperparameters
- List of all ICM hyperparameters
- Explanation of each
- Recommended values
- Tuning guide

## 5. Installation
- Dependencies
- Setup instructions
- Verification tests

## 6. Usage
- Training a curiosity agent (code examples)
- Evaluation (code examples)
- Visualization (code examples)
- Common workflows

## 7. Configuration
- Explanation of curiosity_config.yaml
- How to modify settings
- Example configurations for different scenarios

## 8. Results Interpretation
- How to read training logs
- Understanding intrinsic rewards
- Diagnosing issues (low exploration, high losses, etc.)

## 9. Troubleshooting
- Common issues and solutions
- FAQ
- Debugging tips

## 10. Advanced Usage
- Combining with Rainbow DQN
- Multi-goal environments
- Custom reward shaping
- Transfer learning

## 11. Performance Tips
- GPU optimization
- Batch processing
- Memory management
- Training speed improvements

## 12. References
- Original ICM paper
- Related work
- Further reading

Format in clean Markdown with:
- Clear headings
- Code blocks with syntax highlighting
- Tables where appropriate
- Numbered/bulleted lists
- Inline code formatting
- Links to relevant files

Generate comprehensive guide (~1500 words).
```

---

## 📝 PROMPT 10: Tests

```
@workspace Create tests/test_curiosity_agent.py

Comprehensive unit tests for CuriosityAgent using pytest.

TEST CATEGORIES:

1. Initialization Tests:
   - test_curiosity_agent_init()
   - test_icm_networks_creation()
   - test_hyperparameters_loaded()

2. Network Architecture Tests:
   - test_feature_network_output_shape()
   - test_inverse_model_output_shape()
   - test_forward_model_output_shape()
   - test_network_forward_pass()

3. Intrinsic Reward Tests:
   - test_compute_intrinsic_reward()
   - test_intrinsic_reward_non_negative()
   - test_intrinsic_reward_novel_vs_seen()
   - test_reward_augmentation()

4. Training Tests:
   - test_train_icm()
   - test_icm_loss_computation()
   - test_train_step_with_icm()
   - test_gradient_flow()

5. Integration Tests:
   - test_full_episode_with_curiosity()
   - test_replay_buffer_integration()
   - test_checkpoint_save_load()

6. Edge Cases:
   - test_empty_replay_buffer()
   - test_terminal_state_handling()
   - test_nan_inf_handling()
   - test_invalid_inputs()

7. Comparison Tests:
   - test_curiosity_vs_baseline_exploration()
   - test_intrinsic_reward_impact()

REQUIREMENTS:
- Use pytest fixtures for setup
- Mock environment where appropriate
- Test with different random seeds
- Check tensor shapes and types
- Verify gradient computation
- Test save/load functionality
- Include performance tests (optional)

Example test:
```python
def test_compute_intrinsic_reward():
    agent = CuriosityAgent(state_dim=10, action_dim=4, config={})
    state = np.random.randn(10)
    action = 0
    next_state = np.random.randn(10)
    
    r_int = agent.compute_intrinsic_reward(state, action, next_state)
    
    assert isinstance(r_int, float)
    assert r_int >= 0
    assert not np.isnan(r_int)
    assert not np.isinf(r_int)
```

Generate complete test file (~300 lines) with all tests.
```

---

## 🎯 USAGE INSTRUCTIONS

### Step 1: Create Files in Order
Copy prompts 1-10 and paste into Copilot Chat in this order:

```bash
1. Prompt 1 → curiosity_agent.py (MAIN FILE)
2. Prompt 2 → curiosity_config.yaml
3. Prompt 3 → train_curiosity.py
4. Prompt 4 → evaluate_curiosity.py
5. Prompt 5 → curiosity_plots.py
6. Prompt 6 → Update trainer.py
7. Prompt 7 → train_curiosity_colab.ipynb
8. Prompt 8 → analyze_curiosity_results.ipynb
9. Prompt 9 → ICM_GUIDE.md
10. Prompt 10 → test_curiosity_agent.py
```

### Step 2: Test After Each File
```bash
# After Prompt 1:
python -c "from src.agents.curiosity_agent import CuriosityAgent; print('✓')"

# After Prompt 3:
python scripts/train_curiosity.py --episodes 10

# After Prompt 10:
pytest tests/test_curiosity_agent.py -v
```

### Step 3: Full Training
```bash
# Train curiosity agent
python scripts/train_curiosity.py --episodes 1500

# Evaluate and compare
python scripts/evaluate_curiosity.py \
  --curiosity-model results/models/curiosity/best.pth \
  --baseline-model results/models/dqn/best.pth \
  --episodes 100
```

---

## ⏱️ ESTIMATED TIME

- Prompts 1-5: 2 hours (core implementation)
- Prompts 6-8: 2 hours (integration & notebooks)
- Prompts 9-10: 1 hour (docs & tests)
- Testing & debugging: 2-3 hours
- Training (1500 episodes): 2-3 hours (on GPU)

**Total: 1-2 days of work** 🚀

---

## ✅ SUCCESS CHECKLIST

- [ ] All 10 files generated
- [ ] curiosity_agent.py passes tests
- [ ] Can train for 10 episodes without errors
- [ ] Intrinsic rewards are computed correctly
- [ ] ICM networks train (losses decrease)
- [ ] Checkpoints save/load properly
- [ ] Visualizations generate correctly
- [ ] Colab notebook runs end-to-end
- [ ] Documentation is clear
- [ ] Ready for full training!

---

## 🆘 IF COPILOT MAKES MISTAKES

```
@workspace The code in [file] has this error: [paste error message]

The issue is at line [X]. Please fix it by [explain what's wrong].

Also ensure:
- Type hints are correct
- Imports are complete
- Error handling is robust
```

---

## 📊 EXPECTED RESULTS

After completing all 10 prompts and training, you should see:

### Training Improvements:
- **Exploration**: 30-50% more state coverage vs baseline DQN
- **Success Rate**: 10-20% improvement in sparse reward scenarios
- **Convergence**: Faster learning in early episodes
- **Robustness**: Better generalization to new environments

### Key Metrics:
```
Baseline DQN:
- Success Rate: ~65%
- Avg Reward: ~120
- Coverage: ~45%

Curiosity Agent:
- Success Rate: ~80%
- Avg Reward: ~160
- Coverage: ~70%
```

### Generated Files:
```
projet_RL/
├── src/agents/curiosity_agent.py        (450 lines)
├── configs/curiosity_config.yaml        (120 lines)
├── scripts/train_curiosity.py           (220 lines)
├── scripts/evaluate_curiosity.py        (280 lines)
├── src/visualization/curiosity_plots.py (320 lines)
├── src/training/trainer.py              (UPDATED)
├── notebooks/
│   ├── train_curiosity_colab.ipynb
│   └── analyze_curiosity_results.ipynb
├── docs/ICM_GUIDE.md                    (1500+ lines)
└── tests/test_curiosity_agent.py        (350 lines)
```

---

## 🔬 THEORY RECAP

### ICM Components:

1. **Feature Network (φ)**:
   - Compresses raw state into learnable features
   - Removes task-irrelevant information
   - Makes forward/inverse models easier to train

2. **Inverse Model**:
   - Predicts action from state transitions
   - Forces features to capture controllable aspects
   - Loss: Cross-entropy between predicted and actual action

3. **Forward Model**:
   - Predicts next feature state from current + action
   - Prediction error = intrinsic reward (curiosity)
   - High error = novel/surprising transition = explore!

### Mathematical Formulation:

```
φ: S → F (feature encoding)
g: F_t × F_{t+1} → A (inverse model)
f: F_t × A_t → F_{t+1} (forward model)

Losses:
L_I = CrossEntropy(g(φ(s_t), φ(s_{t+1})), a_t)
L_F = ||f(φ(s_t), a_t) - φ(s_{t+1})||²

Intrinsic Reward:
r_i(s_t, a_t, s_{t+1}) = η/2 × ||f(φ(s_t), a_t) - φ(s_{t+1})||²

Total Reward:
r_total = r_extrinsic + β × r_intrinsic
```

### Why It Works:

- Novel states → high prediction error → high intrinsic reward
- Agent learns to seek novelty → better exploration
- Feature network ignores uncontrollable noise
- Works in sparse reward environments

---

## 🎓 ADVANCED TIPS

### Hyperparameter Tuning:

**curiosity_beta (β)**: Weight of intrinsic reward
- Too low (< 0.1): Not enough exploration
- Too high (> 0.5): Agent ignores extrinsic rewards
- Recommended: 0.2 for navigation, 0.5 for very sparse rewards

**feature_dim**: Size of feature space
- Too small (< 16): Can't capture state complexity
- Too large (> 128): Overfitting, slow training
- Recommended: 32-64 for most tasks

**icm_lr**: ICM learning rate
- Should be similar to or slightly lower than DQN lr
- Recommended: 0.0001 - 0.001

### Common Pitfalls:

1. **ICM learns too fast**: Features stop changing, no more curiosity
   - Solution: Lower icm_lr or add L2 regularization

2. **Intrinsic rewards explode**: Agent gets stuck exploring
   - Solution: Normalize intrinsic rewards or clip them

3. **No improvement over baseline**: Task has dense rewards
   - Solution: ICM not needed, or lower β

4. **ICM losses don't decrease**: Network capacity too small
   - Solution: Increase hidden layer sizes

### Integration with Other Techniques:

- **+ Rainbow DQN**: Better value estimation + better exploration = 🚀
- **+ Prioritized Replay**: Prioritize high intrinsic reward transitions
- **+ Curriculum Learning**: Start with high β, decay over time
- **+ Multi-Agent**: Each agent has own ICM, share feature network

---

## 📚 REFERENCES

### Papers:
1. **ICM Original Paper** (Pathak et al., 2017):
   "Curiosity-driven Exploration by Self-supervised Prediction"
   https://arxiv.org/abs/1705.05363

2. **RND** (Burda et al., 2018):
   "Exploration by Random Network Distillation"
   https://arxiv.org/abs/1810.12894

3. **NGU** (Badia et al., 2020):
   "Never Give Up: Learning Directed Exploration Strategies"
   https://arxiv.org/abs/2002.06038

### Code Implementations:
- OpenAI Baselines: https://github.com/openai/baselines
- Stable-Baselines3: https://github.com/DLR-RM/stable-baselines3
- CleanRL: https://github.com/vwxyzjn/cleanrl

---

## 🚀 READY TO START?

1. **Copy Prompt 1** above
2. **Open GitHub Copilot Chat** in VS Code
3. **Paste the prompt**
4. **Watch the magic happen!** ✨

Good luck with your ICM implementation! 🤖🧠

---

**Last Updated**: December 6, 2025  
**Version**: 1.0  
**Estimated Completion Time**: 1-2 days  
**Difficulty**: Advanced 🔥🔥🔥
