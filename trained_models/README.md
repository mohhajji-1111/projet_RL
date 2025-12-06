# Trained Models

This directory contains saved model checkpoints.

## Structure

- `basic/` - Basic DQN models
- `dynamic/` - Models trained with dynamic obstacles
- `adaptive/` - Adaptive curriculum learning models

## File Format

Models are saved as PyTorch `.pt` files containing:
- Model state dict
- Optimizer state dict
- Training metadata

## Usage

Load a model:
```python
from src.agents import DQNAgent

agent = DQNAgent(state_dim=8, action_dim=4)
agent.load('trained_models/basic/final.pt')
```
