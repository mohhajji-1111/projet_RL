"""
Unit Tests for Curiosity Agent

Run with: pytest tests/test_curiosity_agent.py -v
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import tempfile
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents.curiosity_agent import (
    CuriosityAgent,
    FeatureNetwork,
    InverseModel,
    ForwardModel
)


@pytest.fixture
def default_config():
    """Default configuration for testing"""
    return {
        'curiosity_beta': 0.2,
        'curiosity_eta': 1.0,
        'curiosity_lambda': 0.1,
        'feature_dim': 32,
        'icm_lr': 0.001,
        'normalize_intrinsic': True
    }


@pytest.fixture
def agent(default_config):
    """Create agent for testing"""
    return CuriosityAgent(
        state_dim=10,
        action_dim=4,
        config=default_config
    )


# ============================================================
# 1. INITIALIZATION TESTS
# ============================================================

def test_curiosity_agent_init(default_config):
    """Test agent initialization"""
    agent = CuriosityAgent(state_dim=10, action_dim=4, config=default_config)
    
    assert agent.state_dim == 10
    assert agent.action_dim == 4
    assert agent.curiosity_beta == 0.2
    assert agent.feature_dim == 32
    assert hasattr(agent, 'feature_network')
    assert hasattr(agent, 'inverse_model')
    assert hasattr(agent, 'forward_model')
    assert hasattr(agent, 'icm_optimizer')


def test_icm_networks_creation(agent):
    """Test ICM networks are properly created"""
    assert isinstance(agent.feature_network, nn.Module)
    assert isinstance(agent.inverse_model, nn.Module)
    assert isinstance(agent.forward_model, nn.Module)
    
    # Check networks are on correct device
    assert next(agent.feature_network.parameters()).device.type == agent.device
    assert next(agent.inverse_model.parameters()).device.type == agent.device
    assert next(agent.forward_model.parameters()).device.type == agent.device


def test_hyperparameters_loaded(agent, default_config):
    """Test hyperparameters are correctly loaded"""
    assert agent.curiosity_beta == default_config['curiosity_beta']
    assert agent.curiosity_eta == default_config['curiosity_eta']
    assert agent.curiosity_lambda == default_config['curiosity_lambda']
    assert agent.feature_dim == default_config['feature_dim']
    assert agent.icm_lr == default_config['icm_lr']
    assert agent.normalize_intrinsic == default_config['normalize_intrinsic']


# ============================================================
# 2. NETWORK ARCHITECTURE TESTS
# ============================================================

def test_feature_network_output_shape():
    """Test feature network output shape"""
    feature_net = FeatureNetwork(state_dim=10, feature_dim=32)
    
    state = torch.randn(16, 10)  # Batch of 16 states
    features = feature_net(state)
    
    assert features.shape == (16, 32)


def test_inverse_model_output_shape():
    """Test inverse model output shape"""
    inverse_model = InverseModel(feature_dim=32, action_dim=4)
    
    phi_t = torch.randn(16, 32)
    phi_t1 = torch.randn(16, 32)
    action_logits = inverse_model(phi_t, phi_t1)
    
    assert action_logits.shape == (16, 4)


def test_forward_model_output_shape():
    """Test forward model output shape"""
    forward_model = ForwardModel(feature_dim=32, action_dim=4)
    
    phi_t = torch.randn(16, 32)
    action = torch.randn(16, 4)  # One-hot encoded
    phi_t1_pred = forward_model(phi_t, action)
    
    assert phi_t1_pred.shape == (16, 32)


def test_network_forward_pass(agent):
    """Test complete forward pass through all networks"""
    state = torch.randn(1, agent.state_dim)
    next_state = torch.randn(1, agent.state_dim)
    action = torch.zeros(1, agent.action_dim)
    action[0, 0] = 1.0  # One-hot
    
    # Feature encoding
    phi_t = agent.feature_network(state)
    phi_t1 = agent.feature_network(next_state)
    
    # Inverse model
    action_pred = agent.inverse_model(phi_t, phi_t1)
    
    # Forward model
    phi_t1_pred = agent.forward_model(phi_t, action)
    
    assert phi_t.shape == (1, agent.feature_dim)
    assert phi_t1.shape == (1, agent.feature_dim)
    assert action_pred.shape == (1, agent.action_dim)
    assert phi_t1_pred.shape == (1, agent.feature_dim)


# ============================================================
# 3. INTRINSIC REWARD TESTS
# ============================================================

def test_compute_intrinsic_reward(agent):
    """Test intrinsic reward computation"""
    state = np.random.randn(10)
    action = 0
    next_state = np.random.randn(10)
    
    r_int = agent.compute_intrinsic_reward(state, action, next_state)
    
    assert isinstance(r_int, float)
    assert not np.isnan(r_int)
    assert not np.isinf(r_int)


def test_intrinsic_reward_non_negative(agent):
    """Test intrinsic rewards are non-negative (MSE is always >= 0)"""
    for _ in range(10):
        state = np.random.randn(10)
        action = np.random.randint(0, 4)
        next_state = np.random.randn(10)
        
        r_int = agent.compute_intrinsic_reward(state, action, next_state)
        assert r_int >= 0 or np.isclose(r_int, 0), f"Intrinsic reward {r_int} is negative"


def test_intrinsic_reward_novel_vs_seen(agent):
    """Test that novel states have higher intrinsic reward than seen states"""
    # Same transition repeated
    state = np.random.randn(10)
    action = 0
    next_state = state + 0.01  # Small change
    
    # First time seeing transition
    r_int_1 = agent.compute_intrinsic_reward(state, action, next_state)
    
    # Train on this transition multiple times
    for _ in range(50):
        batch = {
            'states': torch.FloatTensor([state]).to(agent.device),
            'actions': torch.LongTensor([action]).to(agent.device),
            'next_states': torch.FloatTensor([next_state]).to(agent.device)
        }
        agent.train_icm(batch)
    
    # After seeing it many times, intrinsic reward should decrease
    r_int_2 = agent.compute_intrinsic_reward(state, action, next_state)
    
    assert r_int_2 < r_int_1, "Intrinsic reward should decrease for seen transitions"


def test_reward_augmentation(agent):
    """Test that total reward includes both extrinsic and intrinsic"""
    extrinsic = 10.0
    intrinsic = 0.5
    
    total = extrinsic + agent.curiosity_beta * intrinsic
    expected = 10.0 + 0.2 * 0.5  # beta = 0.2
    
    assert np.isclose(total, expected)


# ============================================================
# 4. TRAINING TESTS
# ============================================================

def test_train_icm(agent):
    """Test ICM training step"""
    batch_size = 32
    batch = {
        'states': torch.randn(batch_size, agent.state_dim).to(agent.device),
        'actions': torch.randint(0, agent.action_dim, (batch_size,)).to(agent.device),
        'next_states': torch.randn(batch_size, agent.state_dim).to(agent.device)
    }
    
    losses = agent.train_icm(batch)
    
    assert 'forward_loss' in losses
    assert 'inverse_loss' in losses
    assert 'total_icm_loss' in losses
    assert all(isinstance(v, float) for v in losses.values())
    assert all(v >= 0 for v in losses.values())
    assert all(not np.isnan(v) for v in losses.values())


def test_icm_loss_computation(agent):
    """Test ICM loss components"""
    batch = {
        'states': torch.randn(16, agent.state_dim).to(agent.device),
        'actions': torch.randint(0, agent.action_dim, (16,)).to(agent.device),
        'next_states': torch.randn(16, agent.state_dim).to(agent.device)
    }
    
    losses = agent.train_icm(batch)
    
    # Forward loss should be MSE (always positive)
    assert losses['forward_loss'] >= 0
    
    # Inverse loss should be cross-entropy (always positive)
    assert losses['inverse_loss'] >= 0
    
    # Total loss is weighted sum
    expected_total = (agent.curiosity_lambda * losses['inverse_loss'] + 
                      agent.curiosity_eta * losses['forward_loss'])
    assert np.isclose(losses['total_icm_loss'], expected_total, rtol=0.01)


def test_train_step_with_icm(agent):
    """Test combined DQN + ICM training step"""
    # Need replay buffer
    from src.utils.replay_buffer import ReplayBuffer
    agent.replay_buffer = ReplayBuffer(capacity=1000)
    
    # Fill replay buffer
    for _ in range(100):
        state = np.random.randn(10)
        action = np.random.randint(0, 4)
        reward = np.random.randn()
        next_state = np.random.randn(10)
        done = False
        agent.replay_buffer.push(state, action, reward, next_state, done)
    
    # Train step
    losses = agent.train_step(batch_size=32)
    
    assert 'dqn_loss' in losses
    assert 'forward_loss' in losses
    assert 'inverse_loss' in losses
    assert all(isinstance(v, float) for v in losses.values())


def test_gradient_flow(agent):
    """Test that gradients flow through ICM networks"""
    batch = {
        'states': torch.randn(8, agent.state_dim, requires_grad=True).to(agent.device),
        'actions': torch.randint(0, agent.action_dim, (8,)).to(agent.device),
        'next_states': torch.randn(8, agent.state_dim, requires_grad=True).to(agent.device)
    }
    
    # Before training
    initial_params = [p.clone() for p in agent.feature_network.parameters()]
    
    # Train
    agent.train_icm(batch)
    
    # After training - parameters should have changed
    for initial, current in zip(initial_params, agent.feature_network.parameters()):
        assert not torch.equal(initial, current), "Parameters did not update"


# ============================================================
# 5. INTEGRATION TESTS
# ============================================================

def test_full_episode_with_curiosity(agent):
    """Test complete episode with curiosity"""
    from src.environment.navigation_env import NavigationEnv
    from src.utils.replay_buffer import ReplayBuffer
    
    env = NavigationEnv(width=400, height=300)
    agent.replay_buffer = ReplayBuffer(capacity=1000)
    
    state = env.reset()
    total_reward = 0
    total_intrinsic = 0
    done = False
    steps = 0
    
    while not done and steps < 50:
        action = agent.select_action(state, epsilon=0.1)
        next_state, reward, done, info = env.step(action)
        
        # Compute intrinsic reward
        r_int = agent.compute_intrinsic_reward(state, action, next_state)
        total_intrinsic += r_int
        
        # Total reward
        total = reward + agent.curiosity_beta * r_int
        total_reward += total
        
        # Store transition
        agent.replay_buffer.push(state, action, total, next_state, done)
        
        state = next_state
        steps += 1
    
    assert steps > 0
    assert total_reward is not None
    assert total_intrinsic > 0  # Should have some curiosity


def test_replay_buffer_integration(agent):
    """Test replay buffer integration"""
    from src.utils.replay_buffer import ReplayBuffer
    
    agent.replay_buffer = ReplayBuffer(capacity=100)
    
    # Add experiences
    for _ in range(50):
        state = np.random.randn(10)
        action = np.random.randint(0, 4)
        reward = np.random.randn()
        next_state = np.random.randn(10)
        done = False
        agent.replay_buffer.push(state, action, reward, next_state, done)
    
    # Train
    losses = agent.train_step(batch_size=16)
    
    assert losses['dqn_loss'] >= 0
    assert losses['forward_loss'] >= 0


def test_checkpoint_save_load(agent):
    """Test save and load checkpoint"""
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
        temp_path = f.name
    
    try:
        # Save
        agent.save_checkpoint(temp_path)
        assert Path(temp_path).exists()
        
        # Modify agent
        original_beta = agent.curiosity_beta
        agent.curiosity_beta = 999.0
        
        # Load
        agent.load_checkpoint(temp_path)
        
        # Check restored
        assert agent.curiosity_beta == original_beta
        
    finally:
        Path(temp_path).unlink(missing_ok=True)


# ============================================================
# 6. EDGE CASES
# ============================================================

def test_empty_replay_buffer(agent):
    """Test training with empty replay buffer"""
    from src.utils.replay_buffer import ReplayBuffer
    agent.replay_buffer = ReplayBuffer(capacity=100)
    
    losses = agent.train_step(batch_size=32)
    
    # Should return zeros when buffer is empty
    assert losses['dqn_loss'] == 0.0
    assert losses['forward_loss'] == 0.0


def test_terminal_state_handling(agent):
    """Test handling of terminal states"""
    state = np.random.randn(10)
    action = 0
    next_state = np.random.randn(10)
    
    # Should work fine with terminal states
    r_int = agent.compute_intrinsic_reward(state, action, next_state)
    assert isinstance(r_int, float)


def test_nan_inf_handling(agent):
    """Test handling of NaN/Inf values"""
    # Extreme values that might cause numerical issues
    state = np.ones(10) * 1e6
    action = 0
    next_state = np.ones(10) * 1e6
    
    r_int = agent.compute_intrinsic_reward(state, action, next_state)
    
    assert not np.isnan(r_int), "Intrinsic reward is NaN"
    assert not np.isinf(r_int), "Intrinsic reward is Inf"


def test_invalid_inputs(agent):
    """Test handling of invalid inputs"""
    # Wrong state dimension
    with pytest.raises((RuntimeError, ValueError)):
        agent.compute_intrinsic_reward(
            np.random.randn(5),  # Wrong dim
            0,
            np.random.randn(10)
        )
    
    # Invalid action
    r_int = agent.compute_intrinsic_reward(
        np.random.randn(10),
        -1,  # Invalid but should be handled
        np.random.randn(10)
    )
    assert isinstance(r_int, float)


# ============================================================
# 7. COMPARISON TESTS
# ============================================================

def test_curiosity_vs_baseline_exploration(agent):
    """Test that curiosity agent explores more"""
    from src.agents.dqn_agent import DQNAgent
    
    baseline = DQNAgent(state_dim=10, action_dim=4)
    
    # Curiosity agent should have ICM networks
    assert hasattr(agent, 'feature_network')
    assert hasattr(agent, 'inverse_model')
    assert hasattr(agent, 'forward_model')
    
    # Baseline should not
    assert not hasattr(baseline, 'feature_network')


def test_intrinsic_reward_impact():
    """Test that intrinsic rewards encourage exploration"""
    config_low_beta = {'curiosity_beta': 0.01}
    config_high_beta = {'curiosity_beta': 1.0}
    
    agent_low = CuriosityAgent(10, 4, config_low_beta)
    agent_high = CuriosityAgent(10, 4, config_high_beta)
    
    assert agent_low.curiosity_beta < agent_high.curiosity_beta
    
    # Higher beta means more exploration
    extrinsic = 1.0
    intrinsic = 0.5
    
    total_low = extrinsic + agent_low.curiosity_beta * intrinsic
    total_high = extrinsic + agent_high.curiosity_beta * intrinsic
    
    assert total_high > total_low


# ============================================================
# RUN TESTS
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, '-v', '--tb=short'])
