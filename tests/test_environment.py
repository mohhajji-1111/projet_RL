"""
Unit Tests for Environment
"""
import pytest
import numpy as np
from src.environment import NavigationEnv, ObstacleManager


def test_environment_creation():
    """Test environment can be created"""
    env = NavigationEnv()
    assert env is not None
    assert env.width == 800
    assert env.height == 600


def test_environment_reset():
    """Test environment reset"""
    env = NavigationEnv()
    state, info = env.reset()
    
    assert state.shape == (8,)
    assert 'distance_to_goal' in info
    assert 'robot_pos' in info
    assert 'goal_pos' in info


def test_environment_step():
    """Test environment step"""
    env = NavigationEnv()
    state, _ = env.reset()
    
    action = 0  # Forward
    next_state, reward, terminated, truncated, info = env.step(action)
    
    assert next_state.shape == (8,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_action_space():
    """Test action space"""
    env = NavigationEnv()
    assert env.action_space.n == 4


def test_observation_space():
    """Test observation space"""
    env = NavigationEnv()
    assert env.observation_space.shape == (8,)


def test_goal_reached():
    """Test goal detection"""
    env = NavigationEnv()
    state, _ = env.reset()
    
    # Manually set robot position to goal
    env.robot_pos = env.goal_pos.copy()
    
    action = 0
    _, reward, terminated, _, _ = env.step(action)
    
    # Should terminate with high reward
    assert terminated
    assert reward > 50


if __name__ == '__main__':
    pytest.main([__file__])
