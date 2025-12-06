"""
Curiosity-Driven DQN Agent with Intrinsic Curiosity Module (ICM)

This module implements a DQN agent enhanced with intrinsic curiosity
for better exploration in sparse reward environments.

Reference:
    Pathak et al. (2017) "Curiosity-driven Exploration by Self-supervised Prediction"
    https://arxiv.org/abs/1705.05363
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
from collections import deque
import logging

from .dqn_agent import DQNAgent


class FeatureNetwork(nn.Module):
    """
    Feature encoding network φ(s)
    Compresses raw state into learnable feature representation.
    Removes task-irrelevant information (e.g., background noise).
    """
    
    def __init__(self, state_dim: int, feature_dim: int = 32):
        """
        Args:
            state_dim: Dimension of input state
            feature_dim: Dimension of output features
        """
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, feature_dim),
            nn.ReLU()
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Encode state to features"""
        return self.network(state)


class InverseModel(nn.Module):
    """
    Inverse dynamics model: g(φ(s_t), φ(s_{t+1})) → a_t
    Predicts action from state transition.
    Forces features to capture controllable aspects of the environment.
    """
    
    def __init__(self, feature_dim: int, action_dim: int):
        """
        Args:
            feature_dim: Dimension of encoded features
            action_dim: Number of possible actions
        """
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(feature_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
    def forward(self, phi_t: torch.Tensor, phi_t1: torch.Tensor) -> torch.Tensor:
        """
        Predict action from feature transition
        
        Args:
            phi_t: Features at time t
            phi_t1: Features at time t+1
            
        Returns:
            Action logits
        """
        concat_features = torch.cat([phi_t, phi_t1], dim=-1)
        return self.network(concat_features)


class ForwardModel(nn.Module):
    """
    Forward dynamics model: f(φ(s_t), a_t) → φ(s_{t+1})
    Predicts next feature state from current features and action.
    Prediction error is used as intrinsic reward (curiosity signal).
    """
    
    def __init__(self, feature_dim: int, action_dim: int):
        """
        Args:
            feature_dim: Dimension of encoded features
            action_dim: Number of possible actions
        """
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, feature_dim)
        )
        
    def forward(self, phi_t: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Predict next features from current features and action
        
        Args:
            phi_t: Features at time t
            action: Action taken (one-hot encoded)
            
        Returns:
            Predicted features at time t+1
        """
        concat_input = torch.cat([phi_t, action], dim=-1)
        return self.network(concat_input)


class CuriosityAgent(DQNAgent):
    """
    DQN Agent with Intrinsic Curiosity Module (ICM)
    
    Enhances exploration by adding intrinsic rewards based on prediction error.
    Novel states → high prediction error → high intrinsic reward → exploration!
    
    Total reward: r_total = r_extrinsic + β * r_intrinsic
    where r_intrinsic = ||f(φ(s_t), a_t) - φ(s_{t+1})||²
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Optional[Dict] = None,
        **kwargs
    ):
        """
        Initialize Curiosity Agent
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of discrete actions
            config: Configuration dictionary with ICM parameters:
                - curiosity_beta: Weight of intrinsic reward (default: 0.2)
                - curiosity_eta: Forward loss scale (default: 1.0)
                - curiosity_lambda: Inverse loss scale (default: 0.1)
                - feature_dim: Feature space dimension (default: 32)
                - icm_lr: ICM learning rate (default: 0.001)
                - normalize_intrinsic: Normalize intrinsic rewards (default: True)
            **kwargs: Additional arguments passed to DQNAgent
        """
        # Initialize base DQN agent
        super().__init__(state_dim, action_dim, **kwargs)
        
        # ICM hyperparameters
        config = config or {}
        self.curiosity_beta = config.get('curiosity_beta', 0.2)
        self.curiosity_eta = config.get('curiosity_eta', 1.0)
        self.curiosity_lambda = config.get('curiosity_lambda', 0.1)
        self.feature_dim = config.get('feature_dim', 32)
        self.icm_lr = config.get('icm_lr', 0.001)
        self.normalize_intrinsic = config.get('normalize_intrinsic', True)
        
        # Build ICM networks
        self.feature_network = FeatureNetwork(state_dim, self.feature_dim).to(self.device)
        self.inverse_model = InverseModel(self.feature_dim, action_dim).to(self.device)
        self.forward_model = ForwardModel(self.feature_dim, action_dim).to(self.device)
        
        # ICM optimizer
        icm_params = list(self.feature_network.parameters()) + \
                     list(self.inverse_model.parameters()) + \
                     list(self.forward_model.parameters())
        self.icm_optimizer = optim.Adam(icm_params, lr=self.icm_lr)
        
        # Intrinsic reward statistics (for normalization)
        self.intrinsic_rewards_buffer = deque(maxlen=1000)
        self.intrinsic_reward_mean = 0.0
        self.intrinsic_reward_std = 1.0
        
        # Logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.icm_stats = {
            'forward_loss': [],
            'inverse_loss': [],
            'intrinsic_reward': []
        }
        
        self.logger.info(f"Curiosity Agent initialized with β={self.curiosity_beta}, "
                        f"feature_dim={self.feature_dim}")
    
    def compute_intrinsic_reward(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray
    ) -> float:
        """
        Compute intrinsic reward based on forward model prediction error
        
        The agent is "curious" about states where it cannot predict what happens next.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Resulting state
            
        Returns:
            Intrinsic reward (prediction error)
        """
        self.feature_network.eval()
        self.forward_model.eval()
        
        with torch.no_grad():
            # Convert to tensors
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            next_state_t = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            action_t = torch.zeros(1, self.action_dim).to(self.device)
            action_t[0, action] = 1.0  # One-hot encode
            
            # Encode states
            phi_t = self.feature_network(state_t)
            phi_t1 = self.feature_network(next_state_t)
            
            # Predict next features
            phi_t1_pred = self.forward_model(phi_t, action_t)
            
            # Prediction error = intrinsic reward
            prediction_error = F.mse_loss(phi_t1_pred, phi_t1, reduction='none')
            intrinsic_reward = prediction_error.mean().item()
        
        self.feature_network.train()
        self.forward_model.train()
        
        # Normalize intrinsic reward
        if self.normalize_intrinsic:
            self.intrinsic_rewards_buffer.append(intrinsic_reward)
            if len(self.intrinsic_rewards_buffer) > 10:
                self.intrinsic_reward_mean = np.mean(self.intrinsic_rewards_buffer)
                self.intrinsic_reward_std = np.std(self.intrinsic_rewards_buffer) + 1e-8
                intrinsic_reward = (intrinsic_reward - self.intrinsic_reward_mean) / self.intrinsic_reward_std
        
        # Scale by eta
        intrinsic_reward *= self.curiosity_eta / 2.0
        
        return intrinsic_reward
    
    def train_icm(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Train ICM networks (feature, inverse, forward models)
        
        Args:
            batch: Dictionary containing:
                - states: Current states [batch_size, state_dim]
                - actions: Actions taken [batch_size]
                - next_states: Next states [batch_size, state_dim]
                
        Returns:
            Dictionary with ICM losses
        """
        states = batch['states']
        actions = batch['actions']
        next_states = batch['next_states']
        
        batch_size = states.shape[0]
        
        # Encode states to features
        phi_t = self.feature_network(states)
        phi_t1 = self.feature_network(next_states)
        
        # One-hot encode actions
        actions_onehot = torch.zeros(batch_size, self.action_dim).to(self.device)
        actions_onehot.scatter_(1, actions.unsqueeze(1).long(), 1.0)
        
        # Forward model: predict next features
        phi_t1_pred = self.forward_model(phi_t, actions_onehot)
        forward_loss = F.mse_loss(phi_t1_pred, phi_t1.detach())
        
        # Inverse model: predict action from state transition
        action_logits = self.inverse_model(phi_t, phi_t1)
        inverse_loss = F.cross_entropy(action_logits, actions.long())
        
        # Total ICM loss
        icm_loss = self.curiosity_lambda * inverse_loss + self.curiosity_eta * forward_loss
        
        # Optimize ICM
        self.icm_optimizer.zero_grad()
        icm_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.feature_network.parameters()) +
            list(self.inverse_model.parameters()) +
            list(self.forward_model.parameters()),
            max_norm=1.0
        )
        self.icm_optimizer.step()
        
        # Log statistics
        self.icm_stats['forward_loss'].append(forward_loss.item())
        self.icm_stats['inverse_loss'].append(inverse_loss.item())
        
        return {
            'forward_loss': forward_loss.item(),
            'inverse_loss': inverse_loss.item(),
            'total_icm_loss': icm_loss.item()
        }
    
    def train_step(self, batch_size: int = 64) -> Dict[str, float]:
        """
        Perform one training step (DQN + ICM)
        
        Args:
            batch_size: Number of samples to train on
            
        Returns:
            Dictionary with training losses
        """
        if len(self.replay_buffer) < batch_size:
            return {'dqn_loss': 0.0, 'forward_loss': 0.0, 'inverse_loss': 0.0}
        
        # Sample batch from replay buffer
        batch = self.replay_buffer.sample(batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(batch['states']).to(self.device)
        actions = torch.LongTensor(batch['actions']).to(self.device)
        rewards = torch.FloatTensor(batch['rewards']).to(self.device)
        next_states = torch.FloatTensor(batch['next_states']).to(self.device)
        dones = torch.FloatTensor(batch['dones']).to(self.device)
        
        # Train DQN (from parent class)
        dqn_loss = self._train_dqn_step(states, actions, rewards, next_states, dones)
        
        # Train ICM
        icm_batch = {
            'states': states,
            'actions': actions,
            'next_states': next_states
        }
        icm_losses = self.train_icm(icm_batch)
        
        # Combine results
        return {
            'dqn_loss': dqn_loss,
            **icm_losses
        }
    
    def _train_dqn_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor
    ) -> float:
        """Internal method to train DQN network"""
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update target network
        self._update_target_network()
        
        return loss.item()
    
    def _update_target_network(self):
        """Soft update of target network"""
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save_checkpoint(self, filepath: str):
        """
        Save agent checkpoint including ICM networks
        
        Args:
            filepath: Path to save checkpoint
        """
        checkpoint = {
            # DQN networks
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            
            # ICM networks
            'feature_network': self.feature_network.state_dict(),
            'inverse_model': self.inverse_model.state_dict(),
            'forward_model': self.forward_model.state_dict(),
            'icm_optimizer': self.icm_optimizer.state_dict(),
            
            # Hyperparameters
            'config': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'curiosity_beta': self.curiosity_beta,
                'curiosity_eta': self.curiosity_eta,
                'curiosity_lambda': self.curiosity_lambda,
                'feature_dim': self.feature_dim,
                'gamma': self.gamma,
                'tau': self.tau
            },
            
            # Statistics
            'icm_stats': self.icm_stats,
            'intrinsic_reward_mean': self.intrinsic_reward_mean,
            'intrinsic_reward_std': self.intrinsic_reward_std
        }
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """
        Load agent checkpoint including ICM networks
        
        Args:
            filepath: Path to load checkpoint from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Load DQN networks
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        # Load ICM networks
        self.feature_network.load_state_dict(checkpoint['feature_network'])
        self.inverse_model.load_state_dict(checkpoint['inverse_model'])
        self.forward_model.load_state_dict(checkpoint['forward_model'])
        self.icm_optimizer.load_state_dict(checkpoint['icm_optimizer'])
        
        # Load statistics
        if 'icm_stats' in checkpoint:
            self.icm_stats = checkpoint['icm_stats']
        if 'intrinsic_reward_mean' in checkpoint:
            self.intrinsic_reward_mean = checkpoint['intrinsic_reward_mean']
            self.intrinsic_reward_std = checkpoint['intrinsic_reward_std']
        
        self.logger.info(f"Checkpoint loaded from {filepath}")
    
    def get_icm_stats(self) -> Dict[str, float]:
        """
        Get ICM training statistics
        
        Returns:
            Dictionary with average losses
        """
        if not self.icm_stats['forward_loss']:
            return {'forward_loss': 0.0, 'inverse_loss': 0.0, 'intrinsic_reward': 0.0}
        
        return {
            'forward_loss': np.mean(self.icm_stats['forward_loss'][-100:]),
            'inverse_loss': np.mean(self.icm_stats['inverse_loss'][-100:]),
            'intrinsic_reward': np.mean(self.icm_stats['intrinsic_reward'][-100:]) if self.icm_stats['intrinsic_reward'] else 0.0
        }


def test_curiosity_agent():
    """Test function to verify CuriosityAgent functionality"""
    print("Testing CuriosityAgent...")
    
    # Create agent
    config = {
        'curiosity_beta': 0.2,
        'feature_dim': 32,
        'icm_lr': 0.001
    }
    agent = CuriosityAgent(state_dim=10, action_dim=4, config=config)
    
    # Test intrinsic reward computation
    state = np.random.randn(10)
    action = 0
    next_state = np.random.randn(10)
    
    r_int = agent.compute_intrinsic_reward(state, action, next_state)
    print(f"✓ Intrinsic reward computed: {r_int:.4f}")
    assert isinstance(r_int, float), "Intrinsic reward should be float"
    assert not np.isnan(r_int), "Intrinsic reward should not be NaN"
    
    # Test ICM training
    batch = {
        'states': torch.randn(32, 10),
        'actions': torch.randint(0, 4, (32,)),
        'next_states': torch.randn(32, 10)
    }
    losses = agent.train_icm(batch)
    print(f"✓ ICM training: forward_loss={losses['forward_loss']:.4f}, "
          f"inverse_loss={losses['inverse_loss']:.4f}")
    
    # Test save/load
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
        agent.save_checkpoint(f.name)
        agent.load_checkpoint(f.name)
    print("✓ Save/load checkpoint working")
    
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_curiosity_agent()
