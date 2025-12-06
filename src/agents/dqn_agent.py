"""
Deep Q-Network (DQN) Agent Implementation
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple
from .base_agent import BaseAgent


class DQNNetwork(nn.Module):
    """Deep Q-Network"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = [256, 256]):
        super().__init__()
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, action_dim))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DQNAgent(BaseAgent):
    """Deep Q-Network Agent"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        hidden_dims: list = [256, 256],
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__(state_dim, action_dim, device)
        
        self.gamma = gamma
        self.tau = tau
        
        # Q-Networks
        self.q_network = DQNNetwork(state_dim, action_dim, hidden_dims).to(device)
        self.target_network = DQNNetwork(state_dim, action_dim, hidden_dims).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        
    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """Epsilon-greedy action selection"""
        if np.random.random() < epsilon:
            return np.random.randint(self.action_dim)
        
        with torch.no_grad():
            state_tensor = self.to_tensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax(dim=1).item()
    
    def train_step(self, batch: dict) -> dict:
        """Train on a batch of experiences"""
        states = self.to_tensor(batch['states'])
        actions = torch.LongTensor(batch['actions']).to(self.device)
        rewards = self.to_tensor(batch['rewards'])
        next_states = self.to_tensor(batch['next_states'])
        dones = self.to_tensor(batch['dones'])
        
        # Current Q-values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # Target Q-values
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss
        loss = self.loss_fn(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Soft update target network
        self._soft_update()
        
        return {
            'loss': loss.item(),
            'q_mean': current_q.mean().item(),
            'q_max': current_q.max().item()
        }
    
    def _soft_update(self):
        """Soft update target network"""
        for target_param, param in zip(
            self.target_network.parameters(),
            self.q_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
    
    def save(self, path: str):
        """Save model"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)
    
    def load(self, path: str):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
