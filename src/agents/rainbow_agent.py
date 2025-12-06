"""
Rainbow DQN Agent with advanced features:
- Double DQN
- Dueling Network
- Prioritized Experience Replay support
- Noisy Networks
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
from .base_agent import BaseAgent


class NoisyLinear(nn.Module):
    """Noisy linear layer for exploration"""
    
    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init
        
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))
    
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)


class DuelingNetwork(nn.Module):
    """Dueling DQN Architecture"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list = [512, 256],
        use_noisy: bool = True
    ):
        super().__init__()
        
        self.action_dim = action_dim
        self.use_noisy = use_noisy
        
        # Shared feature extraction
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])
            input_dim = hidden_dim
        
        self.feature_layer = nn.Sequential(*layers)
        
        # Value stream
        if use_noisy:
            self.value_hidden = NoisyLinear(input_dim, 128)
            self.value_output = NoisyLinear(128, 1)
        else:
            self.value_hidden = nn.Linear(input_dim, 128)
            self.value_output = nn.Linear(128, 1)
        
        # Advantage stream
        if use_noisy:
            self.advantage_hidden = NoisyLinear(input_dim, 128)
            self.advantage_output = NoisyLinear(128, action_dim)
        else:
            self.advantage_hidden = nn.Linear(input_dim, 128)
            self.advantage_output = nn.Linear(128, action_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_layer(x)
        
        # Value stream
        value = F.relu(self.value_hidden(features))
        value = self.value_output(value)
        
        # Advantage stream
        advantage = F.relu(self.advantage_hidden(features))
        advantage = self.advantage_output(advantage)
        
        # Combine streams
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values
    
    def reset_noise(self):
        """Reset noise for noisy layers"""
        if self.use_noisy:
            self.value_hidden.reset_noise()
            self.value_output.reset_noise()
            self.advantage_hidden.reset_noise()
            self.advantage_output.reset_noise()


class RainbowAgent(BaseAgent):
    """Rainbow DQN Agent"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 6e-5,
        gamma: float = 0.99,
        tau: float = 0.005,
        hidden_dims: list = [512, 256],
        use_noisy: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__(state_dim, action_dim, device)
        
        self.gamma = gamma
        self.tau = tau
        self.use_noisy = use_noisy
        
        # Networks
        self.q_network = DuelingNetwork(
            state_dim, action_dim, hidden_dims, use_noisy
        ).to(device)
        
        self.target_network = DuelingNetwork(
            state_dim, action_dim, hidden_dims, use_noisy
        ).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """Select action (epsilon ignored if using noisy networks)"""
        if not self.use_noisy and np.random.random() < epsilon:
            return np.random.randint(self.action_dim)
        
        with torch.no_grad():
            state_tensor = self.to_tensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax(dim=1).item()
    
    def train_step(self, batch: dict) -> dict:
        """Train with Double DQN"""
        states = self.to_tensor(batch['states'])
        actions = torch.LongTensor(batch['actions']).to(self.device)
        rewards = self.to_tensor(batch['rewards'])
        next_states = self.to_tensor(batch['next_states'])
        dones = self.to_tensor(batch['dones'])
        
        # Get priorities if available
        weights = batch.get('weights', np.ones(len(states)))
        weights = self.to_tensor(weights)
        
        # Reset noise
        if self.use_noisy:
            self.q_network.reset_noise()
            self.target_network.reset_noise()
        
        # Current Q-values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # Double DQN target
        with torch.no_grad():
            # Select actions using online network
            next_actions = self.q_network(next_states).argmax(1)
            # Evaluate using target network
            next_q = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute TD errors
        td_errors = current_q - target_q
        
        # Weighted loss
        loss = (weights * td_errors.pow(2)).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()
        
        # Soft update
        self._soft_update()
        
        return {
            'loss': loss.item(),
            'q_mean': current_q.mean().item(),
            'q_max': current_q.max().item(),
            'td_error': td_errors.abs().mean().item()
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
