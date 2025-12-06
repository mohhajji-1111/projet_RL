"""
Base Agent Abstract Class for Reinforcement Learning
"""
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class BaseAgent(ABC):
    """Abstract base class for all RL agents"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
    @abstractmethod
    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """Select action based on current state"""
        pass
    
    @abstractmethod
    def train_step(self, batch: dict) -> dict:
        """Perform one training step and return metrics"""
        pass
    
    @abstractmethod
    def save(self, path: str):
        """Save agent model"""
        pass
    
    @abstractmethod
    def load(self, path: str):
        """Load agent model"""
        pass
    
    def to_tensor(self, x: np.ndarray) -> torch.Tensor:
        """Convert numpy array to tensor"""
        return torch.FloatTensor(x).to(self.device)
