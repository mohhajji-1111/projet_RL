"""
Base Trainer Class
"""
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any
import time


class BaseTrainer(ABC):
    """Abstract base trainer"""
    
    def __init__(
        self,
        agent,
        env,
        num_episodes: int = 1000,
        max_steps_per_episode: int = 1000,
        log_interval: int = 10
    ):
        self.agent = agent
        self.env = env
        self.num_episodes = num_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.log_interval = log_interval
        
        # Training stats
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_time = 0
        
    @abstractmethod
    def train(self) -> Dict[str, Any]:
        """Run training"""
        pass
    
    def log_episode(self, episode: int, reward: float, length: int, metrics: dict):
        """Log episode statistics"""
        if episode % self.log_interval == 0:
            avg_reward = np.mean(self.episode_rewards[-self.log_interval:])
            avg_length = np.mean(self.episode_lengths[-self.log_interval:])
            
            print(f"Episode {episode}/{self.num_episodes}")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Avg Length: {avg_length:.1f}")
            
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.4f}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'training_time': self.training_time,
            'mean_reward': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
            'mean_length': np.mean(self.episode_lengths[-100:]) if self.episode_lengths else 0
        }
