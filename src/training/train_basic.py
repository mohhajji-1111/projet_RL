"""
Basic DQN Training Script
"""
import numpy as np
from typing import Dict, Any
import time
from .trainer_base import BaseTrainer
from ..utils.replay_buffer import ReplayBuffer


class BasicTrainer(BaseTrainer):
    """Basic DQN trainer"""
    
    def __init__(
        self,
        agent,
        env,
        num_episodes: int = 1000,
        max_steps_per_episode: int = 1000,
        batch_size: int = 64,
        buffer_size: int = 100000,
        learning_starts: int = 1000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        log_interval: int = 10,
        save_interval: int = 100,
        save_path: str = None
    ):
        super().__init__(agent, env, num_episodes, max_steps_per_episode, log_interval)
        
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.save_interval = save_interval
        self.save_path = save_path
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Training metrics
        self.losses = []
        self.q_values = []
        
    def train(self) -> Dict[str, Any]:
        """Run training"""
        start_time = time.time()
        total_steps = 0
        
        for episode in range(1, self.num_episodes + 1):
            state, _ = self.env.reset()
            episode_reward = 0
            episode_length = 0
            episode_metrics = {'loss': 0, 'q_mean': 0, 'updates': 0}
            
            for step in range(self.max_steps_per_episode):
                # Select action
                action = self.agent.select_action(state, self.epsilon)
                
                # Execute action
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # Store transition
                self.replay_buffer.add(state, action, reward, next_state, done)
                
                # Train if enough samples
                if total_steps >= self.learning_starts and len(self.replay_buffer) >= self.batch_size:
                    batch = self.replay_buffer.sample(self.batch_size)
                    metrics = self.agent.train_step(batch)
                    
                    episode_metrics['loss'] += metrics['loss']
                    episode_metrics['q_mean'] += metrics['q_mean']
                    episode_metrics['updates'] += 1
                    
                    self.losses.append(metrics['loss'])
                    self.q_values.append(metrics['q_mean'])
                
                episode_reward += reward
                episode_length += 1
                total_steps += 1
                state = next_state
                
                if done:
                    break
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            
            # Store episode stats
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            # Average metrics
            if episode_metrics['updates'] > 0:
                episode_metrics['loss'] /= episode_metrics['updates']
                episode_metrics['q_mean'] /= episode_metrics['updates']
            
            episode_metrics['epsilon'] = self.epsilon
            
            # Log progress
            self.log_episode(episode, episode_reward, episode_length, episode_metrics)
            
            # Save model
            if self.save_path and episode % self.save_interval == 0:
                self.agent.save(f"{self.save_path}/episode_{episode}.pt")
        
        self.training_time = time.time() - start_time
        
        # Final save
        if self.save_path:
            self.agent.save(f"{self.save_path}/final.pt")
        
        return self.get_stats()
