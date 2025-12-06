"""
Adaptive Training System with Curriculum Learning
"""
import numpy as np
from typing import Dict, Any, List
import time
from .trainer_base import BaseTrainer
from ..utils.replay_buffer import PrioritizedReplayBuffer


class AdaptiveTrainer(BaseTrainer):
    """Adaptive trainer with curriculum learning and prioritized replay"""
    
    def __init__(
        self,
        agent,
        env,
        num_episodes: int = 1000,
        max_steps_per_episode: int = 1000,
        batch_size: int = 64,
        buffer_size: int = 100000,
        learning_starts: int = 1000,
        curriculum_stages: List[Dict] = None,
        log_interval: int = 10,
        save_interval: int = 100,
        save_path: str = None
    ):
        super().__init__(agent, env, num_episodes, max_steps_per_episode, log_interval)
        
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.save_interval = save_interval
        self.save_path = save_path
        
        # Prioritized replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(buffer_size)
        
        # Curriculum learning stages
        self.curriculum_stages = curriculum_stages or [
            {'episodes': 200, 'difficulty': 'easy', 'epsilon': 1.0},
            {'episodes': 300, 'difficulty': 'medium', 'epsilon': 0.5},
            {'episodes': 500, 'difficulty': 'hard', 'epsilon': 0.1}
        ]
        
        self.current_stage = 0
        self.stage_episode = 0
        
        # Adaptive learning
        self.success_rate_window = 20
        self.recent_successes = []
        
        # Training metrics
        self.losses = []
        self.q_values = []
        self.td_errors = []
        
    def _get_current_stage(self) -> Dict[str, Any]:
        """Get current curriculum stage"""
        if self.current_stage < len(self.curriculum_stages):
            return self.curriculum_stages[self.current_stage]
        return self.curriculum_stages[-1]
    
    def _should_advance_stage(self) -> bool:
        """Check if should advance to next stage"""
        stage = self._get_current_stage()
        
        if self.stage_episode < stage['episodes']:
            return False
        
        # Check success rate
        if len(self.recent_successes) >= self.success_rate_window:
            success_rate = np.mean(self.recent_successes[-self.success_rate_window:])
            return success_rate > 0.7  # Advance if 70% success rate
        
        return False
    
    def _update_environment_difficulty(self, difficulty: str):
        """Update environment based on difficulty"""
        # This is a placeholder - customize based on your environment
        if hasattr(self.env, 'set_difficulty'):
            self.env.set_difficulty(difficulty)
    
    def train(self) -> Dict[str, Any]:
        """Run adaptive training"""
        start_time = time.time()
        total_steps = 0
        
        for episode in range(1, self.num_episodes + 1):
            # Get current stage
            stage = self._get_current_stage()
            epsilon = stage['epsilon']
            
            # Check stage advancement
            if self._should_advance_stage():
                self.current_stage += 1
                self.stage_episode = 0
                print(f"\n🎓 Advanced to stage {self.current_stage + 1}")
                stage = self._get_current_stage()
                self._update_environment_difficulty(stage['difficulty'])
            
            state, _ = self.env.reset()
            episode_reward = 0
            episode_length = 0
            episode_metrics = {'loss': 0, 'q_mean': 0, 'td_error': 0, 'updates': 0}
            episode_success = False
            
            for step in range(self.max_steps_per_episode):
                # Select action (noisy networks don't use epsilon)
                if hasattr(self.agent, 'use_noisy') and self.agent.use_noisy:
                    action = self.agent.select_action(state)
                else:
                    action = self.agent.select_action(state, epsilon)
                
                # Execute action
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Check success
                if terminated and reward > 50:  # Successful completion
                    episode_success = True
                
                # Store transition with initial priority
                self.replay_buffer.add(state, action, reward, next_state, done)
                
                # Train if enough samples
                if total_steps >= self.learning_starts and len(self.replay_buffer) >= self.batch_size:
                    batch = self.replay_buffer.sample(self.batch_size)
                    metrics = self.agent.train_step(batch)
                    
                    # Update priorities if TD errors available
                    if 'td_error' in metrics and hasattr(self.replay_buffer, 'update_priorities'):
                        # Note: would need to track indices from sample to update
                        pass
                    
                    episode_metrics['loss'] += metrics['loss']
                    episode_metrics['q_mean'] += metrics['q_mean']
                    if 'td_error' in metrics:
                        episode_metrics['td_error'] += metrics['td_error']
                    episode_metrics['updates'] += 1
                    
                    self.losses.append(metrics['loss'])
                    self.q_values.append(metrics['q_mean'])
                
                episode_reward += reward
                episode_length += 1
                total_steps += 1
                state = next_state
                
                if done:
                    break
            
            # Track success
            self.recent_successes.append(1 if episode_success else 0)
            
            # Store episode stats
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.stage_episode += 1
            
            # Average metrics
            if episode_metrics['updates'] > 0:
                episode_metrics['loss'] /= episode_metrics['updates']
                episode_metrics['q_mean'] /= episode_metrics['updates']
                if episode_metrics['td_error'] > 0:
                    episode_metrics['td_error'] /= episode_metrics['updates']
            
            episode_metrics['epsilon'] = epsilon
            episode_metrics['stage'] = self.current_stage + 1
            episode_metrics['success_rate'] = np.mean(self.recent_successes[-self.success_rate_window:]) if self.recent_successes else 0
            
            # Log progress
            self.log_episode(episode, episode_reward, episode_length, episode_metrics)
            
            # Save model
            if self.save_path and episode % self.save_interval == 0:
                self.agent.save(f"{self.save_path}/episode_{episode}.pt")
        
        self.training_time = time.time() - start_time
        
        # Final save
        if self.save_path:
            self.agent.save(f"{self.save_path}/final.pt")
        
        stats = self.get_stats()
        stats['final_stage'] = self.current_stage + 1
        return stats
