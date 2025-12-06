"""
Performance Metrics
"""
import numpy as np
from typing import List, Dict, Any
import time


class MetricsTracker:
    """Track and compute performance metrics"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        
        # Episode metrics
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_times: List[float] = []
        
        # Training metrics
        self.losses: List[float] = []
        self.q_values: List[float] = []
        self.td_errors: List[float] = []
        
        # Success tracking
        self.successes: List[bool] = []
        
        # Timing
        self.episode_start_time = None
        self.training_start_time = None
    
    def start_episode(self):
        """Mark episode start"""
        self.episode_start_time = time.time()
    
    def end_episode(
        self,
        reward: float,
        length: int,
        success: bool = False
    ):
        """Record episode completion"""
        episode_time = time.time() - self.episode_start_time if self.episode_start_time else 0
        
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.episode_times.append(episode_time)
        self.successes.append(success)
    
    def add_training_metrics(
        self,
        loss: float = None,
        q_value: float = None,
        td_error: float = None
    ):
        """Add training step metrics"""
        if loss is not None:
            self.losses.append(loss)
        if q_value is not None:
            self.q_values.append(q_value)
        if td_error is not None:
            self.td_errors.append(td_error)
    
    def get_recent_stats(self) -> Dict[str, Any]:
        """Get statistics for recent window"""
        if not self.episode_rewards:
            return {}
        
        n = min(self.window_size, len(self.episode_rewards))
        
        stats = {
            'mean_reward': np.mean(self.episode_rewards[-n:]),
            'std_reward': np.std(self.episode_rewards[-n:]),
            'mean_length': np.mean(self.episode_lengths[-n:]),
            'mean_time': np.mean(self.episode_times[-n:]),
        }
        
        if self.successes:
            stats['success_rate'] = np.mean(self.successes[-n:])
        
        if self.losses:
            stats['mean_loss'] = np.mean(self.losses[-n:])
        
        if self.q_values:
            stats['mean_q'] = np.mean(self.q_values[-n:])
        
        return stats
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get all-time statistics"""
        if not self.episode_rewards:
            return {}
        
        stats = {
            'total_episodes': len(self.episode_rewards),
            'total_steps': sum(self.episode_lengths),
            'mean_reward': np.mean(self.episode_rewards),
            'max_reward': np.max(self.episode_rewards),
            'min_reward': np.min(self.episode_rewards),
            'std_reward': np.std(self.episode_rewards),
            'mean_length': np.mean(self.episode_lengths),
            'total_time': sum(self.episode_times),
        }
        
        if self.successes:
            stats['overall_success_rate'] = np.mean(self.successes)
            stats['recent_success_rate'] = np.mean(self.successes[-self.window_size:])
        
        return stats
    
    def is_improving(self, threshold: float = 0.1) -> bool:
        """Check if performance is improving"""
        if len(self.episode_rewards) < self.window_size * 2:
            return True  # Not enough data
        
        n = self.window_size
        old_mean = np.mean(self.episode_rewards[-2*n:-n])
        new_mean = np.mean(self.episode_rewards[-n:])
        
        improvement = (new_mean - old_mean) / (abs(old_mean) + 1e-6)
        return improvement > threshold
    
    def get_learning_curve(self, smooth_window: int = 10) -> np.ndarray:
        """Get smoothed learning curve"""
        if len(self.episode_rewards) < smooth_window:
            return np.array(self.episode_rewards)
        
        return np.convolve(
            self.episode_rewards,
            np.ones(smooth_window) / smooth_window,
            mode='valid'
        )


class PerformanceMonitor:
    """Monitor real-time performance"""
    
    def __init__(self):
        self.step_times: List[float] = []
        self.forward_times: List[float] = []
        self.backward_times: List[float] = []
        
        self._step_start = None
        self._forward_start = None
        self._backward_start = None
    
    def start_step(self):
        self._step_start = time.perf_counter()
    
    def end_step(self):
        if self._step_start:
            self.step_times.append(time.perf_counter() - self._step_start)
    
    def start_forward(self):
        self._forward_start = time.perf_counter()
    
    def end_forward(self):
        if self._forward_start:
            self.forward_times.append(time.perf_counter() - self._forward_start)
    
    def start_backward(self):
        self._backward_start = time.perf_counter()
    
    def end_backward(self):
        if self._backward_start:
            self.backward_times.append(time.perf_counter() - self._backward_start)
    
    def get_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        return {
            'avg_step_time': np.mean(self.step_times[-100:]) if self.step_times else 0,
            'avg_forward_time': np.mean(self.forward_times[-100:]) if self.forward_times else 0,
            'avg_backward_time': np.mean(self.backward_times[-100:]) if self.backward_times else 0,
            'steps_per_second': 1.0 / (np.mean(self.step_times[-100:]) + 1e-6) if self.step_times else 0,
        }
    
    def reset(self):
        """Reset all timers"""
        self.step_times.clear()
        self.forward_times.clear()
        self.backward_times.clear()
