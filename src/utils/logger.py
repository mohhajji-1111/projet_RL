"""
Logging Utilities
"""
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict
import numpy as np


class TrainingLogger:
    """Logger for training metrics"""
    
    def __init__(
        self,
        log_dir: str = "results/logs",
        experiment_name: str = None
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate experiment name
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"experiment_{timestamp}"
        
        self.experiment_name = experiment_name
        self.log_file = self.log_dir / f"{experiment_name}.log"
        self.metrics_file = self.log_dir / f"{experiment_name}_metrics.json"
        
        # Setup logger
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(self.log_file)
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        
        # Metrics storage
        self.metrics = {
            'episodes': [],
            'rewards': [],
            'lengths': [],
            'losses': [],
            'q_values': []
        }
    
    def log_episode(
        self,
        episode: int,
        reward: float,
        length: int,
        metrics: Dict[str, Any] = None
    ):
        """Log episode results"""
        self.metrics['episodes'].append(episode)
        self.metrics['rewards'].append(reward)
        self.metrics['lengths'].append(length)
        
        msg = f"Episode {episode}: Reward={reward:.2f}, Length={length}"
        
        if metrics:
            for key, value in metrics.items():
                if isinstance(value, (int, float, np.number)):
                    self.metrics.setdefault(key, []).append(float(value))
                    msg += f", {key}={value:.4f}"
        
        self.logger.info(msg)
    
    def log_info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def log_warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
    
    def log_error(self, message: str):
        """Log error message"""
        self.logger.error(message)
    
    def save_metrics(self):
        """Save metrics to JSON file"""
        # Convert numpy types to Python types
        serializable_metrics = {}
        for key, value in self.metrics.items():
            if isinstance(value, list):
                serializable_metrics[key] = [
                    float(v) if isinstance(v, np.number) else v
                    for v in value
                ]
            else:
                serializable_metrics[key] = value
        
        with open(self.metrics_file, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
        
        self.logger.info(f"Metrics saved to {self.metrics_file}")
    
    def load_metrics(self):
        """Load metrics from JSON file"""
        if self.metrics_file.exists():
            with open(self.metrics_file, 'r') as f:
                self.metrics = json.load(f)
            self.logger.info(f"Metrics loaded from {self.metrics_file}")
        else:
            self.logger.warning(f"Metrics file not found: {self.metrics_file}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get training summary"""
        if not self.metrics['rewards']:
            return {}
        
        return {
            'total_episodes': len(self.metrics['rewards']),
            'mean_reward': np.mean(self.metrics['rewards']),
            'std_reward': np.std(self.metrics['rewards']),
            'max_reward': np.max(self.metrics['rewards']),
            'min_reward': np.min(self.metrics['rewards']),
            'mean_length': np.mean(self.metrics['lengths']),
            'final_avg_reward': np.mean(self.metrics['rewards'][-100:]),
        }


class CheckpointManager:
    """Manage model checkpoints"""
    
    def __init__(self, checkpoint_dir: str = "trained_models"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(
        self,
        agent,
        episode: int,
        metrics: Dict[str, Any] = None,
        prefix: str = "checkpoint"
    ):
        """Save checkpoint"""
        checkpoint_path = self.checkpoint_dir / f"{prefix}_episode_{episode}.pt"
        agent.save(str(checkpoint_path))
        
        # Save metadata
        if metrics:
            metadata_path = checkpoint_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump({
                    'episode': episode,
                    'metrics': metrics,
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)
        
        return checkpoint_path
    
    def load_checkpoint(self, agent, checkpoint_path: str):
        """Load checkpoint"""
        agent.load(checkpoint_path)
        
        # Load metadata if exists
        metadata_path = Path(checkpoint_path).with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            return metadata
        
        return None
    
    def get_latest_checkpoint(self, prefix: str = "checkpoint") -> str:
        """Get path to latest checkpoint"""
        checkpoints = list(self.checkpoint_dir.glob(f"{prefix}_episode_*.pt"))
        
        if not checkpoints:
            return None
        
        # Sort by episode number
        checkpoints.sort(
            key=lambda p: int(p.stem.split('_')[-1])
        )
        
        return str(checkpoints[-1])
