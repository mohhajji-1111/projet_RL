"""
Training Script for Curiosity Agent (DQN + ICM)

Train a DQN agent with Intrinsic Curiosity Module for improved exploration.

Usage:
    python scripts/train_curiosity.py --config configs/curiosity_config.yaml
    python scripts/train_curiosity.py --episodes 2000 --device cuda
    python scripts/train_curiosity.py --config configs/curiosity_config.yaml --resume
"""

import argparse
import os
import sys
import yaml
import logging
import signal
from pathlib import Path
from datetime import datetime
from typing import Dict
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.environment.navigation_env import NavigationEnv
from src.agents.curiosity_agent import CuriosityAgent
from src.utils.replay_buffer import ReplayBuffer


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train Curiosity Agent")
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/curiosity_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=None,
        help='Number of episodes (overrides config)'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default=None,
        help='Directory for saving models (overrides config)'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu', 'auto'],
        default='auto',
        help='Device to use for training'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from latest checkpoint'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_random_seeds(seed: int):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def create_directories(config: Dict):
    """Create necessary directories"""
    dirs = [
        config['paths']['models_dir'],
        config['paths']['logs_dir'],
        config['paths']['figures_dir'],
        config['paths']['checkpoints_dir']
    ]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def save_config(config: Dict, save_path: str):
    """Save configuration to file"""
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


class CuriosityTrainer:
    """Trainer for Curiosity Agent"""
    
    def __init__(self, config: Dict, args):
        self.config = config
        self.args = args
        
        # Setup logging
        log_level = logging.DEBUG if args.debug else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(Path(config['paths']['logs_dir']) / 'train.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('CuriosityTrainer')
        
        # Device
        if args.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = args.device
        
        # Create environment
        env_config = config['environment']
        # Note: NavigationEnv doesn't use num_obstacles/max_steps in constructor
        # These are managed internally
        self.env = NavigationEnv(
            width=env_config.get('width', 800),
            height=env_config.get('height', 600),
            render_mode=None
        )
        
        # Create agent
        agent_config = config['agent']
        agent_config['device'] = self.device
        self.agent = CuriosityAgent(
            state_dim=self.env.observation_space.shape[0],
            action_dim=self.env.action_space.n,
            config=agent_config,
            learning_rate=agent_config['learning_rate'],
            gamma=agent_config['gamma'],
            tau=agent_config['tau'],
            hidden_dims=agent_config['hidden_sizes']
        )
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=agent_config['buffer_size']
        )
        self.agent.replay_buffer = self.replay_buffer
        
        # Training parameters
        self.total_episodes = args.episodes or config['training']['total_episodes']
        self.eval_frequency = config['training']['eval_frequency']
        self.save_frequency = config['training']['save_frequency']
        self.log_frequency = config['training']['log_frequency']
        
        # Epsilon decay
        self.epsilon = agent_config['epsilon_start']
        self.epsilon_end = agent_config['epsilon_end']
        self.epsilon_decay = agent_config['epsilon_decay']
        
        # Metrics tracking
        self.episode_rewards = []
        self.episode_intrinsic_rewards = []
        self.episode_steps = []
        self.success_count = 0
        self.eval_results = []
        
        # Best model tracking
        self.best_reward = float('-inf')
        
        # Ctrl+C handling
        self.interrupted = False
        signal.signal(signal.SIGINT, self._signal_handler)
        
        self.logger.info(f"Trainer initialized on device: {self.device}")
        self.logger.info(f"Training for {self.total_episodes} episodes")
    
    def _signal_handler(self, sig, frame):
        """Handle Ctrl+C gracefully"""
        self.logger.warning("\nInterrupted! Saving checkpoint...")
        self.interrupted = True
    
    def train_episode(self, episode: int) -> Dict:
        """Train for one episode"""
        state = self.env.reset()
        episode_reward = 0
        episode_intrinsic_reward = 0
        episode_steps = 0
        done = False
        
        while not done:
            # Select action
            action = self.agent.select_action(state, epsilon=self.epsilon)
            
            # Take step
            next_state, reward, done, info = self.env.step(action)
            
            # Compute intrinsic reward
            intrinsic_reward = self.agent.compute_intrinsic_reward(
                state, action, next_state
            )
            
            # Total reward (extrinsic + intrinsic)
            total_reward = reward + self.agent.curiosity_beta * intrinsic_reward
            
            # Store transition
            self.replay_buffer.push(
                state, action, total_reward, next_state, done
            )
            
            # Train agent
            if len(self.replay_buffer) >= self.config['agent']['min_buffer_size']:
                losses = self.agent.train_step(
                    batch_size=self.config['agent']['batch_size']
                )
            
            # Update metrics
            episode_reward += reward
            episode_intrinsic_reward += intrinsic_reward
            episode_steps += 1
            state = next_state
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return {
            'episode_reward': episode_reward,
            'intrinsic_reward': episode_intrinsic_reward,
            'steps': episode_steps,
            'success': info.get('success', False),
            'epsilon': self.epsilon
        }
    
    def evaluate(self, num_episodes: int = 10) -> Dict:
        """Evaluate agent"""
        rewards = []
        steps = []
        successes = 0
        
        for _ in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_steps = 0
            done = False
            
            while not done:
                action = self.agent.select_action(state, epsilon=0.0)
                next_state, reward, done, info = self.env.step(action)
                episode_reward += reward
                episode_steps += 1
                state = next_state
            
            rewards.append(episode_reward)
            steps.append(episode_steps)
            if info.get('success', False):
                successes += 1
        
        return {
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'avg_steps': np.mean(steps),
            'success_rate': successes / num_episodes
        }
    
    def save_checkpoint(self, episode: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint_dir = Path(self.config['paths']['checkpoints_dir'])
        
        # Save regular checkpoint
        checkpoint_path = checkpoint_dir / f"checkpoint_ep{episode}.pth"
        self.agent.save_checkpoint(str(checkpoint_path))
        
        # Save best model
        if is_best:
            best_path = checkpoint_dir / "best_model.pth"
            self.agent.save_checkpoint(str(best_path))
            self.logger.info(f"New best model saved! (reward: {self.best_reward:.2f})")
    
    def save_metrics(self):
        """Save training metrics to CSV"""
        df = pd.DataFrame({
            'episode': range(1, len(self.episode_rewards) + 1),
            'reward': self.episode_rewards,
            'intrinsic_reward': self.episode_intrinsic_rewards,
            'steps': self.episode_steps
        })
        
        csv_path = Path(self.config['paths']['logs_dir']) / 'training_metrics.csv'
        df.to_csv(csv_path, index=False)
        self.logger.info(f"Metrics saved to {csv_path}")
    
    def train(self):
        """Main training loop"""
        self.logger.info("Starting training...")
        
        progress_bar = tqdm(range(1, self.total_episodes + 1), desc="Training")
        
        for episode in progress_bar:
            if self.interrupted:
                break
            
            # Train one episode
            result = self.train_episode(episode)
            
            # Store metrics
            self.episode_rewards.append(result['episode_reward'])
            self.episode_intrinsic_rewards.append(result['intrinsic_reward'])
            self.episode_steps.append(result['steps'])
            if result['success']:
                self.success_count += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'reward': f"{result['episode_reward']:.1f}",
                'int_rew': f"{result['intrinsic_reward']:.2f}",
                'success': f"{self.success_count}/{episode}",
                'ε': f"{result['epsilon']:.3f}"
            })
            
            # Logging
            if episode % self.log_frequency == 0:
                recent_rewards = self.episode_rewards[-self.log_frequency:]
                recent_success = sum([self.episode_rewards[i] > 0 
                                     for i in range(-self.log_frequency, 0)])
                
                icm_stats = self.agent.get_icm_stats()
                
                self.logger.info(
                    f"Episode {episode}/{self.total_episodes} | "
                    f"Avg Reward: {np.mean(recent_rewards):.2f} | "
                    f"Success Rate: {recent_success/self.log_frequency:.2%} | "
                    f"ICM Forward Loss: {icm_stats['forward_loss']:.4f} | "
                    f"ICM Inverse Loss: {icm_stats['inverse_loss']:.4f}"
                )
            
            # Evaluation
            if episode % self.eval_frequency == 0:
                eval_results = self.evaluate(num_episodes=10)
                self.eval_results.append(eval_results)
                
                self.logger.info(
                    f"Evaluation | "
                    f"Avg Reward: {eval_results['avg_reward']:.2f} ± "
                    f"{eval_results['std_reward']:.2f} | "
                    f"Success Rate: {eval_results['success_rate']:.2%}"
                )
                
                # Check if best model
                if eval_results['avg_reward'] > self.best_reward:
                    self.best_reward = eval_results['avg_reward']
                    self.save_checkpoint(episode, is_best=True)
            
            # Save checkpoint
            if episode % self.save_frequency == 0:
                self.save_checkpoint(episode)
        
        # Save final model and metrics
        self.logger.info("Training completed!")
        final_path = Path(self.config['paths']['models_dir']) / 'final_model.pth'
        self.agent.save_checkpoint(str(final_path))
        self.save_metrics()
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print training summary"""
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        print(f"Total Episodes: {len(self.episode_rewards)}")
        print(f"Average Reward: {np.mean(self.episode_rewards):.2f}")
        print(f"Best Reward: {self.best_reward:.2f}")
        print(f"Success Rate: {self.success_count/len(self.episode_rewards):.2%}")
        print(f"Average Steps: {np.mean(self.episode_steps):.1f}")
        
        if self.eval_results:
            final_eval = self.eval_results[-1]
            print(f"\nFinal Evaluation:")
            print(f"  Reward: {final_eval['avg_reward']:.2f} ± {final_eval['std_reward']:.2f}")
            print(f"  Success Rate: {final_eval['success_rate']:.2%}")
        
        icm_stats = self.agent.get_icm_stats()
        print(f"\nICM Statistics:")
        print(f"  Forward Loss: {icm_stats['forward_loss']:.4f}")
        print(f"  Inverse Loss: {icm_stats['inverse_loss']:.4f}")
        print("="*60)


def main():
    """Main entry point"""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line args
    if args.episodes:
        config['training']['total_episodes'] = args.episodes
    if args.save_dir:
        config['paths']['models_dir'] = args.save_dir
    
    # Set random seeds
    seed = args.seed or config['random_seed']
    set_random_seeds(seed)
    
    # Create directories
    create_directories(config)
    
    # Save config
    config_save_path = Path(config['paths']['logs_dir']) / 'config.yaml'
    save_config(config, str(config_save_path))
    
    # Create trainer
    trainer = CuriosityTrainer(config, args)
    
    try:
        # Train
        trainer.train()
    except Exception as e:
        trainer.logger.error(f"Training failed with error: {e}", exc_info=True)
        raise
    
    print("\n✅ Training completed successfully!")
    print(f"📁 Models saved to: {config['paths']['models_dir']}")
    print(f"📊 Logs saved to: {config['paths']['logs_dir']}")


if __name__ == "__main__":
    main()
