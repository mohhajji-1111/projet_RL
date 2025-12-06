"""
Main Training Script
Run with: python scripts/train.py --config configs/base_config.yaml
"""
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.agents import DQNAgent, RainbowAgent
from src.environment import NavigationEnv
from src.training import BasicTrainer, AdaptiveTrainer
from src.utils import TrainingLogger


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_agent(config: dict) -> object:
    """Create agent from config"""
    agent_config = config['agent']
    agent_type = agent_config['type'].lower()
    
    if agent_type == 'dqn':
        return DQNAgent(
            state_dim=agent_config['state_dim'],
            action_dim=agent_config['action_dim'],
            learning_rate=agent_config['learning_rate'],
            gamma=agent_config['gamma'],
            tau=agent_config['tau'],
            hidden_dims=agent_config['hidden_dims'],
            device=agent_config.get('device', 'cuda')
        )
    elif agent_type == 'rainbow':
        return RainbowAgent(
            state_dim=agent_config['state_dim'],
            action_dim=agent_config['action_dim'],
            learning_rate=agent_config['learning_rate'],
            gamma=agent_config['gamma'],
            tau=agent_config['tau'],
            hidden_dims=agent_config['hidden_dims'],
            use_noisy=agent_config.get('use_noisy', True),
            device=agent_config.get('device', 'cuda')
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def create_environment(config: dict) -> NavigationEnv:
    """Create environment from config"""
    env_config = config['environment']
    
    return NavigationEnv(
        width=env_config['width'],
        height=env_config['height'],
        robot_radius=env_config['robot_radius'],
        goal_radius=env_config['goal_radius'],
        max_speed=env_config['max_speed'],
        render_mode=env_config.get('render_mode')
    )


def create_trainer(agent, env, config: dict) -> object:
    """Create trainer from config"""
    training_config = config['training']
    paths = config['paths']
    
    # Determine trainer type
    if 'curriculum_stages' in training_config:
        return AdaptiveTrainer(
            agent=agent,
            env=env,
            num_episodes=training_config['num_episodes'],
            max_steps_per_episode=training_config['max_steps_per_episode'],
            batch_size=training_config['batch_size'],
            buffer_size=training_config['buffer_size'],
            learning_starts=training_config['learning_starts'],
            curriculum_stages=training_config.get('curriculum_stages'),
            log_interval=training_config['log_interval'],
            save_interval=training_config['save_interval'],
            save_path=paths['save_dir']
        )
    else:
        return BasicTrainer(
            agent=agent,
            env=env,
            num_episodes=training_config['num_episodes'],
            max_steps_per_episode=training_config['max_steps_per_episode'],
            batch_size=training_config['batch_size'],
            buffer_size=training_config['buffer_size'],
            learning_starts=training_config['learning_starts'],
            epsilon_start=training_config['epsilon_start'],
            epsilon_end=training_config['epsilon_end'],
            epsilon_decay=training_config['epsilon_decay'],
            log_interval=training_config['log_interval'],
            save_interval=training_config['save_interval'],
            save_path=paths['save_dir']
        )


def main():
    parser = argparse.ArgumentParser(description='Train RL agent for robot navigation')
    parser.add_argument('--config', type=str, default='configs/base_config.yaml',
                        help='Path to config file')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (overrides config)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (overrides config)')
    
    args = parser.parse_args()
    
    # Load config
    print(f"Loading config from {args.config}")
    config = load_config(args.config)
    
    # Set seed
    seed = args.seed if args.seed is not None else config['experiment']['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"Random seed: {seed}")
    
    # Override device if specified
    if args.device:
        config['agent']['device'] = args.device
    
    device = config['agent']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
        config['agent']['device'] = 'cpu'
    
    print(f"Using device: {device}")
    
    # Create directories
    for path_key in ['save_dir', 'log_dir', 'figure_dir']:
        path = config['paths'][path_key]
        Path(path).mkdir(parents=True, exist_ok=True)
    
    # Create components
    print("\nCreating environment...")
    env = create_environment(config)
    
    print("Creating agent...")
    agent = create_agent(config)
    
    print("Creating trainer...")
    trainer = create_trainer(agent, env, config)
    
    # Setup logger
    logger = TrainingLogger(
        log_dir=config['paths']['log_dir'],
        experiment_name=config['experiment']['name']
    )
    
    # Train
    print(f"\nStarting training: {config['experiment']['name']}")
    print(f"Episodes: {config['training']['num_episodes']}")
    print("-" * 60)
    
    try:
        stats = trainer.train()
        
        # Log summary
        print("\n" + "=" * 60)
        print("Training completed!")
        print(f"Total episodes: {len(stats['episode_rewards'])}")
        print(f"Mean reward (last 100): {stats['mean_reward']:.2f}")
        print(f"Mean length (last 100): {stats['mean_length']:.1f}")
        print(f"Training time: {stats['training_time']:.2f}s")
        print("=" * 60)
        
        # Save final stats
        logger.save_metrics()
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Saving current model...")
        agent.save(f"{config['paths']['save_dir']}/interrupted.pt")
        print("Model saved!")
    
    finally:
        env.close()


if __name__ == '__main__':
    main()
