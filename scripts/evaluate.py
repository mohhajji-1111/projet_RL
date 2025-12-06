"""
Evaluation Script
Run with: python scripts/evaluate.py --model trained_models/basic/final.pt
"""
import argparse
import torch
import numpy as np
from pathlib import Path
import sys
import time

sys.path.append(str(Path(__file__).parent.parent))

from src.agents import DQNAgent, RainbowAgent
from src.environment import NavigationEnv


def load_agent(model_path: str, agent_type: str = 'dqn', device: str = 'cpu'):
    """Load trained agent"""
    if agent_type.lower() == 'dqn':
        agent = DQNAgent(
            state_dim=8,
            action_dim=4,
            device=device
        )
    elif agent_type.lower() == 'rainbow':
        agent = RainbowAgent(
            state_dim=8,
            action_dim=4,
            use_noisy=True,
            device=device
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    agent.load(model_path)
    return agent


def evaluate_agent(
    agent,
    env,
    num_episodes: int = 10,
    render: bool = False,
    epsilon: float = 0.0
):
    """Evaluate agent performance"""
    episode_rewards = []
    episode_lengths = []
    episode_successes = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        success = False
        
        for step in range(1000):
            action = agent.select_action(state, epsilon)
            state, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            if render:
                env.render()
                time.sleep(0.02)
            
            if terminated:
                if reward > 50:  # Successful completion
                    success = True
                break
            
            if truncated:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_successes.append(success)
        
        print(f"Episode {episode + 1:3d}: "
              f"Reward = {episode_reward:7.2f}, "
              f"Length = {episode_length:4d}, "
              f"Success = {success}")
    
    return {
        'rewards': episode_rewards,
        'lengths': episode_lengths,
        'successes': episode_successes,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'success_rate': np.mean(episode_successes)
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained RL agent')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--agent-type', type=str, default='dqn',
                        choices=['dqn', 'rainbow'],
                        help='Type of agent')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of evaluation episodes')
    parser.add_argument('--render', action='store_true',
                        help='Render environment')
    parser.add_argument('--epsilon', type=float, default=0.0,
                        help='Exploration epsilon')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Check model exists
    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        return
    
    # Create environment
    print("Creating environment...")
    render_mode = 'human' if args.render else None
    env = NavigationEnv(render_mode=render_mode)
    
    # Load agent
    print(f"Loading agent from {args.model}...")
    agent = load_agent(args.model, args.agent_type, args.device)
    
    # Evaluate
    print(f"\nEvaluating for {args.episodes} episodes...")
    print("-" * 60)
    
    stats = evaluate_agent(
        agent,
        env,
        num_episodes=args.episodes,
        render=args.render,
        epsilon=args.epsilon
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("Evaluation Results:")
    print(f"Mean Reward:   {stats['mean_reward']:7.2f} ± {stats['std_reward']:.2f}")
    print(f"Mean Length:   {stats['mean_length']:7.1f}")
    print(f"Success Rate:  {stats['success_rate']:7.1%}")
    print("=" * 60)
    
    env.close()


if __name__ == '__main__':
    main()
