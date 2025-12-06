"""
Live Demo with Visualization
Run with: python scripts/demo.py --model trained_models/basic/final.pt
"""
import argparse
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.agents import DQNAgent, RainbowAgent
from src.environment import NavigationEnv
from src.visualization import Renderer, EffectManager


def load_agent(model_path: str, agent_type: str = 'dqn', device: str = 'cpu'):
    """Load trained agent"""
    if agent_type.lower() == 'dqn':
        agent = DQNAgent(state_dim=8, action_dim=4, device=device)
    elif agent_type.lower() == 'rainbow':
        agent = RainbowAgent(state_dim=8, action_dim=4, use_noisy=True, device=device)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    agent.load(model_path)
    return agent


def run_demo(
    agent,
    env,
    renderer: Renderer,
    effects: EffectManager,
    num_episodes: int = 5
):
    """Run demo with visualization"""
    import pygame
    
    for episode in range(num_episodes):
        state, info = env.reset()
        renderer.clear_trajectory()
        effects.clear()
        
        episode_reward = 0
        episode_steps = 0
        running = True
        
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                        return
                    if event.key == pygame.K_r:  # Reset episode
                        break
            
            # Select action
            action = agent.select_action(state, epsilon=0.0)
            
            # Step environment
            next_state, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_steps += 1
            
            # Update effects
            robot_pos = info['robot_pos']
            robot_vel = env.robot_vel
            
            effects.motion_trail(robot_pos, robot_vel)
            effects.update()
            
            # Render
            renderer.clear()
            renderer.draw_grid()
            renderer.draw_trajectory(robot_pos)
            renderer.draw_goal(info['goal_pos'], env.goal_radius)
            renderer.draw_robot(robot_pos, env.robot_angle, env.robot_radius)
            
            # Draw effects
            effects.render(renderer.screen)
            
            # Draw stats
            renderer.draw_stats(
                episode=episode + 1,
                reward=episode_reward,
                distance=info['distance_to_goal'],
                epsilon=0.0
            )
            
            renderer.update()
            
            # Check termination
            if terminated:
                if reward > 50:  # Success
                    effects.goal_reached_effect(robot_pos)
                    for _ in range(60):  # Show effect for 2 seconds
                        effects.update()
                        renderer.clear()
                        renderer.draw_grid()
                        renderer.draw_trajectory(robot_pos)
                        renderer.draw_goal(info['goal_pos'], env.goal_radius)
                        renderer.draw_robot(robot_pos, env.robot_angle, env.robot_radius)
                        effects.render(renderer.screen)
                        renderer.draw_stats(episode + 1, episode_reward, 
                                          info['distance_to_goal'], 0.0)
                        renderer.update()
                
                print(f"  Reward: {episode_reward:.2f}, Steps: {episode_steps}, "
                      f"Success: {reward > 50}")
                break
            
            if truncated:
                print(f"  Reward: {episode_reward:.2f}, Steps: {episode_steps}, "
                      f"Failed (timeout)")
                break
            
            state = next_state


def main():
    parser = argparse.ArgumentParser(description='Live demo of trained agent')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--agent-type', type=str, default='dqn',
                        choices=['dqn', 'rainbow'],
                        help='Type of agent')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of demo episodes')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    if args.seed:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    # Check model exists
    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        return
    
    # Create environment
    print("Creating environment...")
    env = NavigationEnv(render_mode=None)
    
    # Load agent
    print(f"Loading agent from {args.model}...")
    agent = load_agent(args.model, args.agent_type, args.device)
    
    # Create renderer and effects
    renderer = Renderer(width=800, height=600, fps=30)
    effects = EffectManager()
    
    print("\nStarting demo...")
    print("Press Q or ESC to quit, R to reset episode")
    print("-" * 60)
    
    try:
        run_demo(agent, env, renderer, effects, args.episodes)
    except KeyboardInterrupt:
        print("\nDemo interrupted")
    finally:
        renderer.close()
        env.close()
        print("Demo finished!")


if __name__ == '__main__':
    main()
