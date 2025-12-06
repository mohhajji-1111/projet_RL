"""
Quick Start Example - Advanced Training System
Demonstrates basic usage without hyperparameter optimization first
"""

import numpy as np
import torch
from pathlib import Path

# Import your existing environment and agent
from src.environment.navigation_env import NavigationEnv
from src.agents.dqn_agent import DQNAgent

# Import advanced training components
from src.training.distributed_trainer import DeviceManager
from src.training.experiment_tracker import UnifiedTracker
from src.training.curriculum_learning import CurriculumLearningSystem

def quick_start_training():
    """Simple training example using advanced features."""
    
    print("=" * 70)
    print("QUICK START: Advanced Training System")
    print("=" * 70)
    
    # Step 1: Check GPU availability
    print("\n1. Checking GPU availability...")
    DeviceManager.print_device_info()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Using device: {device}")
    
    # Step 2: Initialize environment
    print("\n2. Initializing environment...")
    from src.environment.obstacles import StaticObstacle
    
    # Create some static obstacles
    obstacles = [
        StaticObstacle(x=200, y=200, radius=30),
        StaticObstacle(x=400, y=300, radius=30),
        StaticObstacle(x=600, y=400, radius=30)
    ]
    
    env = NavigationEnv(
        width=800,
        height=600,
        obstacles=obstacles,
        render_mode=None
    )
    print(f"   Environment created")
    print(f"   State dim: {env.observation_space.shape[0]}")
    print(f"   Action dim: {env.action_space.n}")
    
    # Step 3: Create agent and replay buffer
    print("\n3. Creating DQN agent...")
    from src.utils.replay_buffer import ReplayBuffer
    
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        hidden_dims=[256, 256],
        learning_rate=1e-3,
        gamma=0.99,
        device=device
    )
    
    # Create replay buffer
    replay_buffer = ReplayBuffer(capacity=50000)
    batch_size = 64
    
    print(f"   Agent created with 2 hidden layers [256, 256]")
    print(f"   Replay buffer capacity: 50000")
    
    # Step 4: Setup experiment tracking
    print("\n4. Setting up experiment tracking...")
    tracker = UnifiedTracker(
        enable_tensorboard=True,
        enable_wandb=False,  # Set True if you have WandB account
        tensorboard_config={'log_dir': './runs/quick_start'}
    )
    tracker.log_hyperparameters({
        'hidden_dims': [256, 256],
        'learning_rate': 1e-3,
        'gamma': 0.99,
        'batch_size': batch_size,
        'buffer_size': 50000
    })
    print("   ✅ TensorBoard tracking enabled at ./runs/quick_start")
    
    # Step 5: Setup curriculum learning
    print("\n5. Setting up curriculum learning...")
    curriculum = CurriculumLearningSystem()
    print(f"   Starting at: {curriculum.current_stage.name}")
    
    # Step 6: Training loop
    print("\n6. Starting training...")
    print("   Training 100 episodes (quick demo)...")
    print("-" * 70)
    
    num_episodes = 100
    best_reward = -float('inf')
    
    for episode in range(num_episodes):
        # Get curriculum configuration
        env_config = curriculum.get_current_env_config()
        epsilon = curriculum.get_current_epsilon()
        
        # Note: For this simple demo, we're not dynamically updating obstacles
        # In a full implementation, you would recreate the environment with new config
        
        # Train episode
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        collisions = 0
        done = False
        
        while not done:
            # Select action
            action = agent.select_action(state, epsilon)
            
            # Step environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store transition in replay buffer
            replay_buffer.add(state, action, reward, next_state, done)
            
            # Train agent if enough samples
            if len(replay_buffer.buffer) >= batch_size:
                batch = replay_buffer.sample(batch_size)
                loss_info = agent.train_step(batch)
                loss = loss_info.get('loss', 0)
            
            episode_reward += reward
            episode_length += 1
            if info.get('collision', False):
                collisions += 1
            
            state = next_state
        
        # Update curriculum
        success = info.get('success', False)
        curriculum.update(episode, episode_reward, success, collisions)
        
        # Track metrics
        tracker.log_metrics({
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'epsilon': epsilon,
            'curriculum_stage': curriculum.current_stage.value,
            'success': int(success),
            'collisions': collisions
        }, step=episode)
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            checkpoint_dir = Path('checkpoints')
            checkpoint_dir.mkdir(exist_ok=True)
            torch.save(agent.q_network.state_dict(), checkpoint_dir / 'best_model.pt')
        
        # Print progress
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1:3d} | Reward: {episode_reward:6.2f} | "
                  f"Stage: {curriculum.current_stage.name} | "
                  f"ε: {epsilon:.3f} | Success: {success}")
    
    # Step 7: Finalize
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"   Best reward: {best_reward:.2f}")
    print(f"   Final stage: {curriculum.current_stage.name}")
    print(f"   Checkpoints saved to: ./checkpoints/")
    print(f"   TensorBoard logs: ./runs/quick_start")
    print("\nTo view results:")
    print("   tensorboard --logdir=runs/quick_start")
    print("=" * 70)
    
    tracker.finish()
    curriculum.save_progress(Path('results/curriculum_progress.json'))
    
    return agent, best_reward


if __name__ == "__main__":
    try:
        agent, best_reward = quick_start_training()
        print(f"\n✅ SUCCESS! Best reward achieved: {best_reward:.2f}")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
