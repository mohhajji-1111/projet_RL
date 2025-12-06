"""
Extended Training - 500 Episodes
Pour améliorer les performances
"""

import numpy as np
import torch
from pathlib import Path

from src.environment.navigation_env import NavigationEnv
from src.environment.obstacles import StaticObstacle
from src.agents.dqn_agent import DQNAgent
from src.training.distributed_trainer import DeviceManager
from src.training.experiment_tracker import UnifiedTracker
from src.training.curriculum_learning import CurriculumLearningSystem
from src.utils.replay_buffer import ReplayBuffer

def extended_training():
    """Entraînement prolongé - 500 épisodes."""
    
    print("=" * 70)
    print("🚀 ENTRAÎNEMENT PROLONGÉ - 500 ÉPISODES")
    print("=" * 70)
    
    # Setup
    DeviceManager.print_device_info()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Environment avec plus d'obstacles
    obstacles = [
        StaticObstacle(x=200, y=200, radius=30),
        StaticObstacle(x=400, y=300, radius=30),
        StaticObstacle(x=600, y=400, radius=30),
        StaticObstacle(x=300, y=450, radius=25),
        StaticObstacle(x=500, y=150, radius=25),
    ]
    
    env = NavigationEnv(width=800, height=600, obstacles=obstacles, render_mode=None)
    
    # Agent amélioré (réseau plus large)
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        hidden_dims=[512, 512],  # Plus large!
        learning_rate=5e-4,       # Learning rate plus bas
        gamma=0.99,
        device=device
    )
    
    replay_buffer = ReplayBuffer(capacity=100000)  # Buffer plus grand
    batch_size = 128  # Batch plus grand
    
    # Tracking
    tracker = UnifiedTracker(
        enable_tensorboard=True,
        tensorboard_config={'log_dir': './runs/extended_training'}
    )
    
    curriculum = CurriculumLearningSystem()
    
    print("\n🎯 Configuration:")
    print(f"   Device: {device}")
    print(f"   Épisodes: 500")
    print(f"   Hidden layers: [512, 512]")
    print(f"   Batch size: {batch_size}")
    print(f"   Buffer size: 100,000")
    print(f"   Obstacles: {len(obstacles)}")
    print("\n" + "=" * 70)
    
    # Training
    num_episodes = 500
    best_reward = -float('inf')
    episode_rewards = []
    
    for episode in range(num_episodes):
        env_config = curriculum.get_current_env_config()
        epsilon = curriculum.get_current_epsilon()
        
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        collisions = 0
        done = False
        
        while not done:
            action = agent.select_action(state, epsilon)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            replay_buffer.add(state, action, reward, next_state, done)
            
            if len(replay_buffer.buffer) >= batch_size:
                batch = replay_buffer.sample(batch_size)
                loss_info = agent.train_step(batch)
            
            episode_reward += reward
            episode_length += 1
            if info.get('collision', False):
                collisions += 1
            
            state = next_state
        
        episode_rewards.append(episode_reward)
        success = info.get('success', False)
        curriculum.update(episode, episode_reward, success, collisions)
        
        # Logging
        tracker.log_metrics({
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'epsilon': epsilon,
            'curriculum_stage': curriculum.current_stage.value,
            'success': int(success),
            'collisions': collisions,
            'avg_reward_100': np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
        }, step=episode)
        
        # Save best
        if episode_reward > best_reward:
            best_reward = episode_reward
            Path('checkpoints').mkdir(exist_ok=True)
            torch.save(agent.q_network.state_dict(), 'checkpoints/best_extended.pt')
        
        # Progress
        if (episode + 1) % 25 == 0:
            avg_100 = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            print(f"Episode {episode+1:3d} | Reward: {episode_reward:7.2f} | "
                  f"Avg(100): {avg_100:7.2f} | Stage: {curriculum.current_stage.name:20s} | "
                  f"ε: {epsilon:.3f} | Success: {success}")
    
    # Résultats finaux
    print("\n" + "=" * 70)
    print("✅ ENTRAÎNEMENT TERMINÉ!")
    print("=" * 70)
    print(f"   Meilleure récompense: {best_reward:.2f}")
    print(f"   Moyenne (100 derniers): {np.mean(episode_rewards[-100:]):.2f}")
    print(f"   Stage final: {curriculum.current_stage.name}")
    print(f"   Modèle sauvegardé: checkpoints/best_extended.pt")
    print(f"   TensorBoard: tensorboard --logdir=runs/extended_training")
    print("=" * 70)
    
    tracker.finish()
    curriculum.save_progress(Path('results/curriculum_extended.json'))
    
    return agent, best_reward


if __name__ == "__main__":
    try:
        agent, best_reward = extended_training()
        print(f"\n🎉 SUCCÈS! Meilleure récompense: {best_reward:.2f}")
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
