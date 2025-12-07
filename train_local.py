"""
Script d'entraînement local pour l'agent ICM
"""
import sys
import os
from pathlib import Path
import numpy as np
import yaml
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import deque

# Configurer le path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.environment.navigation_env import NavigationEnv
from src.agents.curiosity_agent import CuriosityAgent

def train():
    print("="*60)
    print("🚀 ENTRAÎNEMENT LOCAL - CURIOSITY AGENT")
    print("="*60)
    
    # Charger config
    with open('configs/curiosity_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🔥 Device: {device}")
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Créer environnement
    env = NavigationEnv(width=800, height=600, render_mode=None)
    print(f"\n✅ Environnement créé")
    print(f"   Observation space: {env.observation_space.shape}")
    print(f"   Action space: {env.action_space.n} actions")
    
    # Créer agent
    agent = CuriosityAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        config=config['agent'],
        device=device
    )
    print(f"\n✅ Agent créé")
    print(f"   Feature dim: {config['agent']['feature_dim']}")
    print(f"   Curiosity beta: {config['agent']['curiosity_beta']}")
    
    # Paramètres
    num_episodes = 1500
    save_interval = 100
    
    # Créer dossiers
    os.makedirs('results/models/curiosity', exist_ok=True)
    os.makedirs('results/plots', exist_ok=True)
    
    # Replay buffer (externe)
    replay_buffer = deque(maxlen=10000)
    
    # Métriques
    episode_rewards = []
    episode_lengths = []
    intrinsic_rewards = []
    forward_losses = []
    inverse_losses = []
    success_rate = deque(maxlen=100)
    avg_rewards = deque(maxlen=100)
    best_reward = -float('inf')
    
    print(f"\n🚀 Début de l'entraînement... ({num_episodes} épisodes)\n")
    
    pbar = tqdm(range(num_episodes), desc="Training")
    
    for episode in pbar:
        state, _ = env.reset()
        episode_reward = 0
        episode_intrinsic = 0
        episode_length = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            # Action
            action = agent.select_action(state)
            
            # Step
            next_state, reward, done, truncated, info = env.step(action)
            
            # Récompense intrinsèque
            intrinsic_reward = agent.compute_intrinsic_reward(state, action, next_state)
            combined_reward = reward + config['agent']['curiosity_beta'] * intrinsic_reward
            
            # Stocker dans le replay buffer
            replay_buffer.append((state, action, combined_reward, next_state, done or truncated))
            
            # Train if enough samples
            if len(replay_buffer) >= 128:
                # Sample batch
                indices = np.random.choice(len(replay_buffer), 128, replace=False)
                batch_data = [replay_buffer[i] for i in indices]
                
                states_batch = np.array([x[0] for x in batch_data])
                actions_batch = np.array([x[1] for x in batch_data])
                rewards_batch = np.array([x[2] for x in batch_data])
                next_states_batch = np.array([x[3] for x in batch_data])
                dones_batch = np.array([x[4] for x in batch_data])
                
                batch = {
                    'states': states_batch,
                    'actions': actions_batch,
                    'rewards': rewards_batch,
                    'next_states': next_states_batch,
                    'dones': dones_batch
                }
                
                loss = agent.train_step(batch)
            
            episode_reward += reward
            episode_intrinsic += intrinsic_reward
            episode_length += 1
            state = next_state
        
        # Métriques
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        intrinsic_rewards.append(episode_intrinsic)
        avg_rewards.append(episode_reward)
        success_rate.append(1.0 if info.get('success', False) else 0.0)
        
        icm_stats = agent.get_icm_stats()
        forward_losses.append(icm_stats['forward_loss'])
        inverse_losses.append(icm_stats['inverse_loss'])
        
        # Progress
        pbar.set_postfix({
            'Reward': f"{episode_reward:.2f}",
            'Avg': f"{np.mean(avg_rewards):.2f}",
            'Success': f"{np.mean(success_rate):.1%}",
            'ε': f"{agent.epsilon:.3f}"
        })
        
        # Sauvegarder meilleur
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save_checkpoint('results/models/curiosity/best.pth')
        
        # Checkpoints
        if (episode + 1) % save_interval == 0:
            print(f"\n💾 Checkpoint: episode {episode+1}")
            agent.save_checkpoint(f'results/models/curiosity/checkpoint_{episode+1}.pth')
    
    print("\n" + "="*60)
    print("✅ ENTRAÎNEMENT TERMINÉ!")
    print("="*60)
    print(f"Meilleure récompense: {best_reward:.2f}")
    print(f"Récompense moyenne (100 derniers): {np.mean(avg_rewards):.2f}")
    print(f"Taux de succès final: {np.mean(success_rate):.2%}")
    
    # Sauvegarder les métriques
    metrics = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'intrinsic_rewards': intrinsic_rewards,
        'forward_losses': forward_losses,
        'inverse_losses': inverse_losses
    }
    np.save('results/training_metrics.npy', metrics)
    print(f"\n💾 Métriques sauvegardées dans results/training_metrics.npy")
    
    env.close()

if __name__ == "__main__":
    train()
