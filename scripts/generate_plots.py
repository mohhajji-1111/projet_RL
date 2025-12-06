"""
Generate Visualization Plots
Run with: python scripts/generate_plots.py --log results/logs/experiment_metrics.json
"""
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))


def load_metrics(metrics_path: str) -> dict:
    """Load metrics from JSON file"""
    with open(metrics_path, 'r') as f:
        return json.load(f)


def plot_training_curves(metrics: dict, output_dir: str):
    """Generate training curve plots"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Episode rewards
    if 'rewards' in metrics:
        rewards = metrics['rewards']
        episodes = range(1, len(rewards) + 1)
        
        axes[0, 0].plot(episodes, rewards, alpha=0.3, color='blue', linewidth=0.5)
        
        # Moving average
        window = min(50, len(rewards) // 4)
        if len(rewards) > window:
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            axes[0, 0].plot(range(window, len(rewards) + 1), moving_avg, 
                           color='blue', linewidth=2, label=f'{window}-episode avg')
            axes[0, 0].legend()
        
        axes[0, 0].set_title('Episode Rewards', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True, alpha=0.3)
    
    # Episode lengths
    if 'lengths' in metrics:
        lengths = metrics['lengths']
        episodes = range(1, len(lengths) + 1)
        
        axes[0, 1].plot(episodes, lengths, alpha=0.5, color='green')
        axes[0, 1].set_title('Episode Lengths', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Training loss
    if 'losses' in metrics and metrics['losses']:
        losses = metrics['losses']
        axes[1, 0].plot(losses, alpha=0.5, color='red')
        axes[1, 0].set_title('Training Loss', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Training Step')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Q-values
    if 'q_values' in metrics and metrics['q_values']:
        q_values = metrics['q_values']
        axes[1, 1].plot(q_values, alpha=0.5, color='purple')
        axes[1, 1].set_title('Average Q-Values', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Training Step')
        axes[1, 1].set_ylabel('Q-Value')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'training_curves.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.show()


def plot_statistics(metrics: dict, output_dir: str):
    """Generate statistical plots"""
    if 'rewards' not in metrics:
        return
    
    rewards = np.array(metrics['rewards'])
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Reward distribution
    axes[0].hist(rewards, bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[0].axvline(np.mean(rewards), color='red', linestyle='--', 
                     linewidth=2, label=f'Mean: {np.mean(rewards):.2f}')
    axes[0].set_title('Reward Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Reward')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Rolling statistics
    window = 100
    if len(rewards) > window:
        rolling_mean = np.convolve(rewards, np.ones(window)/window, mode='valid')
        rolling_std = [np.std(rewards[max(0, i-window):i]) 
                       for i in range(window, len(rewards) + 1)]
        
        episodes = range(window, len(rewards) + 1)
        axes[1].plot(episodes, rolling_mean, linewidth=2, label='Mean')
        axes[1].fill_between(episodes, 
                             rolling_mean - rolling_std,
                             rolling_mean + rolling_std,
                             alpha=0.3, label='±1 Std')
        axes[1].set_title(f'Rolling Statistics (window={window})', 
                         fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Reward')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'statistics.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Generate training visualization plots')
    parser.add_argument('--log', type=str, required=True,
                        help='Path to metrics JSON file')
    parser.add_argument('--output', type=str, default='results/figures',
                        help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Check input exists
    if not Path(args.log).exists():
        print(f"Error: Metrics file not found: {args.log}")
        return
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metrics
    print(f"Loading metrics from {args.log}")
    metrics = load_metrics(args.log)
    
    # Generate plots
    print("\nGenerating plots...")
    plot_training_curves(metrics, args.output)
    plot_statistics(metrics, args.output)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
