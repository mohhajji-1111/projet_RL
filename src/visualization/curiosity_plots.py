"""
ICM-Specific Visualization Module

Plotting functions for Intrinsic Curiosity Module analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import animation
from matplotlib.patches import Rectangle, Circle
from typing import List, Dict, Tuple
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Colorblind-friendly palette
COLORS = {
    'curiosity': '#0173B2',
    'baseline': '#DE8F05',
    'success': '#029E73',
    'failure': '#CC78BC',
    'neutral': '#949494'
}


def plot_intrinsic_rewards(
    intrinsic_rewards: List[float],
    save_path: str,
    window: int = 100
):
    """
    Plot intrinsic rewards over training with moving average
    
    Args:
        intrinsic_rewards: List of intrinsic rewards per episode
        save_path: Path to save figure
        window: Moving average window size
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    episodes = range(1, len(intrinsic_rewards) + 1)
    
    # Raw intrinsic rewards
    ax.plot(episodes, intrinsic_rewards, alpha=0.3, color=COLORS['curiosity'], 
            linewidth=0.5, label='Raw')
    
    # Moving average
    if len(intrinsic_rewards) >= window:
        moving_avg = np.convolve(intrinsic_rewards, 
                                 np.ones(window)/window, 
                                 mode='valid')
        ax.plot(range(window, len(intrinsic_rewards) + 1), moving_avg,
                color=COLORS['curiosity'], linewidth=2, 
                label=f'{window}-Episode Moving Average')
    
    # Highlight peaks (top 5%)
    threshold = np.percentile(intrinsic_rewards, 95)
    peaks = [i for i, r in enumerate(intrinsic_rewards) if r > threshold]
    peak_rewards = [intrinsic_rewards[i] for i in peaks]
    ax.scatter([i+1 for i in peaks], peak_rewards, 
               color='red', s=20, alpha=0.6, 
               label='High Curiosity Moments', zorder=5)
    
    ax.set_xlabel('Episode', fontweight='bold')
    ax.set_ylabel('Intrinsic Reward', fontweight='bold')
    ax.set_title('Intrinsic Rewards Over Training', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved intrinsic rewards plot to {save_path}")


def plot_exploration_coverage(
    visited_states: set,
    grid_size: int,
    save_path: str,
    robot_start: Tuple[int, int] = None,
    goals: List[Tuple[int, int]] = None
):
    """
    Plot exploration coverage heatmap
    
    Args:
        visited_states: Set of (x, y) positions visited
        grid_size: Size of grid
        save_path: Path to save figure
        robot_start: Starting position of robot
        goals: List of goal positions
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create heatmap
    heatmap = np.zeros((grid_size, grid_size))
    for x, y in visited_states:
        if 0 <= x < grid_size and 0 <= y < grid_size:
            heatmap[y, x] += 1
    
    # Plot heatmap
    im = ax.imshow(heatmap, cmap='YlOrRd', origin='lower', 
                   interpolation='nearest')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Visit Frequency', fontweight='bold', rotation=270, labelpad=20)
    
    # Overlay robot start
    if robot_start:
        ax.plot(robot_start[0], robot_start[1], 'bs', markersize=15, 
                markeredgewidth=2, markeredgecolor='white', 
                label='Robot Start')
    
    # Overlay goals
    if goals:
        for gx, gy in goals:
            ax.plot(gx, gy, 'g*', markersize=20, 
                    markeredgewidth=2, markeredgecolor='white',
                    label='Goal' if goals.index((gx, gy)) == 0 else '')
    
    # Coverage percentage
    coverage = len(visited_states) / (grid_size * grid_size)
    ax.text(0.5, 1.05, f'Coverage: {coverage:.1%}', 
            transform=ax.transAxes, ha='center', 
            fontsize=12, fontweight='bold')
    
    ax.set_xlabel('X Position', fontweight='bold')
    ax.set_ylabel('Y Position', fontweight='bold')
    ax.set_title('State Space Exploration Coverage', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', bbox_to_anchor=(1.15, 1))
    ax.grid(True, alpha=0.2, color='white', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved exploration coverage plot to {save_path}")


def plot_curiosity_heatmap(
    episode_data: Dict,
    grid_size: int,
    save_path: str
):
    """
    Plot curiosity heatmap showing intrinsic rewards per grid cell
    
    Args:
        episode_data: Dict with 'positions' and 'intrinsic_rewards'
        grid_size: Size of grid
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create curiosity heatmap
    curiosity_map = np.zeros((grid_size, grid_size))
    count_map = np.zeros((grid_size, grid_size))
    
    positions = episode_data['positions']
    intrinsic_rewards = episode_data['intrinsic_rewards']
    
    for pos, r_int in zip(positions, intrinsic_rewards):
        x, y = pos
        if 0 <= x < grid_size and 0 <= y < grid_size:
            curiosity_map[y, x] += r_int
            count_map[y, x] += 1
    
    # Average curiosity per cell
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_curiosity = curiosity_map / count_map
        avg_curiosity[np.isnan(avg_curiosity)] = 0
    
    # Plot
    im = ax.imshow(avg_curiosity, cmap='plasma', origin='lower',
                   interpolation='bilinear')
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Avg Intrinsic Reward', fontweight='bold', 
                   rotation=270, labelpad=20)
    
    # Overlay obstacles and goals if available
    if 'obstacles' in episode_data:
        for ox, oy, ow, oh in episode_data['obstacles']:
            rect = Rectangle((ox-0.5, oy-0.5), ow, oh, 
                           linewidth=2, edgecolor='red', 
                           facecolor='none', linestyle='--')
            ax.add_patch(rect)
    
    if 'goals' in episode_data:
        for gx, gy in episode_data['goals']:
            circle = Circle((gx, gy), 0.4, color='lime', 
                          edgecolor='white', linewidth=2)
            ax.add_patch(circle)
    
    ax.set_xlabel('X Position', fontweight='bold')
    ax.set_ylabel('Y Position', fontweight='bold')
    ax.set_title('Curiosity Heatmap (Where Agent Was Most Curious)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.2, color='white', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved curiosity heatmap to {save_path}")


def plot_icm_losses(
    forward_losses: List[float],
    inverse_losses: List[float],
    save_path: str,
    window: int = 50
):
    """
    Plot ICM forward and inverse losses
    
    Args:
        forward_losses: Forward model losses
        inverse_losses: Inverse model losses
        save_path: Path to save figure
        window: Moving average window
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    episodes = range(1, len(forward_losses) + 1)
    
    # Forward loss (left axis)
    color = COLORS['curiosity']
    ax1.set_xlabel('Training Step', fontweight='bold')
    ax1.set_ylabel('Forward Loss (MSE)', color=color, fontweight='bold')
    
    # Raw
    ax1.plot(episodes, forward_losses, alpha=0.3, color=color, linewidth=0.5)
    
    # Moving average
    if len(forward_losses) >= window:
        fwd_ma = np.convolve(forward_losses, np.ones(window)/window, mode='valid')
        ax1.plot(range(window, len(forward_losses) + 1), fwd_ma,
                color=color, linewidth=2, label='Forward Loss')
    
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    # Inverse loss (right axis)
    ax2 = ax1.twinx()
    color = COLORS['baseline']
    ax2.set_ylabel('Inverse Loss (Cross-Entropy)', color=color, fontweight='bold')
    
    # Raw
    ax2.plot(episodes, inverse_losses, alpha=0.3, color=color, linewidth=0.5)
    
    # Moving average
    if len(inverse_losses) >= window:
        inv_ma = np.convolve(inverse_losses, np.ones(window)/window, mode='valid')
        ax2.plot(range(window, len(inverse_losses) + 1), inv_ma,
                color=color, linewidth=2, label='Inverse Loss')
    
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Title and legend
    ax1.set_title('ICM Training Losses', fontsize=14, fontweight='bold')
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved ICM losses plot to {save_path}")


def plot_reward_comparison(
    extrinsic_rewards: List[float],
    intrinsic_rewards: List[float],
    save_path: str,
    beta: float = 0.2
):
    """
    Plot reward composition (extrinsic + intrinsic)
    
    Args:
        extrinsic_rewards: Extrinsic rewards per episode
        intrinsic_rewards: Intrinsic rewards per episode
        save_path: Path to save figure
        beta: Curiosity beta weight
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    episodes = range(1, len(extrinsic_rewards) + 1)
    
    # Scale intrinsic rewards
    scaled_intrinsic = [r * beta for r in intrinsic_rewards]
    total_rewards = [e + i for e, i in zip(extrinsic_rewards, scaled_intrinsic)]
    
    # Stacked area chart
    ax.fill_between(episodes, 0, extrinsic_rewards, 
                     alpha=0.6, color=COLORS['baseline'], 
                     label='Extrinsic Reward')
    ax.fill_between(episodes, extrinsic_rewards, total_rewards,
                     alpha=0.6, color=COLORS['curiosity'],
                     label=f'Intrinsic Reward (β={beta})')
    
    # Total reward line
    ax.plot(episodes, total_rewards, color='black', linewidth=2,
            label='Total Reward', alpha=0.8)
    
    ax.set_xlabel('Episode', fontweight='bold')
    ax.set_ylabel('Reward', fontweight='bold')
    ax.set_title('Reward Composition (Extrinsic + Intrinsic)', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved reward comparison plot to {save_path}")


def plot_exploration_comparison(
    baseline_coverage: set,
    curiosity_coverage: set,
    grid_size: int,
    save_path: str
):
    """
    Side-by-side exploration comparison
    
    Args:
        baseline_coverage: Set of states visited by baseline
        curiosity_coverage: Set of states visited by curiosity agent
        grid_size: Size of grid
        save_path: Path to save figure
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Baseline heatmap
    baseline_map = np.zeros((grid_size, grid_size))
    for x, y in baseline_coverage:
        if 0 <= x < grid_size and 0 <= y < grid_size:
            baseline_map[y, x] = 1
    
    im1 = ax1.imshow(baseline_map, cmap='Blues', origin='lower', vmin=0, vmax=1)
    ax1.set_title('Baseline DQN Exploration', fontweight='bold')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    baseline_pct = len(baseline_coverage) / (grid_size * grid_size)
    ax1.text(0.5, 1.05, f'Coverage: {baseline_pct:.1%}',
             transform=ax1.transAxes, ha='center', fontweight='bold')
    
    # Curiosity heatmap
    curiosity_map = np.zeros((grid_size, grid_size))
    for x, y in curiosity_coverage:
        if 0 <= x < grid_size and 0 <= y < grid_size:
            curiosity_map[y, x] = 1
    
    im2 = ax2.imshow(curiosity_map, cmap='Oranges', origin='lower', vmin=0, vmax=1)
    ax2.set_title('Curiosity Agent Exploration', fontweight='bold')
    ax2.set_xlabel('X Position')
    ax2.set_ylabel('Y Position')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    curiosity_pct = len(curiosity_coverage) / (grid_size * grid_size)
    ax2.text(0.5, 1.05, f'Coverage: {curiosity_pct:.1%}',
             transform=ax2.transAxes, ha='center', fontweight='bold')
    
    # Difference map (unique to curiosity)
    difference_map = curiosity_map - baseline_map
    difference_map[difference_map < 0] = 0
    
    im3 = ax3.imshow(difference_map, cmap='Greens', origin='lower', vmin=0, vmax=1)
    ax3.set_title('Additional Exploration (Curiosity)', fontweight='bold')
    ax3.set_xlabel('X Position')
    ax3.set_ylabel('Y Position')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    improvement = (curiosity_pct - baseline_pct) / baseline_pct * 100
    ax3.text(0.5, 1.05, f'Improvement: +{improvement:.1f}%',
             transform=ax3.transAxes, ha='center', fontweight='bold', color='green')
    
    plt.suptitle('Exploration Comparison: Baseline vs Curiosity', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved exploration comparison to {save_path}")


def animate_curiosity_episode(
    episode_data: Dict,
    save_path: str,
    fps: int = 10
):
    """
    Create animated video of episode with curiosity visualization
    
    Args:
        episode_data: Dict with 'positions', 'intrinsic_rewards', 'obstacles', 'goals'
        save_path: Path to save MP4 video
        fps: Frames per second
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    grid_size = episode_data.get('grid_size', 10)
    positions = episode_data['positions']
    intrinsic_rewards = episode_data['intrinsic_rewards']
    
    # Setup environment view
    ax1.set_xlim(-0.5, grid_size - 0.5)
    ax1.set_ylim(-0.5, grid_size - 0.5)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Environment', fontweight='bold')
    
    # Draw obstacles
    if 'obstacles' in episode_data:
        for ox, oy, ow, oh in episode_data['obstacles']:
            rect = Rectangle((ox, oy), ow, oh, color='gray', alpha=0.7)
            ax1.add_patch(rect)
    
    # Draw goals
    if 'goals' in episode_data:
        for gx, gy in episode_data['goals']:
            circle = Circle((gx, gy), 0.3, color='gold', edgecolor='black', linewidth=2)
            ax1.add_patch(circle)
    
    # Robot marker
    robot, = ax1.plot([], [], 'bo', markersize=15, markeredgewidth=2, 
                     markeredgecolor='white')
    path_line, = ax1.plot([], [], 'b-', alpha=0.3, linewidth=2)
    
    # Intrinsic reward plot
    ax2.set_xlim(0, len(positions))
    ax2.set_ylim(0, max(intrinsic_rewards) * 1.1 if intrinsic_rewards else 1)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Intrinsic Reward')
    ax2.set_title('Curiosity Over Time', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    reward_line, = ax2.plot([], [], color=COLORS['curiosity'], linewidth=2)
    current_reward = ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    
    def init():
        robot.set_data([], [])
        path_line.set_data([], [])
        reward_line.set_data([], [])
        return robot, path_line, reward_line, current_reward
    
    def animate(frame):
        # Update robot position
        x, y = positions[frame]
        robot.set_data([x], [y])
        
        # Update path
        path_x = [p[0] for p in positions[:frame+1]]
        path_y = [p[1] for p in positions[:frame+1]]
        path_line.set_data(path_x, path_y)
        
        # Update intrinsic reward plot
        reward_line.set_data(range(frame+1), intrinsic_rewards[:frame+1])
        current_reward.set_xdata([frame])
        
        # Update title with current metrics
        fig.suptitle(f'Step {frame+1}/{len(positions)} | '
                    f'Intrinsic Reward: {intrinsic_rewards[frame]:.3f}',
                    fontsize=14, fontweight='bold')
        
        return robot, path_line, reward_line, current_reward
    
    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=len(positions), interval=1000/fps,
        blit=True, repeat=True
    )
    
    # Save animation
    writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
    anim.save(save_path, writer=writer)
    plt.close()
    print(f"✅ Saved animated episode to {save_path}")


# Example usage
if __name__ == "__main__":
    print("ICM Visualization Module")
    print("="*50)
    
    # Generate sample data
    np.random.seed(42)
    episodes = 1000
    intrinsic_rewards = np.random.exponential(0.5, episodes)
    
    output_dir = Path("results/figures/test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test intrinsic rewards plot
    plot_intrinsic_rewards(
        intrinsic_rewards.tolist(),
        str(output_dir / "intrinsic_rewards.png")
    )
    
    # Test exploration coverage
    visited = {(x, y) for x in range(10) for y in range(10) 
               if np.random.rand() > 0.3}
    plot_exploration_coverage(
        visited, 10,
        str(output_dir / "exploration_coverage.png"),
        robot_start=(0, 0),
        goals=[(9, 9)]
    )
    
    print("\n✅ All visualization tests passed!")
