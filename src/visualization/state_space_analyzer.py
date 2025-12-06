"""
State-Space Analysis and Visualization
Advanced analytics for RL state visitation, Q-values, and policy
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import torch
from scipy.stats import entropy
from scipy.ndimage import gaussian_filter
import plotly.graph_objs as go
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap


class StateSpaceAnalyzer:
    """
    Analyze and visualize state-space exploration, Q-value landscapes,
    and policy behavior in reinforcement learning.
    """
    
    def __init__(
        self,
        state_bounds: Tuple[Tuple[float, float], Tuple[float, float]],
        grid_resolution: int = 50,
        dpi: int = 300
    ):
        """
        Initialize state-space analyzer.
        
        Args:
            state_bounds: ((x_min, x_max), (y_min, y_max))
            grid_resolution: Grid size for discretization
            dpi: Resolution for saved figures
        """
        self.x_bounds, self.y_bounds = state_bounds
        self.grid_resolution = grid_resolution
        self.dpi = dpi
        
        # Create state grid
        self.x_grid = np.linspace(self.x_bounds[0], self.x_bounds[1], grid_resolution)
        self.y_grid = np.linspace(self.y_bounds[0], self.y_bounds[1], grid_resolution)
        self.XX, self.YY = np.meshgrid(self.x_grid, self.y_grid)
        
        # Visitation heatmap
        self.visitation_map = np.zeros((grid_resolution, grid_resolution))
        
        # Color scheme
        self.colors = {
            'low': '#2E86AB',
            'mid': '#F18F01',
            'high': '#C73E1D',
            'positive': '#06A77D',
            'negative': '#CC78BC'
        }
    
    def update_visitation(self, states: np.ndarray):
        """
        Update state visitation heatmap with new states.
        
        Args:
            states: Array of (x, y) positions, shape (N, 2)
        """
        for state in states:
            x, y = state[:2]  # Use first two dimensions
            
            # Convert to grid indices
            x_idx = int((x - self.x_bounds[0]) / (self.x_bounds[1] - self.x_bounds[0]) * self.grid_resolution)
            y_idx = int((y - self.y_bounds[0]) / (self.y_bounds[1] - self.y_bounds[0]) * self.grid_resolution)
            
            # Ensure within bounds
            x_idx = np.clip(x_idx, 0, self.grid_resolution - 1)
            y_idx = np.clip(y_idx, 0, self.grid_resolution - 1)
            
            self.visitation_map[y_idx, x_idx] += 1
    
    def visualize_state_visitation(
        self,
        save_path: str = None,
        smoothing: float = 1.0,
        obstacles: List[Dict] = None
    ):
        """
        Create state visitation heatmap.
        
        Args:
            save_path: Path to save figure
            smoothing: Gaussian smoothing sigma
            obstacles: List of obstacle dictionaries
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Apply smoothing
        smoothed = gaussian_filter(self.visitation_map, sigma=smoothing)
        
        # Create heatmap
        im = ax.imshow(
            smoothed,
            extent=[self.x_bounds[0], self.x_bounds[1], self.y_bounds[0], self.y_bounds[1]],
            origin='lower',
            cmap='hot',
            interpolation='bilinear',
            alpha=0.8
        )
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Visit Count', rotation=270, labelpad=20, fontsize=12, fontweight='bold')
        
        # Draw obstacles
        if obstacles:
            for obs in obstacles:
                if obs['type'] == 'circle':
                    circle = Circle(
                        obs['position'],
                        obs['radius'],
                        color='blue',
                        alpha=0.3,
                        edgecolor='blue',
                        linewidth=2
                    )
                    ax.add_patch(circle)
        
        # Add grid
        ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
        
        ax.set_xlabel('X Position (m)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Y Position (m)', fontsize=14, fontweight='bold')
        ax.set_title('State Visitation Heatmap', fontsize=16, fontweight='bold', pad=20)
        
        # Add statistics
        total_states = self.grid_resolution * self.grid_resolution
        visited_states = np.sum(self.visitation_map > 0)
        coverage = visited_states / total_states * 100
        
        stats_text = f'Coverage: {coverage:.1f}% ({visited_states}/{total_states} states)\nTotal visits: {int(self.visitation_map.sum())}'
        ax.text(
            0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9)
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ Saved state visitation: {save_path}")
        
        return fig
    
    def compute_q_value_landscape(
        self,
        agent,
        action: int = 0,
        goal: Optional[Tuple[float, float]] = None
    ) -> np.ndarray:
        """
        Compute Q-values across the state space for a specific action.
        
        Args:
            agent: Trained RL agent with Q-network
            action: Action index to evaluate
            goal: Optional goal position (x, y)
            
        Returns:
            Q-value grid
        """
        q_values = np.zeros((self.grid_resolution, self.grid_resolution))
        
        agent.eval()
        
        for i, x in enumerate(self.x_grid):
            for j, y in enumerate(self.y_grid):
                # Construct state (adjust based on your state representation)
                if goal is not None:
                    # State: [x, y, vx, vy, goal_x, goal_y, distance, angle]
                    distance = np.sqrt((x - goal[0])**2 + (y - goal[1])**2)
                    angle = np.arctan2(goal[1] - y, goal[0] - x)
                    state = np.array([x, y, 0, 0, goal[0], goal[1], distance, angle])
                else:
                    state = np.array([x, y, 0, 0, 0, 0, 0, 0])
                
                # Get Q-values
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    q_vals = agent(state_tensor).cpu().numpy()[0]
                
                q_values[j, i] = q_vals[action]
        
        return q_values
    
    def visualize_q_value_landscape(
        self,
        agent,
        action: int = 0,
        goal: Optional[Tuple[float, float]] = None,
        save_path: str = None,
        view_3d: bool = True
    ):
        """
        Visualize Q-value landscape as 3D surface plot.
        
        Args:
            agent: Trained RL agent
            action: Action index
            goal: Goal position
            save_path: Path to save figure
            view_3d: Create 3D surface plot
        """
        print("Computing Q-value landscape...")
        q_values = self.compute_q_value_landscape(agent, action, goal)
        
        if view_3d:
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Create surface plot
            surf = ax.plot_surface(
                self.XX, self.YY, q_values,
                cmap='viridis',
                linewidth=0,
                antialiased=True,
                alpha=0.8
            )
            
            # Add contour lines at the bottom
            ax.contour(
                self.XX, self.YY, q_values,
                zdir='z',
                offset=q_values.min(),
                cmap='viridis',
                alpha=0.5
            )
            
            ax.set_xlabel('X Position (m)', fontsize=12, fontweight='bold', labelpad=10)
            ax.set_ylabel('Y Position (m)', fontsize=12, fontweight='bold', labelpad=10)
            ax.set_zlabel('Q-Value', fontsize=12, fontweight='bold', labelpad=10)
            ax.set_title(f'Q-Value Landscape (Action {action})', fontsize=16, fontweight='bold', pad=20)
            
            # Add colorbar
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
            
            # Optimize viewing angle
            ax.view_init(elev=30, azim=45)
            
        else:
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Create 2D heatmap
            im = ax.contourf(self.XX, self.YY, q_values, levels=20, cmap='viridis')
            
            # Add contour lines
            contours = ax.contour(self.XX, self.YY, q_values, levels=10, colors='white', alpha=0.3, linewidths=1)
            ax.clabel(contours, inline=True, fontsize=8)
            
            # Mark goal
            if goal:
                ax.plot(goal[0], goal[1], 'r*', markersize=20, label='Goal')
            
            ax.set_xlabel('X Position (m)', fontsize=14, fontweight='bold')
            ax.set_ylabel('Y Position (m)', fontsize=14, fontweight='bold')
            ax.set_title(f'Q-Value Landscape (Action {action})', fontsize=16, fontweight='bold', pad=20)
            
            plt.colorbar(im, ax=ax, label='Q-Value')
            ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ Saved Q-value landscape: {save_path}")
        
        return fig
    
    def visualize_policy_arrows(
        self,
        agent,
        goal: Optional[Tuple[float, float]] = None,
        save_path: str = None,
        subsample: int = 5,
        obstacles: List[Dict] = None
    ):
        """
        Visualize policy as arrow field showing optimal actions.
        
        Args:
            agent: Trained RL agent
            goal: Goal position
            save_path: Path to save figure
            subsample: Subsampling factor for arrows
            obstacles: List of obstacles
        """
        fig, ax = plt.subplots(figsize=(14, 12))
        
        agent.eval()
        
        # Action to direction mapping (adjust for your actions)
        action_dirs = {
            0: (1, 0),      # Forward
            1: (0, 1),      # Rotate left (approximate)
            2: (0, -1),     # Rotate right (approximate)
            3: (-1, 0)      # Backward
        }
        
        # Compute policy for each grid point
        print("Computing policy arrows...")
        for i in range(0, self.grid_resolution, subsample):
            for j in range(0, self.grid_resolution, subsample):
                x = self.x_grid[i]
                y = self.y_grid[j]
                
                # Construct state
                if goal is not None:
                    distance = np.sqrt((x - goal[0])**2 + (y - goal[1])**2)
                    angle = np.arctan2(goal[1] - y, goal[0] - x)
                    state = np.array([x, y, 0, 0, goal[0], goal[1], distance, angle])
                else:
                    state = np.array([x, y, 0, 0, 0, 0, 0, 0])
                
                # Get best action
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    q_vals = agent(state_tensor).cpu().numpy()[0]
                    best_action = np.argmax(q_vals)
                    q_value = q_vals[best_action]
                
                # Get direction
                dx, dy = action_dirs.get(best_action, (0, 0))
                
                # Scale arrow by Q-value confidence
                scale = 0.3
                
                # Color by Q-value
                color = plt.cm.viridis((q_value - q_vals.min()) / (q_vals.max() - q_vals.min() + 1e-8))
                
                # Draw arrow
                arrow = FancyArrowPatch(
                    (x, y),
                    (x + dx * scale, y + dy * scale),
                    arrowstyle='->',
                    color=color,
                    linewidth=2,
                    mutation_scale=15,
                    alpha=0.7
                )
                ax.add_patch(arrow)
        
        # Draw obstacles
        if obstacles:
            for obs in obstacles:
                if obs['type'] == 'circle':
                    circle = Circle(
                        obs['position'],
                        obs['radius'],
                        color='red',
                        alpha=0.3,
                        edgecolor='red',
                        linewidth=2
                    )
                    ax.add_patch(circle)
        
        # Mark goal
        if goal:
            ax.plot(goal[0], goal[1], 'g*', markersize=30, label='Goal', zorder=10)
        
        ax.set_xlim(self.x_bounds)
        ax.set_ylim(self.y_bounds)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        ax.set_xlabel('X Position (m)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Y Position (m)', fontsize=14, fontweight='bold')
        ax.set_title('Policy Visualization (Arrow Field)', fontsize=16, fontweight='bold', pad=20)
        ax.legend(fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ Saved policy arrows: {save_path}")
        
        return fig
    
    def visualize_reward_distribution(
        self,
        rewards: np.ndarray,
        save_path: str = None
    ):
        """
        Visualize reward distribution with histogram and KDE.
        
        Args:
            rewards: Array of rewards
            save_path: Path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Histogram with KDE
        ax1.hist(rewards, bins=50, density=True, alpha=0.7, 
                color=self.colors['mid'], edgecolor='black', label='Histogram')
        
        # KDE
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(rewards)
        x_range = np.linspace(rewards.min(), rewards.max(), 200)
        ax1.plot(x_range, kde(x_range), 'r-', linewidth=3, label='KDE')
        
        # Add statistics lines
        mean_reward = np.mean(rewards)
        median_reward = np.median(rewards)
        
        ax1.axvline(mean_reward, color='green', linestyle='--', linewidth=2, label=f'Mean: {mean_reward:.2f}')
        ax1.axvline(median_reward, color='blue', linestyle='--', linewidth=2, label=f'Median: {median_reward:.2f}')
        
        ax1.set_xlabel('Reward', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Density', fontsize=14, fontweight='bold')
        ax1.set_title('Reward Distribution', fontsize=16, fontweight='bold')
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        bp = ax2.boxplot(
            [rewards],
            vert=True,
            patch_artist=True,
            widths=0.5,
            showmeans=True,
            meanprops=dict(marker='D', markerfacecolor='red', markersize=10)
        )
        
        for patch in bp['boxes']:
            patch.set_facecolor(self.colors['mid'])
            patch.set_alpha(0.7)
        
        ax2.set_ylabel('Reward', fontsize=14, fontweight='bold')
        ax2.set_title('Reward Box Plot', fontsize=16, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_xticklabels(['Episodes'])
        
        # Add statistics text
        stats_text = f"""
        Statistics:
        Mean: {np.mean(rewards):.2f}
        Median: {np.median(rewards):.2f}
        Std: {np.std(rewards):.2f}
        Min: {np.min(rewards):.2f}
        Max: {np.max(rewards):.2f}
        Q1: {np.percentile(rewards, 25):.2f}
        Q3: {np.percentile(rewards, 75):.2f}
        """
        
        ax2.text(
            1.5, 0.5, stats_text,
            transform=ax2.transData,
            fontsize=11, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontfamily='monospace'
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ Saved reward distribution: {save_path}")
        
        return fig
    
    def compute_policy_entropy(self, agent, goal: Optional[Tuple[float, float]] = None) -> float:
        """
        Compute average policy entropy across state space.
        
        Args:
            agent: Trained RL agent
            goal: Goal position
            
        Returns:
            Average entropy
        """
        entropies = []
        
        agent.eval()
        
        for i in range(0, self.grid_resolution, 5):
            for j in range(0, self.grid_resolution, 5):
                x = self.x_grid[i]
                y = self.y_grid[j]
                
                if goal is not None:
                    distance = np.sqrt((x - goal[0])**2 + (y - goal[1])**2)
                    angle = np.arctan2(goal[1] - y, goal[0] - x)
                    state = np.array([x, y, 0, 0, goal[0], goal[1], distance, angle])
                else:
                    state = np.array([x, y, 0, 0, 0, 0, 0, 0])
                
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    q_vals = agent(state_tensor).cpu().numpy()[0]
                    
                    # Convert to probabilities (softmax)
                    probs = np.exp(q_vals) / np.exp(q_vals).sum()
                    
                    # Compute entropy
                    ent = entropy(probs)
                    entropies.append(ent)
        
        return np.mean(entropies)
    
    def visualize_exploration_coverage(
        self,
        save_path: str = None
    ):
        """
        Visualize exploration coverage over time.
        
        Args:
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Calculate coverage percentage
        total_states = self.grid_resolution * self.grid_resolution
        visited = self.visitation_map > 0
        coverage_pct = np.sum(visited) / total_states * 100
        
        # Create visualization
        coverage_map = np.where(visited, 1, 0)
        
        im = ax.imshow(
            coverage_map,
            extent=[self.x_bounds[0], self.x_bounds[1], self.y_bounds[0], self.y_bounds[1]],
            origin='lower',
            cmap='RdYlGn',
            alpha=0.7
        )
        
        ax.set_xlabel('X Position (m)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Y Position (m)', fontsize=14, fontweight='bold')
        ax.set_title(f'Exploration Coverage: {coverage_pct:.1f}%', fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        
        # Add text annotation
        ax.text(
            0.5, 0.95,
            f'{int(np.sum(visited))} / {total_states} states visited',
            transform=ax.transAxes,
            ha='center', va='top',
            fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8)
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ Saved exploration coverage: {save_path}")
        
        return fig
    
    def create_complete_analysis(
        self,
        agent,
        rewards: np.ndarray,
        goal: Optional[Tuple[float, float]] = None,
        obstacles: List[Dict] = None,
        output_dir: str = "results/state_space"
    ):
        """
        Generate complete state-space analysis report.
        
        Args:
            agent: Trained RL agent
            rewards: Array of episode rewards
            goal: Goal position
            obstacles: List of obstacles
            output_dir: Directory to save visualizations
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("🎨 Generating state-space analysis report...")
        
        # 1. State visitation
        self.visualize_state_visitation(
            str(output_path / "01_state_visitation.png"),
            obstacles=obstacles
        )
        
        # 2. Q-value landscapes (all actions)
        for action in range(4):
            self.visualize_q_value_landscape(
                agent, action, goal,
                str(output_path / f"02_q_landscape_action_{action}.png"),
                view_3d=False
            )
        
        # 3. Policy arrows
        self.visualize_policy_arrows(
            agent, goal,
            str(output_path / "03_policy_arrows.png"),
            obstacles=obstacles
        )
        
        # 4. Reward distribution
        self.visualize_reward_distribution(
            rewards,
            str(output_path / "04_reward_distribution.png")
        )
        
        # 5. Exploration coverage
        self.visualize_exploration_coverage(
            str(output_path / "05_exploration_coverage.png")
        )
        
        # 6. Compute policy entropy
        avg_entropy = self.compute_policy_entropy(agent, goal)
        print(f"Average policy entropy: {avg_entropy:.4f}")
        
        print(f"✅ Analysis saved to: {output_path}")


# Convenience function
def analyze_state_space(
    agent,
    trajectory_data: Dict,
    rewards: np.ndarray,
    output_dir: str = "results/state_space"
):
    """Quick function to generate complete state-space analysis."""
    analyzer = StateSpaceAnalyzer(state_bounds=((0, 10), (0, 10)))
    
    # Update visitation
    for traj in trajectory_data:
        positions = np.array(traj['positions'])
        analyzer.update_visitation(positions)
    
    # Generate analysis
    analyzer.create_complete_analysis(agent, rewards, output_dir=output_dir)


if __name__ == "__main__":
    print("State-space analyzer ready!")
    print("Usage: analyzer.visualize_state_visitation('heatmap.png')")
