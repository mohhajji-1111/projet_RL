"""
Trajectory Animation and Visualization
Create MP4, GIF, and interactive HTML animations of robot paths
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle, FancyArrow, Wedge
from matplotlib.collections import LineCollection
import plotly.graph_objs as go
import plotly.express as px
from pathlib import Path
from typing import List, Tuple, Dict
import json
import imageio
from PIL import Image


class TrajectoryAnimator:
    """
    Create animated visualizations of robot trajectories.
    Supports MP4, GIF, and interactive HTML output.
    """
    
    def __init__(
        self,
        environment_size: Tuple[float, float] = (10.0, 10.0),
        fps: int = 30,
        dpi: int = 100
    ):
        """
        Initialize trajectory animator.
        
        Args:
            environment_size: (width, height) of environment
            fps: Frames per second for animations
            dpi: Resolution for saved animations
        """
        self.env_width, self.env_height = environment_size
        self.fps = fps
        self.dpi = dpi
        
        # Color scheme
        self.colors = {
            'robot': '#2E86AB',
            'goal': '#06A77D',
            'obstacle': '#C73E1D',
            'trajectory': '#F18F01',
            'lidar': '#A23B72',
            'collision': '#CC0000',
            'success': '#00CC00',
            'visited': '#FFD700'
        }
    
    def load_trajectory_data(self, log_file: str) -> Dict:
        """
        Load trajectory data from log file.
        
        Args:
            log_file: Path to trajectory log (JSON format)
            
        Returns:
            Dictionary containing trajectory information
        """
        with open(log_file, 'r') as f:
            data = json.load(f)
        return data
    
    def create_matplotlib_animation(
        self,
        trajectory: Dict,
        save_path: str,
        format: str = 'mp4',
        show_lidar: bool = True,
        show_speed: bool = True
    ):
        """
        Create animation using matplotlib.
        
        Args:
            trajectory: Dictionary with trajectory data
            save_path: Path to save animation
            format: 'mp4' or 'gif'
            show_lidar: Show LIDAR rays
            show_speed: Show speed indicator
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Extract data
        positions = np.array(trajectory['positions'])  # (N, 2)
        orientations = np.array(trajectory['orientations'])  # (N,)
        obstacles = trajectory.get('obstacles', [])
        goal = trajectory['goal']
        collisions = trajectory.get('collisions', [])
        success = trajectory.get('success', False)
        lidar_data = trajectory.get('lidar_rays', [])
        speeds = trajectory.get('speeds', np.ones(len(positions)))
        
        # Setup plot
        ax.set_xlim(0, self.env_width)
        ax.set_ylim(0, self.env_height)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X Position (m)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y Position (m)', fontsize=12, fontweight='bold')
        
        title_text = 'Successful Navigation' if success else 'Navigation Attempt'
        title_color = self.colors['success'] if success else self.colors['collision']
        ax.set_title(title_text, fontsize=16, fontweight='bold', color=title_color)
        
        # Draw static elements
        # Goal
        goal_circle = Circle(goal, 0.3, color=self.colors['goal'], alpha=0.3, zorder=1)
        ax.add_patch(goal_circle)
        ax.plot(goal[0], goal[1], 'g*', markersize=20, label='Goal', zorder=2)
        
        # Obstacles
        for obs in obstacles:
            if obs['type'] == 'circle':
                circle = Circle(
                    obs['position'],
                    obs['radius'],
                    color=self.colors['obstacle'],
                    alpha=0.6,
                    zorder=1
                )
                ax.add_patch(circle)
            elif obs['type'] == 'rectangle':
                rect = Rectangle(
                    (obs['position'][0] - obs['width']/2, obs['position'][1] - obs['height']/2),
                    obs['width'],
                    obs['height'],
                    color=self.colors['obstacle'],
                    alpha=0.6,
                    zorder=1
                )
                ax.add_patch(rect)
        
        # Initialize animated elements
        robot_body = Circle((0, 0), 0.2, color=self.colors['robot'], zorder=3)
        ax.add_patch(robot_body)
        
        robot_dir = FancyArrow(0, 0, 0, 0, width=0.05, head_width=0.15, 
                              head_length=0.1, color='white', zorder=4)
        ax.add_patch(robot_dir)
        
        trajectory_line, = ax.plot([], [], color=self.colors['trajectory'], 
                                   linewidth=2, alpha=0.7, label='Trajectory', zorder=2)
        
        lidar_lines = []
        if show_lidar:
            for _ in range(8):  # Assume 8 LIDAR beams
                line, = ax.plot([], [], color=self.colors['lidar'], 
                              linewidth=1, alpha=0.5, zorder=2)
                lidar_lines.append(line)
        
        # Time and speed text
        time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                           fontsize=12, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        if show_speed:
            speed_text = ax.text(0.02, 0.93, '', transform=ax.transAxes,
                               fontsize=12, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        else:
            speed_text = None
        
        # Collision markers
        collision_markers = []
        
        # Heatmap for visited states
        heatmap_data = np.zeros((50, 50))
        
        def init():
            """Initialize animation."""
            robot_body.center = (positions[0, 0], positions[0, 1])
            trajectory_line.set_data([], [])
            time_text.set_text('Time: 0.00s')
            if speed_text:
                speed_text.set_text('Speed: 0.00 m/s')
            return [robot_body, robot_dir, trajectory_line, time_text]
        
        def update(frame):
            """Update animation frame."""
            # Update robot position
            pos = positions[frame]
            ori = orientations[frame]
            
            robot_body.center = (pos[0], pos[1])
            
            # Update direction arrow
            robot_dir.remove()
            dx = 0.3 * np.cos(ori)
            dy = 0.3 * np.sin(ori)
            new_arrow = FancyArrow(
                pos[0], pos[1], dx, dy,
                width=0.05, head_width=0.15, head_length=0.1,
                color='white', zorder=4
            )
            ax.add_patch(new_arrow)
            
            # Update trajectory
            trajectory_line.set_data(positions[:frame+1, 0], positions[:frame+1, 1])
            
            # Color trajectory by time (gradient)
            segments = np.array([positions[:frame+1, 0], positions[:frame+1, 1]]).T.reshape(-1, 1, 2)
            segments = np.concatenate([segments[:-1], segments[1:]], axis=1)
            
            # Update LIDAR
            if show_lidar and frame < len(lidar_data):
                rays = lidar_data[frame]
                for i, (line, ray) in enumerate(zip(lidar_lines, rays)):
                    if ray is not None:
                        line.set_data([pos[0], ray[0]], [pos[1], ray[1]])
                    else:
                        line.set_data([], [])
            
            # Update time text
            time = frame / self.fps
            time_text.set_text(f'Time: {time:.2f}s | Step: {frame}')
            
            # Update speed
            if speed_text and frame < len(speeds):
                speed_text.set_text(f'Speed: {speeds[frame]:.2f} m/s')
            
            # Mark collisions
            if frame in collisions:
                collision_marker = Circle(pos, 0.3, color=self.colors['collision'], 
                                        alpha=0.5, zorder=2)
                ax.add_patch(collision_marker)
                collision_markers.append(collision_marker)
            
            # Update heatmap
            x_idx = int(pos[0] / self.env_width * 50)
            y_idx = int(pos[1] / self.env_height * 50)
            if 0 <= x_idx < 50 and 0 <= y_idx < 50:
                heatmap_data[y_idx, x_idx] += 1
            
            return [robot_body, new_arrow, trajectory_line, time_text]
        
        # Create animation
        anim = animation.FuncAnimation(
            fig, update, init_func=init,
            frames=len(positions),
            interval=1000/self.fps,
            blit=False,
            repeat=True
        )
        
        # Add legend
        ax.legend(loc='upper right', fontsize=10)
        
        # Save animation
        if format == 'mp4':
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=self.fps, bitrate=1800)
            anim.save(save_path, writer=writer, dpi=self.dpi)
            print(f"✓ Saved MP4 animation: {save_path}")
        elif format == 'gif':
            anim.save(save_path, writer='pillow', fps=self.fps, dpi=self.dpi)
            print(f"✓ Saved GIF animation: {save_path}")
        
        plt.close()
        return anim
    
    def create_heatmap_video(
        self,
        trajectories: List[Dict],
        save_path: str,
        format: str = 'mp4'
    ):
        """
        Create heatmap video showing visited states across multiple episodes.
        
        Args:
            trajectories: List of trajectory dictionaries
            save_path: Path to save video
            format: 'mp4' or 'gif'
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Setup heatmap axis
        heatmap_data = np.zeros((50, 50))
        
        # Process all trajectories to build heatmap
        for traj in trajectories:
            positions = np.array(traj['positions'])
            for pos in positions:
                x_idx = int(pos[0] / self.env_width * 50)
                y_idx = int(pos[1] / self.env_height * 50)
                if 0 <= x_idx < 50 and 0 <= y_idx < 50:
                    heatmap_data[y_idx, x_idx] += 1
        
        # Create heatmap
        im = ax1.imshow(
            heatmap_data,
            cmap='hot',
            interpolation='bilinear',
            extent=[0, self.env_width, 0, self.env_height],
            origin='lower',
            alpha=0.7
        )
        
        ax1.set_xlabel('X Position (m)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Y Position (m)', fontsize=12, fontweight='bold')
        ax1.set_title('State Visitation Heatmap', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax1, label='Visit Count')
        
        # Plot individual trajectory on second axis
        ax2.set_xlim(0, self.env_width)
        ax2.set_ylim(0, self.env_height)
        ax2.set_aspect('equal')
        ax2.set_xlabel('X Position (m)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Y Position (m)', fontsize=12, fontweight='bold')
        ax2.set_title('Current Trajectory', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        current_line, = ax2.plot([], [], 'b-', linewidth=2)
        
        def update(frame):
            if frame < len(trajectories):
                traj = trajectories[frame]
                positions = np.array(traj['positions'])
                current_line.set_data(positions[:, 0], positions[:, 1])
                
                # Draw goal
                goal = traj['goal']
                ax2.plot(goal[0], goal[1], 'g*', markersize=20)
            
            return [current_line]
        
        anim = animation.FuncAnimation(
            fig, update,
            frames=len(trajectories),
            interval=200,
            blit=False,
            repeat=True
        )
        
        plt.tight_layout()
        
        if format == 'mp4':
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=5, bitrate=1800)
            anim.save(save_path, writer=writer, dpi=self.dpi)
            print(f"✓ Saved heatmap video: {save_path}")
        elif format == 'gif':
            anim.save(save_path, writer='pillow', fps=5, dpi=self.dpi)
            print(f"✓ Saved heatmap GIF: {save_path}")
        
        plt.close()
    
    def create_plotly_interactive(
        self,
        trajectory: Dict,
        save_path: str
    ):
        """
        Create interactive HTML visualization using Plotly.
        
        Args:
            trajectory: Dictionary with trajectory data
            save_path: Path to save HTML file
        """
        positions = np.array(trajectory['positions'])
        obstacles = trajectory.get('obstacles', [])
        goal = trajectory['goal']
        success = trajectory.get('success', False)
        
        # Create figure
        fig = go.Figure()
        
        # Add trajectory with time-based coloring
        time_steps = np.arange(len(positions))
        
        fig.add_trace(go.Scatter(
            x=positions[:, 0],
            y=positions[:, 1],
            mode='lines+markers',
            name='Trajectory',
            line=dict(color=time_steps, colorscale='Viridis', width=3),
            marker=dict(
                size=8,
                color=time_steps,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Time Step")
            ),
            text=[f'Step {i}<br>Pos: ({positions[i, 0]:.2f}, {positions[i, 1]:.2f})' 
                  for i in range(len(positions))],
            hoverinfo='text'
        ))
        
        # Add start point
        fig.add_trace(go.Scatter(
            x=[positions[0, 0]],
            y=[positions[0, 1]],
            mode='markers',
            name='Start',
            marker=dict(size=15, color='blue', symbol='circle')
        ))
        
        # Add goal
        fig.add_trace(go.Scatter(
            x=[goal[0]],
            y=[goal[1]],
            mode='markers',
            name='Goal',
            marker=dict(size=20, color='green', symbol='star')
        ))
        
        # Add obstacles
        for obs in obstacles:
            if obs['type'] == 'circle':
                theta = np.linspace(0, 2*np.pi, 100)
                x = obs['position'][0] + obs['radius'] * np.cos(theta)
                y = obs['position'][1] + obs['radius'] * np.sin(theta)
                
                fig.add_trace(go.Scatter(
                    x=x, y=y,
                    fill='toself',
                    fillcolor='rgba(255, 0, 0, 0.3)',
                    line=dict(color='red', width=2),
                    name='Obstacle',
                    showlegend=False
                ))
        
        # Update layout
        title_text = '✓ Successful Navigation' if success else '✗ Failed Navigation'
        title_color = 'green' if success else 'red'
        
        fig.update_layout(
            title=dict(
                text=title_text,
                font=dict(size=20, color=title_color)
            ),
            xaxis_title='X Position (m)',
            yaxis_title='Y Position (m)',
            xaxis=dict(range=[0, self.env_width], constrain='domain'),
            yaxis=dict(range=[0, self.env_height], scaleanchor='x', scaleratio=1),
            hovermode='closest',
            template='plotly_white',
            font=dict(family='Arial, Helvetica, sans-serif', size=12),
            showlegend=True,
            legend=dict(x=1.05, y=1),
            width=900,
            height=800
        )
        
        # Save as HTML
        fig.write_html(save_path)
        print(f"✓ Saved interactive HTML: {save_path}")
    
    def create_3d_trajectory(
        self,
        trajectory: Dict,
        save_path: str,
        z_metric: str = 'speed'
    ):
        """
        Create 3D trajectory visualization with metric as Z-axis.
        
        Args:
            trajectory: Dictionary with trajectory data
            save_path: Path to save HTML file
            z_metric: Metric to use for Z-axis ('speed', 'reward', 'q_value')
        """
        positions = np.array(trajectory['positions'])
        
        # Get Z values based on metric
        if z_metric == 'speed':
            z_values = trajectory.get('speeds', np.ones(len(positions)))
        elif z_metric == 'reward':
            z_values = trajectory.get('rewards', np.zeros(len(positions)))
        elif z_metric == 'q_value':
            z_values = trajectory.get('q_values', np.zeros(len(positions)))
        else:
            z_values = np.arange(len(positions))
        
        # Create 3D scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=z_values,
            mode='lines+markers',
            marker=dict(
                size=6,
                color=z_values,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title=z_metric.title())
            ),
            line=dict(color='darkblue', width=4),
            text=[f'Step {i}<br>Pos: ({positions[i, 0]:.2f}, {positions[i, 1]:.2f})<br>{z_metric}: {z_values[i]:.2f}' 
                  for i in range(len(positions))],
            hoverinfo='text'
        )])
        
        fig.update_layout(
            title=f'3D Trajectory (colored by {z_metric})',
            scene=dict(
                xaxis_title='X Position (m)',
                yaxis_title='Y Position (m)',
                zaxis_title=z_metric.title(),
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            font=dict(family='Arial, Helvetica, sans-serif', size=12),
            width=900,
            height=800
        )
        
        fig.write_html(save_path)
        print(f"✓ Saved 3D trajectory: {save_path}")
    
    def create_comparison_animation(
        self,
        trajectories: Dict[str, Dict],
        save_path: str,
        format: str = 'mp4'
    ):
        """
        Create side-by-side comparison of multiple trajectories.
        
        Args:
            trajectories: Dictionary mapping names to trajectory data
            save_path: Path to save animation
            format: 'mp4' or 'gif'
        """
        n_trajs = len(trajectories)
        fig, axes = plt.subplots(1, n_trajs, figsize=(6*n_trajs, 6))
        
        if n_trajs == 1:
            axes = [axes]
        
        # Setup each subplot
        lines = []
        robots = []
        time_texts = []
        
        for idx, (name, traj) in enumerate(trajectories.items()):
            ax = axes[idx]
            ax.set_xlim(0, self.env_width)
            ax.set_ylim(0, self.env_height)
            ax.set_aspect('equal')
            ax.set_title(name, fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Goal
            goal = traj['goal']
            ax.plot(goal[0], goal[1], 'g*', markersize=15)
            
            # Obstacles
            for obs in traj.get('obstacles', []):
                if obs['type'] == 'circle':
                    circle = Circle(obs['position'], obs['radius'], 
                                  color=self.colors['obstacle'], alpha=0.6)
                    ax.add_patch(circle)
            
            # Robot and trajectory
            robot = Circle((0, 0), 0.2, color=self.colors['robot'])
            ax.add_patch(robot)
            robots.append(robot)
            
            line, = ax.plot([], [], 'b-', linewidth=2)
            lines.append(line)
            
            time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                              fontsize=10, verticalalignment='top',
                              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            time_texts.append(time_text)
        
        def update(frame):
            for idx, (name, traj) in enumerate(trajectories.items()):
                positions = np.array(traj['positions'])
                
                if frame < len(positions):
                    pos = positions[frame]
                    robots[idx].center = (pos[0], pos[1])
                    lines[idx].set_data(positions[:frame+1, 0], positions[:frame+1, 1])
                    time_texts[idx].set_text(f'Step: {frame}')
            
            return robots + lines + time_texts
        
        # Find max length
        max_len = max(len(traj['positions']) for traj in trajectories.values())
        
        anim = animation.FuncAnimation(
            fig, update,
            frames=max_len,
            interval=1000/self.fps,
            blit=False,
            repeat=True
        )
        
        plt.tight_layout()
        
        if format == 'mp4':
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=self.fps, bitrate=1800)
            anim.save(save_path, writer=writer, dpi=self.dpi)
            print(f"✓ Saved comparison video: {save_path}")
        elif format == 'gif':
            anim.save(save_path, writer='pillow', fps=self.fps, dpi=self.dpi)
            print(f"✓ Saved comparison GIF: {save_path}")
        
        plt.close()


# Convenience functions
def animate_trajectory(
    log_file: str,
    output_path: str,
    format: str = 'mp4',
    show_lidar: bool = True
):
    """Quick function to animate a single trajectory."""
    animator = TrajectoryAnimator()
    trajectory = animator.load_trajectory_data(log_file)
    animator.create_matplotlib_animation(trajectory, output_path, format, show_lidar)


def create_interactive_trajectory(log_file: str, output_path: str):
    """Quick function to create interactive HTML."""
    animator = TrajectoryAnimator()
    trajectory = animator.load_trajectory_data(log_file)
    animator.create_plotly_interactive(trajectory, output_path)


if __name__ == "__main__":
    # Example usage
    print("Trajectory animator ready!")
    print("Usage: animate_trajectory('log.json', 'output.mp4')")
