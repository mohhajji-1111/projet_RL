"""
Interactive GUI for Training Visualization
"""
import pygame
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
from typing import List, Optional


class TrainingGUI:
    """Interactive GUI for monitoring training"""
    
    def __init__(
        self,
        width: int = 1200,
        height: int = 800,
        plot_width: int = 400
    ):
        self.width = width
        self.height = height
        self.plot_width = plot_width
        
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("RL Training Monitor")
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.title_font = pygame.font.Font(None, 32)
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        # Training data
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.losses: List[float] = []
        self.q_values: List[float] = []
        
        # UI state
        self.paused = False
        self.show_plots = True
        
    def update_data(
        self,
        reward: Optional[float] = None,
        length: Optional[int] = None,
        loss: Optional[float] = None,
        q_value: Optional[float] = None
    ):
        """Update training data"""
        if reward is not None:
            self.episode_rewards.append(reward)
        if length is not None:
            self.episode_lengths.append(length)
        if loss is not None:
            self.losses.append(loss)
        if q_value is not None:
            self.q_values.append(q_value)
    
    def create_plot_surface(
        self,
        data: List[float],
        title: str,
        ylabel: str,
        color: str = 'blue',
        window: int = 100
    ) -> pygame.Surface:
        """Create plot surface using matplotlib"""
        if not data:
            # Create empty plot
            fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
        else:
            # Create plot with data
            fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
            
            # Plot raw data
            episodes = range(len(data))
            ax.plot(episodes, data, alpha=0.3, color=color, linewidth=0.5)
            
            # Plot moving average
            if len(data) > window:
                moving_avg = np.convolve(
                    data,
                    np.ones(window) / window,
                    mode='valid'
                )
                ax.plot(
                    range(window-1, len(data)),
                    moving_avg,
                    color=color,
                    linewidth=2,
                    label=f'{window}-episode avg'
                )
                ax.legend()
            
            ax.set_title(title)
            ax.set_xlabel('Episode')
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
        
        # Convert to pygame surface
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.buffer_rgba()
        size = canvas.get_width_height()
        
        surf = pygame.image.frombuffer(raw_data, size, 'RGBA')
        plt.close(fig)
        
        return surf
    
    def draw_stats_panel(self, x: int, y: int):
        """Draw statistics panel"""
        panel_width = self.plot_width
        panel_height = 200
        
        # Background
        pygame.draw.rect(
            self.screen,
            (255, 255, 255),
            (x, y, panel_width, panel_height)
        )
        pygame.draw.rect(
            self.screen,
            (200, 200, 200),
            (x, y, panel_width, panel_height),
            2
        )
        
        # Title
        title_surf = self.title_font.render("Training Statistics", True, (50, 50, 50))
        self.screen.blit(title_surf, (x + 10, y + 10))
        
        y_offset = y + 50
        line_height = 30
        
        # Statistics
        stats = [
            ("Episodes:", len(self.episode_rewards)),
            ("Avg Reward (last 100):", f"{np.mean(self.episode_rewards[-100:]):.2f}" if self.episode_rewards else "N/A"),
            ("Avg Length (last 100):", f"{np.mean(self.episode_lengths[-100:]):.1f}" if self.episode_lengths else "N/A"),
            ("Avg Loss (last 100):", f"{np.mean(self.losses[-100:]):.4f}" if self.losses else "N/A"),
        ]
        
        for label, value in stats:
            text = f"{label} {value}"
            text_surf = self.font.render(text, True, (50, 50, 50))
            self.screen.blit(text_surf, (x + 20, y_offset))
            y_offset += line_height
    
    def draw_controls(self, x: int, y: int):
        """Draw control instructions"""
        controls = [
            "Controls:",
            "SPACE - Pause/Resume",
            "P - Toggle Plots",
            "R - Reset Stats",
            "Q - Quit"
        ]
        
        y_offset = y
        for text in controls:
            text_surf = self.small_font.render(text, True, (100, 100, 100))
            self.screen.blit(text_surf, (x, y_offset))
            y_offset += 20
    
    def render(self):
        """Render GUI"""
        self.screen.fill((240, 240, 245))
        
        # Environment view area (left side)
        env_area = pygame.Rect(0, 0, self.width - self.plot_width, self.height)
        pygame.draw.rect(self.screen, (255, 255, 255), env_area)
        
        # Plot area (right side)
        plot_x = self.width - self.plot_width
        
        if self.show_plots:
            # Reward plot
            if len(self.episode_rewards) > 0:
                reward_surf = self.create_plot_surface(
                    self.episode_rewards,
                    "Episode Rewards",
                    "Reward",
                    color='green'
                )
                self.screen.blit(reward_surf, (plot_x, 0))
            
            # Loss plot
            if len(self.losses) > 0:
                loss_surf = self.create_plot_surface(
                    self.losses,
                    "Training Loss",
                    "Loss",
                    color='red'
                )
                self.screen.blit(loss_surf, (plot_x, 300))
        
        # Statistics panel
        self.draw_stats_panel(plot_x, self.height - 250)
        
        # Controls
        self.draw_controls(10, self.height - 120)
        
        # Pause indicator
        if self.paused:
            pause_surf = self.title_font.render("PAUSED", True, (255, 0, 0))
            pause_rect = pause_surf.get_rect(center=(self.width // 4, 30))
            self.screen.blit(pause_surf, pause_rect)
        
        pygame.display.flip()
        self.clock.tick(30)
    
    def handle_events(self) -> bool:
        """Handle pygame events. Returns False if should quit"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_p:
                    self.show_plots = not self.show_plots
                elif event.key == pygame.K_r:
                    self.reset_stats()
                elif event.key == pygame.K_q:
                    return False
        
        return True
    
    def reset_stats(self):
        """Reset all statistics"""
        self.episode_rewards.clear()
        self.episode_lengths.clear()
        self.losses.clear()
        self.q_values.clear()
    
    def close(self):
        """Close GUI"""
        pygame.quit()
