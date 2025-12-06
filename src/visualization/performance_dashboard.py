"""
Performance Metrics Dashboard
Statistical analysis and publication-quality performance visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from scipy import stats
from scipy.ndimage import gaussian_filter1d
import json


class PerformanceMetricsDashboard:
    """
    Create comprehensive performance analysis visualizations with
    statistical analysis and publication-quality plots.
    """
    
    def __init__(self, dpi: int = 300):
        """
        Initialize performance metrics dashboard.
        
        Args:
            dpi: Resolution for saved figures
        """
        self.dpi = dpi
        
        # Set style
        sns.set_style("whitegrid")
        sns.set_context("paper", font_scale=1.5)
        
        # Color palette (color-blind friendly)
        self.palette = sns.color_palette("colorblind")
        sns.set_palette(self.palette)
    
    def load_training_data(self, log_file: str) -> pd.DataFrame:
        """
        Load training data from JSON log file.
        
        Args:
            log_file: Path to training log
            
        Returns:
            DataFrame with training metrics
        """
        with open(log_file, 'r') as f:
            data = json.load(f)
        
        return pd.DataFrame(data)
    
    def create_box_plots(
        self,
        data_dict: Dict[str, np.ndarray],
        metric_name: str,
        save_path: str = None
    ):
        """
        Create box plots comparing different configurations.
        
        Args:
            data_dict: Dictionary mapping labels to data arrays
            metric_name: Name of the metric being plotted
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Prepare data
        data_list = list(data_dict.values())
        labels = list(data_dict.keys())
        
        # Create box plot
        bp = ax.boxplot(
            data_list,
            labels=labels,
            patch_artist=True,
            notch=True,
            showmeans=True,
            meanprops=dict(
                marker='D',
                markerfacecolor='red',
                markeredgecolor='red',
                markersize=8,
                label='Mean'
            ),
            medianprops=dict(color='black', linewidth=2),
            widths=0.6
        )
        
        # Color boxes
        for patch, color in zip(bp['boxes'], self.palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Styling
        ax.set_ylabel(metric_name, fontsize=14, fontweight='bold')
        ax.set_title(f'{metric_name} Distribution by Configuration', fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Rotate labels if needed
        if len(labels) > 3:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Add statistical significance tests
        if len(data_list) == 2:
            # T-test
            t_stat, p_value = stats.ttest_ind(data_list[0], data_list[1])
            
            # Add significance annotation
            y_max = max([max(d) for d in data_list])
            y_min = min([min(d) for d in data_list])
            y_range = y_max - y_min
            y = y_max + 0.1 * y_range
            
            sig_marker = self._get_significance_marker(p_value)
            
            ax.plot([1, 2], [y, y], 'k-', linewidth=2)
            ax.text(
                1.5, y,
                f'p={p_value:.4f} {sig_marker}',
                ha='center', va='bottom',
                fontsize=12, fontweight='bold'
            )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ Saved box plot: {save_path}")
        
        return fig
    
    def create_violin_plots(
        self,
        data_dict: Dict[str, np.ndarray],
        metric_name: str,
        save_path: str = None
    ):
        """
        Create violin plots for detailed distribution comparison.
        
        Args:
            data_dict: Dictionary mapping labels to data arrays
            metric_name: Name of the metric
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Prepare data for seaborn
        data_for_plot = []
        for label, values in data_dict.items():
            for v in values:
                data_for_plot.append({'Configuration': label, metric_name: v})
        
        df = pd.DataFrame(data_for_plot)
        
        # Create violin plot
        sns.violinplot(
            data=df,
            x='Configuration',
            y=metric_name,
            palette=self.palette,
            inner='quartile',
            ax=ax
        )
        
        # Overlay swarm plot (if not too many points)
        total_points = sum(len(v) for v in data_dict.values())
        if total_points < 500:
            sns.swarmplot(
                data=df,
                x='Configuration',
                y=metric_name,
                color='black',
                alpha=0.5,
                size=3,
                ax=ax
            )
        
        ax.set_ylabel(metric_name, fontsize=14, fontweight='bold')
        ax.set_xlabel('Configuration', fontsize=14, fontweight='bold')
        ax.set_title(f'{metric_name} Distribution (Violin Plot)', fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='y')
        
        if len(data_dict) > 3:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ Saved violin plot: {save_path}")
        
        return fig
    
    def create_swarm_plot(
        self,
        data_dict: Dict[str, np.ndarray],
        metric_name: str,
        save_path: str = None
    ):
        """
        Create swarm plot showing all individual data points.
        
        Args:
            data_dict: Dictionary mapping labels to data arrays
            metric_name: Name of the metric
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Prepare data
        data_for_plot = []
        for label, values in data_dict.items():
            for v in values:
                data_for_plot.append({'Configuration': label, metric_name: v})
        
        df = pd.DataFrame(data_for_plot)
        
        # Create swarm plot
        sns.swarmplot(
            data=df,
            x='Configuration',
            y=metric_name,
            palette=self.palette,
            size=6,
            alpha=0.7,
            ax=ax
        )
        
        # Add mean markers
        for i, (label, values) in enumerate(data_dict.items()):
            mean_val = np.mean(values)
            ax.plot(i, mean_val, 'r*', markersize=20, markeredgecolor='black', markeredgewidth=1.5, zorder=10)
        
        ax.set_ylabel(metric_name, fontsize=14, fontweight='bold')
        ax.set_xlabel('Configuration', fontsize=14, fontweight='bold')
        ax.set_title(f'{metric_name} - All Episodes (Swarm Plot)', fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='y')
        
        if len(data_dict) > 3:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ Saved swarm plot: {save_path}")
        
        return fig
    
    def create_correlation_matrix(
        self,
        df: pd.DataFrame,
        save_path: str = None
    ):
        """
        Create correlation matrix heatmap.
        
        Args:
            df: DataFrame with multiple metrics
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Compute correlation matrix
        corr = df.corr()
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        # Create heatmap
        sns.heatmap(
            corr,
            mask=mask,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            vmin=-1,
            vmax=1,
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8},
            ax=ax
        )
        
        ax.set_title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ Saved correlation matrix: {save_path}")
        
        return fig
    
    def create_learning_curves_with_ci(
        self,
        runs_dict: Dict[str, List[np.ndarray]],
        metric_name: str,
        confidence: float = 0.95,
        save_path: str = None
    ):
        """
        Create learning curves with confidence intervals from multiple runs.
        
        Args:
            runs_dict: Dictionary mapping labels to list of run arrays
            metric_name: Name of the metric
            confidence: Confidence level for intervals
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for label, runs in runs_dict.items():
            # Convert to array (episodes x runs)
            min_length = min(len(run) for run in runs)
            runs_array = np.array([run[:min_length] for run in runs])
            
            episodes = np.arange(min_length)
            
            # Calculate statistics
            mean = runs_array.mean(axis=0)
            std = runs_array.std(axis=0)
            
            # Smooth
            window = 50
            mean_smooth = pd.Series(mean).rolling(window=window, min_periods=1).mean()
            std_smooth = pd.Series(std).rolling(window=window, min_periods=1).mean()
            
            # Confidence interval
            n_runs = len(runs)
            z_score = stats.norm.ppf((1 + confidence) / 2)
            ci = z_score * std_smooth / np.sqrt(n_runs)
            
            # Plot
            color = self.palette[len(ax.lines) // 3 % len(self.palette)]
            
            ax.plot(episodes, mean_smooth, label=f'{label} (n={n_runs})', linewidth=2.5, color=color)
            ax.fill_between(
                episodes,
                mean_smooth - ci,
                mean_smooth + ci,
                alpha=0.2,
                color=color
            )
        
        ax.set_xlabel('Episode', fontsize=14, fontweight='bold')
        ax.set_ylabel(metric_name, fontsize=14, fontweight='bold')
        ax.set_title(
            f'{metric_name} Learning Curves with {int(confidence*100)}% Confidence Intervals',
            fontsize=16, fontweight='bold', pad=20
        )
        ax.legend(loc='best', fontsize=12, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ Saved learning curves: {save_path}")
        
        return fig
    
    def create_convergence_analysis(
        self,
        data: np.ndarray,
        window: int = 50,
        threshold: float = 0.01,
        save_path: str = None
    ):
        """
        Analyze and visualize convergence of training.
        
        Args:
            data: Training metric over time
            window: Window size for moving average
            threshold: Convergence threshold (std deviation)
            save_path: Path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        episodes = np.arange(len(data))
        
        # Moving average and std
        ma = pd.Series(data).rolling(window=window, min_periods=1).mean()
        std = pd.Series(data).rolling(window=window, min_periods=1).std()
        
        # Plot 1: Raw data with moving average
        ax1.plot(episodes, data, alpha=0.3, color='gray', linewidth=1, label='Raw')
        ax1.plot(episodes, ma, color='blue', linewidth=3, label=f'MA({window})')
        ax1.fill_between(episodes, ma - std, ma + std, alpha=0.2, color='blue')
        
        ax1.set_xlabel('Episode', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Metric Value', fontsize=12, fontweight='bold')
        ax1.set_title('Training Progress', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Convergence indicator (rolling std)
        ax2.plot(episodes, std, color='red', linewidth=2, label='Rolling Std Dev')
        ax2.axhline(threshold, color='green', linestyle='--', linewidth=2, 
                   label=f'Convergence threshold ({threshold})')
        
        # Find convergence point
        converged_idx = np.where(std < threshold)[0]
        if len(converged_idx) > 0:
            convergence_episode = converged_idx[0]
            ax2.axvline(convergence_episode, color='green', linestyle=':', linewidth=2, alpha=0.7)
            ax2.text(
                convergence_episode, ax2.get_ylim()[1] * 0.9,
                f'Converged at episode {convergence_episode}',
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8)
            )
        
        ax2.set_xlabel('Episode', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Standard Deviation', fontsize=12, fontweight='bold')
        ax2.set_title('Convergence Analysis', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ Saved convergence analysis: {save_path}")
        
        return fig
    
    def create_performance_summary_table(
        self,
        data_dict: Dict[str, np.ndarray],
        save_path: str = None
    ):
        """
        Create summary statistics table.
        
        Args:
            data_dict: Dictionary mapping labels to data arrays
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis('off')
        
        # Compute statistics
        stats_data = []
        for label, values in data_dict.items():
            stats_data.append([
                label,
                f"{np.mean(values):.3f}",
                f"{np.median(values):.3f}",
                f"{np.std(values):.3f}",
                f"{np.min(values):.3f}",
                f"{np.max(values):.3f}",
                f"{np.percentile(values, 25):.3f}",
                f"{np.percentile(values, 75):.3f}",
                f"{len(values)}"
            ])
        
        # Create table
        table = ax.table(
            cellText=stats_data,
            colLabels=['Config', 'Mean', 'Median', 'Std', 'Min', 'Max', 'Q1', 'Q3', 'N'],
            cellLoc='center',
            loc='center',
            colWidths=[0.15] + [0.1] * 8
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 3)
        
        # Style header
        for i in range(len(stats_data[0])):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(stats_data) + 1):
            for j in range(len(stats_data[0])):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
        
        ax.set_title('Performance Statistics Summary', fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ Saved summary table: {save_path}")
        
        return fig
    
    def _get_significance_marker(self, p_value: float) -> str:
        """Get statistical significance marker."""
        if p_value < 0.001:
            return '***'
        elif p_value < 0.01:
            return '**'
        elif p_value < 0.05:
            return '*'
        else:
            return 'n.s.'
    
    def create_complete_dashboard(
        self,
        runs_dict: Dict[str, str],
        output_dir: str = "results/performance"
    ):
        """
        Generate complete performance metrics dashboard.
        
        Args:
            runs_dict: Dictionary mapping run names to log file paths
            output_dir: Directory to save all visualizations
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("🎨 Generating performance metrics dashboard...")
        
        # Load all data
        all_data = {}
        for name, path in runs_dict.items():
            all_data[name] = self.load_training_data(path)
        
        # Extract metrics
        rewards = {name: df['reward'].values[-100:] for name, df in all_data.items()}
        success_rates = {name: df['success_rate'].values[-100:] for name, df in all_data.items()}
        steps = {name: df['steps'].values[-100:] for name, df in all_data.items()}
        
        # 1. Box plots
        self.create_box_plots(rewards, 'Episode Reward', 
                             str(output_path / "01_rewards_boxplot.png"))
        self.create_box_plots(success_rates, 'Success Rate',
                             str(output_path / "02_success_boxplot.png"))
        
        # 2. Violin plots
        self.create_violin_plots(rewards, 'Episode Reward',
                                str(output_path / "03_rewards_violin.png"))
        
        # 3. Swarm plot (if not too many points)
        total_points = sum(len(v) for v in rewards.values())
        if total_points < 1000:
            self.create_swarm_plot(rewards, 'Episode Reward',
                                  str(output_path / "04_rewards_swarm.png"))
        
        # 4. Correlation matrix (combine all runs)
        combined_df = pd.concat([df for df in all_data.values()], ignore_index=True)
        if len(combined_df.columns) > 1:
            self.create_correlation_matrix(combined_df[['reward', 'success_rate', 'steps', 'loss']],
                                          str(output_path / "05_correlation_matrix.png"))
        
        # 5. Learning curves with CI
        rewards_list = {name: [df['reward'].values] for name, df in all_data.items()}
        self.create_learning_curves_with_ci(rewards_list, 'Reward',
                                           str(output_path / "06_learning_curves_ci.png"))
        
        # 6. Convergence analysis
        first_run = list(all_data.values())[0]
        self.create_convergence_analysis(first_run['reward'].values,
                                        str(output_path / "07_convergence_analysis.png"))
        
        # 7. Summary table
        self.create_performance_summary_table(rewards,
                                             str(output_path / "08_summary_table.png"))
        
        print(f"✅ Dashboard saved to: {output_path}")


# Convenience function
def generate_performance_dashboard(
    log_files: Dict[str, str],
    output_dir: str = "results/performance"
):
    """Quick function to generate complete performance dashboard."""
    dashboard = PerformanceMetricsDashboard()
    dashboard.create_complete_dashboard(log_files, output_dir)


if __name__ == "__main__":
    print("Performance metrics dashboard ready!")
    print("Usage: dashboard.create_complete_dashboard(runs_dict, 'results/performance')")
