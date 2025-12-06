"""
Comparison Visualizations for RL Training Runs
Side-by-side comparisons with statistical significance testing
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple
import json
from scipy import stats
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')


class ComparisonVisualizer:
    """
    Generate publication-quality comparison plots for different training runs.
    Supports ablation studies, parameter comparisons, and statistical testing.
    """
    
    def __init__(self, style: str = 'seaborn-v0_8-darkgrid', dpi: int = 300):
        """
        Initialize the comparison visualizer.
        
        Args:
            style: Matplotlib style
            dpi: Resolution for saved figures
        """
        plt.style.use('seaborn-v0_8-darkgrid')
        self.dpi = dpi
        
        # Color-blind friendly palette (Wong 2011)
        self.colors = {
            'blue': '#0173B2',
            'orange': '#DE8F05',
            'green': '#029E73',
            'red': '#CC78BC',
            'cyan': '#56B4E9',
            'magenta': '#CA9161',
            'yellow': '#ECE133',
            'purple': '#949494'
        }
        
        self.palette = list(self.colors.values())
        sns.set_palette(self.palette)
        
        # Statistical significance markers
        self.sig_markers = {
            0.05: '*',
            0.01: '**',
            0.001: '***'
        }
    
    def load_run_data(self, log_file: str) -> pd.DataFrame:
        """
        Load training run data from JSON log file.
        
        Args:
            log_file: Path to log file
            
        Returns:
            DataFrame with training metrics
        """
        with open(log_file, 'r') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        return df
    
    def compare_algorithms(
        self,
        runs: Dict[str, str],
        metrics: List[str] = ['reward', 'success_rate', 'steps', 'loss'],
        save_path: str = None
    ):
        """
        Compare different algorithms (e.g., DQN vs Rainbow DQN).
        
        Args:
            runs: Dictionary mapping algorithm names to log file paths
            metrics: List of metrics to compare
            save_path: Path to save the figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        # Load data for all runs
        run_data = {}
        for name, path in runs.items():
            run_data[name] = self.load_run_data(path)
        
        # Plot each metric
        for idx, metric in enumerate(metrics[:4]):
            ax = axes[idx]
            
            for name, df in run_data.items():
                if metric in df.columns:
                    # Plot with confidence interval
                    episodes = df['episode'].values
                    values = df[metric].values
                    
                    # Calculate moving average and std
                    window = 50
                    ma = pd.Series(values).rolling(window=window, min_periods=1).mean()
                    std = pd.Series(values).rolling(window=window, min_periods=1).std()
                    
                    # Plot line with confidence band
                    ax.plot(episodes, ma, label=name, linewidth=2)
                    ax.fill_between(
                        episodes,
                        ma - std,
                        ma + std,
                        alpha=0.2
                    )
            
            # Statistical significance test (last 100 episodes)
            if len(run_data) == 2:
                names = list(run_data.keys())
                data1 = run_data[names[0]][metric].values[-100:]
                data2 = run_data[names[1]][metric].values[-100:]
                
                # Perform t-test
                t_stat, p_value = stats.ttest_ind(data1, data2)
                sig_marker = self._get_significance_marker(p_value)
                
                if sig_marker:
                    ax.text(
                        0.95, 0.95,
                        f'p-value: {p_value:.4f} {sig_marker}',
                        transform=ax.transAxes,
                        ha='right',
                        va='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                    )
            
            ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
            ax.set_title(f'{metric.replace("_", " ").title()} Comparison', fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ Saved comparison plot: {save_path}")
        
        return fig
    
    def compare_environments(
        self,
        runs: Dict[str, str],
        save_path: str = None
    ):
        """
        Compare performance in different environments (e.g., static vs dynamic obstacles).
        
        Args:
            runs: Dictionary mapping environment types to log file paths
            save_path: Path to save the figure
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Load data
        run_data = {}
        for name, path in runs.items():
            run_data[name] = self.load_run_data(path)
        
        metrics = ['reward', 'success_rate', 'steps', 'loss', 'epsilon']
        
        for idx, metric in enumerate(metrics):
            if idx >= 6:
                break
            ax = axes[idx // 3, idx % 3]
            
            # Box plot for each environment
            data_for_plot = []
            labels = []
            
            for name, df in run_data.items():
                if metric in df.columns:
                    # Use last 100 episodes
                    values = df[metric].values[-100:]
                    data_for_plot.append(values)
                    labels.append(name)
            
            if data_for_plot:
                bp = ax.boxplot(
                    data_for_plot,
                    labels=labels,
                    patch_artist=True,
                    notch=True,
                    showmeans=True
                )
                
                # Color boxes
                for patch, color in zip(bp['boxes'], self.palette):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
                ax.set_title(f'{metric.replace("_", " ").title()} by Environment', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add statistical significance
                if len(data_for_plot) == 2:
                    t_stat, p_value = stats.ttest_ind(data_for_plot[0], data_for_plot[1])
                    sig_marker = self._get_significance_marker(p_value)
                    
                    if sig_marker:
                        y_max = max([max(d) for d in data_for_plot])
                        y_range = y_max - min([min(d) for d in data_for_plot])
                        y = y_max + 0.1 * y_range
                        
                        ax.plot([1, 2], [y, y], 'k-', linewidth=1)
                        ax.text(1.5, y, sig_marker, ha='center', va='bottom', fontsize=16)
        
        # Remove empty subplots
        for idx in range(len(metrics), 6):
            fig.delaxes(axes[idx // 3, idx % 3])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ Saved environment comparison: {save_path}")
        
        return fig
    
    def ablation_study(
        self,
        baseline: str,
        ablations: Dict[str, str],
        save_path: str = None
    ):
        """
        Visualize ablation study results.
        
        Args:
            baseline: Path to baseline (full model) log file
            ablations: Dictionary mapping ablation names to log file paths
            save_path: Path to save the figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Load baseline data
        baseline_df = self.load_run_data(baseline)
        baseline_reward = baseline_df['reward'].values[-100:].mean()
        
        # Load ablation data
        ablation_rewards = {}
        for name, path in ablations.items():
            df = self.load_run_data(path)
            ablation_rewards[name] = df['reward'].values[-100:].mean()
        
        # Calculate performance drop
        labels = ['Baseline'] + list(ablation_rewards.keys())
        rewards = [baseline_reward] + list(ablation_rewards.values())
        performance_drop = [0] + [baseline_reward - r for r in ablation_rewards.values()]
        
        # Create bar plot
        x = np.arange(len(labels))
        bars = ax.bar(x, rewards, color=self.palette[:len(labels)], alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Annotate performance drop
        for i, (bar, drop) in enumerate(zip(bars[1:], performance_drop[1:]), 1):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.5,
                f'↓ {drop:.1f}',
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold',
                color='red'
            )
        
        ax.set_xlabel('Configuration', fontsize=14, fontweight='bold')
        ax.set_ylabel('Average Reward (Last 100 Episodes)', fontsize=14, fontweight='bold')
        ax.set_title('Ablation Study: Component Importance', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add baseline reference line
        ax.axhline(y=baseline_reward, color='green', linestyle='--', linewidth=2, label='Baseline Performance')
        ax.legend(fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ Saved ablation study: {save_path}")
        
        return fig
    
    def compare_hyperparameters(
        self,
        runs: Dict[str, str],
        param_name: str,
        save_path: str = None
    ):
        """
        Compare different hyperparameter settings.
        
        Args:
            runs: Dictionary mapping parameter values to log file paths
            param_name: Name of the hyperparameter being compared
            save_path: Path to save the figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        # Load data
        run_data = {}
        for name, path in runs.items():
            run_data[name] = self.load_run_data(path)
        
        metrics = ['reward', 'success_rate', 'steps', 'loss']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            for name, df in run_data.items():
                if metric in df.columns:
                    episodes = df['episode'].values
                    values = df[metric].values
                    
                    # Smooth with moving average
                    window = 50
                    ma = pd.Series(values).rolling(window=window, min_periods=1).mean()
                    
                    ax.plot(episodes, ma, label=f'{param_name}={name}', linewidth=2)
            
            ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
            ax.set_title(f'{metric.replace("_", " ").title()} vs {param_name}', fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ Saved hyperparameter comparison: {save_path}")
        
        return fig
    
    def learning_curve_comparison(
        self,
        runs: Dict[str, str],
        confidence: float = 0.95,
        save_path: str = None
    ):
        """
        Compare learning curves with confidence intervals.
        
        Args:
            runs: Dictionary mapping run names to log file paths
            confidence: Confidence level for intervals
            save_path: Path to save the figure
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for name, path in runs.items():
            df = self.load_run_data(path)
            
            episodes = df['episode'].values
            rewards = df['reward'].values
            
            # Calculate statistics
            window = 50
            ma = pd.Series(rewards).rolling(window=window, min_periods=1).mean()
            std = pd.Series(rewards).rolling(window=window, min_periods=1).std()
            
            # Confidence interval
            z_score = stats.norm.ppf((1 + confidence) / 2)
            ci = z_score * std / np.sqrt(window)
            
            # Plot
            color = self.palette[len(ax.lines) // 3 % len(self.palette)]
            ax.plot(episodes, ma, label=name, linewidth=2.5, color=color)
            ax.fill_between(
                episodes,
                ma - ci,
                ma + ci,
                alpha=0.2,
                color=color
            )
        
        ax.set_xlabel('Episode', fontsize=14, fontweight='bold')
        ax.set_ylabel('Reward', fontsize=14, fontweight='bold')
        ax.set_title(f'Learning Curves with {int(confidence*100)}% Confidence Intervals', fontsize=16, fontweight='bold')
        ax.legend(loc='best', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ Saved learning curve comparison: {save_path}")
        
        return fig
    
    def performance_matrix(
        self,
        runs: Dict[str, str],
        metrics: List[str] = ['reward', 'success_rate', 'steps'],
        save_path: str = None
    ):
        """
        Create a heatmap matrix comparing multiple runs across metrics.
        
        Args:
            runs: Dictionary mapping run names to log file paths
            metrics: List of metrics to compare
            save_path: Path to save the figure
        """
        # Create matrix of average performance
        matrix = []
        row_labels = []
        
        for name, path in runs.items():
            df = self.load_run_data(path)
            row = []
            
            for metric in metrics:
                if metric in df.columns:
                    # Use last 100 episodes
                    avg_value = df[metric].values[-100:].mean()
                    row.append(avg_value)
                else:
                    row.append(0)
            
            matrix.append(row)
            row_labels.append(name)
        
        matrix = np.array(matrix)
        
        # Normalize each column (metric) to 0-1 range
        matrix_norm = (matrix - matrix.min(axis=0)) / (matrix.max(axis=0) - matrix.min(axis=0) + 1e-8)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        
        im = ax.imshow(matrix_norm, cmap='YlGnBu', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(metrics)))
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], fontsize=12)
        ax.set_yticklabels(row_labels, fontsize=12)
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Normalized Performance', rotation=270, labelpad=20, fontsize=12, fontweight='bold')
        
        # Add text annotations
        for i in range(len(row_labels)):
            for j in range(len(metrics)):
                text = ax.text(
                    j, i, f'{matrix[i, j]:.2f}',
                    ha="center", va="center",
                    color="white" if matrix_norm[i, j] > 0.5 else "black",
                    fontsize=10, fontweight='bold'
                )
        
        ax.set_title('Performance Comparison Matrix', fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ Saved performance matrix: {save_path}")
        
        return fig
    
    def _get_significance_marker(self, p_value: float) -> str:
        """Get statistical significance marker."""
        for threshold, marker in sorted(self.sig_markers.items()):
            if p_value < threshold:
                return marker
        return ''
    
    def export_comparison_report(
        self,
        runs: Dict[str, str],
        output_dir: str
    ):
        """
        Generate a complete comparison report with all visualizations.
        
        Args:
            runs: Dictionary mapping run names to log file paths
            output_dir: Directory to save all figures
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("🎨 Generating comparison report...")
        
        # 1. Algorithm comparison
        self.compare_algorithms(runs, save_path=str(output_path / "01_algorithm_comparison.png"))
        
        # 2. Learning curves
        self.learning_curve_comparison(runs, save_path=str(output_path / "02_learning_curves.png"))
        
        # 3. Performance matrix
        self.performance_matrix(runs, save_path=str(output_path / "03_performance_matrix.png"))
        
        # 4. Box plots for final performance
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        metrics = ['reward', 'success_rate', 'steps']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            data = []
            labels = []
            
            for name, path in runs.items():
                df = self.load_run_data(path)
                if metric in df.columns:
                    data.append(df[metric].values[-100:])
                    labels.append(name)
            
            bp = ax.boxplot(data, labels=labels, patch_artist=True, notch=True)
            
            for patch, color in zip(bp['boxes'], self.palette):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
            ax.set_title(f'{metric.replace("_", " ").title()} Distribution', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(str(output_path / "04_distribution_comparison.png"), dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Report saved to: {output_path}")


# Convenience functions
def compare_dqn_vs_rainbow(
    dqn_log: str,
    rainbow_log: str,
    output_dir: str = "results/comparisons"
):
    """Quick comparison of DQN vs Rainbow DQN."""
    viz = ComparisonVisualizer()
    runs = {'DQN': dqn_log, 'Rainbow DQN': rainbow_log}
    viz.export_comparison_report(runs, output_dir)


def compare_learning_rates(
    log_files: Dict[str, str],
    output_dir: str = "results/comparisons"
):
    """Quick comparison of different learning rates."""
    viz = ComparisonVisualizer()
    viz.compare_hyperparameters(log_files, 'Learning Rate', 
                                save_path=f"{output_dir}/learning_rate_comparison.png")


if __name__ == "__main__":
    # Example usage
    viz = ComparisonVisualizer()
    
    # Simulate comparison data
    runs = {
        'DQN': 'results/logs/dqn_training.json',
        'Rainbow DQN': 'results/logs/rainbow_training.json'
    }
    
    # Would generate comparisons if log files existed
    print("Comparison visualizer ready!")
    print("Usage: viz.compare_algorithms(runs, save_path='comparison.png')")
