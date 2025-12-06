"""
Evaluation Script for Curiosity Agent

Evaluate trained Curiosity Agent and compare with baseline DQN.

Usage:
    python scripts/evaluate_curiosity.py --curiosity-model results/models/curiosity/best.pth --baseline-model results/models/dqn/best.pth
    python scripts/evaluate_curiosity.py --curiosity-model results/models/curiosity/best.pth --episodes 100 --render
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
from typing import Dict, List
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.environment.navigation_env import NavigationEnv
from src.agents.curiosity_agent import CuriosityAgent
from src.agents.dqn_agent import DQNAgent


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate Curiosity Agent")
    
    parser.add_argument(
        '--curiosity-model',
        type=str,
        required=True,
        help='Path to trained curiosity agent checkpoint'
    )
    parser.add_argument(
        '--baseline-model',
        type=str,
        default=None,
        help='Path to baseline DQN checkpoint (optional)'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=100,
        help='Number of evaluation episodes'
    )
    parser.add_argument(
        '--render',
        action='store_true',
        help='Render episodes'
    )
    parser.add_argument(
        '--save-video',
        action='store_true',
        help='Save videos of best episodes'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/evaluation',
        help='Output directory for results'
    )
    parser.add_argument(
        '--width',
        type=int,
        default=800,
        help='Environment width'
    )
    parser.add_argument(
        '--height',
        type=int,
        default=600,
        help='Environment height'
    )
    parser.add_argument(
        '--num-obstacles',
        type=int,
        default=5,
        help='Number of obstacles'
    )
    
    return parser.parse_args()


class AgentEvaluator:
    """Evaluator for comparing agents"""
    
    def __init__(self, env: NavigationEnv, output_dir: str):
        self.env = env
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def evaluate_agent(
        self,
        agent,
        num_episodes: int,
        render: bool = False
    ) -> Dict:
        """Evaluate single agent"""
        rewards = []
        steps_list = []
        successes = 0
        collisions = 0
        goals_reached = []
        visited_states = set()
        trajectories = []
        
        for ep in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_steps = 0
            done = False
            trajectory = []
            episode_visited = set()
            
            while not done:
                action = agent.select_action(state, epsilon=0.0)
                next_state, reward, done, info = self.env.step(action)
                
                episode_reward += reward
                episode_steps += 1
                
                # Track trajectory
                pos = tuple(self.env.robot_pos)
                trajectory.append(pos)
                episode_visited.add(pos)
                visited_states.add(pos)
                
                state = next_state
                
                if render:
                    self.env.render()
            
            rewards.append(episode_reward)
            steps_list.append(episode_steps)
            
            if info.get('success', False):
                successes += 1
            if info.get('collision', False):
                collisions += 1
            
            goals_reached.append(info.get('goals_reached', 0))
            trajectories.append(trajectory)
        
        # Calculate metrics
        grid_size = self.env.grid_size
        total_cells = grid_size * grid_size
        coverage = len(visited_states) / total_cells
        
        return {
            'rewards': rewards,
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'median_reward': np.median(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'steps': steps_list,
            'avg_steps': np.mean(steps_list),
            'std_steps': np.std(steps_list),
            'success_rate': successes / num_episodes,
            'collision_rate': collisions / num_episodes,
            'avg_goals': np.mean(goals_reached),
            'coverage': coverage,
            'visited_states': visited_states,
            'trajectories': trajectories
        }
    
    def compare_agents(
        self,
        curiosity_results: Dict,
        baseline_results: Dict
    ) -> Dict:
        """Statistical comparison between agents"""
        # T-test for rewards
        t_stat, p_value = stats.ttest_ind(
            curiosity_results['rewards'],
            baseline_results['rewards']
        )
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            (curiosity_results['std_reward']**2 + baseline_results['std_reward']**2) / 2
        )
        cohens_d = (curiosity_results['avg_reward'] - baseline_results['avg_reward']) / pooled_std
        
        # Improvement percentages
        reward_improvement = (
            (curiosity_results['avg_reward'] - baseline_results['avg_reward']) /
            abs(baseline_results['avg_reward']) * 100
        )
        
        success_improvement = (
            curiosity_results['success_rate'] - baseline_results['success_rate']
        ) * 100
        
        coverage_improvement = (
            (curiosity_results['coverage'] - baseline_results['coverage']) /
            baseline_results['coverage'] * 100
        )
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'reward_improvement': reward_improvement,
            'success_improvement': success_improvement,
            'coverage_improvement': coverage_improvement,
            'significant': p_value < 0.05
        }
    
    def plot_comparison(
        self,
        curiosity_results: Dict,
        baseline_results: Dict = None
    ):
        """Generate comparison plots"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Agent Evaluation Comparison', fontsize=16, fontweight='bold')
        
        # 1. Reward Distribution (Box Plot)
        ax = axes[0, 0]
        data = [curiosity_results['rewards']]
        labels = ['Curiosity']
        if baseline_results:
            data.append(baseline_results['rewards'])
            labels.append('Baseline')
        ax.boxplot(data, labels=labels)
        ax.set_ylabel('Reward')
        ax.set_title('Reward Distribution')
        ax.grid(True, alpha=0.3)
        
        # 2. Success Rate (Bar Chart)
        ax = axes[0, 1]
        success_rates = [curiosity_results['success_rate']]
        labels_sr = ['Curiosity']
        if baseline_results:
            success_rates.append(baseline_results['success_rate'])
            labels_sr.append('Baseline')
        ax.bar(labels_sr, success_rates, color=['#2ecc71', '#e74c3c'])
        ax.set_ylabel('Success Rate')
        ax.set_title('Success Rate Comparison')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')
        
        # 3. Coverage (Bar Chart)
        ax = axes[0, 2]
        coverages = [curiosity_results['coverage']]
        if baseline_results:
            coverages.append(baseline_results['coverage'])
        ax.bar(labels_sr, coverages, color=['#3498db', '#95a5a6'])
        ax.set_ylabel('Coverage (%)')
        ax.set_title('State Space Coverage')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 4. Steps Distribution
        ax = axes[1, 0]
        ax.hist(curiosity_results['steps'], bins=20, alpha=0.6, label='Curiosity', color='#2ecc71')
        if baseline_results:
            ax.hist(baseline_results['steps'], bins=20, alpha=0.6, label='Baseline', color='#e74c3c')
        ax.set_xlabel('Steps per Episode')
        ax.set_ylabel('Frequency')
        ax.set_title('Episode Length Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Exploration Heatmap (Curiosity)
        ax = axes[1, 1]
        grid_size = int(np.sqrt(len(curiosity_results['visited_states']) * 2))
        heatmap = np.zeros((grid_size, grid_size))
        for pos in curiosity_results['visited_states']:
            if pos[0] < grid_size and pos[1] < grid_size:
                heatmap[pos[1], pos[0]] += 1
        sns.heatmap(heatmap, ax=ax, cmap='YlOrRd', cbar_kws={'label': 'Visits'})
        ax.set_title('Curiosity Agent Exploration')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        # 6. Exploration Heatmap (Baseline)
        ax = axes[1, 2]
        if baseline_results:
            heatmap_baseline = np.zeros((grid_size, grid_size))
            for pos in baseline_results['visited_states']:
                if pos[0] < grid_size and pos[1] < grid_size:
                    heatmap_baseline[pos[1], pos[0]] += 1
            sns.heatmap(heatmap_baseline, ax=ax, cmap='Blues', cbar_kws={'label': 'Visits'})
            ax.set_title('Baseline Agent Exploration')
        else:
            ax.text(0.5, 0.5, 'No Baseline', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Baseline Agent Exploration')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        plt.tight_layout()
        
        # Save plot
        save_path = self.output_dir / 'comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Comparison plot saved to: {save_path}")
        plt.close()
    
    def save_report(
        self,
        curiosity_results: Dict,
        baseline_results: Dict = None,
        comparison: Dict = None
    ):
        """Generate evaluation report"""
        report_lines = [
            "# Agent Evaluation Report",
            "",
            "## Curiosity Agent Results",
            "",
            f"- **Average Reward**: {curiosity_results['avg_reward']:.2f} ± {curiosity_results['std_reward']:.2f}",
            f"- **Median Reward**: {curiosity_results['median_reward']:.2f}",
            f"- **Success Rate**: {curiosity_results['success_rate']:.2%}",
            f"- **Collision Rate**: {curiosity_results['collision_rate']:.2%}",
            f"- **Average Steps**: {curiosity_results['avg_steps']:.1f} ± {curiosity_results['std_steps']:.1f}",
            f"- **Average Goals Reached**: {curiosity_results['avg_goals']:.2f}",
            f"- **State Coverage**: {curiosity_results['coverage']:.2%}",
            ""
        ]
        
        if baseline_results:
            report_lines.extend([
                "## Baseline Agent Results",
                "",
                f"- **Average Reward**: {baseline_results['avg_reward']:.2f} ± {baseline_results['std_reward']:.2f}",
                f"- **Median Reward**: {baseline_results['median_reward']:.2f}",
                f"- **Success Rate**: {baseline_results['success_rate']:.2%}",
                f"- **Collision Rate**: {baseline_results['collision_rate']:.2%}",
                f"- **Average Steps**: {baseline_results['avg_steps']:.1f} ± {baseline_results['std_steps']:.1f}",
                f"- **Average Goals Reached**: {baseline_results['avg_goals']:.2f}",
                f"- **State Coverage**: {baseline_results['coverage']:.2%}",
                ""
            ])
        
        if comparison:
            significance = "✅ Yes" if comparison['significant'] else "❌ No"
            report_lines.extend([
                "## Statistical Comparison",
                "",
                f"- **T-statistic**: {comparison['t_statistic']:.3f}",
                f"- **P-value**: {comparison['p_value']:.4f}",
                f"- **Significant**: {significance}",
                f"- **Effect Size (Cohen's d)**: {comparison['cohens_d']:.3f}",
                "",
                "## Improvements",
                "",
                f"- **Reward**: {comparison['reward_improvement']:+.1f}%",
                f"- **Success Rate**: {comparison['success_improvement']:+.1f}%",
                f"- **Coverage**: {comparison['coverage_improvement']:+.1f}%",
                ""
            ])
        
        # Save report
        report_path = self.output_dir / 'evaluation_report.md'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"📄 Report saved to: {report_path}")
    
    def save_metrics_csv(
        self,
        curiosity_results: Dict,
        baseline_results: Dict = None
    ):
        """Save detailed metrics to CSV"""
        # Curiosity metrics
        df_curiosity = pd.DataFrame({
            'agent': 'curiosity',
            'episode': range(1, len(curiosity_results['rewards']) + 1),
            'reward': curiosity_results['rewards'],
            'steps': curiosity_results['steps']
        })
        
        if baseline_results:
            df_baseline = pd.DataFrame({
                'agent': 'baseline',
                'episode': range(1, len(baseline_results['rewards']) + 1),
                'reward': baseline_results['rewards'],
                'steps': baseline_results['steps']
            })
            df = pd.concat([df_curiosity, df_baseline], ignore_index=True)
        else:
            df = df_curiosity
        
        csv_path = self.output_dir / 'evaluation_metrics.csv'
        df.to_csv(csv_path, index=False)
        print(f"💾 Metrics saved to: {csv_path}")


def main():
    """Main evaluation"""
    args = parse_args()
    
    # Create environment
    env = NavigationEnv(
        width=args.width,
        height=args.height,
        render_mode=None
    )
    
    print("🤖 Loading Curiosity Agent...")
    curiosity_agent = CuriosityAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        config={}
    )
    curiosity_agent.load_checkpoint(args.curiosity_model)
    print("✅ Curiosity Agent loaded")
    
    # Load baseline if provided
    baseline_agent = None
    if args.baseline_model:
        print("🤖 Loading Baseline Agent...")
        baseline_agent = DQNAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n
        )
        baseline_agent.load(args.baseline_model)
        print("✅ Baseline Agent loaded")
    
    # Create evaluator
    evaluator = AgentEvaluator(env, args.output_dir)
    
    # Evaluate curiosity agent
    print(f"\n📊 Evaluating Curiosity Agent ({args.episodes} episodes)...")
    curiosity_results = evaluator.evaluate_agent(
        curiosity_agent,
        num_episodes=args.episodes,
        render=args.render
    )
    
    # Evaluate baseline if provided
    baseline_results = None
    comparison = None
    if baseline_agent:
        print(f"\n📊 Evaluating Baseline Agent ({args.episodes} episodes)...")
        baseline_results = evaluator.evaluate_agent(
            baseline_agent,
            num_episodes=args.episodes,
            render=args.render
        )
        
        # Compare agents
        comparison = evaluator.compare_agents(curiosity_results, baseline_results)
    
    # Generate plots
    print("\n📈 Generating plots...")
    evaluator.plot_comparison(curiosity_results, baseline_results)
    
    # Save report
    print("\n📝 Generating report...")
    evaluator.save_report(curiosity_results, baseline_results, comparison)
    
    # Save metrics
    evaluator.save_metrics_csv(curiosity_results, baseline_results)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"\nCuriosity Agent:")
    print(f"  Average Reward: {curiosity_results['avg_reward']:.2f} ± {curiosity_results['std_reward']:.2f}")
    print(f"  Success Rate: {curiosity_results['success_rate']:.2%}")
    print(f"  Coverage: {curiosity_results['coverage']:.2%}")
    
    if baseline_results:
        print(f"\nBaseline Agent:")
        print(f"  Average Reward: {baseline_results['avg_reward']:.2f} ± {baseline_results['std_reward']:.2f}")
        print(f"  Success Rate: {baseline_results['success_rate']:.2%}")
        print(f"  Coverage: {baseline_results['coverage']:.2%}")
        
        if comparison:
            print(f"\nImprovement:")
            print(f"  Reward: {comparison['reward_improvement']:+.1f}%")
            print(f"  Success: {comparison['success_improvement']:+.1f}%")
            print(f"  Coverage: {comparison['coverage_improvement']:+.1f}%")
            print(f"  Significant: {'Yes' if comparison['significant'] else 'No'} (p={comparison['p_value']:.4f})")
    
    print("="*60)
    print(f"\n✅ Evaluation complete! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
