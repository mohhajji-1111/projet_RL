"""
Hyperparameter Optimization using Ray Tune

Features:
- Population Based Training (PBT)
- ASHA scheduler for efficient search
- Bayesian Optimization
- Distributed trials across machines
- TensorBoard integration
- Advanced scheduling algorithms

Author: Advanced Training System
Date: 2025-12-06
"""

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search import ConcurrencyLimiter
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import logging


class RayTuneOptimizer:
    """
    Advanced hyperparameter optimization using Ray Tune.
    
    Features:
    - Multiple search algorithms
    - Advanced schedulers (ASHA, PBT)
    - Distributed training
    - TensorBoard logging
    """
    
    def __init__(
        self,
        experiment_name: str,
        local_dir: str = './ray_results',
        num_cpus: Optional[int] = None,
        num_gpus: Optional[int] = None
    ):
        """
        Initialize Ray Tune optimizer.
        
        Args:
            experiment_name: Name of the experiment
            local_dir: Directory for results
            num_cpus: Number of CPUs to use (None for all)
            num_gpus: Number of GPUs to use (None for all available)
        """
        self.experiment_name = experiment_name
        self.local_dir = Path(local_dir)
        
        # Initialize Ray
        if not ray.is_initialized():
            ray.init(
                num_cpus=num_cpus,
                num_gpus=num_gpus,
                ignore_reinit_error=True
            )
        
        self.logger = logging.getLogger(f'RayTuneOptimizer_{experiment_name}')
        self.logger.setLevel(logging.INFO)
    
    def get_search_space(self) -> Dict[str, Any]:
        """
        Define hyperparameter search space for Ray Tune.
        
        Returns:
            Dictionary with search space definitions
        """
        search_space = {
            # Network architecture
            'num_layers': tune.choice([2, 3, 4]),
            'hidden_dim_0': tune.choice([64, 128, 256, 512]),
            'hidden_dim_1': tune.choice([64, 128, 256, 512]),
            'hidden_dim_2': tune.choice([64, 128, 256, 512]),
            'hidden_dim_3': tune.choice([64, 128, 256, 512]),
            
            # Learning parameters
            'learning_rate': tune.loguniform(1e-5, 1e-2),
            'batch_size': tune.choice([16, 32, 64, 128, 256]),
            
            # RL parameters
            'gamma': tune.uniform(0.95, 0.999),
            'epsilon_decay': tune.uniform(0.990, 0.999),
            'buffer_size': tune.choice([10000, 25000, 50000, 100000]),
            'target_update': tune.choice([100, 250, 500, 1000]),
            
            # Regularization
            'use_dropout': tune.choice([True, False]),
            'dropout_rate': tune.uniform(0.1, 0.5),
            
            # Learning rate schedule
            'use_lr_schedule': tune.choice([True, False]),
            'lr_decay_factor': tune.uniform(0.1, 0.9),
            'lr_decay_steps': tune.choice([100, 200, 300, 500]),
        }
        
        return search_space
    
    def get_asha_scheduler(
        self,
        max_t: int = 500,
        grace_period: int = 50,
        reduction_factor: int = 3
    ) -> ASHAScheduler:
        """
        Create ASHA (Async Successive Halving Algorithm) scheduler.
        
        Args:
            max_t: Maximum number of training iterations
            grace_period: Minimum number of iterations before stopping
            reduction_factor: Factor by which to reduce trials
            
        Returns:
            ASHA scheduler
        """
        scheduler = ASHAScheduler(
            time_attr='training_iteration',
            metric='avg_reward',
            mode='max',
            max_t=max_t,
            grace_period=grace_period,
            reduction_factor=reduction_factor
        )
        return scheduler
    
    def get_pbt_scheduler(
        self,
        time_attr: str = 'training_iteration',
        perturbation_interval: int = 50,
        hyperparam_mutations: Optional[Dict] = None
    ) -> PopulationBasedTraining:
        """
        Create Population Based Training scheduler.
        
        Args:
            time_attr: Time attribute to use
            perturbation_interval: Interval for hyperparameter perturbation
            hyperparam_mutations: Custom mutations
            
        Returns:
            PBT scheduler
        """
        if hyperparam_mutations is None:
            hyperparam_mutations = {
                'learning_rate': tune.loguniform(1e-5, 1e-2),
                'gamma': tune.uniform(0.95, 0.999),
                'epsilon_decay': tune.uniform(0.990, 0.999),
            }
        
        scheduler = PopulationBasedTraining(
            time_attr=time_attr,
            metric='avg_reward',
            mode='max',
            perturbation_interval=perturbation_interval,
            hyperparam_mutations=hyperparam_mutations
        )
        return scheduler
    
    def get_bayesopt_search(
        self,
        max_concurrent: int = 4
    ) -> ConcurrencyLimiter:
        """
        Create Bayesian Optimization search algorithm.
        
        Args:
            max_concurrent: Maximum concurrent trials
            
        Returns:
            Bayesian optimization search with concurrency limiter
        """
        search_alg = BayesOptSearch(
            metric='avg_reward',
            mode='max'
        )
        
        # Limit concurrent trials for efficiency
        search_alg = ConcurrencyLimiter(search_alg, max_concurrent=max_concurrent)
        
        return search_alg
    
    def trainable_function(self, config: Dict[str, Any]):
        """
        Trainable function for Ray Tune.
        
        Args:
            config: Hyperparameter configuration
        """
        # Import here to avoid issues with Ray
        from src.agents.dqn_agent import DQNAgent
        from src.environment.robot_env import RobotNavigationEnv
        
        # Create environment
        env = RobotNavigationEnv(size=(10, 10), num_obstacles=5, num_goals=3)
        
        # Build hidden_dims from config
        num_layers = config['num_layers']
        hidden_dims = [config[f'hidden_dim_{i}'] for i in range(num_layers)]
        
        # Create agent
        agent = DQNAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            hidden_dims=hidden_dims,
            learning_rate=config['learning_rate'],
            gamma=config['gamma'],
            buffer_size=config['buffer_size'],
            batch_size=config['batch_size']
        )
        
        # Training loop
        rewards_history = []
        
        for episode in range(500):  # 500 episodes per trial
            state = env.reset()
            episode_reward = 0
            done = False
            
            # Epsilon decay
            epsilon = max(0.01, 1.0 * (config['epsilon_decay'] ** episode))
            
            while not done:
                action = agent.select_action(state, epsilon)
                next_state, reward, done, _ = env.step(action)
                
                agent.store_transition(state, action, reward, next_state, done)
                if agent.can_train():
                    agent.train_step()
                
                episode_reward += reward
                state = next_state
            
            rewards_history.append(episode_reward)
            
            # Report metrics every 10 episodes
            if episode % 10 == 0:
                avg_reward = np.mean(rewards_history[-100:] if len(rewards_history) >= 100 else rewards_history)
                success_rate = sum(r > 0 for r in rewards_history[-100:]) / min(100, len(rewards_history))
                
                # Report to Ray Tune
                tune.report(
                    training_iteration=episode,
                    avg_reward=avg_reward,
                    success_rate=success_rate,
                    episode_reward=episode_reward
                )
    
    def optimize_with_asha(
        self,
        num_samples: int = 50,
        max_t: int = 500,
        cpus_per_trial: int = 1,
        gpus_per_trial: float = 0
    ):
        """
        Run optimization with ASHA scheduler.
        
        Args:
            num_samples: Number of trials
            max_t: Maximum training iterations
            cpus_per_trial: CPUs per trial
            gpus_per_trial: GPUs per trial (fractional allowed)
        """
        scheduler = self.get_asha_scheduler(max_t=max_t)
        
        reporter = CLIReporter(
            metric_columns=['avg_reward', 'success_rate', 'episode_reward', 'training_iteration'],
            max_report_frequency=30
        )
        
        analysis = tune.run(
            self.trainable_function,
            name=f'{self.experiment_name}_asha',
            config=self.get_search_space(),
            scheduler=scheduler,
            num_samples=num_samples,
            resources_per_trial={'cpu': cpus_per_trial, 'gpu': gpus_per_trial},
            local_dir=str(self.local_dir),
            progress_reporter=reporter,
            verbose=1
        )
        
        return analysis
    
    def optimize_with_pbt(
        self,
        num_samples: int = 8,
        perturbation_interval: int = 50,
        cpus_per_trial: int = 1,
        gpus_per_trial: float = 0
    ):
        """
        Run optimization with Population Based Training.
        
        Args:
            num_samples: Population size
            perturbation_interval: Interval for mutations
            cpus_per_trial: CPUs per trial
            gpus_per_trial: GPUs per trial
        """
        scheduler = self.get_pbt_scheduler(perturbation_interval=perturbation_interval)
        
        reporter = CLIReporter(
            metric_columns=['avg_reward', 'success_rate', 'training_iteration'],
            max_report_frequency=30
        )
        
        analysis = tune.run(
            self.trainable_function,
            name=f'{self.experiment_name}_pbt',
            config=self.get_search_space(),
            scheduler=scheduler,
            num_samples=num_samples,
            resources_per_trial={'cpu': cpus_per_trial, 'gpu': gpus_per_trial},
            local_dir=str(self.local_dir),
            progress_reporter=reporter,
            verbose=1
        )
        
        return analysis
    
    def optimize_with_bayesopt(
        self,
        num_samples: int = 50,
        max_concurrent: int = 4,
        cpus_per_trial: int = 1,
        gpus_per_trial: float = 0
    ):
        """
        Run optimization with Bayesian Optimization.
        
        Args:
            num_samples: Number of trials
            max_concurrent: Maximum concurrent trials
            cpus_per_trial: CPUs per trial
            gpus_per_trial: GPUs per trial
        """
        search_alg = self.get_bayesopt_search(max_concurrent=max_concurrent)
        
        reporter = CLIReporter(
            metric_columns=['avg_reward', 'success_rate', 'training_iteration'],
            max_report_frequency=30
        )
        
        analysis = tune.run(
            self.trainable_function,
            name=f'{self.experiment_name}_bayesopt',
            config=self.get_search_space(),
            search_alg=search_alg,
            num_samples=num_samples,
            resources_per_trial={'cpu': cpus_per_trial, 'gpu': gpus_per_trial},
            local_dir=str(self.local_dir),
            progress_reporter=reporter,
            verbose=1
        )
        
        return analysis
    
    @staticmethod
    def print_results(analysis):
        """Print optimization results."""
        print("\n" + "="*70)
        print("RAY TUNE OPTIMIZATION RESULTS")
        print("="*70)
        
        best_trial = analysis.get_best_trial('avg_reward', 'max', 'last')
        
        print(f"\nBest trial: {best_trial.trial_id}")
        print(f"Best avg_reward: {best_trial.last_result['avg_reward']:.4f}")
        print(f"Best success_rate: {best_trial.last_result['success_rate']:.2%}")
        
        print(f"\n{'='*70}")
        print("BEST CONFIGURATION")
        print("="*70)
        for key, value in best_trial.config.items():
            print(f"  {key}: {value}")
        
        print("="*70 + "\n")
    
    @staticmethod
    def save_best_config(analysis, output_path: Path):
        """
        Save best configuration to JSON.
        
        Args:
            analysis: Ray Tune analysis object
            output_path: Path to save config
        """
        import json
        
        best_trial = analysis.get_best_trial('avg_reward', 'max', 'last')
        best_config = best_trial.config
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(best_config, f, indent=2)
        
        print(f"✅ Best config saved to: {output_path}")


# Example usage
if __name__ == "__main__":
    import logging
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create optimizer
    optimizer = RayTuneOptimizer(
        experiment_name='robot_navigation_raytune',
        local_dir='./ray_results'
    )
    
    # Run optimization with ASHA (recommended for large search spaces)
    print("🚀 Starting optimization with ASHA scheduler...")
    analysis = optimizer.optimize_with_asha(
        num_samples=50,
        cpus_per_trial=1,
        gpus_per_trial=0.25  # Share GPU across 4 trials
    )
    
    # Print and save results
    optimizer.print_results(analysis)
    optimizer.save_best_config(
        analysis,
        Path('results/hyperparameter_optimization/raytune_best_config.json')
    )
    
    # Shutdown Ray
    ray.shutdown()
    
    print("\n✅ Optimization complete!")
    print("📊 View results in TensorBoard:")
    print(f"   tensorboard --logdir {optimizer.local_dir}")
