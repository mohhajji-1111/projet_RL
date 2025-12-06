"""
Hyperparameter Optimization using Optuna

Features:
- Parallel trial execution
- Pruning (early stopping of bad trials)
- Visualization dashboard
- Best configuration export
- Resume optimization
- Cross-validation support

Author: Advanced Training System
Date: 2025-12-06
"""

import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_slice,
    plot_contour
)
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Callable
import json
import logging
from datetime import datetime
import joblib


class OptunaOptimizer:
    """
    Hyperparameter optimization using Optuna.
    
    Features:
    - Flexible search space definition
    - Parallel trials
    - Pruning for efficiency
    - Visualization of results
    - Best config export
    """
    
    def __init__(
        self,
        study_name: str,
        storage: Optional[str] = None,
        direction: str = 'maximize',
        pruner: Optional[optuna.pruners.BasePruner] = None,
        sampler: Optional[optuna.samplers.BaseSampler] = None,
        load_if_exists: bool = True
    ):
        """
        Initialize Optuna optimizer.
        
        Args:
            study_name: Name of the study
            storage: Database URL for persistence (e.g., 'sqlite:///optuna.db')
            direction: 'maximize' or 'minimize'
            pruner: Pruner for early stopping
            sampler: Sampling algorithm
            load_if_exists: Load existing study if available
        """
        self.study_name = study_name
        self.storage = storage
        self.direction = direction
        
        # Default pruner: MedianPruner
        if pruner is None:
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=50,
                interval_steps=10
            )
        
        # Default sampler: TPE (Tree-structured Parzen Estimator)
        if sampler is None:
            sampler = optuna.samplers.TPESampler(
                n_startup_trials=10,
                multivariate=True
            )
        
        # Create or load study
        self.study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction=direction,
            pruner=pruner,
            sampler=sampler,
            load_if_exists=load_if_exists
        )
        
        self.logger = logging.getLogger(f'OptunaOptimizer_{study_name}')
        self.logger.setLevel(logging.INFO)
    
    def define_search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Define hyperparameter search space.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of hyperparameters
        """
        # Network architecture
        num_layers = trial.suggest_int('num_layers', 2, 4)
        hidden_dims = []
        for i in range(num_layers):
            dim = trial.suggest_categorical(f'hidden_dim_{i}', [64, 128, 256, 512])
            hidden_dims.append(dim)
        
        # Learning parameters
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256])
        
        # RL parameters
        gamma = trial.suggest_float('gamma', 0.95, 0.999)
        epsilon_decay = trial.suggest_float('epsilon_decay', 0.990, 0.999)
        buffer_size = trial.suggest_int('buffer_size', 10000, 100000, step=10000)
        target_update = trial.suggest_int('target_update', 100, 1000, step=100)
        
        # Optional: Dropout for regularization
        use_dropout = trial.suggest_categorical('use_dropout', [True, False])
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5) if use_dropout else 0.0
        
        # Optional: Learning rate schedule
        use_lr_schedule = trial.suggest_categorical('use_lr_schedule', [True, False])
        lr_decay_factor = trial.suggest_float('lr_decay_factor', 0.1, 0.9) if use_lr_schedule else 1.0
        lr_decay_steps = trial.suggest_int('lr_decay_steps', 100, 500, step=50) if use_lr_schedule else 1000
        
        config = {
            'hidden_dims': hidden_dims,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'gamma': gamma,
            'epsilon_decay': epsilon_decay,
            'buffer_size': buffer_size,
            'target_update': target_update,
            'use_dropout': use_dropout,
            'dropout_rate': dropout_rate,
            'use_lr_schedule': use_lr_schedule,
            'lr_decay_factor': lr_decay_factor,
            'lr_decay_steps': lr_decay_steps,
        }
        
        return config
    
    def objective_function(
        self,
        trial: optuna.Trial,
        train_fn: Callable,
        num_episodes: int = 500,
        eval_interval: int = 50
    ) -> float:
        """
        Objective function to optimize.
        
        Args:
            trial: Optuna trial
            train_fn: Training function that takes config and returns performance
            num_episodes: Number of episodes to train
            eval_interval: Interval for reporting intermediate values
            
        Returns:
            Performance metric (e.g., average reward)
        """
        # Get hyperparameters
        config = self.define_search_space(trial)
        
        # Train with these hyperparameters
        try:
            performance_history = train_fn(
                config=config,
                num_episodes=num_episodes,
                trial=trial,
                eval_interval=eval_interval
            )
            
            # Report final performance
            final_performance = performance_history[-1]
            
            return final_performance
            
        except optuna.exceptions.TrialPruned:
            raise
        except Exception as e:
            self.logger.error(f"Trial failed: {e}")
            return float('-inf') if self.direction == 'maximize' else float('inf')
    
    def optimize(
        self,
        train_fn: Callable,
        n_trials: int = 100,
        num_episodes: int = 500,
        eval_interval: int = 50,
        n_jobs: int = 1,
        timeout: Optional[int] = None,
        show_progress_bar: bool = True
    ):
        """
        Run optimization.
        
        Args:
            train_fn: Training function
            n_trials: Number of trials to run
            num_episodes: Episodes per trial
            eval_interval: Evaluation interval
            n_jobs: Number of parallel jobs (-1 for all CPUs)
            timeout: Timeout in seconds
            show_progress_bar: Show progress bar
        """
        self.logger.info(f"Starting optimization: {n_trials} trials, {n_jobs} parallel jobs")
        
        # Define objective wrapper
        def objective(trial):
            return self.objective_function(
                trial=trial,
                train_fn=train_fn,
                num_episodes=num_episodes,
                eval_interval=eval_interval
            )
        
        # Run optimization
        self.study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=n_jobs,
            timeout=timeout,
            show_progress_bar=show_progress_bar
        )
        
        self.logger.info("Optimization complete!")
        self.print_results()
    
    def print_results(self):
        """Print optimization results."""
        print("\n" + "="*70)
        print("OPTIMIZATION RESULTS")
        print("="*70)
        
        print(f"\nStudy: {self.study_name}")
        print(f"Direction: {self.direction}")
        print(f"Number of trials: {len(self.study.trials)}")
        print(f"Best trial: #{self.study.best_trial.number}")
        print(f"Best value: {self.study.best_value:.4f}")
        
        print(f"\n{'='*70}")
        print("BEST HYPERPARAMETERS")
        print("="*70)
        for key, value in self.study.best_params.items():
            print(f"  {key}: {value}")
        
        print("="*70 + "\n")
    
    def save_results(self, output_dir: Path):
        """
        Save optimization results.
        
        Args:
            output_dir: Directory to save results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save best config as JSON
        best_config_path = output_dir / f'{self.study_name}_best_config.json'
        with open(best_config_path, 'w') as f:
            json.dump(self.study.best_params, f, indent=2)
        
        self.logger.info(f"Best config saved to: {best_config_path}")
        
        # Save study
        study_path = output_dir / f'{self.study_name}_study.pkl'
        joblib.dump(self.study, study_path)
        
        self.logger.info(f"Study saved to: {study_path}")
        
        # Save all trials as CSV
        df = self.study.trials_dataframe()
        csv_path = output_dir / f'{self.study_name}_trials.csv'
        df.to_csv(csv_path, index=False)
        
        self.logger.info(f"Trials saved to: {csv_path}")
    
    def visualize(self, output_dir: Path):
        """
        Create and save visualizations.
        
        Args:
            output_dir: Directory to save plots
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Optimization history
            fig = plot_optimization_history(self.study)
            fig.write_html(str(output_dir / f'{self.study_name}_history.html'))
            
            # Parameter importances
            fig = plot_param_importances(self.study)
            fig.write_html(str(output_dir / f'{self.study_name}_importances.html'))
            
            # Parallel coordinate plot
            fig = plot_parallel_coordinate(self.study)
            fig.write_html(str(output_dir / f'{self.study_name}_parallel.html'))
            
            # Slice plot
            fig = plot_slice(self.study)
            fig.write_html(str(output_dir / f'{self.study_name}_slice.html'))
            
            # Contour plot (for 2D visualization)
            if len(self.study.best_params) >= 2:
                params = list(self.study.best_params.keys())[:2]
                fig = plot_contour(self.study, params=params)
                fig.write_html(str(output_dir / f'{self.study_name}_contour.html'))
            
            self.logger.info(f"Visualizations saved to: {output_dir}")
            
        except Exception as e:
            self.logger.warning(f"Could not create some visualizations: {e}")
    
    def get_best_config(self) -> Dict[str, Any]:
        """Get best hyperparameter configuration."""
        return self.study.best_params
    
    def get_best_value(self) -> float:
        """Get best performance value."""
        return self.study.best_value
    
    @staticmethod
    def load_study(study_name: str, storage: str) -> 'OptunaOptimizer':
        """
        Load existing study.
        
        Args:
            study_name: Name of study to load
            storage: Database URL
            
        Returns:
            OptunaOptimizer instance with loaded study
        """
        optimizer = OptunaOptimizer(
            study_name=study_name,
            storage=storage,
            load_if_exists=True
        )
        return optimizer


# Example training function that works with Optuna
def train_with_config(
    config: Dict[str, Any],
    num_episodes: int,
    trial: Optional[optuna.Trial] = None,
    eval_interval: int = 50
) -> float:
    """
    Example training function for Optuna.
    
    Args:
        config: Hyperparameter configuration
        num_episodes: Number of episodes to train
        trial: Optuna trial for pruning
        eval_interval: Evaluation interval
        
    Returns:
        Final performance metric
    """
    # This is a template - replace with actual training logic
    from src.agents.dqn_agent import DQNAgent
    from src.environment.robot_env import RobotNavigationEnv
    
    # Create environment and agent with config
    env = RobotNavigationEnv(size=(10, 10), num_obstacles=5, num_goals=3)
    
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        hidden_dims=config['hidden_dims'],
        learning_rate=config['learning_rate'],
        gamma=config['gamma'],
        buffer_size=config['buffer_size'],
        batch_size=config['batch_size']
    )
    
    # Training loop
    rewards = []
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            epsilon = max(0.01, 1.0 * (config['epsilon_decay'] ** episode))
            action = agent.select_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            
            agent.store_transition(state, action, reward, next_state, done)
            if agent.can_train():
                agent.train_step()
            
            episode_reward += reward
            state = next_state
        
        rewards.append(episode_reward)
        
        # Report intermediate value for pruning
        if trial is not None and episode % eval_interval == 0:
            intermediate_value = np.mean(rewards[-eval_interval:])
            trial.report(intermediate_value, episode)
            
            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
    
    # Return final performance (e.g., average of last 100 episodes)
    final_performance = np.mean(rewards[-100:])
    return final_performance


# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create optimizer
    optimizer = OptunaOptimizer(
        study_name='robot_navigation_hpo',
        storage='sqlite:///optuna_studies.db',
        direction='maximize'
    )
    
    # Run optimization
    optimizer.optimize(
        train_fn=train_with_config,
        n_trials=50,
        num_episodes=500,
        n_jobs=3,  # 3 parallel trials
        show_progress_bar=True
    )
    
    # Save results
    output_dir = Path('results/hyperparameter_optimization')
    optimizer.save_results(output_dir)
    optimizer.visualize(output_dir)
    
    print(f"\n✅ Best configuration saved to: {output_dir}")
    print(f"✅ Visualizations available in: {output_dir}")
