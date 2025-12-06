"""
Curriculum Learning System for Progressive Training

Implements 4-stage curriculum:
Stage 1: Basic Navigation (Episodes 0-500)
Stage 2: Obstacle Avoidance (Episodes 500-1000)
Stage 3: Multi-Goal Planning (Episodes 1000-1500)
Stage 4: Full Challenge (Episodes 1500-2000)

Features:
- Automatic stage progression based on performance
- Configurable progression criteria
- Stage-specific environment configuration
- Progress tracking and visualization

Author: Advanced Training System
Date: 2025-12-06
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
import logging
from enum import Enum


class Stage(Enum):
    """Training stages."""
    BASIC_NAVIGATION = 1
    OBSTACLE_AVOIDANCE = 2
    MULTI_GOAL_PLANNING = 3
    FULL_CHALLENGE = 4


@dataclass
class StageConfig:
    """Configuration for a training stage."""
    name: str
    episode_range: Tuple[int, int]
    env_config: Dict
    epsilon_config: Dict
    progression_criteria: Dict
    description: str


@dataclass
class ProgressionCriteria:
    """Criteria for progressing to next stage."""
    min_success_rate: float
    evaluation_window: int
    min_avg_reward: float
    max_collision_rate: float


class CurriculumLearningSystem:
    """
    Manages curriculum learning with progressive difficulty.
    
    Features:
    - 4 predefined stages
    - Automatic progression based on performance
    - Configurable criteria
    - Progress tracking
    """
    
    def __init__(
        self,
        base_env_config: Optional[Dict] = None,
        custom_stages: Optional[List[StageConfig]] = None,
        enable_auto_progression: bool = True
    ):
        """
        Initialize curriculum learning system.
        
        Args:
            base_env_config: Base environment configuration
            custom_stages: Custom stage definitions (overrides default)
            enable_auto_progression: Enable automatic stage progression
        """
        self.base_env_config = base_env_config or {}
        self.enable_auto_progression = enable_auto_progression
        
        # Define stages
        if custom_stages:
            self.stages = custom_stages
        else:
            self.stages = self._create_default_stages()
        
        # Current state
        self.current_stage = Stage.BASIC_NAVIGATION
        self.current_episode = 0
        self.stage_history = []
        
        # Performance tracking
        self.episode_rewards = []
        self.episode_successes = []
        self.episode_collisions = []
        
        self.logger = logging.getLogger('CurriculumLearning')
        self.logger.setLevel(logging.INFO)
        
        self.logger.info(f"Curriculum Learning initialized with {len(self.stages)} stages")
    
    def _create_default_stages(self) -> List[StageConfig]:
        """Create default 4-stage curriculum."""
        stages = [
            StageConfig(
                name="Stage 1: Basic Navigation",
                episode_range=(0, 500),
                env_config={
                    'size': (10, 10),
                    'num_obstacles': 3,
                    'num_goals': 1,
                    'dynamic_obstacles': False,
                    'max_steps': 200,
                },
                epsilon_config={
                    'start': 1.0,
                    'end': 0.3,
                    'decay': 0.995
                },
                progression_criteria={
                    'min_success_rate': 0.70,
                    'evaluation_window': 50,
                    'min_avg_reward': 5.0,
                    'max_collision_rate': 0.30
                },
                description="Learn basic movement and simple goal reaching"
            ),
            StageConfig(
                name="Stage 2: Obstacle Avoidance",
                episode_range=(500, 1000),
                env_config={
                    'size': (10, 10),
                    'num_obstacles': 5,
                    'num_goals': 1,
                    'dynamic_obstacles': True,
                    'max_steps': 200,
                },
                epsilon_config={
                    'start': 0.5,
                    'end': 0.1,
                    'decay': 0.995
                },
                progression_criteria={
                    'min_success_rate': 0.70,
                    'evaluation_window': 50,
                    'min_avg_reward': 8.0,
                    'max_collision_rate': 0.20
                },
                description="Master collision avoidance with dynamic obstacles"
            ),
            StageConfig(
                name="Stage 3: Multi-Goal Planning",
                episode_range=(1000, 1500),
                env_config={
                    'size': (10, 10),
                    'num_obstacles': 5,
                    'num_goals': 3,
                    'dynamic_obstacles': True,
                    'max_steps': 300,
                },
                epsilon_config={
                    'start': 0.3,
                    'end': 0.05,
                    'decay': 0.997
                },
                progression_criteria={
                    'min_success_rate': 0.70,
                    'evaluation_window': 50,
                    'min_avg_reward': 12.0,
                    'max_collision_rate': 0.20
                },
                description="Learn sequential goal reaching and path planning"
            ),
            StageConfig(
                name="Stage 4: Full Challenge",
                episode_range=(1500, 2000),
                env_config={
                    'size': (10, 10),
                    'num_obstacles': 7,
                    'num_goals': 4,
                    'dynamic_obstacles': True,
                    'max_steps': 400,
                },
                epsilon_config={
                    'start': 0.2,
                    'end': 0.01,
                    'decay': 0.998
                },
                progression_criteria={
                    'min_success_rate': 0.60,  # Lower threshold for final stage
                    'evaluation_window': 50,
                    'min_avg_reward': 15.0,
                    'max_collision_rate': 0.25
                },
                description="Final challenge with maximum difficulty"
            )
        ]
        
        return stages
    
    def get_current_stage_config(self) -> StageConfig:
        """Get configuration for current stage."""
        stage_idx = self.current_stage.value - 1
        return self.stages[stage_idx]
    
    def get_current_env_config(self) -> Dict:
        """Get environment configuration for current stage."""
        stage_config = self.get_current_stage_config()
        config = {**self.base_env_config, **stage_config.env_config}
        return config
    
    def get_current_epsilon(self) -> float:
        """Calculate epsilon for current episode in current stage."""
        stage_config = self.get_current_stage_config()
        epsilon_config = stage_config.epsilon_config
        
        # Episodes within current stage
        stage_start = stage_config.episode_range[0]
        episodes_in_stage = self.current_episode - stage_start
        
        epsilon = max(
            epsilon_config['end'],
            epsilon_config['start'] * (epsilon_config['decay'] ** episodes_in_stage)
        )
        
        return epsilon
    
    def update(
        self,
        episode: int,
        reward: float,
        success: bool,
        collisions: int
    ) -> bool:
        """
        Update curriculum state and check for progression.
        
        Args:
            episode: Current episode number
            reward: Episode reward
            success: Whether episode was successful
            collisions: Number of collisions
            
        Returns:
            True if stage progressed, False otherwise
        """
        self.current_episode = episode
        self.episode_rewards.append(reward)
        self.episode_successes.append(success)
        self.episode_collisions.append(collisions)
        
        # Check if should progress to next stage
        if self.enable_auto_progression:
            if self._should_progress():
                return self._progress_to_next_stage()
        
        return False
    
    def _should_progress(self) -> bool:
        """Check if performance criteria met for progression."""
        # Don't progress if already in final stage
        if self.current_stage == Stage.FULL_CHALLENGE:
            return False
        
        stage_config = self.get_current_stage_config()
        criteria = stage_config.progression_criteria
        window = criteria['evaluation_window']
        
        # Need enough episodes for evaluation
        if len(self.episode_rewards) < window:
            return False
        
        # Get recent performance
        recent_rewards = self.episode_rewards[-window:]
        recent_successes = self.episode_successes[-window:]
        recent_collisions = self.episode_collisions[-window:]
        
        # Calculate metrics
        avg_reward = np.mean(recent_rewards)
        success_rate = sum(recent_successes) / len(recent_successes)
        collision_rate = sum(recent_collisions) / len(recent_collisions)
        
        # Check all criteria
        meets_success_rate = success_rate >= criteria['min_success_rate']
        meets_avg_reward = avg_reward >= criteria['min_avg_reward']
        meets_collision_rate = collision_rate <= criteria['max_collision_rate']
        
        if meets_success_rate and meets_avg_reward and meets_collision_rate:
            self.logger.info(
                f"✅ Progression criteria met: "
                f"Success={success_rate:.2%} (>={criteria['min_success_rate']:.2%}), "
                f"Reward={avg_reward:.2f} (>={criteria['min_avg_reward']:.2f}), "
                f"Collisions={collision_rate:.2f} (<={criteria['max_collision_rate']:.2f})"
            )
            return True
        
        return False
    
    def _progress_to_next_stage(self) -> bool:
        """Progress to next stage."""
        if self.current_stage == Stage.FULL_CHALLENGE:
            return False
        
        # Move to next stage
        next_stage = Stage(self.current_stage.value + 1)
        old_stage = self.current_stage
        self.current_stage = next_stage
        
        # Record transition
        self.stage_history.append({
            'from_stage': old_stage.name,
            'to_stage': next_stage.name,
            'episode': self.current_episode,
            'performance': {
                'avg_reward': np.mean(self.episode_rewards[-50:]),
                'success_rate': sum(self.episode_successes[-50:]) / 50,
                'collision_rate': sum(self.episode_collisions[-50:]) / 50
            }
        })
        
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"🎓 STAGE PROGRESSION: {old_stage.name} → {next_stage.name}")
        self.logger.info(f"{'='*70}\n")
        
        # Reset performance tracking for new stage
        self.episode_rewards = []
        self.episode_successes = []
        self.episode_collisions = []
        
        return True
    
    def force_stage(self, stage: Stage):
        """
        Manually set current stage.
        
        Args:
            stage: Stage to set
        """
        self.current_stage = stage
        self.logger.info(f"Stage manually set to: {stage.name}")
    
    def get_progress_summary(self) -> Dict:
        """Get current progress summary."""
        stage_config = self.get_current_stage_config()
        
        # Calculate recent performance
        window = 50
        recent_data = min(len(self.episode_rewards), window)
        
        if recent_data > 0:
            avg_reward = np.mean(self.episode_rewards[-recent_data:])
            success_rate = sum(self.episode_successes[-recent_data:]) / recent_data
            collision_rate = sum(self.episode_collisions[-recent_data:]) / recent_data
        else:
            avg_reward = 0.0
            success_rate = 0.0
            collision_rate = 0.0
        
        summary = {
            'current_stage': self.current_stage.name,
            'current_episode': self.current_episode,
            'stage_config': {
                'name': stage_config.name,
                'episode_range': stage_config.episode_range,
                'description': stage_config.description
            },
            'recent_performance': {
                'avg_reward': avg_reward,
                'success_rate': success_rate,
                'collision_rate': collision_rate,
                'evaluation_window': recent_data
            },
            'progression_criteria': stage_config.progression_criteria,
            'stage_history': self.stage_history
        }
        
        return summary
    
    def print_progress(self):
        """Print current progress."""
        summary = self.get_progress_summary()
        
        print(f"\n{'='*70}")
        print(f"CURRICULUM LEARNING PROGRESS")
        print(f"{'='*70}")
        print(f"\nCurrent Stage: {summary['current_stage']}")
        print(f"Episode: {summary['current_episode']}")
        print(f"Description: {summary['stage_config']['description']}")
        
        print(f"\nRecent Performance (last {summary['recent_performance']['evaluation_window']} episodes):")
        print(f"  Average Reward: {summary['recent_performance']['avg_reward']:.2f}")
        print(f"  Success Rate: {summary['recent_performance']['success_rate']:.2%}")
        print(f"  Collision Rate: {summary['recent_performance']['collision_rate']:.2f}")
        
        print(f"\nProgression Criteria:")
        criteria = summary['progression_criteria']
        print(f"  Min Success Rate: {criteria['min_success_rate']:.2%}")
        print(f"  Min Avg Reward: {criteria['min_avg_reward']:.2f}")
        print(f"  Max Collision Rate: {criteria['max_collision_rate']:.2f}")
        
        if summary['stage_history']:
            print(f"\nStage History:")
            for i, transition in enumerate(summary['stage_history'], 1):
                print(f"  {i}. Episode {transition['episode']}: {transition['from_stage']} → {transition['to_stage']}")
        
        print(f"{'='*70}\n")
    
    def save_progress(self, output_path: Path):
        """
        Save curriculum progress to JSON.
        
        Args:
            output_path: Path to save progress
        """
        summary = self.get_progress_summary()
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Progress saved to: {output_path}")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create curriculum system
    curriculum = CurriculumLearningSystem()
    
    # Simulate training
    print("🎓 Starting curriculum learning simulation...\n")
    
    for episode in range(2000):
        # Get current environment config
        env_config = curriculum.get_current_env_config()
        epsilon = curriculum.get_current_epsilon()
        
        # Simulate episode (replace with actual training)
        # Performance improves over time
        base_success = min(0.8, episode / 2000)
        reward = np.random.normal(base_success * 20, 5)
        success = np.random.random() < base_success
        collisions = np.random.randint(0, 3)
        
        # Update curriculum
        progressed = curriculum.update(episode, reward, success, collisions)
        
        if progressed:
            curriculum.print_progress()
        
        # Print progress every 100 episodes
        if episode % 100 == 0:
            print(f"Episode {episode}: Stage={curriculum.current_stage.name}, "
                  f"Epsilon={epsilon:.3f}, Reward={reward:.2f}")
    
    # Final summary
    curriculum.print_progress()
    curriculum.save_progress(Path('results/curriculum_progress.json'))
