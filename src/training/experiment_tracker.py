"""
Experiment Tracking Integration - Unified Interface

Supports multiple tracking backends:
- Weights & Biases (WandB)
- TensorBoard
- MLflow

Features:
- Unified API for all backends
- Automatic metric aggregation
- Video/image logging
- System metrics tracking
- Model checkpointing
- Hyperparameter logging

Author: Advanced Training System
Date: 2025-12-06
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import numpy as np
import torch
from abc import ABC, abstractmethod
import logging


class ExperimentTracker(ABC):
    """Abstract base class for experiment trackers."""
    
    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log scalar metrics."""
        pass
    
    @abstractmethod
    def log_hyperparameters(self, params: Dict[str, Any]):
        """Log hyperparameters."""
        pass
    
    @abstractmethod
    def log_model(self, model: torch.nn.Module, name: str):
        """Log model architecture."""
        pass
    
    @abstractmethod
    def finish(self):
        """Finish tracking."""
        pass


class WandBTracker(ExperimentTracker):
    """Weights & Biases tracker."""
    
    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        config: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        entity: Optional[str] = None
    ):
        """
        Initialize WandB tracker.
        
        Args:
            project: Project name
            name: Run name
            config: Configuration dictionary
            tags: Tags for organization
            notes: Notes for this run
            entity: WandB entity (username or team)
        """
        try:
            import wandb
            self.wandb = wandb
            
            self.run = wandb.init(
                project=project,
                name=name,
                config=config,
                tags=tags,
                notes=notes,
                entity=entity
            )
            
            self.logger = logging.getLogger('WandBTracker')
            self.logger.info(f"WandB tracking initialized: {self.run.url}")
            
        except ImportError:
            raise ImportError("wandb not installed. Install with: pip install wandb")
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics to WandB."""
        self.wandb.log(metrics, step=step)
    
    def log_hyperparameters(self, params: Dict[str, Any]):
        """Log hyperparameters."""
        self.wandb.config.update(params)
    
    def log_model(self, model: torch.nn.Module, name: str):
        """Log model to WandB."""
        # Save model artifact
        artifact = self.wandb.Artifact(name, type='model')
        
        # Save model file
        model_path = Path(f'{name}.pt')
        torch.save(model.state_dict(), model_path)
        artifact.add_file(str(model_path))
        
        self.run.log_artifact(artifact)
        
        # Clean up
        model_path.unlink()
    
    def log_video(self, video_array: np.ndarray, name: str, fps: int = 30):
        """
        Log video to WandB.
        
        Args:
            video_array: Video array (T, H, W, C)
            name: Video name
            fps: Frames per second
        """
        self.wandb.log({name: self.wandb.Video(video_array, fps=fps, format='mp4')})
    
    def log_image(self, image: np.ndarray, name: str):
        """Log image to WandB."""
        self.wandb.log({name: self.wandb.Image(image)})
    
    def log_histogram(self, values: np.ndarray, name: str, step: int):
        """Log histogram."""
        self.wandb.log({name: self.wandb.Histogram(values)}, step=step)
    
    def watch_model(self, model: torch.nn.Module, log: str = 'gradients', log_freq: int = 100):
        """Watch model gradients and parameters."""
        self.wandb.watch(model, log=log, log_freq=log_freq)
    
    def finish(self):
        """Finish WandB run."""
        self.run.finish()
        self.logger.info("WandB tracking finished")


class TensorBoardTracker(ExperimentTracker):
    """TensorBoard tracker."""
    
    def __init__(
        self,
        log_dir: str,
        comment: str = ''
    ):
        """
        Initialize TensorBoard tracker.
        
        Args:
            log_dir: Directory for TensorBoard logs
            comment: Comment to append to run name
        """
        try:
            from torch.utils.tensorboard import SummaryWriter
            
            self.log_dir = Path(log_dir)
            self.log_dir.mkdir(parents=True, exist_ok=True)
            
            self.writer = SummaryWriter(log_dir=str(self.log_dir), comment=comment)
            
            self.logger = logging.getLogger('TensorBoardTracker')
            self.logger.info(f"TensorBoard logging to: {self.log_dir}")
            
        except ImportError:
            raise ImportError("tensorboard not installed. Install with: pip install tensorboard")
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log scalar metrics."""
        for name, value in metrics.items():
            self.writer.add_scalar(name, value, step)
    
    def log_hyperparameters(self, params: Dict[str, Any]):
        """Log hyperparameters as text."""
        hparam_str = '\n'.join([f'{k}: {v}' for k, v in params.items()])
        self.writer.add_text('hyperparameters', hparam_str, 0)
    
    def log_model(self, model: torch.nn.Module, name: str):
        """Log model graph to TensorBoard."""
        # Note: Requires dummy input
        self.logger.warning("TensorBoard model logging requires dummy input")
    
    def log_histogram(self, values: Union[np.ndarray, torch.Tensor], name: str, step: int):
        """Log histogram."""
        self.writer.add_histogram(name, values, step)
    
    def log_image(self, image: np.ndarray, name: str, step: int):
        """Log image."""
        self.writer.add_image(name, image, step, dataformats='HWC')
    
    def log_images(self, images: np.ndarray, name: str, step: int):
        """Log multiple images as grid."""
        self.writer.add_images(name, images, step, dataformats='NHWC')
    
    def log_figure(self, figure, name: str, step: int):
        """Log matplotlib figure."""
        self.writer.add_figure(name, figure, step)
    
    def log_text(self, text: str, name: str, step: int):
        """Log text."""
        self.writer.add_text(name, text, step)
    
    def log_graph(self, model: torch.nn.Module, input_to_model):
        """Log model graph."""
        self.writer.add_graph(model, input_to_model)
    
    def finish(self):
        """Close TensorBoard writer."""
        self.writer.close()
        self.logger.info("TensorBoard writer closed")


class MLflowTracker(ExperimentTracker):
    """MLflow tracker."""
    
    def __init__(
        self,
        experiment_name: str,
        tracking_uri: Optional[str] = None,
        run_name: Optional[str] = None
    ):
        """
        Initialize MLflow tracker.
        
        Args:
            experiment_name: Experiment name
            tracking_uri: MLflow tracking server URI
            run_name: Run name
        """
        try:
            import mlflow
            
            self.mlflow = mlflow
            
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            
            mlflow.set_experiment(experiment_name)
            self.run = mlflow.start_run(run_name=run_name)
            
            self.logger = logging.getLogger('MLflowTracker')
            self.logger.info(f"MLflow tracking initialized: {experiment_name}")
            
        except ImportError:
            raise ImportError("mlflow not installed. Install with: pip install mlflow")
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics."""
        for name, value in metrics.items():
            self.mlflow.log_metric(name, value, step=step)
    
    def log_hyperparameters(self, params: Dict[str, Any]):
        """Log hyperparameters."""
        self.mlflow.log_params(params)
    
    def log_model(self, model: torch.nn.Module, name: str):
        """Log PyTorch model."""
        self.mlflow.pytorch.log_model(model, name)
    
    def log_artifact(self, file_path: str):
        """Log artifact file."""
        self.mlflow.log_artifact(file_path)
    
    def set_tags(self, tags: Dict[str, str]):
        """Set tags."""
        self.mlflow.set_tags(tags)
    
    def finish(self):
        """End MLflow run."""
        self.mlflow.end_run()
        self.logger.info("MLflow run ended")


class UnifiedTracker:
    """
    Unified interface for multiple experiment trackers.
    
    Logs to all configured backends simultaneously.
    """
    
    def __init__(
        self,
        enable_wandb: bool = False,
        enable_tensorboard: bool = True,
        enable_mlflow: bool = False,
        wandb_config: Optional[Dict] = None,
        tensorboard_config: Optional[Dict] = None,
        mlflow_config: Optional[Dict] = None
    ):
        """
        Initialize unified tracker.
        
        Args:
            enable_wandb: Enable WandB tracking
            enable_tensorboard: Enable TensorBoard tracking
            enable_mlflow: Enable MLflow tracking
            wandb_config: WandB configuration
            tensorboard_config: TensorBoard configuration
            mlflow_config: MLflow configuration
        """
        self.trackers: List[ExperimentTracker] = []
        
        if enable_wandb and wandb_config:
            self.trackers.append(WandBTracker(**wandb_config))
        
        if enable_tensorboard and tensorboard_config:
            self.trackers.append(TensorBoardTracker(**tensorboard_config))
        
        if enable_mlflow and mlflow_config:
            self.trackers.append(MLflowTracker(**mlflow_config))
        
        self.logger = logging.getLogger('UnifiedTracker')
        self.logger.info(f"Initialized {len(self.trackers)} trackers")
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics to all trackers."""
        for tracker in self.trackers:
            tracker.log_metrics(metrics, step)
    
    def log_hyperparameters(self, params: Dict[str, Any]):
        """Log hyperparameters to all trackers."""
        for tracker in self.trackers:
            tracker.log_hyperparameters(params)
    
    def log_model(self, model: torch.nn.Module, name: str):
        """Log model to all trackers."""
        for tracker in self.trackers:
            try:
                tracker.log_model(model, name)
            except Exception as e:
                self.logger.warning(f"Could not log model to {type(tracker).__name__}: {e}")
    
    def finish(self):
        """Finish all trackers."""
        for tracker in self.trackers:
            tracker.finish()


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize unified tracker
    tracker = UnifiedTracker(
        enable_wandb=False,  # Set to True if you have WandB account
        enable_tensorboard=True,
        enable_mlflow=False,  # Set to True if you have MLflow server
        tensorboard_config={'log_dir': './runs/experiment_1'},
    )
    
    # Log hyperparameters
    tracker.log_hyperparameters({
        'learning_rate': 1e-3,
        'batch_size': 64,
        'gamma': 0.99
    })
    
    # Training loop
    for step in range(100):
        metrics = {
            'reward': np.random.randn(),
            'loss': np.random.rand(),
            'success_rate': np.random.rand()
        }
        tracker.log_metrics(metrics, step)
    
    # Finish tracking
    tracker.finish()
    
    print("\n✅ Experiment tracking complete!")
    print("View TensorBoard: tensorboard --logdir ./runs")
