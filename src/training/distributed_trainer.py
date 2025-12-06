"""
Multi-GPU Distributed Training System

Supports:
- DataParallel (simple multi-GPU)
- DistributedDataParallel (recommended for multi-GPU)
- Mixed precision training (AMP)
- Gradient accumulation
- Automatic batch size finding
- Multi-node training (future-ready)

Author: Advanced Training System
Date: 2025-12-06
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DataParallel as DP
from torch.cuda.amp import GradScaler, autocast
from typing import Optional, Dict, Any, List, Tuple
import logging
from pathlib import Path
import json
import time
import psutil
import GPUtil


class DistributedTrainer:
    """
    Wrapper for distributed training across multiple GPUs.
    
    Features:
    - Automatic device detection
    - Mixed precision training (AMP)
    - Gradient accumulation
    - Synchronized batch normalization
    - Distributed checkpointing
    - Multi-GPU metrics aggregation
    """
    
    def __init__(
        self,
        model: nn.Module,
        use_amp: bool = True,
        gradient_accumulation_steps: int = 1,
        find_unused_parameters: bool = False,
        backend: str = 'nccl',
        sync_bn: bool = True
    ):
        """
        Initialize distributed trainer.
        
        Args:
            model: PyTorch model to train
            use_amp: Enable mixed precision training
            gradient_accumulation_steps: Number of steps to accumulate gradients
            find_unused_parameters: For DDP, set True if not all parameters are used
            backend: 'nccl' for GPU, 'gloo' for CPU
            sync_bn: Synchronize batch normalization across GPUs
        """
        self.model = model
        self.use_amp = use_amp
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.find_unused_parameters = find_unused_parameters
        self.backend = backend
        self.sync_bn = sync_bn
        
        # Device setup
        self.device = self._setup_device()
        self.world_size = self._get_world_size()
        self.rank = self._get_rank()
        self.local_rank = self._get_local_rank()
        self.is_distributed = self.world_size > 1
        
        # Mixed precision scaler
        self.scaler = GradScaler() if use_amp and torch.cuda.is_available() else None
        
        # Logging
        self.logger = self._setup_logger()
        
        # Move model to device and wrap for distributed training
        self._setup_model()
        
        self.logger.info(f"Distributed Trainer initialized:")
        self.logger.info(f"  - World size: {self.world_size}")
        self.logger.info(f"  - Rank: {self.rank}")
        self.logger.info(f"  - Local rank: {self.local_rank}")
        self.logger.info(f"  - Device: {self.device}")
        self.logger.info(f"  - Mixed precision: {self.use_amp}")
        self.logger.info(f"  - Gradient accumulation: {self.gradient_accumulation_steps}")
    
    def _setup_device(self) -> torch.device:
        """Setup and return the appropriate device."""
        if torch.cuda.is_available():
            # Use local rank if in distributed mode
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            device = torch.device(f'cuda:{local_rank}')
            torch.cuda.set_device(device)
            return device
        else:
            return torch.device('cpu')
    
    def _get_world_size(self) -> int:
        """Get total number of processes."""
        if 'WORLD_SIZE' in os.environ:
            return int(os.environ['WORLD_SIZE'])
        return 1
    
    def _get_rank(self) -> int:
        """Get global rank of current process."""
        if 'RANK' in os.environ:
            return int(os.environ['RANK'])
        return 0
    
    def _get_local_rank(self) -> int:
        """Get local rank (GPU id on current node)."""
        if 'LOCAL_RANK' in os.environ:
            return int(os.environ['LOCAL_RANK'])
        return 0
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger (only main process logs by default)."""
        logger = logging.getLogger(f'DistributedTrainer_Rank{self.rank}')
        logger.setLevel(logging.INFO if self.rank == 0 else logging.WARNING)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f'[Rank {self.rank}] %(asctime)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _setup_model(self):
        """Setup model for distributed training."""
        # Convert BatchNorm to SyncBatchNorm for distributed training
        if self.is_distributed and self.sync_bn and torch.cuda.is_available():
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.logger.info("Converted BatchNorm to SyncBatchNorm")
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Wrap model for distributed training
        if self.is_distributed:
            if self.backend == 'nccl' and torch.cuda.is_available():
                # Use DistributedDataParallel (recommended)
                self.model = DDP(
                    self.model,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    find_unused_parameters=self.find_unused_parameters
                )
                self.logger.info("Model wrapped with DistributedDataParallel")
            else:
                # Fallback to DataParallel
                self.model = DP(self.model)
                self.logger.info("Model wrapped with DataParallel")
        else:
            self.logger.info("Single device training (no wrapping)")
    
    def training_step(
        self,
        batch: Any,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        step: int
    ) -> Dict[str, float]:
        """
        Perform one training step with gradient accumulation and mixed precision.
        
        Args:
            batch: Input batch
            optimizer: Optimizer
            criterion: Loss function
            step: Current training step
            
        Returns:
            Dictionary with loss and other metrics
        """
        # Zero gradients at the start of accumulation
        if step % self.gradient_accumulation_steps == 0:
            optimizer.zero_grad()
        
        # Mixed precision forward pass
        with autocast(enabled=self.use_amp and torch.cuda.is_available()):
            output = self.model(batch)
            loss = criterion(output, batch)
            
            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps
        
        # Backward pass with gradient scaling
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Optimizer step (only after accumulating gradients)
        if (step + 1) % self.gradient_accumulation_steps == 0:
            if self.scaler is not None:
                # Unscale gradients and clip
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Optimizer step
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
        
        # Return metrics (unscaled loss)
        return {
            'loss': loss.item() * self.gradient_accumulation_steps
        }
    
    def reduce_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Aggregate metrics across all processes.
        
        Args:
            metrics: Dictionary of metrics from current process
            
        Returns:
            Averaged metrics across all processes
        """
        if not self.is_distributed:
            return metrics
        
        reduced_metrics = {}
        for key, value in metrics.items():
            tensor = torch.tensor(value, device=self.device)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            reduced_metrics[key] = tensor.item() / self.world_size
        
        return reduced_metrics
    
    def save_checkpoint(
        self,
        path: Path,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        extra_state: Optional[Dict] = None
    ):
        """
        Save checkpoint (only main process saves).
        
        Args:
            path: Path to save checkpoint
            epoch: Current epoch
            optimizer: Optimizer state
            scheduler: Learning rate scheduler state
            extra_state: Additional state to save
        """
        if self.rank != 0:
            return
        
        # Get model state dict (unwrap DDP/DP)
        if isinstance(self.model, (DDP, DP)):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        if extra_state is not None:
            checkpoint.update(extra_state)
        
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(
        self,
        path: Path,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        map_location: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load checkpoint.
        
        Args:
            path: Path to checkpoint
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
            map_location: Device to map checkpoint to
            
        Returns:
            Checkpoint dictionary
        """
        if map_location is None:
            map_location = str(self.device)
        
        checkpoint = torch.load(path, map_location=map_location)
        
        # Load model state
        if isinstance(self.model, (DDP, DP)):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load scaler state
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.logger.info(f"Checkpoint loaded from {path}")
        return checkpoint
    
    def barrier(self):
        """Synchronize all processes."""
        if self.is_distributed:
            dist.barrier()
    
    def is_main_process(self) -> bool:
        """Check if this is the main process."""
        return self.rank == 0
    
    @staticmethod
    def cleanup():
        """Cleanup distributed training."""
        if dist.is_initialized():
            dist.destroy_process_group()


class DeviceManager:
    """
    Utility for managing devices and finding optimal batch size.
    """
    
    @staticmethod
    def get_device_info() -> Dict[str, Any]:
        """Get information about available devices."""
        info = {
            'cpu_count': psutil.cpu_count(),
            'cpu_percent': psutil.cpu_percent(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'memory_available_gb': psutil.virtual_memory().available / (1024**3),
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': 0,
            'gpus': []
        }
        
        if torch.cuda.is_available():
            info['cuda_device_count'] = torch.cuda.device_count()
            info['cuda_version'] = torch.version.cuda
            
            try:
                gpus = GPUtil.getGPUs()
                for gpu in gpus:
                    info['gpus'].append({
                        'id': gpu.id,
                        'name': gpu.name,
                        'memory_total_mb': gpu.memoryTotal,
                        'memory_used_mb': gpu.memoryUsed,
                        'memory_free_mb': gpu.memoryFree,
                        'gpu_util_percent': gpu.load * 100,
                        'temperature_c': gpu.temperature
                    })
            except Exception as e:
                logging.warning(f"Could not get GPU info: {e}")
        
        return info
    
    @staticmethod
    def print_device_info():
        """Print device information."""
        info = DeviceManager.get_device_info()
        
        print("\n" + "="*60)
        print("DEVICE INFORMATION")
        print("="*60)
        
        print(f"\nCPU:")
        print(f"  Cores: {info['cpu_count']}")
        print(f"  Usage: {info['cpu_percent']}%")
        print(f"  Memory: {info['memory_available_gb']:.1f}GB / {info['memory_total_gb']:.1f}GB")
        
        print(f"\nCUDA:")
        print(f"  Available: {info['cuda_available']}")
        
        if info['cuda_available']:
            print(f"  Version: {info.get('cuda_version', 'N/A')}")
            print(f"  Device count: {info['cuda_device_count']}")
            
            for gpu in info['gpus']:
                print(f"\n  GPU {gpu['id']}: {gpu['name']}")
                print(f"    Memory: {gpu['memory_used_mb']:.0f}MB / {gpu['memory_total_mb']:.0f}MB")
                print(f"    Utilization: {gpu['gpu_util_percent']:.1f}%")
                print(f"    Temperature: {gpu['temperature_c']}°C")
        
        print("="*60 + "\n")
    
    @staticmethod
    def find_optimal_batch_size(
        model: nn.Module,
        input_shape: Tuple[int, ...],
        device: torch.device,
        min_batch_size: int = 1,
        max_batch_size: int = 2048,
        criterion: Optional[nn.Module] = None
    ) -> int:
        """
        Find the largest batch size that fits in memory.
        
        Args:
            model: Model to test
            input_shape: Shape of a single input (without batch dimension)
            device: Device to test on
            min_batch_size: Minimum batch size to try
            max_batch_size: Maximum batch size to try
            criterion: Loss function (optional)
            
        Returns:
            Optimal batch size
        """
        model = model.to(device)
        model.train()
        
        if criterion is not None:
            criterion = criterion.to(device)
        
        def try_batch_size(batch_size: int) -> bool:
            """Try a specific batch size."""
            try:
                # Clear cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Create dummy input
                dummy_input = torch.randn(batch_size, *input_shape, device=device)
                
                # Forward pass
                output = model(dummy_input)
                
                # Backward pass if criterion provided
                if criterion is not None:
                    dummy_target = torch.randn_like(output)
                    loss = criterion(output, dummy_target)
                    loss.backward()
                
                # Clear memory
                del dummy_input, output
                if criterion is not None:
                    del loss
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                return True
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    return False
                raise
        
        # Binary search for optimal batch size
        low, high = min_batch_size, max_batch_size
        optimal = min_batch_size
        
        while low <= high:
            mid = (low + high) // 2
            
            if try_batch_size(mid):
                optimal = mid
                low = mid + 1
            else:
                high = mid - 1
        
        # Use 80% of maximum to be safe
        optimal = int(optimal * 0.8)
        optimal = max(optimal, min_batch_size)
        
        return optimal


def launch_distributed(
    fn,
    world_size: int,
    backend: str = 'nccl',
    *args,
    **kwargs
):
    """
    Launch distributed training across multiple processes.
    
    Args:
        fn: Function to run in each process
        world_size: Number of processes
        backend: 'nccl' or 'gloo'
        *args: Arguments to pass to fn
        **kwargs: Keyword arguments to pass to fn
    """
    if world_size == 1:
        # Single process
        fn(0, world_size, *args, **kwargs)
    else:
        # Multi-process
        mp.spawn(
            fn,
            args=(world_size, *args),
            nprocs=world_size,
            join=True
        )


def setup_distributed(rank: int, world_size: int, backend: str = 'nccl'):
    """
    Setup distributed process group.
    
    Args:
        rank: Rank of current process
        world_size: Total number of processes
        backend: 'nccl' or 'gloo'
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    dist.init_process_group(backend, rank=rank, world_size=world_size)


# Example usage
if __name__ == "__main__":
    # Print device information
    DeviceManager.print_device_info()
    
    # Example: Find optimal batch size
    from torch import nn
    
    model = nn.Sequential(
        nn.Linear(10, 128),
        nn.ReLU(),
        nn.Linear(128, 1)
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if torch.cuda.is_available():
        optimal_bs = DeviceManager.find_optimal_batch_size(
            model=model,
            input_shape=(10,),
            device=device,
            criterion=nn.MSELoss()
        )
        print(f"Optimal batch size: {optimal_bs}")
