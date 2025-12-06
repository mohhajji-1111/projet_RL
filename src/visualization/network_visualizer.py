"""
Neural Network Visualization Tools
Visualize network architecture, weights, activations, and gradients
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import networkx as nx
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from collections import defaultdict


class NetworkVisualizer:
    """
    Visualize neural network architecture, weights, activations, and gradients.
    Publication-quality visualizations for Q-networks.
    """
    
    def __init__(self, model: nn.Module, dpi: int = 300):
        """
        Initialize network visualizer.
        
        Args:
            model: PyTorch neural network model
            dpi: Resolution for saved figures
        """
        self.model = model
        self.dpi = dpi
        self.activations = {}
        self.gradients = {}
        
        # Color scheme
        self.colors = {
            'input': '#2E86AB',
            'hidden': '#F18F01',
            'output': '#06A77D',
            'positive': '#00CC00',
            'negative': '#CC0000',
            'neutral': '#888888'
        }
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks to capture activations and gradients."""
        
        def forward_hook(name):
            def hook(module, input, output):
                self.activations[name] = output.detach().cpu()
            return hook
        
        def backward_hook(name):
            def hook(module, grad_input, grad_output):
                self.gradients[name] = grad_output[0].detach().cpu()
            return hook
        
        # Register hooks for all layers
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                module.register_forward_hook(forward_hook(name))
                module.register_full_backward_hook(backward_hook(name))
    
    def visualize_architecture(self, save_path: str = None):
        """
        Visualize the network architecture as a diagram.
        
        Args:
            save_path: Path to save the figure
        """
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.axis('off')
        
        # Extract layer information
        layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                layers.append({
                    'name': name,
                    'type': 'Linear',
                    'in_features': module.in_features,
                    'out_features': module.out_features
                })
            elif isinstance(module, nn.Conv2d):
                layers.append({
                    'name': name,
                    'type': 'Conv2d',
                    'in_channels': module.in_channels,
                    'out_channels': module.out_channels,
                    'kernel_size': module.kernel_size
                })
        
        if not layers:
            print("No visualizable layers found")
            return
        
        # Layout parameters
        layer_spacing = 2.5
        max_neurons = 12  # Max neurons to display per layer
        neuron_radius = 0.2
        
        # Draw layers
        for i, layer in enumerate(layers):
            x = i * layer_spacing
            
            if layer['type'] == 'Linear':
                n_units = min(layer['out_features'], max_neurons)
                color = self.colors['hidden'] if i < len(layers)-1 else self.colors['output']
                
                # Draw neurons
                for j in range(n_units):
                    y = (j - n_units/2) * 0.5
                    circle = plt.Circle((x, y), neuron_radius, color=color, alpha=0.7, zorder=2)
                    ax.add_patch(circle)
                
                # Add ellipsis if truncated
                if layer['out_features'] > max_neurons:
                    y = (max_neurons/2) * 0.5 + 0.5
                    ax.text(x, y, '...', ha='center', va='center', fontsize=16)
                
                # Layer label
                ax.text(
                    x, -n_units/2*0.5 - 0.8,
                    f"{layer['name']}\n{layer['out_features']} units",
                    ha='center', va='top',
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
                )
            
            # Draw connections to previous layer
            if i > 0:
                prev_layer = layers[i-1]
                prev_x = (i-1) * layer_spacing
                
                prev_n = min(prev_layer.get('out_features', 10), max_neurons)
                curr_n = min(layer.get('out_features', 10), max_neurons)
                
                # Draw sample connections (not all to avoid clutter)
                for j in range(min(3, curr_n)):
                    for k in range(min(3, prev_n)):
                        y1 = (k - prev_n/2) * 0.5
                        y2 = (j - curr_n/2) * 0.5
                        
                        arrow = FancyArrowPatch(
                            (prev_x + neuron_radius, y1),
                            (x - neuron_radius, y2),
                            arrowstyle='-',
                            color='gray',
                            alpha=0.3,
                            linewidth=0.5,
                            zorder=1
                        )
                        ax.add_patch(arrow)
        
        # Add input indicator
        ax.text(
            -0.5, 0,
            'Input\nState',
            ha='right', va='center',
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=self.colors['input'], alpha=0.7)
        )
        
        # Add output indicator
        last_x = (len(layers)-1) * layer_spacing
        ax.text(
            last_x + 0.5, 0,
            'Output\nQ-Values',
            ha='left', va='center',
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=self.colors['output'], alpha=0.7)
        )
        
        # Set limits
        ax.set_xlim(-1.5, (len(layers)-1)*layer_spacing + 1.5)
        ax.set_ylim(-4, 4)
        ax.set_aspect('equal')
        
        plt.title('Neural Network Architecture', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
            print(f"✓ Saved architecture diagram: {save_path}")
        
        return fig
    
    def visualize_weight_distributions(self, save_path: str = None):
        """
        Visualize weight distributions for each layer.
        
        Args:
            save_path: Path to save the figure
        """
        # Collect weights from all layers
        weights = {}
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                weights[name] = param.detach().cpu().numpy().flatten()
        
        if not weights:
            print("No weights found")
            return
        
        # Create subplots
        n_layers = len(weights)
        fig, axes = plt.subplots(n_layers, 2, figsize=(14, 4*n_layers))
        
        if n_layers == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (name, w) in enumerate(weights.items()):
            # Histogram
            ax1 = axes[idx, 0]
            ax1.hist(w, bins=50, color=self.colors['hidden'], alpha=0.7, edgecolor='black')
            ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
            ax1.axvline(np.mean(w), color='green', linestyle='--', linewidth=2, label=f'Mean: {np.mean(w):.4f}')
            ax1.set_xlabel('Weight Value', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
            ax1.set_title(f'{name} - Distribution', fontsize=12, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Statistics
            ax2 = axes[idx, 1]
            ax2.axis('off')
            
            stats_text = f"""
            Layer: {name}
            
            Statistics:
            • Mean: {np.mean(w):.6f}
            • Std: {np.std(w):.6f}
            • Min: {np.min(w):.6f}
            • Max: {np.max(w):.6f}
            • L1 Norm: {np.abs(w).sum():.6f}
            • L2 Norm: {np.sqrt((w**2).sum()):.6f}
            
            Percentiles:
            • 25%: {np.percentile(w, 25):.6f}
            • 50%: {np.percentile(w, 50):.6f}
            • 75%: {np.percentile(w, 75):.6f}
            """
            
            ax2.text(
                0.1, 0.5, stats_text,
                fontsize=10, verticalalignment='center',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ Saved weight distributions: {save_path}")
        
        return fig
    
    def visualize_activations(
        self,
        input_sample: torch.Tensor,
        save_path: str = None
    ):
        """
        Visualize activation patterns for a given input.
        
        Args:
            input_sample: Input tensor to pass through network
            save_path: Path to save the figure
        """
        # Forward pass to populate activations
        self.model.eval()
        with torch.no_grad():
            _ = self.model(input_sample)
        
        if not self.activations:
            print("No activations captured")
            return
        
        # Create subplots
        n_layers = len(self.activations)
        fig, axes = plt.subplots(n_layers, 1, figsize=(12, 4*n_layers))
        
        if n_layers == 1:
            axes = [axes]
        
        for idx, (name, activation) in enumerate(self.activations.items()):
            ax = axes[idx]
            
            # Get activation values
            act = activation.numpy().flatten()
            
            # Create bar plot
            x = np.arange(len(act))
            colors = [self.colors['positive'] if a > 0 else self.colors['negative'] for a in act]
            
            ax.bar(x, act, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
            ax.axhline(0, color='black', linewidth=1)
            
            ax.set_xlabel('Neuron Index', fontsize=12, fontweight='bold')
            ax.set_ylabel('Activation Value', fontsize=12, fontweight='bold')
            ax.set_title(f'{name} - Activation Pattern', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add statistics
            stats_text = f'Mean: {np.mean(act):.4f} | Std: {np.std(act):.4f} | Active: {(act > 0).sum()}/{len(act)}'
            ax.text(
                0.5, 0.95, stats_text,
                transform=ax.transAxes,
                ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ Saved activation patterns: {save_path}")
        
        return fig
    
    def visualize_gradient_flow(self, save_path: str = None):
        """
        Visualize gradient flow to detect vanishing/exploding gradients.
        
        Args:
            save_path: Path to save the figure
        """
        if not self.gradients:
            print("No gradients captured. Perform a backward pass first.")
            return
        
        # Collect gradient statistics
        layer_names = []
        mean_grads = []
        max_grads = []
        
        for name, grad in self.gradients.items():
            if grad is not None:
                layer_names.append(name)
                mean_grads.append(grad.abs().mean().item())
                max_grads.append(grad.abs().max().item())
        
        if not layer_names:
            print("No gradient data available")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        x = np.arange(len(layer_names))
        
        # Mean gradient plot
        bars1 = ax1.bar(x, mean_grads, color=self.colors['hidden'], alpha=0.7, edgecolor='black')
        ax1.set_ylabel('Mean |Gradient|', fontsize=12, fontweight='bold')
        ax1.set_title('Mean Gradient Magnitude by Layer', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(layer_names, rotation=45, ha='right')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Color bars based on gradient health
        for bar, grad in zip(bars1, mean_grads):
            if grad < 1e-5:
                bar.set_color(self.colors['negative'])  # Vanishing
            elif grad > 1.0:
                bar.set_color(self.colors['negative'])  # Exploding
            else:
                bar.set_color(self.colors['positive'])  # Healthy
        
        # Add reference lines
        ax1.axhline(1e-5, color='red', linestyle='--', linewidth=2, label='Vanishing threshold')
        ax1.axhline(1.0, color='orange', linestyle='--', linewidth=2, label='Large gradient')
        ax1.legend()
        
        # Max gradient plot
        bars2 = ax2.bar(x, max_grads, color=self.colors['output'], alpha=0.7, edgecolor='black')
        ax2.set_ylabel('Max |Gradient|', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Layer', fontsize=12, fontweight='bold')
        ax2.set_title('Maximum Gradient Magnitude by Layer', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(layer_names, rotation=45, ha='right')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add health indicators
        for i, (mean_g, max_g) in enumerate(zip(mean_grads, max_grads)):
            if mean_g < 1e-5:
                status = "⚠ VANISHING"
                color = 'red'
            elif max_g > 100:
                status = "⚠ EXPLODING"
                color = 'red'
            else:
                status = "✓ HEALTHY"
                color = 'green'
            
            ax2.text(i, max_g * 1.5, status, ha='center', va='bottom', 
                    fontsize=8, color=color, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ Saved gradient flow: {save_path}")
        
        return fig
    
    def visualize_weight_matrix(
        self,
        layer_name: str,
        save_path: str = None
    ):
        """
        Visualize weight matrix as heatmap for a specific layer.
        
        Args:
            layer_name: Name of the layer to visualize
            save_path: Path to save the figure
        """
        # Find the layer
        layer = None
        for name, module in self.model.named_modules():
            if name == layer_name and isinstance(module, nn.Linear):
                layer = module
                break
        
        if layer is None:
            print(f"Layer {layer_name} not found or not Linear")
            return
        
        # Get weight matrix
        weights = layer.weight.detach().cpu().numpy()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Heatmap
        im1 = ax1.imshow(weights, cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
        ax1.set_xlabel('Input Features', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Output Features', fontsize=12, fontweight='bold')
        ax1.set_title(f'{layer_name} - Weight Matrix', fontsize=14, fontweight='bold')
        plt.colorbar(im1, ax=ax1, label='Weight Value')
        
        # SVD analysis
        U, S, Vt = np.linalg.svd(weights, full_matrices=False)
        
        ax2.plot(S, 'o-', linewidth=2, markersize=8, color=self.colors['hidden'])
        ax2.set_xlabel('Singular Value Index', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Singular Value', fontsize=12, fontweight='bold')
        ax2.set_title(f'{layer_name} - Singular Values (Rank Analysis)', fontsize=14, fontweight='bold')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        # Add effective rank
        total = S.sum()
        cumsum = np.cumsum(S)
        effective_rank = np.searchsorted(cumsum, 0.95 * total) + 1
        
        ax2.axvline(effective_rank, color='red', linestyle='--', linewidth=2, 
                   label=f'95% Energy at rank {effective_rank}')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ Saved weight matrix: {save_path}")
        
        return fig
    
    def create_complete_report(self, output_dir: str):
        """
        Generate a complete visualization report of the network.
        
        Args:
            output_dir: Directory to save all visualizations
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("🎨 Generating network visualization report...")
        
        # 1. Architecture
        self.visualize_architecture(str(output_path / "01_architecture.png"))
        
        # 2. Weight distributions
        self.visualize_weight_distributions(str(output_path / "02_weight_distributions.png"))
        
        # 3. Sample activation (need a dummy input)
        # Get input size from first layer
        first_linear = None
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                first_linear = module
                break
        
        if first_linear:
            dummy_input = torch.randn(1, first_linear.in_features)
            self.visualize_activations(dummy_input, str(output_path / "03_activations.png"))
            
            # Perform backward pass to get gradients
            output = self.model(dummy_input)
            loss = output.sum()
            loss.backward()
            
            # 4. Gradient flow
            self.visualize_gradient_flow(str(output_path / "04_gradient_flow.png"))
        
        # 5. Weight matrices for each layer
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                safe_name = name.replace('.', '_')
                self.visualize_weight_matrix(name, str(output_path / f"05_weights_{safe_name}.png"))
        
        print(f"✅ Report saved to: {output_path}")


# Convenience functions
def visualize_network(model: nn.Module, output_dir: str = "results/network_vis"):
    """Quick function to generate complete network visualization."""
    viz = NetworkVisualizer(model)
    viz.create_complete_report(output_dir)


if __name__ == "__main__":
    # Example with a simple network
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 4)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            return self.fc3(x)
    
    model = SimpleNet()
    viz = NetworkVisualizer(model)
    
    print("Network visualizer ready!")
    print("Usage: viz.visualize_architecture('architecture.png')")
