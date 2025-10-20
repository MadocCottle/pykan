"""Track per-node importance during training for adaptive densification.

This module tracks the importance of individual nodes (neurons) in KAN layers
over time, enabling selective grid densification where it matters most.

Key Features:
- Per-node gradient magnitude tracking
- Per-node activation variance tracking
- Per-node weight magnitude tracking
- Exponential moving average for stability
- Importance ranking for selective densification

Reference:
- Plan Section: Extension 2 - Adaptive Densification Based on Node Importance
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


class NodeImportanceTracker:
    """Track importance of individual nodes across KAN layers.

    Args:
        model: KAN model to track
        ema_alpha: Exponential moving average smoothing factor (0-1)
        track_interval: How often to update importance (every N steps)

    Example:
        >>> tracker = NodeImportanceTracker(kan_model)
        >>> for epoch in range(epochs):
        ...     loss.backward()
        ...     tracker.update(step=epoch)
        ...     optimizer.step()
        >>> importance = tracker.get_importance_rankings()
    """

    def __init__(
        self,
        model: nn.Module,
        ema_alpha: float = 0.1,
        track_interval: int = 1
    ):
        self.model = model
        self.ema_alpha = ema_alpha
        self.track_interval = track_interval
        self.step_count = 0

        # Storage for importance metrics
        # Format: {layer_idx: {node_idx: value}}
        self.gradient_magnitude = defaultdict(dict)
        self.activation_variance = defaultdict(dict)
        self.weight_magnitude = defaultdict(dict)

        # Hooks for tracking activations
        self.activation_hooks = []
        self.activations = {}

        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks to capture activations."""

        def get_activation(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook

        # Register hooks on each layer
        if hasattr(self.model, 'layers'):
            for i, layer in enumerate(self.model.layers):
                hook = layer.register_forward_hook(get_activation(f'layer_{i}'))
                self.activation_hooks.append(hook)

    def update(self, step: Optional[int] = None):
        """Update importance metrics.

        Should be called after loss.backward() but before optimizer.step()

        Args:
            step: Optional step counter (uses internal counter if None)
        """
        if step is not None:
            self.step_count = step
        else:
            self.step_count += 1

        # Only update at specified intervals
        if self.step_count % self.track_interval != 0:
            return

        # Update gradient-based importance
        self._update_gradient_importance()

        # Update activation-based importance
        self._update_activation_importance()

        # Update weight-based importance
        self._update_weight_importance()

    def _update_gradient_importance(self):
        """Update importance based on gradient magnitudes."""

        if not hasattr(self.model, 'layers'):
            return

        for layer_idx, layer in enumerate(self.model.layers):
            # Get gradients of layer parameters
            for name, param in layer.named_parameters():
                if param.grad is None:
                    continue

                grad = param.grad.detach()

                # Compute per-node gradient magnitude
                if 'scale' in name or 'bias' in name:
                    # These are per-output-node parameters
                    if len(grad.shape) >= 1:
                        node_grads = torch.norm(grad, dim=tuple(range(1, len(grad.shape))))

                        for node_idx, grad_mag in enumerate(node_grads):
                            # Exponential moving average
                            key = (layer_idx, node_idx)
                            if key in self.gradient_magnitude.get(layer_idx, {}):
                                old_val = self.gradient_magnitude[layer_idx][node_idx]
                                new_val = (self.ema_alpha * grad_mag.item() +
                                          (1 - self.ema_alpha) * old_val)
                            else:
                                new_val = grad_mag.item()

                            self.gradient_magnitude[layer_idx][node_idx] = new_val

    def _update_activation_importance(self):
        """Update importance based on activation variance."""

        for name, activation in self.activations.items():
            if not name.startswith('layer_'):
                continue

            layer_idx = int(name.split('_')[1])

            # Compute variance per output dimension (node)
            if len(activation.shape) == 2:
                # (batch, nodes)
                node_vars = torch.var(activation, dim=0)

                for node_idx, var in enumerate(node_vars):
                    # Exponential moving average
                    if node_idx in self.activation_variance.get(layer_idx, {}):
                        old_val = self.activation_variance[layer_idx][node_idx]
                        new_val = (self.ema_alpha * var.item() +
                                  (1 - self.ema_alpha) * old_val)
                    else:
                        new_val = var.item()

                    self.activation_variance[layer_idx][node_idx] = new_val

    def _update_weight_importance(self):
        """Update importance based on weight magnitudes."""

        if not hasattr(self.model, 'layers'):
            return

        for layer_idx, layer in enumerate(self.model.layers):
            # Get weight parameters
            for name, param in layer.named_parameters():
                if 'weight' not in name and 'coef' not in name:
                    continue

                weights = param.detach()

                # Compute per-node weight magnitude
                # Assuming first dimension is output dimension
                if len(weights.shape) >= 2:
                    node_weights = torch.norm(weights, dim=tuple(range(1, len(weights.shape))))

                    for node_idx, weight_mag in enumerate(node_weights):
                        # Exponential moving average
                        if node_idx in self.weight_magnitude.get(layer_idx, {}):
                            old_val = self.weight_magnitude[layer_idx][node_idx]
                            new_val = (self.ema_alpha * weight_mag.item() +
                                      (1 - self.ema_alpha) * old_val)
                        else:
                            new_val = weight_mag.item()

                        self.weight_magnitude[layer_idx][node_idx] = new_val

    def get_node_importance(
        self,
        layer_idx: int,
        node_idx: int,
        method: str = 'combined'
    ) -> float:
        """Get importance score for a specific node.

        Args:
            layer_idx: Layer index
            node_idx: Node index within layer
            method: Importance method ('gradient', 'activation', 'weight', 'combined')

        Returns:
            Importance score
        """
        if method == 'gradient':
            return self.gradient_magnitude.get(layer_idx, {}).get(node_idx, 0.0)
        elif method == 'activation':
            return self.activation_variance.get(layer_idx, {}).get(node_idx, 0.0)
        elif method == 'weight':
            return self.weight_magnitude.get(layer_idx, {}).get(node_idx, 0.0)
        elif method == 'combined':
            # Average of all methods (normalized)
            grad_imp = self.gradient_magnitude.get(layer_idx, {}).get(node_idx, 0.0)
            act_imp = self.activation_variance.get(layer_idx, {}).get(node_idx, 0.0)
            weight_imp = self.weight_magnitude.get(layer_idx, {}).get(node_idx, 0.0)

            # Normalize by max in each category
            grad_max = max(self.gradient_magnitude.get(layer_idx, {}).values(), default=1.0)
            act_max = max(self.activation_variance.get(layer_idx, {}).values(), default=1.0)
            weight_max = max(self.weight_magnitude.get(layer_idx, {}).values(), default=1.0)

            grad_norm = grad_imp / grad_max if grad_max > 0 else 0
            act_norm = act_imp / act_max if act_max > 0 else 0
            weight_norm = weight_imp / weight_max if weight_max > 0 else 0

            return (grad_norm + act_norm + weight_norm) / 3.0
        else:
            raise ValueError(f"Unknown method: {method}")

    def get_layer_importance_rankings(
        self,
        layer_idx: int,
        method: str = 'combined',
        k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """Get ranked list of nodes by importance for a layer.

        Args:
            layer_idx: Layer index
            method: Importance method
            k: Number of top nodes to return (None = all)

        Returns:
            List of (node_idx, importance_score) tuples, sorted by importance
        """
        # Get all nodes in layer
        if method == 'gradient':
            nodes = self.gradient_magnitude.get(layer_idx, {})
        elif method == 'activation':
            nodes = self.activation_variance.get(layer_idx, {})
        elif method == 'weight':
            nodes = self.weight_magnitude.get(layer_idx, {})
        elif method == 'combined':
            # Get all node indices from any metric
            all_indices = set()
            all_indices.update(self.gradient_magnitude.get(layer_idx, {}).keys())
            all_indices.update(self.activation_variance.get(layer_idx, {}).keys())
            all_indices.update(self.weight_magnitude.get(layer_idx, {}).keys())

            nodes = {idx: self.get_node_importance(layer_idx, idx, 'combined')
                    for idx in all_indices}
        else:
            raise ValueError(f"Unknown method: {method}")

        # Sort by importance
        rankings = sorted(nodes.items(), key=lambda x: x[1], reverse=True)

        if k is not None:
            rankings = rankings[:k]

        return rankings

    def get_importance_rankings(
        self,
        method: str = 'combined',
        k_per_layer: Optional[int] = None
    ) -> Dict[int, List[Tuple[int, float]]]:
        """Get importance rankings for all layers.

        Args:
            method: Importance method
            k_per_layer: Number of top nodes per layer (None = all)

        Returns:
            Dictionary mapping layer_idx to ranked node list
        """
        rankings = {}

        # Get all layer indices
        all_layers = set()
        all_layers.update(self.gradient_magnitude.keys())
        all_layers.update(self.activation_variance.keys())
        all_layers.update(self.weight_magnitude.keys())

        for layer_idx in sorted(all_layers):
            rankings[layer_idx] = self.get_layer_importance_rankings(
                layer_idx, method, k_per_layer
            )

        return rankings

    def reset(self):
        """Reset all tracked statistics."""
        self.gradient_magnitude.clear()
        self.activation_variance.clear()
        self.weight_magnitude.clear()
        self.activations.clear()
        self.step_count = 0

    def cleanup(self):
        """Remove forward hooks."""
        for hook in self.activation_hooks:
            hook.remove()
        self.activation_hooks.clear()


# =============================================================================
# Example Usage and Testing
# =============================================================================

if __name__ == '__main__':
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from section1.models.kan_variants import RBF_KAN

    print("="*70)
    print("Testing Node Importance Tracker")
    print("="*70)

    # Create model
    torch.manual_seed(42)
    model = RBF_KAN(input_dim=3, hidden_dim=10, output_dim=1, depth=3)

    # Create tracker
    tracker = NodeImportanceTracker(model, ema_alpha=0.1, track_interval=1)

    # Training simulation
    X = torch.randn(20, 3)
    y = torch.randn(20, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    print("\nTraining for 50 steps...")
    for step in range(50):
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()

        # Update importance tracking
        tracker.update(step)

        optimizer.step()

        if (step + 1) % 10 == 0:
            print(f"Step {step+1:3d}: Loss = {loss.item():.6f}")

    # Get importance rankings
    print("\n" + "="*70)
    print("Importance Rankings")
    print("="*70)

    rankings = tracker.get_importance_rankings(method='combined', k_per_layer=5)

    for layer_idx, node_rankings in rankings.items():
        print(f"\nLayer {layer_idx} - Top 5 Most Important Nodes:")
        for rank, (node_idx, importance) in enumerate(node_rankings[:5], 1):
            print(f"  {rank}. Node {node_idx}: {importance:.6f}")

    # Test specific node importance
    if 0 in rankings and len(rankings[0]) > 0:
        top_node = rankings[0][0][0]
        print(f"\nDetailed importance for Layer 0, Node {top_node}:")
        print(f"  Gradient: {tracker.get_node_importance(0, top_node, 'gradient'):.6f}")
        print(f"  Activation: {tracker.get_node_importance(0, top_node, 'activation'):.6f}")
        print(f"  Weight: {tracker.get_node_importance(0, top_node, 'weight'):.6f}")
        print(f"  Combined: {tracker.get_node_importance(0, top_node, 'combined'):.6f}")

    # Cleanup
    tracker.cleanup()

    print("\n" + "="*70)
    print("Tracker test complete!")
    print("="*70)
