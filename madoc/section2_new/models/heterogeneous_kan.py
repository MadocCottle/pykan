"""Heterogeneous KAN with mixed basis functions.

This module implements Extension 3: Heterogeneous Basis Functions. Instead of
using a single basis type (B-spline, Chebyshev, Fourier, etc.) throughout the
network, different edges can use different basis functions optimized for the
characteristics of the data flowing through them.

Key Features:
- Mixed-basis KAN layers with edge-specific basis selection
- Fixed basis assignment (manual or heuristic)
- Learnable basis selection via Gumbel-softmax
- Automatic basis selection based on signal characteristics

Reference:
- Plan Section: Extension 3 - Heterogeneous Basis Functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Union
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "section1"))

from section1.models.kan_variants import (
    ChebyshevBasis, FourierBasis, RBFBasis
)


class HeterogeneousKANLayer(nn.Module):
    """KAN layer with heterogeneous basis functions.

    Each edge (input_dim × output_dim) can use a different basis function,
    allowing the network to adapt to different signal characteristics.

    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        basis_config: Dict mapping edge IDs to basis types, or 'learnable'
        basis_params: Dict of basis-specific parameters (degree, grid_size, etc.)
        base_fun: Base activation function (default: SiLU)

    Example:
        >>> layer = HeterogeneousKANLayer(
        ...     input_dim=2,
        ...     output_dim=3,
        ...     basis_config={(0, 0): 'fourier', (0, 1): 'rbf', 'default': 'chebyshev'}
        ... )
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        basis_config: Union[str, Dict] = 'rbf',
        basis_params: Optional[Dict] = None,
        base_fun: Optional[nn.Module] = None
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Default basis parameters
        if basis_params is None:
            basis_params = {
                'chebyshev': {'degree': 8},
                'fourier': {'n_frequencies': 5},
                'rbf': {'n_centers': 10},
                'wavelet': {'n_wavelets': 8}
            }
        self.basis_params = basis_params

        # Base activation
        if base_fun is None:
            self.base_fun = nn.SiLU()
        else:
            self.base_fun = base_fun

        # Initialize basis functions for each edge
        self.basis_modules = nn.ModuleDict()
        self.basis_assignments = {}

        # Parse basis configuration
        if isinstance(basis_config, str):
            # All edges use same basis
            if basis_config == 'learnable':
                self.learnable_basis = True
                self._setup_learnable_basis()
            else:
                self.learnable_basis = False
                for i in range(input_dim):
                    for j in range(output_dim):
                        edge_id = f"{i}_{j}"
                        self.basis_assignments[edge_id] = basis_config
                        self.basis_modules[edge_id] = self._create_basis(basis_config)
        else:
            # Custom basis assignment per edge
            self.learnable_basis = False
            default_basis = basis_config.get('default', 'rbf')

            for i in range(input_dim):
                for j in range(output_dim):
                    edge_tuple = (i, j)
                    edge_id = f"{i}_{j}"

                    # Check if this edge has custom basis
                    if edge_tuple in basis_config:
                        basis_type = basis_config[edge_tuple]
                    elif i in basis_config:  # Config by input dimension
                        basis_type = basis_config[i]
                    else:
                        basis_type = default_basis

                    self.basis_assignments[edge_id] = basis_type
                    self.basis_modules[edge_id] = self._create_basis(basis_type)

        # Learnable coefficients for each edge
        self.coefficients = nn.ParameterDict()
        for edge_id, basis in self.basis_modules.items():
            n_basis = self._get_num_basis(basis)
            self.coefficients[edge_id] = nn.Parameter(
                torch.randn(n_basis) * 0.1
            )

        # Scaling parameters
        self.scale_base = nn.Parameter(torch.ones(output_dim, input_dim))
        self.scale_sp = nn.Parameter(torch.ones(output_dim, input_dim))

    def _create_basis(self, basis_type: str) -> nn.Module:
        """Create a basis module.

        Args:
            basis_type: Type of basis ('chebyshev', 'fourier', 'rbf')

        Returns:
            Basis module
        """
        if basis_type == 'chebyshev':
            degree = self.basis_params.get('chebyshev', {}).get('degree', 8)
            return ChebyshevBasis(degree=degree)
        elif basis_type == 'fourier':
            n_freq = self.basis_params.get('fourier', {}).get('n_frequencies', 5)
            return FourierBasis(n_frequencies=n_freq)
        elif basis_type == 'rbf':
            n_centers = self.basis_params.get('rbf', {}).get('n_centers', 10)
            return RBFBasis(n_centers=n_centers)
        else:
            raise ValueError(f"Unknown basis type: {basis_type}")

    def _get_num_basis(self, basis: nn.Module) -> int:
        """Get number of basis functions.

        Args:
            basis: Basis module

        Returns:
            Number of basis functions
        """
        if hasattr(basis, 'num_basis'):
            return basis.num_basis
        elif hasattr(basis, 'n_frequencies'):
            return 2 * basis.n_frequencies + 1
        elif hasattr(basis, 'n_centers'):
            return basis.n_centers
        else:
            raise ValueError(f"Cannot determine number of basis functions for {type(basis)}")

    def _setup_learnable_basis(self):
        """Setup learnable basis selection using Gumbel-softmax."""
        # Available basis types
        self.basis_types = ['chebyshev', 'fourier', 'rbf']
        n_basis_types = len(self.basis_types)

        # Learnable logits for basis selection (per edge)
        self.basis_logits = nn.Parameter(
            torch.randn(self.output_dim, self.input_dim, n_basis_types)
        )

        # Create all basis types
        for i in range(self.input_dim):
            for j in range(self.output_dim):
                edge_id = f"{i}_{j}"
                for basis_type in self.basis_types:
                    module_id = f"{edge_id}_{basis_type}"
                    self.basis_modules[module_id] = self._create_basis(basis_type)

    def forward(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Forward pass with heterogeneous basis functions.

        Args:
            x: Input tensor (batch_size, input_dim)
            temperature: Temperature for Gumbel-softmax (if learnable)

        Returns:
            Output tensor (batch_size, output_dim)
        """
        batch_size = x.shape[0]

        # Normalize inputs
        x_norm = torch.tanh(x)

        # Base activation
        base = self.base_fun(x_norm)  # (batch_size, input_dim)

        # Compute basis activations for each edge
        output = torch.zeros(batch_size, self.output_dim, device=x.device)

        if self.learnable_basis:
            # Learnable basis selection with Gumbel-softmax
            for i in range(self.input_dim):
                for j in range(self.output_dim):
                    # Get basis weights via Gumbel-softmax
                    logits = self.basis_logits[j, i]
                    weights = F.gumbel_softmax(logits, tau=temperature, hard=False)

                    # Weighted combination of basis outputs
                    edge_output = 0.0
                    for k, basis_type in enumerate(self.basis_types):
                        edge_id = f"{i}_{j}"
                        module_id = f"{edge_id}_{basis_type}"
                        basis = self.basis_modules[module_id]

                        # Evaluate basis
                        basis_vals = basis(x_norm[:, i])  # (batch_size, n_basis)
                        coef_key = f"{i}_{j}_{basis_type}"
                        if coef_key not in self.coefficients:
                            self.coefficients[coef_key] = nn.Parameter(
                                torch.randn(self._get_num_basis(basis)) * 0.1
                            )
                        coef = self.coefficients[coef_key]

                        # Weighted contribution
                        edge_output = edge_output + weights[k] * torch.matmul(basis_vals, coef)

                    # Add scaled base and basis components
                    output[:, j] += (self.scale_base[j, i] * base[:, i] +
                                   self.scale_sp[j, i] * edge_output)
        else:
            # Fixed basis assignment
            for i in range(self.input_dim):
                for j in range(self.output_dim):
                    edge_id = f"{i}_{j}"
                    basis = self.basis_modules[edge_id]

                    # Evaluate basis
                    basis_vals = basis(x_norm[:, i])  # (batch_size, n_basis)
                    coef = self.coefficients[edge_id]

                    # Basis contribution
                    basis_output = torch.matmul(basis_vals, coef)

                    # Add scaled base and basis components
                    output[:, j] += (self.scale_base[j, i] * base[:, i] +
                                   self.scale_sp[j, i] * basis_output)

        return output

    def get_basis_usage(self) -> Dict[str, str]:
        """Get current basis assignment for each edge.

        Returns:
            Dictionary mapping edge IDs to basis types
        """
        if self.learnable_basis:
            # Get most likely basis from logits
            usage = {}
            for i in range(self.input_dim):
                for j in range(self.output_dim):
                    logits = self.basis_logits[j, i]
                    best_idx = torch.argmax(logits).item()
                    edge_id = f"{i}_{j}"
                    usage[edge_id] = self.basis_types[best_idx]
            return usage
        else:
            return self.basis_assignments.copy()


class HeterogeneousBasisKAN(nn.Module):
    """Complete KAN network with heterogeneous basis functions.

    Args:
        layer_dims: List of layer dimensions [input_dim, hidden_dims..., output_dim]
        basis_config: Basis configuration for each layer (list or single config)
        basis_params: Basis parameters

    Example:
        >>> kan = HeterogeneousBasisKAN(
        ...     layer_dims=[2, 5, 5, 1],
        ...     basis_config=[
        ...         {0: 'fourier', 'default': 'rbf'},  # Layer 0: input is periodic
        ...         'rbf',  # Layer 1: all RBF
        ...         'chebyshev'  # Layer 2: all Chebyshev
        ...     ]
        ... )
    """

    def __init__(
        self,
        layer_dims: List[int],
        basis_config: Union[str, Dict, List] = 'rbf',
        basis_params: Optional[Dict] = None
    ):
        super().__init__()
        self.layer_dims = layer_dims
        self.n_layers = len(layer_dims) - 1

        # Parse basis config
        if isinstance(basis_config, (str, dict)):
            # Same config for all layers
            basis_configs = [basis_config] * self.n_layers
        else:
            # Per-layer config
            basis_configs = basis_config
            if len(basis_configs) != self.n_layers:
                raise ValueError(
                    f"basis_config length ({len(basis_configs)}) must match "
                    f"number of layers ({self.n_layers})"
                )

        # Create layers
        self.layers = nn.ModuleList()
        for i in range(self.n_layers):
            layer = HeterogeneousKANLayer(
                input_dim=layer_dims[i],
                output_dim=layer_dims[i + 1],
                basis_config=basis_configs[i],
                basis_params=basis_params
            )
            self.layers.append(layer)

    def forward(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor (batch_size, input_dim)
            temperature: Temperature for Gumbel-softmax (if learnable)

        Returns:
            Output tensor (batch_size, output_dim)
        """
        for layer in self.layers:
            x = layer(x, temperature)
        return x

    def get_all_basis_usage(self) -> Dict[int, Dict[str, str]]:
        """Get basis usage for all layers.

        Returns:
            Dictionary mapping layer index to basis usage
        """
        usage = {}
        for i, layer in enumerate(self.layers):
            usage[i] = layer.get_basis_usage()
        return usage


# =============================================================================
# Example Usage and Testing
# =============================================================================

if __name__ == '__main__':
    print("="*70)
    print("Testing Heterogeneous Basis KAN")
    print("="*70)

    # Test 1: Fixed basis assignment
    print("\n1. Fixed Basis Assignment")
    print("-"*70)

    kan_fixed = HeterogeneousBasisKAN(
        layer_dims=[2, 5, 1],
        basis_config=[
            {0: 'fourier', 1: 'rbf'},  # Input 0: Fourier, Input 1: RBF
            'chebyshev'  # All Chebyshev in second layer
        ]
    )

    X = torch.randn(10, 2)
    y = kan_fixed(X)
    print(f"Input shape: {X.shape}")
    print(f"Output shape: {y.shape}")

    usage = kan_fixed.get_all_basis_usage()
    print("\nBasis Usage:")
    for layer_idx, layer_usage in usage.items():
        print(f"  Layer {layer_idx}:")
        for edge_id, basis_type in sorted(layer_usage.items())[:5]:  # Show first 5
            print(f"    Edge {edge_id}: {basis_type}")

    # Test 2: Training with heterogeneous basis
    print("\n2. Training Heterogeneous KAN")
    print("-"*70)

    # Generate synthetic data: y = sin(x0) + x1^2
    torch.manual_seed(42)
    X_train = torch.randn(100, 2)
    y_train = (torch.sin(X_train[:, 0]) + X_train[:, 1] ** 2).reshape(-1, 1)

    kan = HeterogeneousBasisKAN(
        layer_dims=[2, 10, 1],
        basis_config='rbf'  # All RBF for simplicity
    )

    optimizer = torch.optim.Adam(kan.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    print("Training...")
    for epoch in range(100):
        optimizer.zero_grad()
        y_pred = kan(X_train)
        loss = loss_fn(y_pred, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d}: Loss = {loss.item():.6f}")

    print("\nTraining complete!")

    # Test prediction
    X_test = torch.randn(10, 2)
    y_test = (torch.sin(X_test[:, 0]) + X_test[:, 1] ** 2).reshape(-1, 1)
    with torch.no_grad():
        y_pred = kan(X_test)
        test_loss = loss_fn(y_pred, y_test)
    print(f"Test Loss: {test_loss.item():.6f}")
