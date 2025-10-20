"""
Model definitions for PDE solving.

Includes:
- MLP with various activations
- SIREN (Sinusoidal Representation Networks)
- KAN wrapper utilities
"""

import torch
import torch.nn as nn
import numpy as np


# ============= SIREN =============
class Sine(nn.Module):
    """Sine activation function with frequency parameter."""

    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class SIREN(nn.Module):
    """
    Sinusoidal Representation Networks (SIREN).

    Particularly effective for learning smooth functions and solving PDEs.

    Reference:
        Sitzmann et al., "Implicit Neural Representations with Periodic Activation Functions"
        NeurIPS 2020
    """

    def __init__(self, in_features, hidden_features, hidden_layers, out_features,
                 outermost_linear=True, first_omega_0=30.0, hidden_omega_0=30.0):
        """
        Args:
            in_features: Input dimension
            hidden_features: Hidden layer width
            hidden_layers: Number of hidden layers
            out_features: Output dimension
            outermost_linear: If True, don't apply sine to final layer
            first_omega_0: Frequency for first layer
            hidden_omega_0: Frequency for hidden layers
        """
        super().__init__()

        self.net = []
        self.net.append(nn.Linear(in_features, hidden_features))
        self.net.append(Sine(first_omega_0))

        for i in range(hidden_layers):
            self.net.append(nn.Linear(hidden_features, hidden_features))
            self.net.append(Sine(hidden_omega_0))

        final_linear = nn.Linear(hidden_features, out_features)
        self.net.append(final_linear)

        self.net = nn.Sequential(*self.net)
        self._init_weights()

    def _init_weights(self):
        """Initialize weights according to SIREN paper."""
        with torch.no_grad():
            # First layer
            self.net[0].weight.uniform_(-1 / self.net[0].in_features,
                                        1 / self.net[0].in_features)
            # Hidden layers
            for i in range(2, len(self.net) - 1, 2):
                self.net[i].weight.uniform_(-np.sqrt(6 / self.net[i].in_features) / 30.0,
                                            np.sqrt(6 / self.net[i].in_features) / 30.0)
            # Final layer
            self.net[-1].weight.uniform_(-np.sqrt(6 / self.net[-1].in_features) / 30.0,
                                         np.sqrt(6 / self.net[-1].in_features) / 30.0)

    def forward(self, x):
        return self.net(x)


# ============= Standard MLP =============
class MLP(nn.Module):
    """
    Standard Multi-Layer Perceptron with configurable activation.
    """

    def __init__(self, in_features=1, hidden_features=64, hidden_layers=2, out_features=1, activation='tanh'):
        """
        Args:
            in_features: Input dimension
            hidden_features: Hidden layer width
            hidden_layers: Number of hidden layers (including input and output layers in total depth)
            out_features: Output dimension
            activation: Activation function ('tanh', 'relu', 'silu', 'gelu')
        """
        super().__init__()
        layers = []

        # Input layer
        layers.append(nn.Linear(in_features, hidden_features))

        # Activation function
        if activation == 'tanh':
            act_fn = nn.Tanh()
        elif activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'silu':
            act_fn = nn.SiLU()
        elif activation == 'gelu':
            act_fn = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        layers.append(act_fn)

        # Hidden layers
        for _ in range(hidden_layers - 2):
            layers.append(nn.Linear(hidden_features, hidden_features))
            layers.append(act_fn)

        # Output layer
        layers.append(nn.Linear(hidden_features, out_features))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ============= Modified MLP for PDEs (with better initialization) =============
class PDEMLP(nn.Module):
    """
    MLP with Xavier initialization, better suited for PDE solving.
    """

    def __init__(self, in_features=1, hidden_features=64, hidden_layers=3, out_features=1,
                 activation='tanh', use_batchnorm=False):
        """
        Args:
            in_features: Input dimension
            hidden_features: Hidden layer width
            hidden_layers: Number of hidden layers
            out_features: Output dimension
            activation: Activation function
            use_batchnorm: Whether to use batch normalization
        """
        super().__init__()

        layers = []

        # Input layer
        layers.append(nn.Linear(in_features, hidden_features))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_features))

        # Activation
        if activation == 'tanh':
            act_fn = nn.Tanh()
        elif activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'silu':
            act_fn = nn.SiLU()
        elif activation == 'gelu':
            act_fn = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        layers.append(act_fn)

        # Hidden layers
        for _ in range(hidden_layers - 2):
            layers.append(nn.Linear(hidden_features, hidden_features))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_features))
            layers.append(act_fn)

        # Output layer
        layers.append(nn.Linear(hidden_features, out_features))

        self.network = nn.Sequential(*layers)

        # Xavier initialization
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.network(x)


# ============= Model Factory =============
def create_model(model_type, in_features, hidden_features, hidden_layers, out_features,
                 activation='tanh', device='cpu', **kwargs):
    """
    Factory function to create models.

    Args:
        model_type: 'mlp', 'pde_mlp', or 'siren'
        in_features: Input dimension
        hidden_features: Hidden layer width
        hidden_layers: Number of hidden layers
        out_features: Output dimension
        activation: Activation function (for MLP)
        device: Device to create model on
        **kwargs: Additional arguments for specific model types

    Returns:
        Model instance
    """
    if model_type == 'mlp':
        model = MLP(in_features, hidden_features, hidden_layers, out_features, activation)
    elif model_type == 'pde_mlp':
        model = PDEMLP(in_features, hidden_features, hidden_layers, out_features, activation, **kwargs)
    elif model_type == 'siren':
        first_omega_0 = kwargs.get('first_omega_0', 30.0)
        hidden_omega_0 = kwargs.get('hidden_omega_0', 30.0)
        model = SIREN(in_features, hidden_features, hidden_layers, out_features,
                      first_omega_0=first_omega_0, hidden_omega_0=hidden_omega_0)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model.to(device)


# ============= KAN Model Wrapper =============
class KANWrapper:
    """
    Wrapper for KAN model creation with common configurations.
    """

    @staticmethod
    def create_kan(width, grid=5, k=3, device='cpu', seed=0, **kwargs):
        """
        Create a KAN model.

        Args:
            width: List of layer widths, e.g., [2, 5, 5, 1]
            grid: Grid size
            k: Spline order
            device: Device
            seed: Random seed
            **kwargs: Additional KAN arguments

        Returns:
            KAN model instance
        """
        from kan import KAN

        model = KAN(width=width, grid=grid, k=k, seed=seed, device=device, **kwargs)
        return model

    @staticmethod
    def create_progressive_kan(width, grids=[5, 10, 20], k=3, device='cpu', seed=0):
        """
        Create a KAN model with progressive grid refinement setup.

        Args:
            width: List of layer widths
            grids: List of grid sizes for progressive refinement
            k: Spline order
            device: Device
            seed: Random seed

        Returns:
            Tuple of (initial_model, grids_list)
        """
        from kan import KAN

        model = KAN(width=width, grid=grids[0], k=k, seed=seed, device=device)
        return model, grids


def count_parameters(model):
    """
    Count the number of trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)