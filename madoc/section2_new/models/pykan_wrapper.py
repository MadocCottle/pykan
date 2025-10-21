"""PyKAN compatibility wrapper for section2_new.

This module provides a compatibility layer between pykan's MultKAN and section2_new's
API conventions, allowing seamless integration of pykan's B-spline KAN implementation
with section2_new's ensemble, evolution, and other advanced features.

Reference:
    Liu, Ziming, et al. "KAN: Kolmogorov-Arnold Networks."
    arXiv preprint arXiv:2404.19756 (2024).
    https://arxiv.org/abs/2404.19756

    Liu, Ziming, et al. "KAN 2.0: Kolmogorov-Arnold Networks Meet Science."
    arXiv preprint arXiv:2408.10205 (2024).
    https://arxiv.org/abs/2408.10205
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
from typing import Optional, List

# Add pykan to path (parent.parent.parent.parent = pykan/)
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from kan.MultKAN import KAN as MultKAN


class PyKANCompatible(MultKAN):
    """Wrapper to make pykan's MultKAN compatible with section2_new API.

    This class adapts MultKAN's constructor to match the section2_new convention
    of (input_dim, hidden_dim, output_dim, depth) instead of MultKAN's width list.

    Args:
        input_dim: Input dimension
        hidden_dim: Hidden layer width
        output_dim: Output dimension
        depth: Number of layers (total layers including input and output)
        grid_size: Grid size for B-spline knots (maps to MultKAN's 'grid')
        spline_order: B-spline order (maps to MultKAN's 'k'), default 3 (cubic)
        device: Torch device ('cpu', 'cuda', or 'mps')
        **kwargs: Additional arguments passed to MultKAN

    Example:
        >>> # Section2_new style
        >>> model = PyKANCompatible(
        ...     input_dim=3,
        ...     hidden_dim=16,
        ...     output_dim=1,
        ...     depth=3,
        ...     grid_size=5,
        ...     device='cpu'
        ... )
        >>> # Equivalent to MultKAN with width=[3, 16, 16, 1]

    Notes:
        - This wrapper allows section2_new code to use MultKAN without modification
        - MultKAN provides B-spline basis functions with adaptive grid management
        - MultKAN supports symbolic reasoning, pruning, and interpretability features
        - For other basis types (Chebyshev, Fourier, Wavelet, RBF), use the
          corresponding classes from section1.models.kan_variants
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 1,
        depth: int = 3,
        grid_size: int = 5,
        spline_order: int = 3,
        device: str = 'cpu',
        **kwargs
    ):
        # Convert section2_new API to MultKAN width list
        # depth=3 means: input -> hidden -> hidden -> output (3 layers total)
        # So we need depth+1 elements in width list
        if depth < 2:
            raise ValueError(f"Depth must be >= 2 (got {depth}). Depth=2 means input->output.")

        # Build width list: [input, hidden, hidden, ..., output]
        # Number of hidden layers = depth - 1
        width = [input_dim] + [hidden_dim] * (depth - 1) + [output_dim]

        # Store for compatibility
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.depth = depth
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Filter kwargs to only include MultKAN-supported parameters
        # Remove basis-specific params (RBF: n_centers, Chebyshev: degree, etc.)
        unsupported_params = {'n_centers', 'degree', 'num_frequencies', 'wavelet_type'}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in unsupported_params}

        # Initialize MultKAN
        super().__init__(
            width=width,
            grid=grid_size,
            k=spline_order,
            device=device,
            **filtered_kwargs
        )

    def __repr__(self):
        return (
            f"PyKANCompatible(input_dim={self.input_dim}, "
            f"hidden_dim={self.hidden_dim}, output_dim={self.output_dim}, "
            f"depth={self.depth}, grid_size={self.grid_size}, "
            f"spline_order={self.spline_order}, device={self.device})"
        )


def create_pykan_model(
    input_dim: int,
    hidden_dim: int = 64,
    output_dim: int = 1,
    depth: int = 3,
    grid_size: int = 5,
    spline_order: int = 3,
    device: str = 'cpu',
    use_speed_mode: bool = False,
    **kwargs
) -> MultKAN:
    """Factory function to create a MultKAN model with section2_new API.

    Args:
        input_dim: Input dimension
        hidden_dim: Hidden layer width
        output_dim: Output dimension
        depth: Number of layers
        grid_size: Grid size for B-spline knots
        spline_order: B-spline order (default 3 = cubic)
        device: Torch device
        use_speed_mode: If True, call .speed() for faster inference
        **kwargs: Additional MultKAN arguments

    Returns:
        MultKAN model instance

    Example:
        >>> model = create_pykan_model(3, 16, 1, depth=3, grid_size=5)
        >>> x = torch.randn(10, 3)
        >>> y = model(x)
    """
    model = PyKANCompatible(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        depth=depth,
        grid_size=grid_size,
        spline_order=spline_order,
        device=device,
        **kwargs
    )

    if use_speed_mode:
        model = model.speed()

    return model


# Alias for convenience
BSplineKAN = PyKANCompatible


__all__ = [
    'PyKANCompatible',
    'BSplineKAN',
    'create_pykan_model',
    'MultKAN',  # Re-export original MultKAN
]
