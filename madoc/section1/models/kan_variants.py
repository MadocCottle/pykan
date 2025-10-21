"""
KAN Variants with Different Basis Functions

This module provides custom KAN implementations using different mathematical basis functions.
Implementations are inspired by and adapted from external KAN research projects.

================================================================================
USED PAPERS & REPOSITORIES:
================================================================================

1. FastKAN: Li, Ziyao (2024). "FastKAN: Very Fast Implementation of KAN"
   Repository: https://github.com/ZiyaoLi/fast-kan
   License: Apache License 2.0
   Paper: "Kolmogorov-Arnold Networks are Radial Basis Function Networks"
          https://arxiv.org/html/2405.06721
   Used for: RBF_KAN, RBFBasis implementations
   Status: USED - RBF basis functions with Gaussian kernels
   Performance: 3.33x faster than efficient_kan (742us -> 223us on V100)

2. ChebyKAN: SynodicMonth (2024). "Chebyshev Polynomial-Based KAN"
   Repository: https://github.com/SynodicMonth/ChebyKAN
   Paper: "Chebyshev Polynomial-Based Kolmogorov-Arnold Networks: An Efficient
           Architecture for Nonlinear Function Approximation"
          arXiv:2405.07200v1
   License: MIT
   Used for: ChebyshevKAN, ChebyshevBasis implementations
   Status: USED - Chebyshev polynomial basis functions
   Note: Uses tanh normalization to [-1,1] and LayerNorm to avoid vanishing gradients

3. FourierKAN: GistNoesis (2024). "Fourier Kolmogorov-Arnold Network"
   Repository: https://github.com/GistNoesis/FourierKAN
   License: MIT
   Used for: FourierKAN, FourierBasis implementations
   Status: USED - Fourier series basis functions
   Note: Higher frequency terms may make training difficult; smooth_initialization recommended

================================================================================
UNUSED PAPERS (for reference):
================================================================================

4. WaveletKAN: Da1sypetals (2024). "CUDA Wavelet KAN"
   Paper: "Kolmogorov-Arnold Networks with Modified Activation Function" arXiv:2405.12832
   Repository: https://github.com/Da1sypetals/cuda-Wavelet-KAN
   License: MIT
   Status: NOT IMPLEMENTED - CUDA-optimized wavelet basis (Mexican hat wavelets)
   Reason: Requires CUDA compilation; placeholder implementation provided

5. TorchKAN: Bhattacharjee, S. (2024). "TorchKAN with Legendre/Chebyshev"
   Repository: https://github.com/1ssb/torchkan
   PyPI: pip install TorchKAN
   License: MIT
   Status: NOT IMPLEMENTED - Alternative Chebyshev/Legendre implementation
   Reason: Redundant with ChebyKAN implementation

================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional


# ============================================================================
# BASIS FUNCTION CLASSES
# ============================================================================

class RBFBasis(nn.Module):
    """
    Radial Basis Function (RBF) with Gaussian kernels.

    Based on FastKAN implementation:
    - Uses Gaussian RBF: φ(r) = exp(-(r - c)²/σ²)
    - Centers distributed uniformly across input range
    - Fixed or learnable widths

    References:
        FastKAN (Li, Ziyao, 2024): https://github.com/ZiyaoLi/fast-kan
    """

    def __init__(self,
                 n_centers: int = 10,
                 input_range: tuple = (-1.0, 1.0),
                 learnable_centers: bool = False,
                 learnable_widths: bool = True):
        """
        Args:
            n_centers: Number of RBF centers
            input_range: Range for center placement (min, max)
            learnable_centers: Whether centers are trainable
            learnable_widths: Whether widths are trainable
        """
        super().__init__()
        self.n_centers = n_centers

        # Initialize centers uniformly
        centers = torch.linspace(input_range[0], input_range[1], n_centers)
        if learnable_centers:
            self.centers = nn.Parameter(centers)
        else:
            self.register_buffer('centers', centers)

        # Initialize widths (inverse of variance)
        # Default: overlap neighboring RBFs
        width = 2.0 / (n_centers - 1) if n_centers > 1 else 1.0
        widths = torch.ones(n_centers) * width

        if learnable_widths:
            self.log_widths = nn.Parameter(torch.log(widths))
        else:
            self.register_buffer('log_widths', torch.log(widths))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [..., 1]
        Returns:
            RBF activations [..., n_centers]
        """
        # x: [..., 1], centers: [n_centers]
        # Compute distances
        distances = x - self.centers.view(1, -1)  # [..., n_centers]

        # Apply Gaussian kernel with learnable widths
        widths = torch.exp(self.log_widths)
        rbf_values = torch.exp(-0.5 * (distances / widths) ** 2)

        return rbf_values


class ChebyshevBasis(nn.Module):
    """
    Chebyshev polynomial basis functions of the first kind.

    Based on ChebyKAN implementation:
    - Uses trigonometric definition for numerical stability
    - Input normalized to [-1, 1] using tanh
    - Orthogonal polynomials on [-1, 1]

    Chebyshev polynomials defined recursively:
        T_0(x) = 1
        T_1(x) = x
        T_{n+1}(x) = 2x * T_n(x) - T_{n-1}(x)

    Or via trigonometric form: T_n(x) = cos(n * arccos(x))

    References:
        ChebyKAN (SynodicMonth, 2024): https://github.com/SynodicMonth/ChebyKAN
        arXiv:2405.07200v1
    """

    def __init__(self, degree: int = 5, normalize_input: bool = True):
        """
        Args:
            degree: Maximum polynomial degree
            normalize_input: Apply tanh to normalize inputs to [-1, 1]
        """
        super().__init__()
        self.degree = degree
        self.normalize_input = normalize_input

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [..., 1]
        Returns:
            Chebyshev polynomial values [..., degree+1]
        """
        if self.normalize_input:
            x = torch.tanh(x)

        # Clamp to [-1, 1] for numerical stability
        x = torch.clamp(x, -1.0, 1.0)

        # Use trigonometric definition: T_n(x) = cos(n * arccos(x))
        acos_x = torch.acos(x)

        chebyshev_values = []
        for n in range(self.degree + 1):
            T_n = torch.cos(n * acos_x)
            chebyshev_values.append(T_n)

        return torch.cat(chebyshev_values, dim=-1)


class FourierBasis(nn.Module):
    """
    Fourier series basis functions (sines and cosines).

    Based on FourierKAN implementation:
    - Uses sin and cos at multiple frequencies
    - Can use smooth initialization (Brownian noise) for training stability
    - Memory usage proportional to gridsize

    Basis: {1, sin(ωx), cos(ωx), sin(2ωx), cos(2ωx), ...}
    where ω is the fundamental frequency

    References:
        FourierKAN (GistNoesis, 2024): https://github.com/GistNoesis/FourierKAN
    """

    def __init__(self,
                 num_frequencies: int = 5,
                 base_frequency: float = 1.0,
                 include_constant: bool = True):
        """
        Args:
            num_frequencies: Number of frequency components
            base_frequency: Fundamental frequency ω
            include_constant: Include constant term (DC component)
        """
        super().__init__()
        self.num_frequencies = num_frequencies
        self.base_frequency = base_frequency
        self.include_constant = include_constant

        # Calculate output dimension
        # constant + (sin + cos) * num_frequencies
        self.output_dim = (1 if include_constant else 0) + 2 * num_frequencies

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [..., 1]
        Returns:
            Fourier basis values [..., output_dim]
        """
        components = []

        if self.include_constant:
            components.append(torch.ones_like(x))

        for k in range(1, self.num_frequencies + 1):
            omega_k = k * self.base_frequency
            components.append(torch.sin(omega_k * x))
            components.append(torch.cos(omega_k * x))

        return torch.cat(components, dim=-1)


# ============================================================================
# KAN LAYER IMPLEMENTATIONS
# ============================================================================

class KANLayer(nn.Module):
    """
    Base KAN layer with customizable basis functions.

    Each edge has its own basis function representation:
        f(x) = Σ_i w_i * φ_i(x)

    where φ_i are basis functions (RBF, Chebyshev, Fourier, etc.)
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 basis_class: type,
                 basis_kwargs: dict,
                 use_base_activation: bool = True,
                 base_activation: nn.Module = nn.SiLU()):
        """
        Args:
            input_dim: Number of input features
            output_dim: Number of output features
            basis_class: Basis function class (RBFBasis, ChebyshevBasis, etc.)
            basis_kwargs: Arguments for basis function initialization
            use_base_activation: Add residual base activation
            base_activation: Activation function for residual
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_base_activation = use_base_activation

        # Create basis functions for each input
        self.basis_functions = nn.ModuleList([
            basis_class(**basis_kwargs) for _ in range(input_dim)
        ])

        # Determine basis output dimension
        dummy_input = torch.zeros(1, 1)
        basis_out_dim = self.basis_functions[0](dummy_input).shape[-1]

        # Weights for each edge: [output_dim, input_dim, basis_out_dim]
        self.weights = nn.Parameter(
            torch.randn(output_dim, input_dim, basis_out_dim) * 0.1
        )

        # Optional base activation (residual connection)
        if use_base_activation:
            self.base_activation = base_activation
            self.base_weights = nn.Parameter(
                torch.randn(output_dim, input_dim) * 0.1
            )

        # Layer normalization for training stability (from ChebyKAN)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch, input_dim]
        Returns:
            Output tensor [batch, output_dim]
        """
        batch_size = x.shape[0]

        # Apply basis functions to each input dimension
        basis_outputs = []
        for i in range(self.input_dim):
            x_i = x[:, i:i+1]  # [batch, 1]
            basis_out = self.basis_functions[i](x_i)  # [batch, basis_dim]
            basis_outputs.append(basis_out)

        basis_outputs = torch.stack(basis_outputs, dim=1)  # [batch, input_dim, basis_dim]

        # Apply weights: [batch, input_dim, basis_dim] @ [output_dim, input_dim, basis_dim]
        # For each output dimension, sum over input dimensions
        output = torch.zeros(batch_size, self.output_dim, device=x.device)

        for j in range(self.output_dim):
            # weights[j]: [input_dim, basis_dim]
            # Compute: sum_i (w_ji · φ_i(x_i))
            for i in range(self.input_dim):
                output[:, j] += torch.sum(
                    basis_outputs[:, i, :] * self.weights[j, i, :],
                    dim=-1
                )

        # Add base activation if enabled
        if self.use_base_activation:
            base_out = self.base_activation(x) @ self.base_weights.t()
            output = output + base_out

        # Layer normalization
        output = self.layer_norm(output)

        return output


# ============================================================================
# FULL KAN NETWORK IMPLEMENTATIONS
# ============================================================================

class RBF_KAN(nn.Module):
    """
    KAN using Radial Basis Functions (Gaussian RBF).

    Fast approximation of B-spline KAN using RBF kernels.
    3.33x faster than efficient_kan according to FastKAN benchmarks.

    Attributes:
        layers: List of KAN layers (exposed for adaptive densification)

    References:
        FastKAN: https://github.com/ZiyaoLi/fast-kan
        "Kolmogorov-Arnold Networks are Radial Basis Function Networks"
    """

    def __init__(self,
                 layers_hidden: List[int],
                 n_centers: int = 10,
                 learnable_centers: bool = False,
                 learnable_widths: bool = True,
                 input_range: tuple = (-1.0, 1.0),
                 device: str = 'cpu'):
        """
        Args:
            layers_hidden: Layer dimensions [input_dim, hidden1, ..., output_dim]
            n_centers: Number of RBF centers per edge
            learnable_centers: Whether RBF centers are trainable
            learnable_widths: Whether RBF widths are trainable
            input_range: Range for RBF center placement
            device: Device to run on
        """
        super().__init__()
        self.layers_hidden = layers_hidden
        self.n_centers = n_centers
        self.device = device

        basis_kwargs = {
            'n_centers': n_centers,
            'input_range': input_range,
            'learnable_centers': learnable_centers,
            'learnable_widths': learnable_widths
        }

        # Create KAN layers
        self.layers = nn.ModuleList()
        for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:]):
            layer = KANLayer(
                input_dim=in_dim,
                output_dim=out_dim,
                basis_class=RBFBasis,
                basis_kwargs=basis_kwargs
            )
            self.layers.append(layer)

        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch, input_dim]
        Returns:
            Output tensor [batch, output_dim]
        """
        for layer in self.layers:
            x = layer(x)
        return x


class ChebyshevKAN(nn.Module):
    """
    KAN using Chebyshev polynomial basis functions.

    Efficient orthogonal polynomial basis on [-1, 1].
    Uses tanh normalization and LayerNorm for training stability.

    Attributes:
        layers: List of KAN layers (exposed for adaptive densification)

    References:
        ChebyKAN: https://github.com/SynodicMonth/ChebyKAN
        "Chebyshev Polynomial-Based KAN" arXiv:2405.07200v1
    """

    def __init__(self,
                 layers_hidden: List[int],
                 degree: int = 5,
                 normalize_input: bool = True,
                 device: str = 'cpu'):
        """
        Args:
            layers_hidden: Layer dimensions [input_dim, hidden1, ..., output_dim]
            degree: Maximum Chebyshev polynomial degree
            normalize_input: Apply tanh normalization to [-1, 1]
            device: Device to run on
        """
        super().__init__()
        self.layers_hidden = layers_hidden
        self.degree = degree
        self.device = device

        basis_kwargs = {
            'degree': degree,
            'normalize_input': normalize_input
        }

        # Create KAN layers
        self.layers = nn.ModuleList()
        for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:]):
            layer = KANLayer(
                input_dim=in_dim,
                output_dim=out_dim,
                basis_class=ChebyshevBasis,
                basis_kwargs=basis_kwargs
            )
            self.layers.append(layer)

        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch, input_dim]
        Returns:
            Output tensor [batch, output_dim]
        """
        for layer in self.layers:
            x = layer(x)
        return x


class FourierKAN(nn.Module):
    """
    KAN using Fourier series basis functions.

    Uses sine and cosine terms at multiple frequencies.
    Higher frequency terms may make training difficult.

    Attributes:
        layers: List of KAN layers (exposed for adaptive densification)

    References:
        FourierKAN: https://github.com/GistNoesis/FourierKAN
    """

    def __init__(self,
                 layers_hidden: List[int],
                 num_frequencies: int = 5,
                 base_frequency: float = 1.0,
                 include_constant: bool = True,
                 device: str = 'cpu'):
        """
        Args:
            layers_hidden: Layer dimensions [input_dim, hidden1, ..., output_dim]
            num_frequencies: Number of frequency components
            base_frequency: Fundamental frequency ω
            include_constant: Include DC component
            device: Device to run on
        """
        super().__init__()
        self.layers_hidden = layers_hidden
        self.num_frequencies = num_frequencies
        self.device = device

        basis_kwargs = {
            'num_frequencies': num_frequencies,
            'base_frequency': base_frequency,
            'include_constant': include_constant
        }

        # Create KAN layers
        self.layers = nn.ModuleList()
        for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:]):
            layer = KANLayer(
                input_dim=in_dim,
                output_dim=out_dim,
                basis_class=FourierBasis,
                basis_kwargs=basis_kwargs
            )
            self.layers.append(layer)

        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch, input_dim]
        Returns:
            Output tensor [batch, output_dim]
        """
        for layer in self.layers:
            x = layer(x)
        return x


class WaveletKAN(nn.Module):
    """
    Placeholder for Wavelet-based KAN.

    NOT IMPLEMENTED: Requires CUDA compilation for optimal performance.
    Falls back to RBF_KAN for compatibility.

    References (UNUSED):
        WaveletKAN: https://github.com/Da1sypetals/cuda-Wavelet-KAN
        "KAN with Modified Activation Function" arXiv:2405.12832
    """

    def __init__(self,
                 layers_hidden: List[int],
                 wavelet_type: str = 'mexican_hat',
                 device: str = 'cpu',
                 **kwargs):
        """
        Args:
            layers_hidden: Layer dimensions
            wavelet_type: Type of wavelet (not used, falls back to RBF)
            device: Device to run on
        """
        super().__init__()
        print(f"⚠ WaveletKAN not implemented, falling back to RBF_KAN")

        # Fallback to RBF implementation
        self.model = RBF_KAN(
            layers_hidden=layers_hidden,
            device=device,
            **{k: v for k, v in kwargs.items() if k in ['n_centers', 'learnable_centers', 'learnable_widths']}
        )

        # Expose layers attribute for compatibility
        self.layers = self.model.layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ============================================================================
# COMPATIBILITY UTILITIES
# ============================================================================

def create_kan(variant: str, layers_hidden: List[int], device: str = 'cpu', **kwargs):
    """
    Factory function to create KAN variants.

    Args:
        variant: One of ['rbf', 'chebyshev', 'fourier', 'wavelet', 'bspline']
        layers_hidden: Layer dimensions [input_dim, hidden1, ..., output_dim]
        device: Device to run on
        **kwargs: Additional arguments for specific variants
            - n_centers: For RBF (default: 10)
            - degree: For Chebyshev (default: 5)
            - num_frequencies: For Fourier (default: 5)

    Returns:
        KAN model instance
    """
    variant = variant.lower()

    if variant in ['rbf', 'radial']:
        return RBF_KAN(
            layers_hidden=layers_hidden,
            n_centers=kwargs.get('n_centers', 10),
            device=device
        )

    elif variant in ['chebyshev', 'cheby']:
        return ChebyshevKAN(
            layers_hidden=layers_hidden,
            degree=kwargs.get('degree', 5),
            device=device
        )

    elif variant in ['fourier', 'fft']:
        return FourierKAN(
            layers_hidden=layers_hidden,
            num_frequencies=kwargs.get('num_frequencies', 5),
            device=device
        )

    elif variant in ['wavelet']:
        return WaveletKAN(
            layers_hidden=layers_hidden,
            device=device,
            **kwargs
        )

    elif variant in ['bspline', 'pykan']:
        # Fall back to PyKAN's MultKAN
        try:
            from kan.MultKAN import KAN as MultKAN
            return MultKAN(
                width=layers_hidden,
                device=device
            )
        except ImportError:
            raise ImportError("PyKAN not available. Install with: pip install pykan")

    else:
        raise ValueError(f"Unknown KAN variant: {variant}. "
                        f"Choose from: rbf, chebyshev, fourier, wavelet, bspline")
