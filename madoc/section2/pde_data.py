"""
PDE Data Generation Module

Provides data generation for various PDEs:
- 2D Poisson equation
- 1D heat equation
- 1D wave equation
- Burgers equation
- More PDEs can be added
"""

import torch
import numpy as np
from kan import create_dataset


# ============= 2D Poisson Equation =============

def poisson_2d_solution(freq_x=1.0, freq_y=1.0):
    """
    2D Poisson equation: ∇²u = f
    Solution: u(x,y) = sin(π * freq_x * x) * sin(π * freq_y * y)

    Args:
        freq_x: Frequency in x direction
        freq_y: Frequency in y direction

    Returns:
        Tuple of (solution_func, source_func, gradient_func)
    """
    sol_fun = lambda x: torch.sin(torch.pi * freq_x * x[:, [0]]) * torch.sin(torch.pi * freq_y * x[:, [1]])

    source_fun = lambda x: -(torch.pi * freq_x) ** 2 * torch.sin(torch.pi * freq_x * x[:, [0]]) * torch.sin(
        torch.pi * freq_y * x[:, [1]]) \
                           - (torch.pi * freq_y) ** 2 * torch.sin(torch.pi * freq_x * x[:, [0]]) * torch.sin(
        torch.pi * freq_y * x[:, [1]])

    # Gradient function for H1 norm computation
    grad_fun = lambda x: torch.stack([
        torch.pi * freq_x * torch.cos(torch.pi * freq_x * x[:, 0]) * torch.sin(torch.pi * freq_y * x[:, 1]),
        torch.pi * freq_y * torch.sin(torch.pi * freq_x * x[:, 0]) * torch.cos(torch.pi * freq_y * x[:, 1])
    ], dim=1)

    return sol_fun, source_fun, grad_fun


def poisson_2d_highfreq():
    """
    2D Poisson with higher frequency (default from tutorial).

    Returns:
        Tuple of (solution_func, source_func, gradient_func)
    """
    return poisson_2d_solution(freq_x=1.0, freq_y=1.0)


def poisson_2d_multiscale():
    """
    2D Poisson with multiple frequencies.

    Returns:
        Tuple of (solution_func, source_func, gradient_func)
    """
    sol_fun = lambda x: (torch.sin(torch.pi * x[:, [0]]) * torch.sin(torch.pi * x[:, [1]]) +
                         0.3 * torch.sin(3 * torch.pi * x[:, [0]]) * torch.sin(3 * torch.pi * x[:, [1]]))

    source_fun = lambda x: (
            -2 * (torch.pi) ** 2 * torch.sin(torch.pi * x[:, [0]]) * torch.sin(torch.pi * x[:, [1]]) -
            2 * (3 * torch.pi) ** 2 * 0.3 * torch.sin(3 * torch.pi * x[:, [0]]) * torch.sin(3 * torch.pi * x[:, [1]])
    )

    grad_fun = lambda x: torch.stack([
        torch.pi * torch.cos(torch.pi * x[:, 0]) * torch.sin(torch.pi * x[:, 1]) +
        0.3 * 3 * torch.pi * torch.cos(3 * torch.pi * x[:, 0]) * torch.sin(3 * torch.pi * x[:, 1]),
        torch.pi * torch.sin(torch.pi * x[:, 0]) * torch.cos(torch.pi * x[:, 1]) +
        0.3 * 3 * torch.pi * torch.sin(3 * torch.pi * x[:, 0]) * torch.cos(3 * torch.pi * x[:, 1])
    ], dim=1)

    return sol_fun, source_fun, grad_fun


# ============= 1D Poisson Equation =============

def poisson_1d_solution(freq=1.0):
    """
    1D Poisson equation: d²u/dx² = f

    Args:
        freq: Frequency parameter

    Returns:
        Tuple of (solution_func, source_func, gradient_func)
    """
    sol_fun = lambda x: torch.sin(torch.pi * freq * x[:, [0]])
    source_fun = lambda x: -(torch.pi * freq) ** 2 * torch.sin(torch.pi * freq * x[:, [0]])
    grad_fun = lambda x: torch.pi * freq * torch.cos(torch.pi * freq * x[:, 0]).unsqueeze(1)

    return sol_fun, source_fun, grad_fun


def poisson_1d_highfreq():
    """1D Poisson with high frequency"""
    return poisson_1d_solution(freq=3.0)


# ============= Heat Equation (1D) =============

def heat_1d_solution(alpha=1.0, freq=1.0):
    """
    1D Heat equation: ∂u/∂t = α * ∂²u/∂x²
    Solution: u(x,t) = exp(-α * (π*freq)² * t) * sin(π*freq*x)

    Args:
        alpha: Thermal diffusivity
        freq: Frequency parameter

    Returns:
        Tuple of (solution_func, source_func)
    """
    # Note: input x is [x, t] concatenated
    sol_fun = lambda xt: torch.exp(-alpha * (torch.pi * freq) ** 2 * xt[:, [1]]) * torch.sin(
        torch.pi * freq * xt[:, [0]])

    # For heat equation, source is typically 0 (homogeneous)
    source_fun = lambda xt: torch.zeros_like(xt[:, [0]])

    return sol_fun, source_fun, None


# ============= Burgers Equation =============

def burgers_solution_simple():
    """
    Simple traveling wave solution to Burgers equation: ∂u/∂t + u*∂u/∂x = ν*∂²u/∂x²

    Returns:
        Tuple of (solution_func, None, None)
    """
    # Simple shock wave solution (approximate)
    # u(x,t) = x/t for viscous Burgers with specific IC
    # For simplicity, use a tanh profile

    sol_fun = lambda xt: 0.5 * (1 - torch.tanh((xt[:, [0]] - xt[:, [1]]) / 0.1))

    return sol_fun, None, None


# ============= Helmholtz Equation =============

def helmholtz_2d_solution(k=1.0):
    """
    2D Helmholtz equation: ∇²u + k²u = f

    Args:
        k: Wave number

    Returns:
        Tuple of (solution_func, source_func, gradient_func)
    """
    sol_fun = lambda x: torch.sin(torch.pi * x[:, [0]]) * torch.sin(torch.pi * x[:, [1]])

    # ∇²u + k²u = f => f = -2π²*sin(πx)sin(πy) + k²*sin(πx)sin(πy)
    source_fun = lambda x: (-2 * torch.pi ** 2 + k ** 2) * torch.sin(torch.pi * x[:, [0]]) * torch.sin(
        torch.pi * x[:, [1]])

    grad_fun = lambda x: torch.stack([
        torch.pi * torch.cos(torch.pi * x[:, 0]) * torch.sin(torch.pi * x[:, 1]),
        torch.pi * torch.sin(torch.pi * x[:, 0]) * torch.cos(torch.pi * x[:, 1])
    ], dim=1)

    return sol_fun, source_fun, grad_fun


# ============= Data Generation Utilities =============

def create_pde_dataset_2d(solution_func, ranges=[-1, 1], train_num=1000, test_num=1000, device='cpu', seed=0):
    """
    Create dataset for 2D PDE using pykan's create_dataset.

    Args:
        solution_func: Function that takes (batch, 2) tensor and returns (batch, 1) tensor
        ranges: Range for both x and y
        train_num: Number of training points
        test_num: Number of test points
        device: Device to create tensors on
        seed: Random seed

    Returns:
        Dataset dictionary compatible with pykan
    """
    dataset = create_dataset(
        solution_func,
        n_var=2,
        ranges=ranges,
        train_num=train_num,
        test_num=test_num,
        device=device,
        seed=seed
    )
    return dataset


def create_pde_dataset_1d(solution_func, ranges=[-1, 1], train_num=1000, test_num=1000, device='cpu', seed=0):
    """
    Create dataset for 1D PDE.

    Args:
        solution_func: Function that takes (batch, 1) tensor and returns (batch, 1) tensor
        ranges: Range for x
        train_num: Number of training points
        test_num: Number of test points
        device: Device to create tensors on
        seed: Random seed

    Returns:
        Dataset dictionary
    """
    dataset = create_dataset(
        solution_func,
        n_var=1,
        ranges=ranges,
        train_num=train_num,
        test_num=test_num,
        device=device,
        seed=seed
    )
    return dataset


def create_boundary_points_2d(ranges=[-1, 1], n_points=51, device='cpu'):
    """
    Create boundary points for 2D domain.

    Args:
        ranges: [min, max] for domain
        n_points: Number of points along each boundary edge
        device: Device to create tensors on

    Returns:
        Tensor of boundary points (4*n_points, 2)
    """
    x_mesh = torch.linspace(ranges[0], ranges[1], steps=n_points)
    y_mesh = torch.linspace(ranges[0], ranges[1], steps=n_points)
    X, Y = torch.meshgrid(x_mesh, y_mesh, indexing="ij")

    helper = lambda X, Y: torch.stack([X.reshape(-1, ), Y.reshape(-1, )]).permute(1, 0)

    # 4 sides of the boundary
    xb1 = helper(X[0], Y[0])  # Bottom edge
    xb2 = helper(X[-1], Y[0])  # Top edge
    xb3 = helper(X[:, 0], Y[:, 0])  # Left edge
    xb4 = helper(X[:, 0], Y[:, -1])  # Right edge

    x_b = torch.cat([xb1, xb2, xb3, xb4], dim=0)
    return x_b.to(device)


def create_interior_points_2d(ranges=[-1, 1], n_points=51, mode='mesh', device='cpu', seed=0):
    """
    Create interior points for 2D domain.

    Args:
        ranges: [min, max] for domain
        n_points: Number of points along each dimension (for mesh) or total points (for random)
        mode: 'mesh' or 'random'
        device: Device to create tensors on
        seed: Random seed for random mode

    Returns:
        Tensor of interior points
    """
    torch.manual_seed(seed)

    if mode == 'mesh':
        x_mesh = torch.linspace(ranges[0], ranges[1], steps=n_points)
        y_mesh = torch.linspace(ranges[0], ranges[1], steps=n_points)
        X, Y = torch.meshgrid(x_mesh, y_mesh, indexing="ij")
        x_i = torch.stack([X.reshape(-1, ), Y.reshape(-1, )]).permute(1, 0)
    elif mode == 'random':
        x_i = torch.rand((n_points ** 2, 2)) * (ranges[1] - ranges[0]) + ranges[0]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return x_i.to(device)


# ============= PDE Problem Registry =============

PDE_PROBLEMS = {
    '2d_poisson': poisson_2d_highfreq,
    '2d_poisson_multiscale': poisson_2d_multiscale,
    '1d_poisson': poisson_1d_solution,
    '1d_poisson_highfreq': poisson_1d_highfreq,
    '1d_heat': heat_1d_solution,
    'burgers': burgers_solution_simple,
    '2d_helmholtz': helmholtz_2d_solution,
}


def get_pde_problem(name, **kwargs):
    """
    Get PDE problem by name.

    Args:
        name: Name of PDE problem
        **kwargs: Additional arguments to pass to the problem function

    Returns:
        Tuple of (solution_func, source_func, gradient_func)
    """
    if name not in PDE_PROBLEMS:
        raise ValueError(f"Unknown PDE problem: {name}. Available: {list(PDE_PROBLEMS.keys())}")

    problem_func = PDE_PROBLEMS[name]
    return problem_func(**kwargs)