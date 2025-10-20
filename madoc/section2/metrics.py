"""
Metrics for evaluating PDE solutions.

This module provides:
- MSE error computation on dense test sets
- H1 norm and semi-norm calculations
- Sobolev norm computations
- PDE residual calculations
"""

import torch
from torch import autograd
import numpy as np


def batch_jacobian(func, x, create_graph=False):
    """
    Compute Jacobian for batch of inputs.

    Args:
        func: Function to differentiate
        x: Input tensor of shape (batch, n_vars)
        create_graph: Whether to create computation graph for higher derivatives

    Returns:
        Jacobian tensor of shape (batch, n_outputs, n_vars)
    """
    def _func_sum(x):
        return func(x).sum(dim=0)
    return autograd.functional.jacobian(_func_sum, x, create_graph=create_graph).permute(1, 0, 2)


def compute_mse_error(model, x_test, y_test):
    """
    Compute MSE error on test set.

    Args:
        model: Neural network model
        x_test: Test inputs (batch, n_vars)
        y_test: True outputs (batch, n_outputs)

    Returns:
        MSE error as scalar tensor
    """
    with torch.no_grad():
        y_pred = model(x_test)
        mse = torch.mean((y_pred - y_test) ** 2)
    return mse


def compute_mae_error(model, x_test, y_test):
    """
    Compute Mean Absolute Error on test set.

    Args:
        model: Neural network model
        x_test: Test inputs (batch, n_vars)
        y_test: True outputs (batch, n_outputs)

    Returns:
        MAE as scalar tensor
    """
    with torch.no_grad():
        y_pred = model(x_test)
        mae = torch.mean(torch.abs(y_pred - y_test))
    return mae


def compute_max_error(model, x_test, y_test):
    """
    Compute maximum absolute error (L-infinity norm).

    Args:
        model: Neural network model
        x_test: Test inputs (batch, n_vars)
        y_test: True outputs (batch, n_outputs)

    Returns:
        Maximum error as scalar tensor
    """
    with torch.no_grad():
        y_pred = model(x_test)
        max_err = torch.max(torch.abs(y_pred - y_test))
    return max_err


def compute_gradient(model, x, create_graph=False):
    """
    Compute gradient of model output with respect to inputs.

    Args:
        model: Neural network model
        x: Input tensor (batch, n_vars)
        create_graph: Whether to create computation graph

    Returns:
        Gradient tensor of shape (batch, n_vars)
    """
    grad_fun = lambda x: batch_jacobian(model, x, create_graph=create_graph)[:, 0, :]
    return grad_fun(x)


def compute_laplacian(model, x):
    """
    Compute Laplacian of model output.

    Args:
        model: Neural network model
        x: Input tensor (batch, n_vars)

    Returns:
        Laplacian tensor of shape (batch, 1)
    """
    # First derivatives
    sol_D1_fun = lambda x: batch_jacobian(model, x, create_graph=True)[:, 0, :]
    sol_D1 = sol_D1_fun(x)

    # Second derivatives
    sol_D2 = batch_jacobian(sol_D1_fun, x, create_graph=True)[:, :, :]

    # Laplacian is trace of Hessian
    lap = torch.sum(torch.diagonal(sol_D2, dim1=1, dim2=2), dim=1, keepdim=True)

    return lap


def compute_h1_seminorm_squared(model, x, true_gradient=None):
    """
    Compute H1 semi-norm squared: ||∇u||^2_{L2}

    Args:
        model: Neural network model
        x: Input tensor (batch, n_vars)
        true_gradient: Optional true gradient for error computation

    Returns:
        H1 semi-norm squared as scalar tensor
    """
    grad = compute_gradient(model, x, create_graph=False)

    if true_gradient is not None:
        # Error in gradient
        grad_error = grad - true_gradient
        seminorm_sq = torch.mean(torch.sum(grad_error ** 2, dim=1))
    else:
        # Just the semi-norm of the solution
        seminorm_sq = torch.mean(torch.sum(grad ** 2, dim=1))

    return seminorm_sq


def compute_h1_norm_squared(model, x, y_true=None, true_gradient=None):
    """
    Compute H1 norm squared: ||u||^2_{L2} + ||∇u||^2_{L2}

    Args:
        model: Neural network model
        x: Input tensor (batch, n_vars)
        y_true: True solution values (for error computation)
        true_gradient: True gradient values (for error computation)

    Returns:
        H1 norm squared as scalar tensor
    """
    # L2 norm term
    if y_true is not None:
        y_pred = model(x)
        l2_term = torch.mean((y_pred - y_true) ** 2)
    else:
        y_pred = model(x)
        l2_term = torch.mean(y_pred ** 2)

    # H1 semi-norm term (gradient)
    h1_semi = compute_h1_seminorm_squared(model, x, true_gradient)

    return l2_term + h1_semi


def compute_h1_seminorm(model, x, true_gradient=None):
    """
    Compute H1 semi-norm: sqrt(||∇u||^2_{L2})

    Args:
        model: Neural network model
        x: Input tensor (batch, n_vars)
        true_gradient: Optional true gradient for error computation

    Returns:
        H1 semi-norm as scalar tensor
    """
    return torch.sqrt(compute_h1_seminorm_squared(model, x, true_gradient))


def compute_h1_norm(model, x, y_true=None, true_gradient=None):
    """
    Compute H1 norm: sqrt(||u||^2_{L2} + ||∇u||^2_{L2})

    Args:
        model: Neural network model
        x: Input tensor (batch, n_vars)
        y_true: True solution values
        true_gradient: True gradient values

    Returns:
        H1 norm as scalar tensor
    """
    return torch.sqrt(compute_h1_norm_squared(model, x, y_true, true_gradient))


def compute_relative_l2_error(model, x_test, y_test):
    """
    Compute relative L2 error: ||u_pred - u_true||_L2 / ||u_true||_L2

    Args:
        model: Neural network model
        x_test: Test inputs
        y_test: True outputs

    Returns:
        Relative L2 error as scalar tensor
    """
    with torch.no_grad():
        y_pred = model(x_test)
        error_norm = torch.sqrt(torch.mean((y_pred - y_test) ** 2))
        true_norm = torch.sqrt(torch.mean(y_test ** 2))
        rel_error = error_norm / (true_norm + 1e-10)  # Add small epsilon to avoid division by zero
    return rel_error


def compute_pde_residual_poisson_2d(model, x, source_fun):
    """
    Compute PDE residual for 2D Poisson equation: ∇²u - f = 0

    Args:
        model: Neural network model
        x: Input points (batch, 2)
        source_fun: Source function f(x)

    Returns:
        Mean squared residual
    """
    lap = compute_laplacian(model, x)
    source = source_fun(x)
    residual = torch.mean((lap - source) ** 2)
    return residual


def create_dense_test_set(ranges, n_points_per_dim, device='cpu'):
    """
    Create a dense mesh grid for testing.

    Args:
        ranges: List [min, max] or array of shape (n_dims, 2)
        n_points_per_dim: Number of points per dimension
        device: Device to create tensors on

    Returns:
        Tensor of shape (n_points_per_dim^n_dims, n_dims)
    """
    if isinstance(ranges, list) and len(ranges) == 2:
        # Same range for all dimensions - need to infer n_dims from context
        # This will be used for 1D or with explicit dimension count
        ranges = np.array(ranges)
        n_dims = 1
    else:
        ranges = np.array(ranges)
        if ranges.ndim == 1:
            n_dims = 1
            ranges = ranges.reshape(1, 2)
        else:
            n_dims = ranges.shape[0]

    # Create mesh grid
    grids = []
    for i in range(n_dims):
        grids.append(torch.linspace(ranges[i, 0], ranges[i, 1], n_points_per_dim))

    if n_dims == 1:
        x = grids[0].reshape(-1, 1)
    else:
        mesh = torch.meshgrid(*grids, indexing="ij")
        x = torch.stack([m.reshape(-1) for m in mesh], dim=1)

    return x.to(device)


class MetricsTracker:
    """
    Track multiple metrics during training.
    """

    def __init__(self, model, test_dataset, solution_func=None, gradient_func=None, source_func=None):
        """
        Initialize metrics tracker.

        Args:
            model: Neural network model
            test_dataset: Dictionary with 'test_input' and 'test_label'
            solution_func: Ground truth solution function (optional)
            gradient_func: Ground truth gradient function (optional)
            source_func: Source function for PDE residual (optional)
        """
        self.model = model
        self.test_dataset = test_dataset
        self.solution_func = solution_func
        self.gradient_func = gradient_func
        self.source_func = source_func

        self.metrics_history = {
            'mse_error': [],
            'mae_error': [],
            'max_error': [],
            'relative_l2_error': [],
            'h1_seminorm': [],
            'h1_norm': [],
        }

        if source_func is not None:
            self.metrics_history['pde_residual'] = []

    def compute_all_metrics(self):
        """
        Compute all available metrics.

        Returns:
            Dictionary of metric values
        """
        x_test = self.test_dataset['test_input']
        y_test = self.test_dataset['test_label']

        metrics = {}

        # Basic errors
        metrics['mse_error'] = compute_mse_error(self.model, x_test, y_test).item()
        metrics['mae_error'] = compute_mae_error(self.model, x_test, y_test).item()
        metrics['max_error'] = compute_max_error(self.model, x_test, y_test).item()
        metrics['relative_l2_error'] = compute_relative_l2_error(self.model, x_test, y_test).item()

        # H1 metrics
        true_grad = self.gradient_func(x_test) if self.gradient_func is not None else None
        metrics['h1_seminorm'] = compute_h1_seminorm(self.model, x_test, true_grad).item()
        metrics['h1_norm'] = compute_h1_norm(self.model, x_test, y_test, true_grad).item()

        # PDE residual
        if self.source_func is not None:
            metrics['pde_residual'] = compute_pde_residual_poisson_2d(self.model, x_test, self.source_func).item()

        return metrics

    def log_metrics(self):
        """
        Compute and log all metrics to history.
        """
        metrics = self.compute_all_metrics()

        for key, value in metrics.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)

        return metrics

    def get_history(self):
        """
        Get full metrics history.

        Returns:
            Dictionary of metric histories
        """
        return self.metrics_history