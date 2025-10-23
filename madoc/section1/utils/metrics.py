"""Evaluation metrics for model performance assessment"""
import torch
import torch.nn as nn
import numpy as np


def count_parameters(model):
    """
    Count the total number of trainable parameters in a model.
    Works with both standard PyTorch models (MLP, SIREN) and KAN models.

    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def dense_mse_error(model, true_function, n_var=1, ranges=[-1, 1],
                    num_samples=10000, device='cpu'):
    """
    Compute MSE error by densely sampling from the true function.

    This metric samples from the function being fit much more densely than
    the MSE loss used during training, providing a better idea of how the
    model is performing across the entire input domain.

    Args:
    -----
        model : nn.Module or KAN
            The trained model to evaluate
        true_function : callable
            The ground truth function f(x) that the model is trying to approximate
        n_var : int
            Number of input variables. Default: 1
        ranges : list or np.array; shape (2,) or (n_var, 2)
            The range of input variables for sampling. Default: [-1, 1]
        num_samples : int
            Number of dense samples to evaluate. Default: 10000
        device : str
            Device to perform computation on. Default: 'cpu'

    Returns:
    --------
        mse_error : float
            The mean squared error computed over dense samples

    Example:
    --------
    >>> from kan import KAN, create_dataset
    >>> f = lambda x: torch.sin(2 * torch.pi * x)
    >>> dataset = create_dataset(f, n_var=1, train_num=100)
    >>> model = KAN(width=[1, 5, 1], grid=5, k=3, device='cpu')
    >>> model.fit(dataset, opt="LBFGS", steps=20)
    >>> error = dense_mse_error(model, f, n_var=1, num_samples=10000)
    >>> print(f"Dense MSE Error: {error:.6f}")
    """
    # Parse ranges
    if isinstance(ranges, list) and len(ranges) == 2 and not isinstance(ranges[0], (list, np.ndarray)):
        # Single range for all variables
        ranges = np.array([ranges for _ in range(n_var)])
    else:
        ranges = np.array(ranges)
        if ranges.shape == (2,):
            ranges = np.array([ranges for _ in range(n_var)])

    # Generate dense samples
    if n_var == 1:
        x = torch.linspace(ranges[0, 0], ranges[0, 1], num_samples, device=device).reshape(-1, 1)
    else:
        # For multiple variables, use grid or random sampling
        # Using random sampling to avoid curse of dimensionality
        x = torch.zeros(num_samples, n_var, device=device)
        for i in range(n_var):
            x[:, i] = torch.rand(num_samples, device=device) * (ranges[i, 1] - ranges[i, 0]) + ranges[i, 0]

    # Compute predictions and true values
    model.eval()
    with torch.no_grad():
        y_pred = model(x)
        y_true = true_function(x)

        # Ensure shapes match
        if y_pred.shape != y_true.shape:
            if y_pred.dim() == 1:
                y_pred = y_pred.reshape(-1, 1)
            if y_true.dim() == 1:
                y_true = y_true.reshape(-1, 1)

        # Compute MSE
        mse = nn.MSELoss()(y_pred, y_true).item()

    return mse


def dense_mse_error_from_dataset(model, dataset, true_function,
                                  num_samples=10000, device='cpu'):
    """
    Compute dense MSE error using ranges inferred from dataset.

    This is a convenience wrapper that infers the input ranges from the
    training dataset, then computes the dense MSE error.

    Args:
    -----
        model : nn.Module or KAN
            The trained model to evaluate
        dataset : dict
            Dataset dictionary with 'train_input' key
        true_function : callable
            The ground truth function f(x)
        num_samples : int
            Number of dense samples. Default: 10000
        device : str
            Device to perform computation on. Default: 'cpu'

    Returns:
    --------
        mse_error : float
            The mean squared error computed over dense samples

    Example:
    --------
    >>> from kan import KAN, create_dataset
    >>> f = lambda x: torch.sin(2 * torch.pi * x)
    >>> dataset = create_dataset(f, n_var=1, train_num=100)
    >>> model = KAN(width=[1, 5, 1], grid=5, k=3, device='cpu')
    >>> model.fit(dataset, opt="LBFGS", steps=20)
    >>> error = dense_mse_error_from_dataset(model, dataset, f, num_samples=10000)
    >>> print(f"Dense MSE Error: {error:.6f}")
    """
    # Infer n_var and ranges from dataset
    train_input = dataset['train_input']
    n_var = train_input.shape[1]

    # Compute ranges from training data with small buffer
    ranges = []
    for i in range(n_var):
        min_val = train_input[:, i].min().item()
        max_val = train_input[:, i].max().item()
        # Add small buffer (5%) to ensure coverage
        buffer = (max_val - min_val) * 0.05
        ranges.append([min_val - buffer, max_val + buffer])

    return dense_mse_error(model, true_function, n_var=n_var,
                          ranges=ranges, num_samples=num_samples, device=device)


def evaluate_all_models(models, datasets, true_functions, num_samples=10000, device='cpu'):
    """
    Evaluate multiple models on multiple datasets using dense MSE error.

    Args:
    -----
        models : dict or list
            Dictionary or list of trained models
        datasets : list
            List of dataset dictionaries
        true_functions : list
            List of ground truth functions corresponding to each dataset
        num_samples : int
            Number of dense samples per evaluation. Default: 10000
        device : str
            Device to perform computation on. Default: 'cpu'

    Returns:
    --------
        results : dict
            Dictionary mapping model/dataset indices to dense MSE errors

    Example:
    --------
    >>> errors = evaluate_all_models(
    ...     models={0: model1, 1: model2},
    ...     datasets=[dataset1, dataset2],
    ...     true_functions=[f1, f2],
    ...     num_samples=10000
    ... )
    """
    results = {}

    # Handle both dict and list inputs for models
    if isinstance(models, dict):
        model_items = models.items()
    else:
        model_items = enumerate(models)

    for model_idx, model in model_items:
        results[model_idx] = {}
        for data_idx, (dataset, true_func) in enumerate(zip(datasets, true_functions)):
            error = dense_mse_error_from_dataset(
                model, dataset, true_func,
                num_samples=num_samples, device=device
            )
            results[model_idx][data_idx] = error
            print(f"Model {model_idx}, Dataset {data_idx}: Dense MSE = {error:.6e}")

    return results


def linf_error(model, true_function, n_var=1, ranges=[-1, 1],
               num_samples=10000, device='cpu'):
    """
    Compute L∞ (maximum pointwise) error.

    The L∞ norm measures the worst-case error: the maximum absolute difference
    between predicted and true values across the entire domain. This is critical
    for safety-critical applications and reveals localized failures.

    Args:
    -----
        model : nn.Module or KAN
            The trained model to evaluate
        true_function : callable
            The ground truth function f(x) that the model is trying to approximate
        n_var : int
            Number of input variables. Default: 1
        ranges : list or np.array; shape (2,) or (n_var, 2)
            The range of input variables for sampling. Default: [-1, 1]
        num_samples : int
            Number of dense samples to evaluate. Default: 10000
        device : str
            Device to perform computation on. Default: 'cpu'

    Returns:
    --------
        linf_error : float
            The maximum absolute error: max|y_pred - y_true|

    Example:
    --------
    >>> from kan import KAN, create_dataset
    >>> f = lambda x: torch.sin(2 * torch.pi * x)
    >>> dataset = create_dataset(f, n_var=1, train_num=100)
    >>> model = KAN(width=[1, 5, 1], grid=5, k=3, device='cpu')
    >>> model.fit(dataset, opt="LBFGS", steps=20)
    >>> error = linf_error(model, f, n_var=1, num_samples=10000)
    >>> print(f"L∞ Error: {error:.6f}")
    """
    # Parse ranges
    if isinstance(ranges, list) and len(ranges) == 2 and not isinstance(ranges[0], (list, np.ndarray)):
        # Single range for all variables
        ranges = np.array([ranges for _ in range(n_var)])
    else:
        ranges = np.array(ranges)
        if ranges.shape == (2,):
            ranges = np.array([ranges for _ in range(n_var)])

    # Generate dense samples (same as dense_mse_error)
    if n_var == 1:
        x = torch.linspace(ranges[0, 0], ranges[0, 1], num_samples, device=device).reshape(-1, 1)
    else:
        # For multiple variables, use random sampling
        x = torch.zeros(num_samples, n_var, device=device)
        for i in range(n_var):
            x[:, i] = torch.rand(num_samples, device=device) * (ranges[i, 1] - ranges[i, 0]) + ranges[i, 0]

    # Compute predictions and true values
    model.eval()
    with torch.no_grad():
        y_pred = model(x)
        y_true = true_function(x)

        # Ensure shapes match
        if y_pred.shape != y_true.shape:
            if y_pred.dim() == 1:
                y_pred = y_pred.reshape(-1, 1)
            if y_true.dim() == 1:
                y_true = y_true.reshape(-1, 1)

        # Compute L∞ (max absolute error)
        abs_errors = torch.abs(y_pred - y_true)
        linf = torch.max(abs_errors).item()

    return linf


def linf_error_from_dataset(model, dataset, true_function,
                            num_samples=10000, device='cpu'):
    """
    Compute L∞ error using ranges inferred from dataset.

    This is a convenience wrapper that infers the input ranges from the
    training dataset, then computes the L∞ error.

    Args:
    -----
        model : nn.Module or KAN
            The trained model to evaluate
        dataset : dict
            Dataset dictionary with 'train_input' key
        true_function : callable
            The ground truth function f(x)
        num_samples : int
            Number of dense samples. Default: 10000
        device : str
            Device to perform computation on. Default: 'cpu'

    Returns:
    --------
        linf_error : float
            The maximum absolute error

    Example:
    --------
    >>> from kan import KAN, create_dataset
    >>> f = lambda x: torch.sin(2 * torch.pi * x)
    >>> dataset = create_dataset(f, n_var=1, train_num=100)
    >>> model = KAN(width=[1, 5, 1], grid=5, k=3, device='cpu')
    >>> model.fit(dataset, opt="LBFGS", steps=20)
    >>> error = linf_error_from_dataset(model, dataset, f, num_samples=10000)
    >>> print(f"L∞ Error: {error:.6f}")
    """
    # Infer n_var and ranges from dataset
    train_input = dataset['train_input']
    n_var = train_input.shape[1]

    # Compute ranges from training data with small buffer
    ranges = []
    for i in range(n_var):
        min_val = train_input[:, i].min().item()
        max_val = train_input[:, i].max().item()
        # Add small buffer (5%) to ensure coverage
        buffer = (max_val - min_val) * 0.05
        ranges.append([min_val - buffer, max_val + buffer])

    return linf_error(model, true_function, n_var=n_var,
                      ranges=ranges, num_samples=num_samples, device=device)


def h1_seminorm_error(model, true_function, n_var=2, ranges=[-1, 1],
                      num_samples=10000, device='cpu', eps=1e-4):
    """
    Compute H¹ seminorm error: ||∇u - ∇u_true||_L²

    The H¹ seminorm measures the error in the GRADIENT (derivatives) of the solution,
    not just pointwise values. This is critical for PDE problems where equations
    fundamentally involve derivatives (e.g., Poisson: -∇²u = f).

    A model can have low L² error but terrible gradient estimation, which this
    metric reveals.

    Implementation uses finite differences:
        ∂u/∂x_i ≈ (u(x + ε*e_i) - u(x - ε*e_i)) / (2ε)

    Args:
    -----
        model : nn.Module or KAN
            The trained model to evaluate
        true_function : callable
            The ground truth function f(x)
        n_var : int
            Number of input variables (dimensions). Default: 2
        ranges : list or np.array; shape (2,) or (n_var, 2)
            The range of input variables. Default: [-1, 1]
        num_samples : int
            Number of samples to evaluate. Default: 10000
        device : str
            Device to perform computation on. Default: 'cpu'
        eps : float
            Finite difference step size. Default: 1e-4

    Returns:
    --------
        h1_error : float
            The H¹ seminorm: sqrt(∫ |∇u - ∇u_true|² dx)

    Example:
    --------
    >>> from kan import KAN, create_dataset
    >>> f = lambda x: torch.sin(torch.pi*x[:,[0]]) * torch.sin(torch.pi*x[:,[1]])
    >>> dataset = create_dataset(f, n_var=2, train_num=1000)
    >>> model = KAN(width=[2, 5, 1], grid=10, k=3, device='cpu')
    >>> model.fit(dataset, opt="LBFGS", steps=50)
    >>> error = h1_seminorm_error(model, f, n_var=2, num_samples=10000)
    >>> print(f"H¹ Seminorm Error: {error:.6e}")

    Note:
    -----
    For n_var=1, this reduces to:
        sqrt(∫ (du/dx - du_true/dx)² dx)
    For n_var=2 (common for 2D PDEs):
        sqrt(∫ (∂u/∂x - ∂u_true/∂x)² + (∂u/∂y - ∂u_true/∂y)² dx dy)
    """
    # Parse ranges
    if isinstance(ranges, list) and len(ranges) == 2 and not isinstance(ranges[0], (list, np.ndarray)):
        ranges = np.array([ranges for _ in range(n_var)])
    else:
        ranges = np.array(ranges)
        if ranges.shape == (2,):
            ranges = np.array([ranges for _ in range(n_var)])

    # Generate dense samples (interior points, away from boundaries for finite diff)
    if n_var == 1:
        x = torch.linspace(ranges[0, 0] + eps, ranges[0, 1] - eps, num_samples, device=device).reshape(-1, 1)
    else:
        # Random sampling for higher dimensions
        x = torch.zeros(num_samples, n_var, device=device)
        for i in range(n_var):
            # Sample interior (avoid boundaries where finite diff fails)
            x[:, i] = torch.rand(num_samples, device=device) * (ranges[i, 1] - ranges[i, 0] - 2*eps) + (ranges[i, 0] + eps)

    model.eval()
    gradient_squared_error_sum = 0.0

    with torch.no_grad():
        # Compute gradients using central finite differences for each dimension
        for dim in range(n_var):
            # Create perturbation vector
            perturbation = torch.zeros_like(x)
            perturbation[:, dim] = eps

            # Forward differences: u(x + eps*e_i)
            x_plus = x + perturbation
            y_pred_plus = model(x_plus)
            y_true_plus = true_function(x_plus)

            # Backward differences: u(x - eps*e_i)
            x_minus = x - perturbation
            y_pred_minus = model(x_minus)
            y_true_minus = true_function(x_minus)

            # Central difference: (u(x+ε) - u(x-ε)) / (2ε)
            grad_pred = (y_pred_plus - y_pred_minus) / (2 * eps)
            grad_true = (y_true_plus - y_true_minus) / (2 * eps)

            # Ensure shapes match
            if grad_pred.shape != grad_true.shape:
                if grad_pred.dim() == 1:
                    grad_pred = grad_pred.reshape(-1, 1)
                if grad_true.dim() == 1:
                    grad_true = grad_true.reshape(-1, 1)

            # Accumulate squared gradient error for this dimension
            gradient_squared_error_sum += torch.sum((grad_pred - grad_true) ** 2).item()

    # H¹ seminorm = sqrt(mean of sum of squared gradient errors)
    h1_seminorm = np.sqrt(gradient_squared_error_sum / num_samples)

    return h1_seminorm


def h1_seminorm_error_from_dataset(model, dataset, true_function,
                                   num_samples=10000, device='cpu', eps=1e-4):
    """
    Compute H¹ seminorm error using ranges inferred from dataset.

    Args:
    -----
        model : nn.Module or KAN
            The trained model to evaluate
        dataset : dict
            Dataset dictionary with 'train_input' key
        true_function : callable
            The ground truth function f(x)
        num_samples : int
            Number of dense samples. Default: 10000
        device : str
            Device to perform computation on. Default: 'cpu'
        eps : float
            Finite difference step size. Default: 1e-4

    Returns:
    --------
        h1_error : float
            The H¹ seminorm error

    Example:
    --------
    >>> from kan import KAN, create_dataset
    >>> f = lambda x: torch.sin(torch.pi*x[:,[0]]) * torch.sin(torch.pi*x[:,[1]])
    >>> dataset = create_dataset(f, n_var=2, train_num=1000)
    >>> model = KAN(width=[2, 5, 1], grid=10, k=3, device='cpu')
    >>> model.fit(dataset, opt="LBFGS", steps=50)
    >>> error = h1_seminorm_error_from_dataset(model, dataset, f)
    >>> print(f"H¹ Seminorm Error: {error:.6e}")
    """
    # Infer n_var and ranges from dataset
    train_input = dataset['train_input']
    n_var = train_input.shape[1]

    # Compute ranges from training data with small buffer
    ranges = []
    for i in range(n_var):
        min_val = train_input[:, i].min().item()
        max_val = train_input[:, i].max().item()
        # Add buffer (5%) to ensure coverage
        buffer = (max_val - min_val) * 0.05
        ranges.append([min_val - buffer, max_val + buffer])

    return h1_seminorm_error(model, true_function, n_var=n_var,
                            ranges=ranges, num_samples=num_samples,
                            device=device, eps=eps)