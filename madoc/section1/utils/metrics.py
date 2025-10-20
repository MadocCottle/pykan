"""Evaluation metrics for model performance assessment"""
import torch
import torch.nn as nn
import numpy as np


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