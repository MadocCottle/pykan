"""
Reusable optimizer test functions for Section 2.

This module contains functions for testing different KAN optimizers and
adaptive density approaches. These functions are used across section2_1
(optimizer comparison) and section2_2 (adaptive density).
"""

import time
import pandas as pd
import torch
import torch.nn as nn
from copy import deepcopy
from kan import KAN
from .metrics import count_parameters, dense_mse_error_from_dataset

try:
    import torch_levenberg_marquardt as tlm
    HAS_LM = True
except ImportError:
    HAS_LM = False
    print("Warning: torch-levenberg-marquardt not installed. LM optimizer will not be available.")


def detect_kan_threshold(test_losses, patience=2, threshold=0.05):
    """Detect when KAN starts overfitting (test loss increases)

    Args:
        test_losses: List of test losses over training
        patience: Number of consecutive increases needed to confirm overfitting
        threshold: Relative increase in loss to count as degradation (default 5%)

    Returns:
        Index of epoch where threshold was reached (before degradation started)
    """
    if len(test_losses) < patience + 1:
        return len(test_losses) - 1

    best_loss = float('inf')
    worse_count = 0
    best_epoch = 0

    for i, loss in enumerate(test_losses):
        if loss < best_loss:
            best_loss = loss
            best_epoch = i
            worse_count = 0
        elif loss > best_loss * (1 + threshold):
            worse_count += 1
            if worse_count >= patience:
                # Return the epoch where we had the best loss (before degradation)
                return best_epoch
        else:
            worse_count = 0

    # If never degraded significantly, return the epoch with best loss
    return best_epoch


def run_kan_optimizer_tests(datasets, grids, epochs, device, optimizer_name, true_functions=None, dataset_names=None):
    """Run KAN tests with specified optimizer

    Args:
        datasets: List of datasets
        grids: List of grid sizes to test
        epochs: Number of training epochs per grid
        device: Device to run on
        optimizer_name: Name of optimizer ('Adam', 'LBFGS', etc.)
        true_functions: List of true functions for each dataset (required)
        dataset_names: List of descriptive names for each dataset (optional)

    Returns:
        Tuple of (DataFrame, checkpoints_dict, threshold_time) where:
        - DataFrame has columns: dataset_idx, dataset_name, grid_size, epoch, train_loss, test_loss,
                                 dense_mse, total_time, time_per_epoch, num_params, optimizer
        - checkpoints_dict: Dict[dataset_idx -> {'at_threshold': {...}, 'final': {...}}]
        - threshold_time: Average time to reach interpolation threshold across datasets
    """
    print(f"kan_{optimizer_name.lower()}")
    rows = []
    checkpoints = {}
    threshold_times = []  # Track threshold time for each dataset

    # Generate default names if not provided
    if dataset_names is None:
        dataset_names = [f"dataset_{i}" for i in range(len(datasets))]

    for i, dataset in enumerate(datasets):
        train_losses = []
        test_losses = []
        dense_mse_errors = []
        n_var = dataset['train_input'].shape[1]
        true_func = true_functions[i] if true_functions else None
        dataset_name = dataset_names[i]

        dataset_start_time = time.time()
        grid_times = []
        grid_param_counts = []

        for j, grid_size in enumerate(grids):
            grid_start_time = time.time()
            if j == 0:
                model = KAN(width=[n_var, 5, 1], grid=grid_size, k=3, seed=1, device=device)
            else:
                model = model.refine(grid_size)
            num_params = count_parameters(model)
            grid_param_counts.append(num_params)

            # Train with specified optimizer
            train_results = model.fit(dataset, opt=optimizer_name, steps=epochs, log=1)
            train_losses += train_results['train_loss']
            test_losses += train_results['test_loss']

            # Compute dense MSE once at the end of this grid's training
            with torch.no_grad():
                dense_mse_final = dense_mse_error_from_dataset(model, dataset, true_func,
                                                        num_samples=10000, device=device)
            # Store the same final dense MSE for all epochs in this grid
            for _ in range(epochs):
                dense_mse_errors.append(dense_mse_final)

            grid_time = time.time() - grid_start_time
            grid_times.append(grid_time)
            print(f"  Dataset {i} ({dataset_name}), grid {grid_size}: {grid_time:.2f}s total, {grid_time/epochs:.3f}s/epoch, {num_params} params")

        total_dataset_time = time.time() - dataset_start_time

        # Detect interpolation threshold based on test loss
        threshold_epoch = detect_kan_threshold(test_losses)

        # Calculate cumulative time up to threshold
        cumulative_time_at_threshold = 0
        cumulative_epochs_at_threshold = 0
        threshold_grid_idx = 0

        for j, grid_size in enumerate(grids):
            if cumulative_epochs_at_threshold + epochs > threshold_epoch:
                # Threshold is within this grid
                epochs_into_this_grid = threshold_epoch - cumulative_epochs_at_threshold
                cumulative_time_at_threshold += grid_times[j] * (epochs_into_this_grid / epochs)
                threshold_grid_idx = j
                break
            else:
                cumulative_time_at_threshold += grid_times[j]
                cumulative_epochs_at_threshold += epochs
        else:
            # Threshold not found, use final grid
            cumulative_time_at_threshold = sum(grid_times)
            threshold_grid_idx = len(grids) - 1

        threshold_times.append(cumulative_time_at_threshold)

        # Compute dense MSE for final model (already computed above)
        final_dense_mse = dense_mse_final

        # Note: We can't perfectly recover the model at threshold_epoch without saving during training
        # So we'll compute threshold metrics on the final model (limitation noted in section1)
        threshold_test_loss = test_losses[threshold_epoch]
        threshold_train_loss = train_losses[threshold_epoch]

        # Compute dense MSE at threshold (use final model as approximation)
        with torch.no_grad():
            threshold_dense_mse = dense_mse_error_from_dataset(model, dataset, true_func,
                                                               num_samples=10000, device=device)

        print(f"    Interpolation threshold detected at epoch {threshold_epoch} (time: {cumulative_time_at_threshold:.2f}s)")
        print(f"    Test loss at threshold: {threshold_test_loss:.6e}")
        print(f"    Dense MSE at threshold: {threshold_dense_mse:.6e} (computed on final model)")

        # Store checkpoints for this dataset
        checkpoints[i] = {
            'at_threshold': {
                'model': deepcopy(model),  # Note: This is final model, not actual threshold model
                'epoch': threshold_epoch,
                'time': cumulative_time_at_threshold,
                'train_loss': threshold_train_loss,
                'test_loss': threshold_test_loss,
                'dense_mse': threshold_dense_mse,
                'grid_idx': threshold_grid_idx,
                'grid_size': grids[threshold_grid_idx],
                'num_params': grid_param_counts[threshold_grid_idx],
                'optimizer': optimizer_name
            },
            'final': {
                'model': model,
                'epoch': len(train_losses) - 1,
                'time': total_dataset_time,
                'train_loss': train_losses[-1],
                'test_loss': test_losses[-1],
                'dense_mse': final_dense_mse,
                'grid_size': grids[-1],
                'num_params': grid_param_counts[-1],
                'optimizer': optimizer_name
            }
        }

        # Create rows for each grid and epoch
        for j, grid_size in enumerate(grids):
            for epoch_in_grid in range(epochs):
                global_epoch = j * epochs + epoch_in_grid
                rows.append({
                    'dataset_idx': i,
                    'dataset_name': dataset_name,
                    'grid_size': grid_size,
                    'epoch': global_epoch,
                    'train_loss': train_losses[global_epoch],
                    'test_loss': test_losses[global_epoch],
                    'dense_mse': dense_mse_errors[global_epoch],
                    'total_time': grid_times[j],
                    'time_per_epoch': grid_times[j] / epochs,
                    'num_params': grid_param_counts[j],
                    'optimizer': optimizer_name
                })

        print(f"  Dataset {i} ({dataset_name}) complete: {total_dataset_time:.2f}s total")

    # Calculate average threshold time across datasets
    avg_threshold_time = sum(threshold_times) / len(threshold_times) if threshold_times else 0
    print(f"\n  Average {optimizer_name} interpolation threshold time: {avg_threshold_time:.2f}s")

    return pd.DataFrame(rows), checkpoints, avg_threshold_time


def run_kan_lm_tests(datasets, grids, epochs, device, true_functions=None, dataset_names=None):
    """Run KAN tests with torch-levenberg-marquardt optimizer

    Args:
        datasets: List of datasets
        grids: List of grid sizes to test
        epochs: Number of training epochs per grid
        device: Device to run on
        true_functions: List of true functions for each dataset (required)
        dataset_names: List of descriptive names for each dataset (optional)

    Returns:
        Tuple of (DataFrame, checkpoints_dict, threshold_time) where:
        - DataFrame has columns: dataset_idx, dataset_name, grid_size, epoch, train_loss, test_loss,
                                 dense_mse, total_time, time_per_epoch, num_params, optimizer
        - checkpoints_dict: Dict[dataset_idx -> {'at_threshold': {...}, 'final': {...}}]
        - threshold_time: Average time to reach interpolation threshold across datasets
    """
    if not HAS_LM:
        raise ImportError("torch-levenberg-marquardt is not installed. Please install it to use LM optimizer.")

    print("kan_lm")
    rows = []
    checkpoints = {}
    threshold_times = []  # Track threshold time for each dataset

    # Generate default names if not provided
    if dataset_names is None:
        dataset_names = [f"dataset_{i}" for i in range(len(datasets))]

    for i, dataset in enumerate(datasets):
        train_losses = []
        test_losses = []
        dense_mse_errors = []
        n_var = dataset['train_input'].shape[1]
        true_func = true_functions[i] if true_functions else None
        dataset_name = dataset_names[i]

        dataset_start_time = time.time()
        grid_times = []
        grid_param_counts = []

        for j, grid_size in enumerate(grids):
            grid_start_time = time.time()
            if j == 0:
                model = KAN(width=[n_var, 5, 1], grid=grid_size, k=3, seed=1, device=device)
            else:
                model = model.refine(grid_size)
            num_params = count_parameters(model)
            grid_param_counts.append(num_params)

            # Train with torch-levenberg-marquardt optimizer
            lm_module = tlm.training.LevenbergMarquardtModule(
                model=model,
                loss_fn=tlm.loss.MSELoss(),
                learning_rate=1.0,
                attempts_per_step=10,
                solve_method='qr'
            )

            for epoch in range(epochs):
                # Perform one training step with LM
                inputs = dataset['train_input']
                targets = dataset['train_label']

                outputs, loss, stop_training, logs = lm_module.training_step(inputs, targets)

                # Track losses
                with torch.no_grad():
                    train_pred = model(dataset['train_input'])
                    test_pred = model(dataset['test_input'])
                    criterion = nn.MSELoss()
                    train_loss = criterion(train_pred, dataset['train_label']).item()
                    test_loss = criterion(test_pred, dataset['test_label']).item()
                    train_losses.append(train_loss)
                    test_losses.append(test_loss)

            # Compute dense MSE once at the end of this grid's training
            with torch.no_grad():
                dense_mse_final = dense_mse_error_from_dataset(model, dataset, true_func,
                                                        num_samples=10000, device=device)
            # Store the same final dense MSE for all epochs in this grid
            for _ in range(epochs):
                dense_mse_errors.append(dense_mse_final)

            grid_time = time.time() - grid_start_time
            grid_times.append(grid_time)
            print(f"  Dataset {i} ({dataset_name}), grid {grid_size}: {grid_time:.2f}s total, {grid_time/epochs:.3f}s/epoch, {num_params} params")

        total_dataset_time = time.time() - dataset_start_time

        # Detect interpolation threshold based on test loss
        threshold_epoch = detect_kan_threshold(test_losses)

        # Calculate cumulative time up to threshold
        cumulative_time_at_threshold = 0
        cumulative_epochs_at_threshold = 0
        threshold_grid_idx = 0

        for j, grid_size in enumerate(grids):
            if cumulative_epochs_at_threshold + epochs > threshold_epoch:
                # Threshold is within this grid
                epochs_into_this_grid = threshold_epoch - cumulative_epochs_at_threshold
                cumulative_time_at_threshold += grid_times[j] * (epochs_into_this_grid / epochs)
                threshold_grid_idx = j
                break
            else:
                cumulative_time_at_threshold += grid_times[j]
                cumulative_epochs_at_threshold += epochs
        else:
            # Threshold not found, use final grid
            cumulative_time_at_threshold = sum(grid_times)
            threshold_grid_idx = len(grids) - 1

        threshold_times.append(cumulative_time_at_threshold)

        # Compute dense MSE for final model (already computed above)
        final_dense_mse = dense_mse_final

        # Note: We can't perfectly recover the model at threshold_epoch without saving during training
        # So we'll compute threshold metrics on the final model (limitation noted in section1)
        threshold_test_loss = test_losses[threshold_epoch]
        threshold_train_loss = train_losses[threshold_epoch]

        # Compute dense MSE at threshold (use final model as approximation)
        with torch.no_grad():
            threshold_dense_mse = dense_mse_error_from_dataset(model, dataset, true_func,
                                                               num_samples=10000, device=device)

        print(f"    Interpolation threshold detected at epoch {threshold_epoch} (time: {cumulative_time_at_threshold:.2f}s)")
        print(f"    Test loss at threshold: {threshold_test_loss:.6e}")
        print(f"    Dense MSE at threshold: {threshold_dense_mse:.6e} (computed on final model)")

        # Store checkpoints for this dataset
        checkpoints[i] = {
            'at_threshold': {
                'model': deepcopy(model),  # Note: This is final model, not actual threshold model
                'epoch': threshold_epoch,
                'time': cumulative_time_at_threshold,
                'train_loss': threshold_train_loss,
                'test_loss': threshold_test_loss,
                'dense_mse': threshold_dense_mse,
                'grid_idx': threshold_grid_idx,
                'grid_size': grids[threshold_grid_idx],
                'num_params': grid_param_counts[threshold_grid_idx],
                'optimizer': 'LM'
            },
            'final': {
                'model': model,
                'epoch': len(train_losses) - 1,
                'time': total_dataset_time,
                'train_loss': train_losses[-1],
                'test_loss': test_losses[-1],
                'dense_mse': final_dense_mse,
                'grid_size': grids[-1],
                'num_params': grid_param_counts[-1],
                'optimizer': 'LM'
            }
        }

        # Create rows for each grid and epoch
        for j, grid_size in enumerate(grids):
            for epoch_in_grid in range(epochs):
                global_epoch = j * epochs + epoch_in_grid
                rows.append({
                    'dataset_idx': i,
                    'dataset_name': dataset_name,
                    'grid_size': grid_size,
                    'epoch': global_epoch,
                    'train_loss': train_losses[global_epoch],
                    'test_loss': test_losses[global_epoch],
                    'dense_mse': dense_mse_errors[global_epoch],
                    'total_time': grid_times[j],
                    'time_per_epoch': grid_times[j] / epochs,
                    'num_params': grid_param_counts[j],
                    'optimizer': 'LM'
                })

        print(f"  Dataset {i} ({dataset_name}) complete: {total_dataset_time:.2f}s total")

    # Calculate average threshold time across datasets
    avg_threshold_time = sum(threshold_times) / len(threshold_times) if threshold_times else 0
    print(f"\n  Average LM interpolation threshold time: {avg_threshold_time:.2f}s")

    return pd.DataFrame(rows), checkpoints, avg_threshold_time


def run_kan_adaptive_density_test(datasets, grids, epochs, device, use_regular_refine=False,
                                   attribution_threshold=1e-2, true_functions=None, dataset_names=None):
    """
    Run KAN tests with adaptive density based on attribution scores.

    Args:
        datasets: List of datasets
        grids: List of grid sizes to test
        epochs: Number of training epochs per grid
        device: Device to run on
        use_regular_refine: If True, use regular refinement; if False, use adaptive only
        attribution_threshold: Threshold for attribution scores
        true_functions: List of true functions for each dataset (required)
        dataset_names: List of descriptive names for each dataset (optional)

    Returns:
        Tuple of (DataFrame, checkpoints_dict, threshold_time) where:
        - DataFrame contains training metrics
        - checkpoints_dict: Dict[dataset_idx -> {'at_threshold': {...}, 'final': {...}}]
        - threshold_time: Average time to reach interpolation threshold across datasets
    """
    approach = "adaptive+regular" if use_regular_refine else "adaptive_only"
    print(f"kan_adaptive_density ({approach})")
    rows = []
    checkpoints = {}
    threshold_times = []  # Track threshold time for each dataset

    # Generate default names if not provided
    if dataset_names is None:
        dataset_names = [f"dataset_{i}" for i in range(len(datasets))]

    for i, dataset in enumerate(datasets):
        train_losses = []
        test_losses = []
        dense_mse_errors = []
        n_var = dataset['train_input'].shape[1]
        true_func = true_functions[i] if true_functions else None
        dataset_name = dataset_names[i]

        dataset_start_time = time.time()
        grid_times = []
        grid_param_counts = []
        attribution_stats = []

        for j, grid_size in enumerate(grids):
            grid_start_time = time.time()

            if j == 0:
                # Initial model
                model = KAN(width=[n_var, 5, 1], grid=grid_size, k=3, seed=1, device=device)
            else:
                # Apply adaptive or combined densification
                if use_regular_refine:
                    # Test 2: Adaptive + Regular
                    # First do regular refinement
                    model = model.refine(grid_size)
                    # Then check if we should do additional densification on important nodes
                    # (In practice, since all neurons share grid size, this is already done)
                    model, stats = adaptive_densify_model(model, dataset,
                                                         threshold=attribution_threshold,
                                                         new_grid_size=None,  # Already refined
                                                         device=device)
                else:
                    # Test 1: Adaptive only (as alternative to regular)
                    # Only refine if there are important neurons
                    model, stats = adaptive_densify_model(model, dataset,
                                                         threshold=attribution_threshold,
                                                         new_grid_size=grid_size,
                                                         device=device)

                attribution_stats.append(stats if 'stats' in locals() else {})

            num_params = count_parameters(model)
            grid_param_counts.append(num_params)

            # Train with Adam optimizer
            train_results = model.fit(dataset, opt="Adam", steps=epochs, log=1)
            train_losses += train_results['train_loss']
            test_losses += train_results['test_loss']

            # Compute dense MSE once at the end of this grid's training
            with torch.no_grad():
                dense_mse_final = dense_mse_error_from_dataset(model, dataset, true_func,
                                                        num_samples=10000, device=device)
            # Store the same final dense MSE for all epochs in this grid
            for _ in range(epochs):
                dense_mse_errors.append(dense_mse_final)

            grid_time = time.time() - grid_start_time
            grid_times.append(grid_time)
            print(f"  Dataset {i} ({dataset_name}), grid {grid_size}: {grid_time:.2f}s total, "
                  f"{grid_time/epochs:.3f}s/epoch, {num_params} params")

        total_dataset_time = time.time() - dataset_start_time

        # Detect interpolation threshold based on test loss
        threshold_epoch = detect_kan_threshold(test_losses)

        # Calculate cumulative time up to threshold
        cumulative_time_at_threshold = 0
        cumulative_epochs_at_threshold = 0
        threshold_grid_idx = 0

        for j, grid_size in enumerate(grids):
            if cumulative_epochs_at_threshold + epochs > threshold_epoch:
                # Threshold is within this grid
                epochs_into_this_grid = threshold_epoch - cumulative_epochs_at_threshold
                cumulative_time_at_threshold += grid_times[j] * (epochs_into_this_grid / epochs)
                threshold_grid_idx = j
                break
            else:
                cumulative_time_at_threshold += grid_times[j]
                cumulative_epochs_at_threshold += epochs
        else:
            # Threshold not found, use final grid
            cumulative_time_at_threshold = sum(grid_times)
            threshold_grid_idx = len(grids) - 1

        threshold_times.append(cumulative_time_at_threshold)

        # Compute dense MSE for final model (already computed above)
        final_dense_mse = dense_mse_final

        # Threshold metrics
        threshold_test_loss = test_losses[threshold_epoch]
        threshold_train_loss = train_losses[threshold_epoch]

        # Compute dense MSE at threshold (use final model as approximation)
        with torch.no_grad():
            threshold_dense_mse = dense_mse_error_from_dataset(model, dataset, true_func,
                                                               num_samples=10000, device=device)

        print(f"    Interpolation threshold detected at epoch {threshold_epoch} (time: {cumulative_time_at_threshold:.2f}s)")
        print(f"    Test loss at threshold: {threshold_test_loss:.6e}")
        print(f"    Dense MSE at threshold: {threshold_dense_mse:.6e} (computed on final model)")

        # Store checkpoints for this dataset
        checkpoints[i] = {
            'at_threshold': {
                'model': deepcopy(model),
                'epoch': threshold_epoch,
                'time': cumulative_time_at_threshold,
                'train_loss': threshold_train_loss,
                'test_loss': threshold_test_loss,
                'dense_mse': threshold_dense_mse,
                'grid_idx': threshold_grid_idx,
                'grid_size': grids[threshold_grid_idx],
                'num_params': grid_param_counts[threshold_grid_idx],
                'approach': approach
            },
            'final': {
                'model': model,
                'epoch': len(train_losses) - 1,
                'time': total_dataset_time,
                'train_loss': train_losses[-1],
                'test_loss': test_losses[-1],
                'dense_mse': final_dense_mse,
                'grid_size': grids[-1],
                'num_params': grid_param_counts[-1],
                'approach': approach
            }
        }

        # Create rows for each grid and epoch
        for j, grid_size in enumerate(grids):
            for epoch_in_grid in range(epochs):
                global_epoch = j * epochs + epoch_in_grid
                row = {
                    'dataset_idx': i,
                    'dataset_name': dataset_name,
                    'grid_size': grid_size,
                    'epoch': global_epoch,
                    'train_loss': train_losses[global_epoch],
                    'test_loss': test_losses[global_epoch],
                    'dense_mse': dense_mse_errors[global_epoch],
                    'total_time': grid_times[j],
                    'time_per_epoch': grid_times[j] / epochs,
                    'num_params': grid_param_counts[j],
                    'approach': approach,
                    'attribution_threshold': attribution_threshold
                }

                # Add attribution stats if available
                if j > 0 and j - 1 < len(attribution_stats):
                    stats = attribution_stats[j - 1]
                    row['important_neurons'] = stats.get('important_neurons', None)
                    row['total_neurons'] = stats.get('total_neurons', None)

                rows.append(row)

        print(f"  Dataset {i} ({dataset_name}) complete: {total_dataset_time:.2f}s total")

    # Calculate average threshold time across datasets
    avg_threshold_time = sum(threshold_times) / len(threshold_times) if threshold_times else 0
    print(f"\n  Average {approach} interpolation threshold time: {avg_threshold_time:.2f}s")

    return pd.DataFrame(rows), checkpoints, avg_threshold_time


def run_kan_baseline_test(datasets, grids, epochs, device, true_functions=None, dataset_names=None):
    """Run baseline KAN tests with regular refinement for comparison

    Returns:
        Tuple of (DataFrame, checkpoints_dict, threshold_time) where:
        - DataFrame contains training metrics
        - checkpoints_dict: Dict[dataset_idx -> {'at_threshold': {...}, 'final': {...}}]
        - threshold_time: Average time to reach interpolation threshold across datasets
    """
    print("kan_baseline (regular refinement)")
    rows = []
    checkpoints = {}
    threshold_times = []  # Track threshold time for each dataset

    if dataset_names is None:
        dataset_names = [f"dataset_{i}" for i in range(len(datasets))]

    for i, dataset in enumerate(datasets):
        train_losses = []
        test_losses = []
        dense_mse_errors = []
        n_var = dataset['train_input'].shape[1]
        true_func = true_functions[i] if true_functions else None
        dataset_name = dataset_names[i]

        dataset_start_time = time.time()
        grid_times = []
        grid_param_counts = []

        for j, grid_size in enumerate(grids):
            grid_start_time = time.time()
            if j == 0:
                model = KAN(width=[n_var, 5, 1], grid=grid_size, k=3, seed=1, device=device)
            else:
                model = model.refine(grid_size)
            num_params = count_parameters(model)
            grid_param_counts.append(num_params)

            train_results = model.fit(dataset, opt="Adam", steps=epochs, log=1)
            train_losses += train_results['train_loss']
            test_losses += train_results['test_loss']

            # Compute dense MSE once at the end of this grid's training
            with torch.no_grad():
                dense_mse_final = dense_mse_error_from_dataset(model, dataset, true_func,
                                                        num_samples=10000, device=device)
            # Store the same final dense MSE for all epochs in this grid
            for _ in range(epochs):
                dense_mse_errors.append(dense_mse_final)

            grid_time = time.time() - grid_start_time
            grid_times.append(grid_time)
            print(f"  Dataset {i} ({dataset_name}), grid {grid_size}: {grid_time:.2f}s total, "
                  f"{grid_time/epochs:.3f}s/epoch, {num_params} params")

        total_dataset_time = time.time() - dataset_start_time

        # Detect interpolation threshold based on test loss
        threshold_epoch = detect_kan_threshold(test_losses)

        # Calculate cumulative time up to threshold
        cumulative_time_at_threshold = 0
        cumulative_epochs_at_threshold = 0
        threshold_grid_idx = 0

        for j, grid_size in enumerate(grids):
            if cumulative_epochs_at_threshold + epochs > threshold_epoch:
                # Threshold is within this grid
                epochs_into_this_grid = threshold_epoch - cumulative_epochs_at_threshold
                cumulative_time_at_threshold += grid_times[j] * (epochs_into_this_grid / epochs)
                threshold_grid_idx = j
                break
            else:
                cumulative_time_at_threshold += grid_times[j]
                cumulative_epochs_at_threshold += epochs
        else:
            # Threshold not found, use final grid
            cumulative_time_at_threshold = sum(grid_times)
            threshold_grid_idx = len(grids) - 1

        threshold_times.append(cumulative_time_at_threshold)

        # Compute dense MSE for final model (already computed above)
        final_dense_mse = dense_mse_final

        # Threshold metrics
        threshold_test_loss = test_losses[threshold_epoch]
        threshold_train_loss = train_losses[threshold_epoch]

        # Compute dense MSE at threshold (use final model as approximation)
        with torch.no_grad():
            threshold_dense_mse = dense_mse_error_from_dataset(model, dataset, true_func,
                                                               num_samples=10000, device=device)

        print(f"    Interpolation threshold detected at epoch {threshold_epoch} (time: {cumulative_time_at_threshold:.2f}s)")
        print(f"    Test loss at threshold: {threshold_test_loss:.6e}")
        print(f"    Dense MSE at threshold: {threshold_dense_mse:.6e} (computed on final model)")

        # Store checkpoints for this dataset
        checkpoints[i] = {
            'at_threshold': {
                'model': deepcopy(model),
                'epoch': threshold_epoch,
                'time': cumulative_time_at_threshold,
                'train_loss': threshold_train_loss,
                'test_loss': threshold_test_loss,
                'dense_mse': threshold_dense_mse,
                'grid_idx': threshold_grid_idx,
                'grid_size': grids[threshold_grid_idx],
                'num_params': grid_param_counts[threshold_grid_idx],
                'approach': 'baseline'
            },
            'final': {
                'model': model,
                'epoch': len(train_losses) - 1,
                'time': total_dataset_time,
                'train_loss': train_losses[-1],
                'test_loss': test_losses[-1],
                'dense_mse': final_dense_mse,
                'grid_size': grids[-1],
                'num_params': grid_param_counts[-1],
                'approach': 'baseline'
            }
        }

        for j, grid_size in enumerate(grids):
            for epoch_in_grid in range(epochs):
                global_epoch = j * epochs + epoch_in_grid
                rows.append({
                    'dataset_idx': i,
                    'dataset_name': dataset_name,
                    'grid_size': grid_size,
                    'epoch': global_epoch,
                    'train_loss': train_losses[global_epoch],
                    'test_loss': test_losses[global_epoch],
                    'dense_mse': dense_mse_errors[global_epoch],
                    'total_time': grid_times[j],
                    'time_per_epoch': grid_times[j] / epochs,
                    'num_params': grid_param_counts[j],
                    'approach': 'baseline',
                    'attribution_threshold': None
                })

        print(f"  Dataset {i} ({dataset_name}) complete: {total_dataset_time:.2f}s total")

    # Calculate average threshold time across datasets
    avg_threshold_time = sum(threshold_times) / len(threshold_times) if threshold_times else 0
    print(f"\n  Average baseline interpolation threshold time: {avg_threshold_time:.2f}s")

    return pd.DataFrame(rows), checkpoints, avg_threshold_time


def adaptive_densify_model(model, dataset, threshold=1e-2, new_grid_size=None, device='cpu'):
    """
    Selectively densify the model based on attribution scores.

    Args:
        model: The KAN model
        dataset: Dataset to use for computing activations
        threshold: Attribution score threshold
        new_grid_size: New grid size (number of intervals)
        device: Device to run on

    Returns:
        model: The refined model (note: due to pykan limitations, all neurons get refined)
        stats: Dictionary with densification statistics
    """
    # Get activations and attribution scores
    if model.acts is None:
        model.get_act(dataset)
    model.attribute()

    # Identify important neurons per layer
    important_neurons_per_layer = []
    total_important = 0
    total_neurons = 0

    for layer_idx in range(model.depth):
        node_scores = model.node_scores[layer_idx]
        important_mask = node_scores > threshold
        important_neurons_per_layer.append(important_mask)
        total_important += important_mask.sum().item()
        total_neurons += len(node_scores)

    stats = {
        'total_neurons': total_neurons,
        'important_neurons': total_important,
        'threshold': threshold,
        'new_grid_size': new_grid_size
    }

    # Due to pykan architecture, all neurons in a layer must have the same grid size
    # So we apply regular refinement but only when there are important neurons
    if total_important > 0 and new_grid_size is not None:
        model = model.refine(new_grid_size)

    return model, stats


def print_optimizer_summary(all_results, dataset_names):
    """
    Print a summary table of best dense MSE for each optimizer/approach.

    Args:
        all_results: Dictionary mapping optimizer/approach names to DataFrames
        dataset_names: List of dataset names
    """
    print("\n" + "="*80)
    print("BEST DENSE MSE SUMMARY")
    print("="*80)

    # Header
    header = f"{'Dataset':<25}"
    for opt_name in all_results.keys():
        header += f"{opt_name.upper():<15}"
    print(header)
    print("-"*80)

    # For each dataset
    num_datasets = len(dataset_names)
    for dataset_idx in range(num_datasets):
        dataset_name = dataset_names[dataset_idx]
        row = f"{dataset_name:<25}"

        for opt_name, df in all_results.items():
            # Get final dense_mse for this dataset
            df_dataset = df[df['dataset_idx'] == dataset_idx]

            if len(df_dataset) > 0:
                # Get minimum dense_mse across all configurations
                min_mse = df_dataset['dense_mse'].min()
                row += f"{min_mse:<15.6e}"
            else:
                row += f"{'N/A':<15}"

        print(row)

    print("="*80)
