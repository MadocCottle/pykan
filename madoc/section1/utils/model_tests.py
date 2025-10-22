"""Model training and testing utilities for Section 1 experiments"""
from kan import *
from . import trad_nn as tnn
from .metrics import dense_mse_error_from_dataset, count_parameters
import time
import pandas as pd

def train_model(model, dataset, epochs, device, true_function):
    """Train any model using LBFGS optimizer and return loss history with timing

    Args:
        model: Model to train
        dataset: Dataset dict with train/test inputs and labels
        epochs: Number of training epochs
        device: Device to run on
        true_function: True function for computing dense MSE (required)

    Returns: train_losses, test_losses, total_time, time_per_epoch, dense_mse_errors
    """
    optimizer = torch.optim.LBFGS(model.parameters())
    criterion = nn.MSELoss()

    train_losses = []
    test_losses = []
    dense_mse_errors = []

    start_time = time.time()

    for epoch in range(epochs):
        def closure():
            """LBFGS optimizer requires closure that computes loss"""
            optimizer.zero_grad()
            train_pred = model(dataset['train_input'])
            loss = criterion(train_pred, dataset['train_label'])
            loss.backward()
            return loss

        optimizer.step(closure)

        with torch.no_grad():
            # Track training and test loss after each epoch
            train_pred = model(dataset['train_input'])
            test_pred = model(dataset['test_input'])
            train_loss = criterion(train_pred, dataset['train_label']).item()
            test_loss = criterion(test_pred, dataset['test_label']).item()
            train_losses.append(train_loss)
            test_losses.append(test_loss)

            # Always compute dense MSE on broader domain
            dense_mse = dense_mse_error_from_dataset(model, dataset, true_function,
                                                     num_samples=10000, device=device)
            dense_mse_errors.append(dense_mse)

    total_time = time.time() - start_time
    time_per_epoch = total_time / epochs if epochs > 0 else 0

    return train_losses, test_losses, total_time, time_per_epoch, dense_mse_errors

def run_mlp_tests(datasets, depths, activations, epochs, device, true_functions, dataset_names=None):
    """Train MLPs with varying depths and activations across multiple datasets

    Args:
        datasets: List of datasets
        depths: List of network depths to test
        activations: List of activation functions to test
        epochs: Number of training epochs
        device: Device to run on
        true_functions: List of true functions for each dataset (required)
        dataset_names: List of descriptive names for each dataset (optional)

    Returns:
        DataFrame with columns: dataset_idx, dataset_name, depth, activation, epoch, train_loss, test_loss,
                                dense_mse, total_time, time_per_epoch, num_params
    """
    print("mlp")
    rows = []

    # Generate default names if not provided
    if dataset_names is None:
        dataset_names = [f"dataset_{i}" for i in range(len(datasets))]

    for i, dataset in enumerate(datasets):
        n_var = dataset['train_input'].shape[1]
        true_func = true_functions[i]
        dataset_name = dataset_names[i]

        for d in depths:
            for act in activations:
                # Create MLP with fixed width=5 and varying depth/activation
                model = tnn.MLP(in_features=n_var, width=5, depth=d, activation=act).to(device)
                num_params = count_parameters(model)
                train_loss, test_loss, total_time, time_per_epoch, dense_mse = train_model(
                    model, dataset, epochs, device, true_func
                )

                # Create row for each epoch
                for epoch in range(epochs):
                    rows.append({
                        'dataset_idx': i,
                        'dataset_name': dataset_name,
                        'depth': d,
                        'activation': act,
                        'epoch': epoch,
                        'train_loss': train_loss[epoch],
                        'test_loss': test_loss[epoch],
                        'dense_mse': dense_mse[epoch],
                        'total_time': total_time,
                        'time_per_epoch': time_per_epoch,
                        'num_params': num_params
                    })

                print(f"  Dataset {i} ({dataset_name}), depth {d}, {act}: {total_time:.2f}s total, {time_per_epoch:.3f}s/epoch, {num_params} params")

    return pd.DataFrame(rows)

def run_siren_tests(datasets, depths, epochs, device, true_functions, dataset_names=None):
    """Train SIREN models with varying depths across multiple datasets

    Args:
        datasets: List of datasets
        depths: List of network depths to test
        epochs: Number of training epochs
        device: Device to run on
        true_functions: List of true functions for each dataset (required)
        dataset_names: List of descriptive names for each dataset (optional)

    Returns:
        DataFrame with columns: dataset_idx, dataset_name, depth, epoch, train_loss, test_loss,
                                dense_mse, total_time, time_per_epoch, num_params
    """
    print("siren")
    rows = []

    # Generate default names if not provided
    if dataset_names is None:
        dataset_names = [f"dataset_{i}" for i in range(len(datasets))]

    for i, dataset in enumerate(datasets):
        n_var = dataset['train_input'].shape[1]
        true_func = true_functions[i]
        dataset_name = dataset_names[i]

        for d in depths:
            # SIREN with hidden_layers=d-2 (accounting for input and output layers)
            model = tnn.SIREN(in_features=n_var, hidden_features=5, hidden_layers=d-2, out_features=1).to(device)
            num_params = count_parameters(model)
            train_loss, test_loss, total_time, time_per_epoch, dense_mse = train_model(
                model, dataset, epochs, device, true_func
            )

            # Create row for each epoch
            for epoch in range(epochs):
                rows.append({
                    'dataset_idx': i,
                    'dataset_name': dataset_name,
                    'depth': d,
                    'epoch': epoch,
                    'train_loss': train_loss[epoch],
                    'test_loss': test_loss[epoch],
                    'dense_mse': dense_mse[epoch],
                    'total_time': total_time,
                    'time_per_epoch': time_per_epoch,
                    'num_params': num_params
                })

            print(f"  Dataset {i} ({dataset_name}), depth {d}: {total_time:.2f}s total, {time_per_epoch:.3f}s/epoch, {num_params} params")

    return pd.DataFrame(rows)

def run_kan_grid_tests(datasets, grids, epochs, device, prune, true_functions, dataset_names=None):
    """Train KAN models with grid refinement, optionally with pruning

    Args:
        datasets: List of datasets
        grids: List of grid sizes to test
        epochs: Number of training epochs per grid
        device: Device to run on
        prune: Whether to apply pruning after training
        true_functions: List of true functions for each dataset (required)
        dataset_names: List of descriptive names for each dataset (optional)

    Returns:
        Tuple of (results_df, models, pruned_models) if prune=True
        Tuple of (results_df, models) if prune=False
        where results_df has columns: dataset_idx, dataset_name, grid_size, epoch, train_loss, test_loss,
                                      dense_mse, total_time, time_per_epoch, num_params
                                      (and pruned rows if prune=True)
    """
    print("kan_pruning" if prune else "kan")
    rows = []
    models = {}
    pruned_models = {} if prune else None

    # Generate default names if not provided
    if dataset_names is None:
        dataset_names = [f"dataset_{i}" for i in range(len(datasets))]

    for i, dataset in enumerate(datasets):
        train_losses = []
        test_losses = []
        dense_mse_errors = []
        n_var = dataset['train_input'].shape[1]
        true_func = true_functions[i]
        dataset_name = dataset_names[i]

        dataset_start_time = time.time()
        grid_times = []
        grid_param_counts = []

        for j, grid_size in enumerate(grids):
            grid_start_time = time.time()
            if j == 0:
                # Initialize KAN on first grid
                model = KAN(width=[n_var, 5, 1], grid=grid_size, k=3, seed=1, device=device)
            else:
                # Refine to next grid size (preserves learned splines)
                model = model.refine(grid_size)
            num_params = count_parameters(model)
            grid_param_counts.append(num_params)
            train_results = model.fit(dataset, opt="LBFGS", steps=epochs, log=1)
            train_losses += train_results['train_loss']
            test_losses += train_results['test_loss']

            # Compute dense MSE for each epoch in this grid
            for epoch_in_grid in range(epochs):
                with torch.no_grad():
                    dense_mse = dense_mse_error_from_dataset(model, dataset, true_func,
                                                            num_samples=10000, device=device)
                    dense_mse_errors.append(dense_mse)

            grid_time = time.time() - grid_start_time
            grid_times.append(grid_time)
            print(f"  Dataset {i} ({dataset_name}), grid {grid_size}: {grid_time:.2f}s total, {grid_time/epochs:.3f}s/epoch, {num_params} params")

        total_dataset_time = time.time() - dataset_start_time
        models[i] = model

        # Create rows for regular training
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
                    'is_pruned': False
                })

        if prune:
            # Apply pruning to remove insignificant edges and nodes
            prune_start_time = time.time()
            model_pruned = model.prune(node_th=1e-2, edge_th=3e-2)
            pruned_models[i] = model_pruned
            num_params_pruned = count_parameters(model_pruned)
            prune_time = time.time() - prune_start_time
            with torch.no_grad():
                train_loss_pruned = nn.MSELoss()(model_pruned(dataset['train_input']), dataset['train_label']).item()
                test_loss_pruned = nn.MSELoss()(model_pruned(dataset['test_input']), dataset['test_label']).item()
                dense_mse_pruned = dense_mse_error_from_dataset(model_pruned, dataset, true_func,
                                                               num_samples=10000, device=device)
            print(f"  Dataset {i} ({dataset_name}), pruning: {prune_time:.2f}s, {num_params_pruned} params")

            # Add pruned result row (single row, no epoch tracking for pruned)
            rows.append({
                'dataset_idx': i,
                'dataset_name': dataset_name,
                'grid_size': grids[-1],  # Associate with final grid
                'epoch': len(grids) * epochs,  # Epoch after all training
                'train_loss': train_loss_pruned,
                'test_loss': test_loss_pruned,
                'dense_mse': dense_mse_pruned,
                'total_time': prune_time,
                'time_per_epoch': 0,  # Pruning doesn't have epochs
                'num_params': num_params_pruned,
                'is_pruned': True
            })

        print(f"  Dataset {i} ({dataset_name}) complete: {total_dataset_time:.2f}s total")

    results_df = pd.DataFrame(rows)
    return (results_df, models, pruned_models) if prune else (results_df, models)
