"""Model training and testing utilities for Section 1 experiments"""
from kan import *
from . import trad_nn as tnn
from .metrics import dense_mse_error_from_dataset, count_parameters
import time
import pandas as pd

def print_best_dense_mse_summary(all_results, dataset_names):
    """Print a summary table showing best dense MSE per model type per dataset

    Args:
        all_results: Dict mapping model_type -> DataFrame with results
        dataset_names: List of dataset names
    """
    print("\n" + "="*80)
    print("BEST DENSE MSE SUMMARY (across all hyperparameters)")
    print("="*80)

    # Create header
    model_types = list(all_results.keys())
    header = f"{'Dataset':<25}"
    for model_type in model_types:
        header += f"{model_type.upper():<20}"
    print(header)
    print("-" * 80)

    # For each dataset, find best dense MSE for each model type
    for i, dataset_name in enumerate(dataset_names):
        row = f"{dataset_name:<25}"
        for model_type in model_types:
            df = all_results[model_type]
            # Filter for this dataset and non-pruned results
            dataset_df = df[df['dataset_idx'] == i]
            if 'is_pruned' in dataset_df.columns:
                dataset_df = dataset_df[dataset_df['is_pruned'] == False]

            if len(dataset_df) > 0:
                # Get minimum dense MSE across all configurations
                best_dense_mse = dataset_df['dense_mse'].min()
                row += f"{best_dense_mse:<20.6e}"
            else:
                row += f"{'N/A':<20}"
        print(row)

    # Also print pruned KAN results if available
    if 'kan_pruning' in all_results:
        df = all_results['kan_pruning']
        pruned_df = df[df['is_pruned'] == True] if 'is_pruned' in df.columns else pd.DataFrame()
        if len(pruned_df) > 0:
            print("\n" + "-" * 80)
            print("KAN PRUNED RESULTS:")
            print("-" * 80)
            for i, dataset_name in enumerate(dataset_names):
                dataset_pruned = pruned_df[pruned_df['dataset_idx'] == i]
                if len(dataset_pruned) > 0:
                    pruned_mse = dataset_pruned['dense_mse'].values[0]
                    print(f"{dataset_name:<25}{pruned_mse:.6e}")

    print("="*80 + "\n")

def train_model(model, dataset, epochs, device, true_function):
    """Train any model using LBFGS optimizer and return loss history with timing

    Args:
        model: Model to train
        dataset: Dataset dict with train/test inputs and labels
        epochs: Number of training epochs
        device: Device to run on
        true_function: True function for computing dense MSE (required)

    Returns: train_losses, test_losses, total_time, time_per_epoch, final_dense_mse
    """
    # Use standard PyTorch LBFGS with stable parameters
    # Line search helps prevent divergence
    optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=20,
                                  tolerance_grad=1e-7, tolerance_change=1e-9,
                                  history_size=50, line_search_fn="strong_wolfe")
    criterion = nn.MSELoss()

    train_losses = []
    test_losses = []

    start_time = time.time()

    for epoch in range(epochs):
        def closure():
            """LBFGS optimizer requires closure that computes loss"""
            optimizer.zero_grad()
            train_pred = model(dataset['train_input'])
            loss = criterion(train_pred, dataset['train_label'])
            loss.backward()

            # Add gradient clipping to prevent explosions
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            return loss

        optimizer.step(closure)

        with torch.no_grad():
            # Track training and test loss after each epoch
            train_pred = model(dataset['train_input'])
            test_pred = model(dataset['test_input'])
            train_loss = criterion(train_pred, dataset['train_label']).item()
            test_loss = criterion(test_pred, dataset['test_label']).item()

            # Check for NaN/Inf and stop training if detected
            import math
            if math.isnan(train_loss) or math.isinf(train_loss):
                print(f"    WARNING: Training diverged at epoch {epoch+1} (loss={train_loss:.6e})")
                # Fill remaining epochs with nan
                for _ in range(epoch, epochs):
                    train_losses.append(float('nan'))
                    test_losses.append(float('nan'))
                break

            train_losses.append(train_loss)
            test_losses.append(test_loss)

    total_time = time.time() - start_time
    time_per_epoch = total_time / epochs if epochs > 0 else 0

    # Compute dense MSE only once at the end
    with torch.no_grad():
        final_dense_mse = dense_mse_error_from_dataset(model, dataset, true_function,
                                                        num_samples=10000, device=device)
    print(f"    Final Dense MSE = {final_dense_mse:.6e}")

    return train_losses, test_losses, total_time, time_per_epoch, final_dense_mse

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
        Tuple of (DataFrame, models_dict) where:
        - DataFrame has columns: dataset_idx, dataset_name, depth, activation, epoch, train_loss, test_loss,
                                 dense_mse, total_time, time_per_epoch, num_params
        - models_dict maps dataset_idx -> trained model (best model per dataset: lowest final dense_mse)
    """
    print("mlp")
    rows = []
    models = {}
    best_models_info = {}  # Track best model per dataset

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
                print(f"  Training MLP: Dataset {i} ({dataset_name}), depth={d}, activation={act}")
                train_loss, test_loss, total_time, time_per_epoch, final_dense_mse = train_model(
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
                        'dense_mse': final_dense_mse,  # Same value for all epochs
                        'total_time': total_time,
                        'time_per_epoch': time_per_epoch,
                        'num_params': num_params
                    })

                print(f"  Completed: {total_time:.2f}s total, {time_per_epoch:.3f}s/epoch, {num_params} params")

                # Track best model per dataset (lowest final dense_mse)
                # Filter out NaN/Inf values to avoid saving diverged models
                import math
                if not math.isnan(final_dense_mse) and not math.isinf(final_dense_mse):
                    if i not in best_models_info or final_dense_mse < best_models_info[i]['dense_mse']:
                        best_models_info[i] = {
                            'model': model,
                            'dense_mse': final_dense_mse,
                            'depth': d,
                            'activation': act
                        }

    # Store best models
    for i, info in best_models_info.items():
        models[i] = info['model']
        print(f"  Best MLP for dataset {i}: depth={info['depth']}, activation={info['activation']}, dense_mse={info['dense_mse']:.6e}")

    return pd.DataFrame(rows), models

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
        Tuple of (DataFrame, models_dict) where:
        - DataFrame has columns: dataset_idx, dataset_name, depth, epoch, train_loss, test_loss,
                                 dense_mse, total_time, time_per_epoch, num_params
        - models_dict maps dataset_idx -> trained model (best model per dataset: lowest final dense_mse)
    """
    print("siren")
    rows = []
    models = {}
    best_models_info = {}  # Track best model per dataset

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
            print(f"  Training SIREN: Dataset {i} ({dataset_name}), depth={d}")
            train_loss, test_loss, total_time, time_per_epoch, final_dense_mse = train_model(
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
                    'dense_mse': final_dense_mse,  # Same value for all epochs
                    'total_time': total_time,
                    'time_per_epoch': time_per_epoch,
                    'num_params': num_params
                })

            print(f"  Completed: {total_time:.2f}s total, {time_per_epoch:.3f}s/epoch, {num_params} params")

            # Track best model per dataset (lowest final dense_mse)
            # Filter out NaN/Inf values to avoid saving diverged models
            import math
            if not math.isnan(final_dense_mse) and not math.isinf(final_dense_mse):
                if i not in best_models_info or final_dense_mse < best_models_info[i]['dense_mse']:
                    best_models_info[i] = {
                        'model': model,
                        'dense_mse': final_dense_mse,
                        'depth': d
                    }

    # Store best models
    for i, info in best_models_info.items():
        models[i] = info['model']
        print(f"  Best SIREN for dataset {i}: depth={info['depth']}, dense_mse={info['dense_mse']:.6e}")

    return pd.DataFrame(rows), models

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
                print(f"  Training KAN: Dataset {i} ({dataset_name}), grid={grid_size}")
            else:
                # Refine to next grid size (preserves learned splines)
                model = model.refine(grid_size)
                print(f"  Refining KAN: Dataset {i} ({dataset_name}), grid={grid_size}")
            num_params = count_parameters(model)
            grid_param_counts.append(num_params)
            train_results = model.fit(dataset, opt="LBFGS", steps=epochs, log=1)
            train_losses += train_results['train_loss']
            test_losses += train_results['test_loss']

            grid_time = time.time() - grid_start_time
            grid_times.append(grid_time)
            print(f"  Grid {grid_size} completed: {grid_time:.2f}s total, {grid_time/epochs:.3f}s/epoch, {num_params} params")

        # Compute dense MSE only once at the end
        with torch.no_grad():
            final_dense_mse = dense_mse_error_from_dataset(model, dataset, true_func,
                                                           num_samples=10000, device=device)
        print(f"    Final Dense MSE = {final_dense_mse:.6e}")

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
                    'dense_mse': final_dense_mse,  # Same value for all epochs
                    'total_time': grid_times[j],
                    'time_per_epoch': grid_times[j] / epochs,
                    'num_params': grid_param_counts[j],
                    'is_pruned': False
                })

        if prune:
            # Apply pruning to remove insignificant edges and nodes
            print(f"  Pruning KAN: Dataset {i} ({dataset_name})")
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
            print(f"  Pruning completed: {prune_time:.2f}s, {num_params_pruned} params, Dense MSE = {dense_mse_pruned:.6e}")

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
