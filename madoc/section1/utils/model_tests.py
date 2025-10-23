"""Model training and testing utilities for Section 1 experiments"""
from kan import *
from . import trad_nn as tnn
from .metrics import (dense_mse_error_from_dataset, count_parameters,
                      linf_error_from_dataset, h1_seminorm_error_from_dataset)
import time
import pandas as pd
from copy import deepcopy

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


def compute_all_metrics(model, dataset, true_function, device='cpu',
                        compute_h1=False, num_samples=10000):
    """
    Compute all evaluation metrics for a model.

    Args:
        model: Trained model
        dataset: Dataset dict with 'train_input' key
        true_function: Ground truth function
        device: Device to compute on
        compute_h1: Whether to compute H¹ seminorm (expensive, mainly for 2D PDEs)
        num_samples: Number of samples for evaluation

    Returns:
        Dict with keys: 'dense_mse', 'linf_error', and optionally 'h1_seminorm'
    """
    metrics = {}

    with torch.no_grad():
        # L² norm (dense MSE) - always computed
        metrics['dense_mse'] = dense_mse_error_from_dataset(
            model, dataset, true_function,
            num_samples=num_samples, device=device
        )

        # L∞ norm (max error) - cheap to add
        metrics['linf_error'] = linf_error_from_dataset(
            model, dataset, true_function,
            num_samples=num_samples, device=device
        )

        # H¹ seminorm (gradient error) - only for PDEs (expensive)
        if compute_h1:
            n_var = dataset['train_input'].shape[1]
            if n_var >= 2:  # Only makes sense for multi-dimensional problems
                metrics['h1_seminorm'] = h1_seminorm_error_from_dataset(
                    model, dataset, true_function,
                    num_samples=num_samples, device=device
                )
            else:
                metrics['h1_seminorm'] = None
        else:
            metrics['h1_seminorm'] = None

    return metrics


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
            # Skip if dataframe is empty
            if len(df) == 0:
                row += f"{'N/A':<20}"
                continue
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
    """Train any model using appropriate optimizer and return loss history with timing

    Args:
        model: Model to train
        dataset: Dataset dict with train/test inputs and labels
        epochs: Number of training epochs
        device: Device to run on
        true_function: True function for computing dense MSE (required)

    Returns: train_losses, test_losses, total_time, time_per_epoch, final_dense_mse
    """
    # Detect model type for optimizer selection
    is_siren = hasattr(model, 'net') and any(isinstance(m, tnn.Sine) for m in model.modules())

    # SIREN requires Adam optimizer with lower learning rate for stability
    # MLPs work well with LBFGS
    if is_siren:
        # Adam optimizer with learning rate schedule for SIREN stability
        # Start with lr=1e-4 for first half, then 1e-5 for second half
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs//2), gamma=0.1)
        use_closure = False
        grad_clip_norm = 0.1  # Tighter clipping for SIREN
    else:
        # LBFGS for MLPs - more stable with line search
        optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=20,
                                      tolerance_grad=1e-7, tolerance_change=1e-9,
                                      history_size=50, line_search_fn="strong_wolfe")
        scheduler = None
        use_closure = True
        grad_clip_norm = 1.0

    criterion = nn.MSELoss()

    train_losses = []
    test_losses = []

    start_time = time.time()

    for epoch in range(epochs):
        if use_closure:
            # LBFGS requires closure
            def closure():
                """LBFGS optimizer requires closure that computes loss"""
                optimizer.zero_grad()
                train_pred = model(dataset['train_input'])
                loss = criterion(train_pred, dataset['train_label'])
                loss.backward()

                # Add gradient clipping to prevent explosions
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

                return loss

            optimizer.step(closure)
        else:
            # Adam-style training
            optimizer.zero_grad()
            train_pred = model(dataset['train_input'])
            loss = criterion(train_pred, dataset['train_label'])
            loss.backward()

            # Gradient clipping is critical for SIREN stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

            optimizer.step()
            if scheduler:
                scheduler.step()

        with torch.no_grad():
            # Track training and test loss after each epoch
            train_pred = model(dataset['train_input'])
            test_pred = model(dataset['test_input'])
            train_loss = criterion(train_pred, dataset['train_label']).item()
            test_loss = criterion(test_pred, dataset['test_label']).item()

            # Check for NaN/Inf and stop training if detected
            import math
            if math.isnan(train_loss) or math.isinf(train_loss) or math.isnan(test_loss) or math.isinf(test_loss):
                print(f"    WARNING: Training diverged at epoch {epoch+1} (train_loss={train_loss:.6e}, test_loss={test_loss:.6e})")
                # Fill remaining epochs with nan
                for _ in range(epoch, epochs):
                    train_losses.append(float('nan'))
                    test_losses.append(float('nan'))
                break

            train_losses.append(train_loss)
            test_losses.append(test_loss)

            # Print loss after each epoch (show LR for SIREN)
            if is_siren and scheduler:
                current_lr = scheduler.get_last_lr()[0]
                print(f"    Epoch {epoch+1}/{epochs}: Train Loss = {train_loss:.6e}, Test Loss = {test_loss:.6e}, LR = {current_lr:.2e}")
            else:
                print(f"    Epoch {epoch+1}/{epochs}: Train Loss = {train_loss:.6e}, Test Loss = {test_loss:.6e}")

    total_time = time.time() - start_time
    time_per_epoch = total_time / epochs if epochs > 0 else 0

    # Compute dense MSE only once at the end (handle NaN case)
    with torch.no_grad():
        try:
            final_dense_mse = dense_mse_error_from_dataset(model, dataset, true_function,
                                                            num_samples=10000, device=device)
            # Check if result is valid
            import math
            if math.isnan(final_dense_mse) or math.isinf(final_dense_mse):
                final_dense_mse = float('nan')
        except:
            final_dense_mse = float('nan')

    print(f"    Final Dense MSE = {final_dense_mse:.6e}")

    return train_losses, test_losses, total_time, time_per_epoch, final_dense_mse

def run_mlp_tests(datasets, depths, activations, epochs, device, true_functions, dataset_names=None, kan_threshold_time=None):
    """Train MLPs with varying depths and activations across multiple datasets

    Args:
        datasets: List of datasets
        depths: List of network depths to test
        activations: List of activation functions to test
        epochs: Number of training epochs
        device: Device to run on
        true_functions: List of true functions for each dataset (required)
        dataset_names: List of descriptive names for each dataset (optional)
        kan_threshold_time: Time at which KAN reached interpolation threshold (optional)

    Returns:
        Tuple of (DataFrame, checkpoints) where:
        - DataFrame has columns: dataset_idx, dataset_name, depth, activation, epoch, train_loss, test_loss,
                                 dense_mse, total_time, time_per_epoch, num_params
        - checkpoints: Dict[dataset_idx -> {'at_kan_threshold_time': {...}, 'final': {...}}]
    """
    print("mlp")
    rows = []
    checkpoints = {}
    best_models_info = {}  # Track best model per dataset

    # Generate default names if not provided
    if dataset_names is None:
        dataset_names = [f"dataset_{i}" for i in range(len(datasets))]

    for i, dataset in enumerate(datasets):
        n_var = dataset['train_input'].shape[1]
        true_func = true_functions[i]
        dataset_name = dataset_names[i]

        # Track threshold checkpoint per dataset
        threshold_checkpoint_saved = False
        threshold_checkpoint = None

        for d in depths:
            for act in activations:
                # Create MLP with fixed width=5 and varying depth/activation
                model = tnn.MLP(in_features=n_var, width=5, depth=d, activation=act).to(device)
                num_params = count_parameters(model)
                print(f"  Training MLP: Dataset {i} ({dataset_name}), depth={d}, activation={act}")
                train_loss, test_loss, total_time, time_per_epoch, final_dense_mse = train_model(
                    model, dataset, epochs, device, true_func
                )

                # If KAN threshold time provided, check if we should save a checkpoint
                # We save from the best performing configuration at the time closest to threshold
                if kan_threshold_time and not threshold_checkpoint_saved:
                    # Find the epoch where cumulative time >= kan_threshold_time
                    for epoch in range(epochs):
                        cumulative_time = time_per_epoch * (epoch + 1)
                        if cumulative_time >= kan_threshold_time:
                            # Save this as the threshold checkpoint for this dataset
                            # (use the best model's checkpoint at this time)
                            threshold_epoch = epoch
                            with torch.no_grad():
                                threshold_dense_mse = dense_mse_error_from_dataset(model, dataset, true_func,
                                                                                   num_samples=10000, device=device)

                            if threshold_checkpoint is None or test_loss[threshold_epoch] < threshold_checkpoint['test_loss']:
                                threshold_checkpoint = {
                                    'model': deepcopy(model),
                                    'epoch': threshold_epoch,
                                    'time': cumulative_time,
                                    'train_loss': train_loss[threshold_epoch],
                                    'test_loss': test_loss[threshold_epoch],
                                    'dense_mse': threshold_dense_mse,
                                    'depth': d,
                                    'activation': act,
                                    'num_params': num_params
                                }
                            break

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
                            'activation': act,
                            'train_loss': train_loss[-1],
                            'test_loss': test_loss[-1],
                            'total_time': total_time,
                            'num_params': num_params
                        }

        # Store checkpoints for this dataset
        if i in best_models_info:
            checkpoints[i] = {
                'final': best_models_info[i]
            }
            if threshold_checkpoint:
                checkpoints[i]['at_kan_threshold_time'] = threshold_checkpoint
                print(f"  MLP checkpoint at KAN threshold time ({kan_threshold_time:.2f}s): depth={threshold_checkpoint['depth']}, activation={threshold_checkpoint['activation']}, dense_mse={threshold_checkpoint['dense_mse']:.6e}")

            print(f"  Best MLP for dataset {i}: depth={best_models_info[i]['depth']}, activation={best_models_info[i]['activation']}, dense_mse={best_models_info[i]['dense_mse']:.6e}")

    return pd.DataFrame(rows), checkpoints

def run_siren_tests(datasets, depths, epochs, device, true_functions, dataset_names=None, kan_threshold_time=None):
    """Train SIREN models with varying depths across multiple datasets

    Args:
        datasets: List of datasets
        depths: List of network depths to test
        epochs: Number of training epochs
        device: Device to run on
        true_functions: List of true functions for each dataset (required)
        dataset_names: List of descriptive names for each dataset (optional)
        kan_threshold_time: Time at which KAN reached interpolation threshold (optional)

    Returns:
        Tuple of (DataFrame, checkpoints) where:
        - DataFrame has columns: dataset_idx, dataset_name, depth, epoch, train_loss, test_loss,
                                 dense_mse, total_time, time_per_epoch, num_params
        - checkpoints: Dict[dataset_idx -> {'at_kan_threshold_time': {...}, 'final': {...}}]
    """
    print("siren")
    rows = []
    checkpoints = {}
    best_models_info = {}  # Track best model per dataset

    # Generate default names if not provided
    if dataset_names is None:
        dataset_names = [f"dataset_{i}" for i in range(len(datasets))]

    for i, dataset in enumerate(datasets):
        n_var = dataset['train_input'].shape[1]
        true_func = true_functions[i]
        dataset_name = dataset_names[i]

        # Track threshold checkpoint per dataset
        threshold_checkpoint = None

        for d in depths:
            # SIREN with hidden_layers=d-2 (accounting for input and output layers)
            model = tnn.SIREN(in_features=n_var, hidden_features=5, hidden_layers=d-2, out_features=1).to(device)
            num_params = count_parameters(model)
            print(f"  Training SIREN: Dataset {i} ({dataset_name}), depth={d}")
            train_loss, test_loss, total_time, time_per_epoch, final_dense_mse = train_model(
                model, dataset, epochs, device, true_func
            )

            # If KAN threshold time provided, check if we should save a checkpoint
            if kan_threshold_time:
                # Find the epoch where cumulative time >= kan_threshold_time
                for epoch in range(epochs):
                    cumulative_time = time_per_epoch * (epoch + 1)
                    if cumulative_time >= kan_threshold_time:
                        threshold_epoch = epoch
                        with torch.no_grad():
                            threshold_dense_mse = dense_mse_error_from_dataset(model, dataset, true_func,
                                                                               num_samples=10000, device=device)

                        if threshold_checkpoint is None or test_loss[threshold_epoch] < threshold_checkpoint['test_loss']:
                            threshold_checkpoint = {
                                'model': deepcopy(model),
                                'epoch': threshold_epoch,
                                'time': cumulative_time,
                                'train_loss': train_loss[threshold_epoch],
                                'test_loss': test_loss[threshold_epoch],
                                'dense_mse': threshold_dense_mse,
                                'depth': d,
                                'num_params': num_params
                            }
                        break

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
                        'depth': d,
                        'train_loss': train_loss[-1],
                        'test_loss': test_loss[-1],
                        'total_time': total_time,
                        'num_params': num_params
                    }

        # Store checkpoints for this dataset
        if i in best_models_info:
            checkpoints[i] = {
                'final': best_models_info[i]
            }
            if threshold_checkpoint:
                checkpoints[i]['at_kan_threshold_time'] = threshold_checkpoint
                print(f"  SIREN checkpoint at KAN threshold time ({kan_threshold_time:.2f}s): depth={threshold_checkpoint['depth']}, dense_mse={threshold_checkpoint['dense_mse']:.6e}")

            print(f"  Best SIREN for dataset {i}: depth={best_models_info[i]['depth']}, dense_mse={best_models_info[i]['dense_mse']:.6e}")

    return pd.DataFrame(rows), checkpoints

def run_kan_grid_tests(datasets, grids, epochs, device, prune, true_functions, dataset_names=None, steps_per_grid=200):
    """Train KAN models with grid refinement, optionally with pruning

    Args:
        datasets: List of datasets
        grids: List of grid sizes to test
        epochs: Total epoch budget (training stops when cumulative epochs reach this limit)
        device: Device to run on
        prune: Whether to apply pruning after training
        true_functions: List of true functions for each dataset (required)
        dataset_names: List of descriptive names for each dataset (optional)
        steps_per_grid: Number of training epochs per grid size (default=200)
                       Training will complete as many grids as possible within the epochs budget

    Returns:
        Tuple of (results_df, checkpoints, threshold_time, pruned_models) if prune=True
        Tuple of (results_df, checkpoints, threshold_time) if prune=False

        where:
        - results_df has columns: dataset_idx, dataset_name, grid_size, epoch, train_loss, test_loss,
                                  dense_mse, total_time, time_per_epoch, num_params
        - checkpoints: Dict[dataset_idx -> {'at_threshold': {...}, 'final': {...}}]
        - threshold_time: Average time to reach interpolation threshold across datasets
        - pruned_models: Dict (only if prune=True)
    """
    print("kan_pruning" if prune else "kan")
    rows = []
    checkpoints = {}
    pruned_models = {} if prune else None
    threshold_times = []  # Track threshold time for each dataset

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
        cumulative_epochs = 0
        grids_completed = []

        for j, grid_size in enumerate(grids):
            # Calculate how many epochs we can afford for this grid
            remaining_budget = epochs - cumulative_epochs
            if remaining_budget <= 0:
                print(f"  Epoch budget exhausted ({cumulative_epochs}/{epochs} epochs used)")
                print(f"  Completed {len(grids_completed)}/{len(grids)} grids within budget")
                break

            steps_for_this_grid = min(steps_per_grid, remaining_budget)

            grid_start_time = time.time()

            try:
                if j == 0:
                    # Initialize KAN on first grid
                    model = KAN(width=[n_var, 5, 1], grid=grid_size, k=3, seed=1, device=device)
                    print(f"  Training KAN: Dataset {i} ({dataset_name}), grid={grid_size} (budget: {cumulative_epochs}/{epochs} epochs used, training {steps_for_this_grid} epochs)")
                else:
                    # Refine to next grid size (preserves learned splines)
                    model = model.refine(grid_size)
                    print(f"  Refining KAN: Dataset {i} ({dataset_name}), grid={grid_size} (budget: {cumulative_epochs}/{epochs} epochs used, training {steps_for_this_grid} epochs)")

                num_params = count_parameters(model)
                grid_param_counts.append(num_params)

                train_results = model.fit(dataset, opt="LBFGS", steps=steps_for_this_grid, log=1)

                # Check if training produced valid results
                if not train_results or 'train_loss' not in train_results or 'test_loss' not in train_results:
                    raise RuntimeError("Training returned invalid results (missing loss values)")

                train_losses += train_results['train_loss']
                test_losses += train_results['test_loss']

                # Check for NaN/Inf in losses
                import math
                if any(math.isnan(loss) or math.isinf(loss) for loss in train_results['train_loss']):
                    raise RuntimeError("Training produced NaN/Inf values in train loss")
                if any(math.isnan(loss) or math.isinf(loss) for loss in train_results['test_loss']):
                    raise RuntimeError("Training produced NaN/Inf values in test loss")

                # Print losses for each epoch in this grid
                for epoch_in_grid in range(steps_for_this_grid):
                    global_epoch = cumulative_epochs + epoch_in_grid + 1
                    train_loss_val = train_results['train_loss'][epoch_in_grid]
                    test_loss_val = train_results['test_loss'][epoch_in_grid]
                    print(f"    Epoch {global_epoch}/{epochs}: Train Loss = {train_loss_val:.6e}, Test Loss = {test_loss_val:.6e}")

                cumulative_epochs += steps_for_this_grid
                grids_completed.append((grid_size, steps_for_this_grid))
                grid_time = time.time() - grid_start_time
                grid_times.append(grid_time)
                print(f"  Grid {grid_size} completed: {grid_time:.2f}s total, {grid_time/steps_for_this_grid:.3f}s/epoch, {num_params} params")

            except Exception as e:
                # Catch any training errors and log them
                grid_time = time.time() - grid_start_time
                print(f"  ERROR: KAN training failed at grid {grid_size} for dataset {i} ({dataset_name})")
                print(f"  Error type: {type(e).__name__}")
                print(f"  Error message: {str(e)}")
                print(f"  Stopping training for this dataset. Completed {len(grids_completed)}/{len(grids)} grids.")
                # Break out of grid loop but continue with other datasets
                break

        # Compute dense MSE and detect interpolation threshold (only if at least one grid completed)
        if grids_completed:
            # Detect interpolation threshold based on test loss
            threshold_epoch = detect_kan_threshold(test_losses)

            # Calculate cumulative time up to threshold
            cumulative_time_at_threshold = 0
            cumulative_epochs_at_threshold = 0
            threshold_grid_idx = 0

            for j, (grid_size, steps_trained) in enumerate(grids_completed):
                if cumulative_epochs_at_threshold + steps_trained > threshold_epoch:
                    # Threshold is within this grid
                    epochs_into_this_grid = threshold_epoch - cumulative_epochs_at_threshold
                    cumulative_time_at_threshold += grid_times[j] * (epochs_into_this_grid / steps_trained)
                    threshold_grid_idx = j
                    break
                else:
                    cumulative_time_at_threshold += grid_times[j]
                    cumulative_epochs_at_threshold += steps_trained
            else:
                # Threshold not found, use final grid
                cumulative_time_at_threshold = sum(grid_times)
                threshold_grid_idx = len(grids_completed) - 1

            threshold_times.append(cumulative_time_at_threshold)

            # Compute dense MSE for final model
            with torch.no_grad():
                final_dense_mse = dense_mse_error_from_dataset(model, dataset, true_func,
                                                               num_samples=10000, device=device)
            print(f"    Final Dense MSE = {final_dense_mse:.6e}")

            # Note: We can't perfectly recover the model at threshold_epoch without saving during training
            # So we'll compute a threshold checkpoint on the final model but report the threshold metrics
            threshold_test_loss = test_losses[threshold_epoch]
            threshold_train_loss = train_losses[threshold_epoch]

            # Compute dense MSE at threshold (use final model as approximation - limitation noted)
            # In practice, the model at threshold would need to be saved during training
            # For now, we note this as the threshold metrics on the final model
            with torch.no_grad():
                threshold_dense_mse = dense_mse_error_from_dataset(model, dataset, true_func,
                                                                   num_samples=10000, device=device)

            print(f"    Interpolation threshold detected at epoch {threshold_epoch} (time: {cumulative_time_at_threshold:.2f}s)")
            print(f"    Test loss at threshold: {threshold_test_loss:.6e}")
            print(f"    Dense MSE at threshold: {threshold_dense_mse:.6e} (computed on final model)")

            total_dataset_time = time.time() - dataset_start_time

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
                    'grid_size': grids_completed[threshold_grid_idx][0] if threshold_grid_idx < len(grids_completed) else grids_completed[-1][0],
                    'num_params': grid_param_counts[threshold_grid_idx] if threshold_grid_idx < len(grid_param_counts) else grid_param_counts[-1]
                },
                'final': {
                    'model': model,
                    'epoch': cumulative_epochs,
                    'time': total_dataset_time,
                    'train_loss': train_losses[-1],
                    'test_loss': test_losses[-1],
                    'dense_mse': final_dense_mse,
                    'grid_size': grids_completed[-1][0],
                    'num_params': grid_param_counts[-1]
                }
            }
        else:
            # No grids completed - skip this dataset
            print(f"  Warning: No grids completed for dataset {i} ({dataset_name}) - skipping")
            continue

        # Create rows for regular training (only for completed grids)
        cumulative_global_epoch = 0
        for j, (grid_size, steps_trained) in enumerate(grids_completed):
            for epoch_in_grid in range(steps_trained):
                global_epoch = cumulative_global_epoch + epoch_in_grid
                rows.append({
                    'dataset_idx': i,
                    'dataset_name': dataset_name,
                    'grid_size': grid_size,
                    'epoch': epoch_in_grid,  # Use epoch within grid, not global epoch
                    'train_loss': train_losses[global_epoch],
                    'test_loss': test_losses[global_epoch],
                    'dense_mse': final_dense_mse,  # Same value for all epochs
                    'total_time': grid_times[j],
                    'time_per_epoch': grid_times[j] / steps_trained,
                    'num_params': grid_param_counts[j],
                    'is_pruned': False
                })
            cumulative_global_epoch += steps_trained

        if prune:
            # KAN Paper pruning workflow (Section 3.2.1):
            # Stage 1: Train with sparsification regularization
            # Stage 2: Prune based on attribution scores
            # Stage 3: Retrain the pruned network with grid extension

            try:
                print(f"  === KAN Pruning Workflow (Paper-aligned) ===")
                print(f"  Stage 1: Sparsification training with regularization (lamb=1e-2)")

                # Stage 1: Train with sparsification for 200 steps (paper default)
                sparsification_steps = min(200, steps_per_grid)
                sparsification_start_time = time.time()

                # The model.fit() with lamb parameter applies L1 regularization on activations
                # This encourages sparsity before pruning
                sparsification_results = model.fit(dataset, opt="LBFGS", steps=sparsification_steps,
                                                   lamb=1e-2, log=1)

                sparsification_time = time.time() - sparsification_start_time
                print(f"  Sparsification complete: {sparsification_time:.2f}s")
                print(f"  Final sparsification loss: train={sparsification_results['train_loss'][-1]:.6e}, test={sparsification_results['test_loss'][-1]:.6e}")

                # Stage 2: Apply pruning based on attribution scores
                print(f"  Stage 2: Pruning insignificant nodes and edges")
                prune_start_time = time.time()

                # Compute attribution scores before pruning (required by prune() method)
                model.forward(dataset['train_input'])
                model.attribute()

                # Prune with thresholds from paper (node_th=1e-2, edge_th not explicitly mentioned but 3e-2 is reasonable)
                model_pruned = model.prune(node_th=1e-2, edge_th=3e-2)
                pruned_models[i] = model_pruned
                num_params_pruned = count_parameters(model_pruned)
                prune_time = time.time() - prune_start_time

                print(f"  Pruning complete: {prune_time:.2f}s")
                print(f"  Parameters: {count_parameters(model)} -> {num_params_pruned} (removed {count_parameters(model) - num_params_pruned})")

                # Compute metrics immediately after pruning (before retraining)
                with torch.no_grad():
                    train_loss_after_prune = nn.MSELoss()(model_pruned(dataset['train_input']), dataset['train_label']).item()
                    test_loss_after_prune = nn.MSELoss()(model_pruned(dataset['test_input']), dataset['test_label']).item()
                    dense_mse_after_prune = dense_mse_error_from_dataset(model_pruned, dataset, true_func,
                                                                         num_samples=10000, device=device)
                print(f"  Metrics after pruning (before retrain): Dense MSE = {dense_mse_after_prune:.6e}")

                # Stage 3: Retrain the pruned network with grid extension
                # Paper mentions continuing training with grid extension after pruning
                print(f"  Stage 3: Retraining pruned network")
                retrain_steps = min(200, steps_per_grid)
                retrain_start_time = time.time()

                # Retrain without regularization to allow the pruned network to optimize
                retrain_results = model_pruned.fit(dataset, opt="LBFGS", steps=retrain_steps, log=1)

                retrain_time = time.time() - retrain_start_time
                print(f"  Retraining complete: {retrain_time:.2f}s")
                print(f"  Final retrain loss: train={retrain_results['train_loss'][-1]:.6e}, test={retrain_results['test_loss'][-1]:.6e}")

                # Compute final metrics after retraining
                with torch.no_grad():
                    train_loss_pruned = nn.MSELoss()(model_pruned(dataset['train_input']), dataset['train_label']).item()
                    test_loss_pruned = nn.MSELoss()(model_pruned(dataset['test_input']), dataset['test_label']).item()
                    dense_mse_pruned = dense_mse_error_from_dataset(model_pruned, dataset, true_func,
                                                                   num_samples=10000, device=device)

                total_pruning_time = sparsification_time + prune_time + retrain_time
                print(f"  === Pruning workflow complete ===")
                print(f"  Total time: {total_pruning_time:.2f}s (sparsify: {sparsification_time:.2f}s, prune: {prune_time:.2f}s, retrain: {retrain_time:.2f}s)")
                print(f"  Final pruned model: {num_params_pruned} params, Dense MSE = {dense_mse_pruned:.6e}")

                # Add pruned result row (single row, no epoch tracking for pruned)
                rows.append({
                    'dataset_idx': i,
                    'dataset_name': dataset_name,
                    'grid_size': grids_completed[-1][0] if grids_completed else grids[0],  # Associate with final completed grid
                    'epoch': cumulative_epochs + sparsification_steps + retrain_steps,  # Include all training steps
                    'train_loss': train_loss_pruned,
                    'test_loss': test_loss_pruned,
                    'dense_mse': dense_mse_pruned,
                    'total_time': total_pruning_time,
                    'time_per_epoch': 0,  # Pruning workflow is multi-stage
                    'num_params': num_params_pruned,
                    'is_pruned': True
                })

            except Exception as e:
                # Catch any pruning errors and log them
                print(f"  ERROR: KAN pruning workflow failed for dataset {i} ({dataset_name})")
                print(f"  Error type: {type(e).__name__}")
                print(f"  Error message: {str(e)}")
                print(f"  Skipping pruned results for this dataset, but keeping unpruned model.")

        print(f"  Dataset {i} ({dataset_name}) complete: {total_dataset_time:.2f}s total")

    # Calculate average threshold time across datasets
    avg_threshold_time = sum(threshold_times) / len(threshold_times) if threshold_times else 0
    print(f"\n  Average KAN interpolation threshold time: {avg_threshold_time:.2f}s")

    results_df = pd.DataFrame(rows)
    return (results_df, checkpoints, avg_threshold_time, pruned_models) if prune else (results_df, checkpoints, avg_threshold_time)
