import sys
from pathlib import Path

# Add pykan to path (parent directory of madoc)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from kan import *
from utils import data_funcs as dfs
from utils import save_run, track_time, print_timing_summary, print_optimizer_summary
from utils import run_kan_adaptive_density_test, run_kan_baseline_test
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Section 2.2 High-D: Adaptive Density on Higher-Dimensional Poisson PDEs')
parser.add_argument('--dim', type=int, required=True, choices=[3, 4, 10, 100],
                    help='Dimension of the Poisson PDE (3, 4, 10, or 100)')
parser.add_argument('--architecture', type=str, required=True, choices=['shallow', 'deep'],
                    help='KAN architecture: shallow or deep')
parser.add_argument('--epochs', type=int, default=10,
                    help='Number of epochs for training per grid (default: 10)')
args = parser.parse_args()

dim = args.dim
architecture = args.architecture
epochs = args.epochs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print(f"Running {dim}D Poisson PDE with {architecture} architecture and {epochs} epochs per grid")

# ============= Configuration =============

# Architecture configuration based on dimension and depth choice
ARCHITECTURES = {
    3: {'shallow': [3, 5, 1], 'deep': [3, 3, 2, 1]},
    4: {'shallow': [4, 5, 1], 'deep': [4, 4, 2, 1]},
    10: {'shallow': [10, 5, 1], 'deep': [10, 10, 5, 1]},
    100: {'shallow': [100, 1, 1], 'deep': [100, 10, 1]}
}

# Grid sizes based on dimension (reduced for higher dimensions)
GRIDS = {
    3: np.array([3, 5, 10, 20, 50, 100]),
    4: np.array([3, 5, 10, 20, 50, 100]),
    10: np.array([3, 5, 10, 20, 50]),
    100: np.array([3, 5, 10, 20])
}

# Test functions by dimension
FUNCTIONS = {
    3: dfs.f_poisson_3d_sin,
    4: dfs.f_poisson_4d_sin,
    10: dfs.f_poisson_10d_sin,
    100: dfs.f_poisson_100d_sin
}

# Dataset names
DATASET_NAMES = {
    3: 'poisson_3d_sin',
    4: 'poisson_4d_sin',
    10: 'poisson_10d_sin',
    100: 'poisson_100d_sin'
}

# Get configuration for this run
kan_width = ARCHITECTURES[dim][architecture]
grids = GRIDS[dim]
true_function = FUNCTIONS[dim]
dataset_name = DATASET_NAMES[dim]

print(f"\nConfiguration:")
print(f"  Dimension: {dim}D")
print(f"  Architecture: {kan_width} ({architecture})")
print(f"  Grid sizes: {grids.tolist()}")
print(f"  Function: {dataset_name}")

# ============= Create Dataset =============
print(f"\nCreating {dim}D dataset...")
dataset = create_dataset(true_function, n_var=dim, train_num=1000, test_num=1000)

print("\n" + "="*60)
print(f"Starting Section 2.2 High-D Adaptive Density ({dim}D, {architecture})")
print("="*60 + "\n")

timers = {}

# ============= Custom test functions with architecture support =============

def run_kan_adaptive_density_test_custom_arch(datasets, grids, epochs, device, use_regular_refine=False,
                                               attribution_threshold=1e-2, true_functions=None,
                                               dataset_names=None, kan_width=None, dim=None):
    """Modified version that accepts custom KAN architecture"""
    import time
    import pandas as pd
    import torch
    import torch.nn as nn
    from kan import KAN
    from utils.metrics import count_parameters, dense_mse_error_from_dataset
    from utils.optimizer_tests import adaptive_densify_model

    approach = "adaptive+regular" if use_regular_refine else "adaptive_only"
    print(f"kan_adaptive_density ({approach}) custom_arch_{kan_width}")
    rows = []
    models = {}

    # Handle single dataset case
    datasets_list = [datasets] if not isinstance(datasets, list) else datasets
    true_functions_list = [true_functions] if not isinstance(true_functions, list) and true_functions is not None else true_functions
    dataset_names_list = [dataset_names] if not isinstance(dataset_names, list) and dataset_names is not None else dataset_names

    if dataset_names_list is None:
        dataset_names_list = [f"dataset_{i}" for i in range(len(datasets_list))]

    for i, dataset in enumerate(datasets_list):
        train_losses = []
        test_losses = []
        dense_mse_errors = []
        n_var = dataset['train_input'].shape[1]
        true_func = true_functions_list[i] if true_functions_list else None
        dataset_name = dataset_names_list[i]

        dataset_start_time = time.time()
        grid_times = []
        grid_param_counts = []
        attribution_stats = []

        for j, grid_size in enumerate(grids):
            grid_start_time = time.time()

            if j == 0:
                # Initial model with custom architecture
                model = KAN(width=kan_width, grid=grid_size, k=3, seed=1, device=device)
            else:
                # Apply adaptive or combined densification
                if use_regular_refine:
                    # Test 2: Adaptive + Regular
                    model = model.refine(grid_size)
                    model, stats = adaptive_densify_model(model, dataset,
                                                         threshold=attribution_threshold,
                                                         new_grid_size=None,
                                                         device=device)
                else:
                    # Test 1: Adaptive only
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

            # Compute dense MSE
            dense_samples = 10000 if dim <= 4 else (5000 if dim == 10 else 1000)
            with torch.no_grad():
                dense_mse_final = dense_mse_error_from_dataset(model, dataset, true_func,
                                                        num_samples=dense_samples, device=device)
            for _ in range(epochs):
                dense_mse_errors.append(dense_mse_final)

            grid_time = time.time() - grid_start_time
            grid_times.append(grid_time)
            print(f"  Dataset {i} ({dataset_name}), grid {grid_size}: {grid_time:.2f}s total, "
                  f"{grid_time/epochs:.3f}s/epoch, {num_params} params")

        total_dataset_time = time.time() - dataset_start_time
        models[i] = model

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
                    'attribution_threshold': attribution_threshold,
                    'dimension': n_var,
                    'architecture': str(kan_width),
                    'architecture_type': architecture
                }

                # Add attribution stats if available
                if j > 0 and j - 1 < len(attribution_stats):
                    stats = attribution_stats[j - 1]
                    row['important_neurons'] = stats.get('important_neurons', None)
                    row['total_neurons'] = stats.get('total_neurons', None)

                rows.append(row)

        print(f"  Dataset {i} ({dataset_name}) complete: {total_dataset_time:.2f}s total")

    return pd.DataFrame(rows), models


def run_kan_baseline_test_custom_arch(datasets, grids, epochs, device, true_functions=None,
                                       dataset_names=None, kan_width=None, dim=None):
    """Modified baseline test with custom architecture"""
    import time
    import pandas as pd
    import torch
    import torch.nn as nn
    from kan import KAN
    from utils.metrics import count_parameters, dense_mse_error_from_dataset

    print(f"kan_baseline (regular refinement) custom_arch_{kan_width}")
    rows = []
    models = {}

    # Handle single dataset case
    datasets_list = [datasets] if not isinstance(datasets, list) else datasets
    true_functions_list = [true_functions] if not isinstance(true_functions, list) and true_functions is not None else true_functions
    dataset_names_list = [dataset_names] if not isinstance(dataset_names, list) and dataset_names is not None else dataset_names

    if dataset_names_list is None:
        dataset_names_list = [f"dataset_{i}" for i in range(len(datasets_list))]

    for i, dataset in enumerate(datasets_list):
        train_losses = []
        test_losses = []
        dense_mse_errors = []
        n_var = dataset['train_input'].shape[1]
        true_func = true_functions_list[i] if true_functions_list else None
        dataset_name = dataset_names_list[i]

        dataset_start_time = time.time()
        grid_times = []
        grid_param_counts = []

        for j, grid_size in enumerate(grids):
            grid_start_time = time.time()
            if j == 0:
                model = KAN(width=kan_width, grid=grid_size, k=3, seed=1, device=device)
            else:
                model = model.refine(grid_size)
            num_params = count_parameters(model)
            grid_param_counts.append(num_params)

            train_results = model.fit(dataset, opt="Adam", steps=epochs, log=1)
            train_losses += train_results['train_loss']
            test_losses += train_results['test_loss']

            # Compute dense MSE
            dense_samples = 10000 if dim <= 4 else (5000 if dim == 10 else 1000)
            with torch.no_grad():
                dense_mse_final = dense_mse_error_from_dataset(model, dataset, true_func,
                                                        num_samples=dense_samples, device=device)
            for _ in range(epochs):
                dense_mse_errors.append(dense_mse_final)

            grid_time = time.time() - grid_start_time
            grid_times.append(grid_time)
            print(f"  Dataset {i} ({dataset_name}), grid {grid_size}: {grid_time:.2f}s total, "
                  f"{grid_time/epochs:.3f}s/epoch, {num_params} params")

        total_dataset_time = time.time() - dataset_start_time
        models[i] = model

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
                    'attribution_threshold': None,
                    'dimension': n_var,
                    'architecture': str(kan_width),
                    'architecture_type': architecture
                })

        print(f"  Dataset {i} ({dataset_name}) complete: {total_dataset_time:.2f}s total")

    return pd.DataFrame(rows), models


# ============= Training with different approaches =============

print("Test 1: Training KANs with adaptive density (alternative to regular densification)...")
adaptive_only_results, adaptive_only_models = track_time(
    timers, "KAN adaptive density only",
    run_kan_adaptive_density_test_custom_arch,
    dataset, grids, epochs, device, False, 1e-2, true_function, dataset_name, kan_width, dim
)

print("\nTest 2: Training KANs with adaptive + regular densification...")
adaptive_regular_results, adaptive_regular_models = track_time(
    timers, "KAN adaptive + regular density",
    run_kan_adaptive_density_test_custom_arch,
    dataset, grids, epochs, device, True, 1e-2, true_function, dataset_name, kan_width, dim
)

print("\nTraining baseline KANs (regular refinement only for comparison)...")
baseline_results, baseline_models = track_time(
    timers, "KAN baseline",
    run_kan_baseline_test_custom_arch,
    dataset, grids, epochs, device, true_function, dataset_name, kan_width, dim
)

# Print timing summary
print_timing_summary(timers, f"Section 2.2 High-D ({dim}D, {architecture})", num_datasets=1)

all_results = {
    'adaptive_only': adaptive_only_results,
    'adaptive_regular': adaptive_regular_results,
    'baseline': baseline_results
}

print(f"\nResults summary:")
for model_type, df in all_results.items():
    print(f"  {model_type}: {df.shape[0]} rows, {df.shape[1]} columns")

# Print approach summary table
print_optimizer_summary(all_results, [dataset_name])

# Save results
run_name = f'section2_2_highd_{dim}d_{architecture}'
save_run(all_results, run_name,
         models={
             'adaptive_only': adaptive_only_models,
             'adaptive_regular': adaptive_regular_models,
             'baseline': baseline_models
         },
         epochs=epochs, device=str(device),
         dimension=dim, architecture=architecture, kan_width=kan_width)

print(f"\n{'='*60}")
print(f"Experiment complete! Results saved to: {run_name}")
print(f"{'='*60}")
