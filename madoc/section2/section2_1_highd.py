import sys
from pathlib import Path

# Add pykan to path (parent directory of madoc)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from kan import *
from utils import data_funcs as dfs
from utils import save_run, track_time, print_timing_summary, print_optimizer_summary
from utils import run_kan_optimizer_tests, run_kan_lm_tests
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Section 2.1 High-D: Optimizer Comparison on Higher-Dimensional Poisson PDEs')
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
print(f"Starting Section 2.1 High-D Optimizer Comparison ({dim}D, {architecture})")
print("="*60 + "\n")

timers = {}

# ============= Override KAN creation to use custom architecture =============
# We need to modify the test functions to accept a custom width parameter
# For now, we'll create a wrapper that passes the correct architecture

def run_kan_optimizer_tests_custom_arch(datasets, grids, epochs, device, optimizer_name,
                                         true_functions=None, dataset_names=None, kan_width=None):
    """Modified version of run_kan_optimizer_tests that accepts custom KAN architecture"""
    import time
    import pandas as pd
    import torch
    import torch.nn as nn
    from kan import KAN
    from utils.metrics import count_parameters, dense_mse_error_from_dataset

    print(f"kan_{optimizer_name.lower()}_custom_arch_{kan_width}")
    rows = []
    models = {}

    # Handle single dataset case
    datasets_list = [datasets] if not isinstance(datasets, list) else datasets
    true_functions_list = [true_functions] if not isinstance(true_functions, list) and true_functions is not None else true_functions
    dataset_names_list = [dataset_names] if not isinstance(dataset_names, list) and dataset_names is not None else dataset_names

    # Generate default names if not provided
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
                # Use custom architecture width
                model = KAN(width=kan_width, grid=grid_size, k=3, seed=1, device=device)
            else:
                model = model.refine(grid_size)
            num_params = count_parameters(model)
            grid_param_counts.append(num_params)

            # Train with specified optimizer
            train_results = model.fit(dataset, opt=optimizer_name, steps=epochs, log=1)
            train_losses += train_results['train_loss']
            test_losses += train_results['test_loss']

            # Compute dense MSE once at the end of this grid's training
            # Reduce dense_mse samples for very high dimensions
            dense_samples = 10000 if dim <= 4 else (5000 if dim == 10 else 1000)
            with torch.no_grad():
                dense_mse_final = dense_mse_error_from_dataset(model, dataset, true_func,
                                                        num_samples=dense_samples, device=device)
            # Store the same final dense MSE for all epochs in this grid
            for _ in range(epochs):
                dense_mse_errors.append(dense_mse_final)

            grid_time = time.time() - grid_start_time
            grid_times.append(grid_time)
            print(f"  Dataset {i} ({dataset_name}), grid {grid_size}: {grid_time:.2f}s total, {grid_time/epochs:.3f}s/epoch, {num_params} params")

        total_dataset_time = time.time() - dataset_start_time
        models[i] = model

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
                    'optimizer': optimizer_name,
                    'dimension': n_var,
                    'architecture': str(kan_width),
                    'architecture_type': architecture
                })

        print(f"  Dataset {i} ({dataset_name}) complete: {total_dataset_time:.2f}s total")

    return pd.DataFrame(rows), models


def run_kan_lm_tests_custom_arch(datasets, grids, epochs, device, true_functions=None,
                                   dataset_names=None, kan_width=None):
    """Modified version of run_kan_lm_tests that accepts custom KAN architecture"""
    import time
    import pandas as pd
    import torch
    import torch.nn as nn
    from kan import KAN
    from utils.metrics import count_parameters, dense_mse_error_from_dataset

    try:
        import torch_levenberg_marquardt as tlm
    except ImportError:
        raise ImportError("torch-levenberg-marquardt is not installed. Please install it to use LM optimizer.")

    print(f"kan_lm_custom_arch_{kan_width}")
    rows = []
    models = {}

    # Handle single dataset case
    datasets_list = [datasets] if not isinstance(datasets, list) else datasets
    true_functions_list = [true_functions] if not isinstance(true_functions, list) and true_functions is not None else true_functions
    dataset_names_list = [dataset_names] if not isinstance(dataset_names, list) and dataset_names is not None else dataset_names

    # Generate default names if not provided
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
                # Use custom architecture width
                model = KAN(width=kan_width, grid=grid_size, k=3, seed=1, device=device)
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
            # Reduce dense_mse samples for very high dimensions
            dense_samples = 10000 if dim <= 4 else (5000 if dim == 10 else 1000)
            with torch.no_grad():
                dense_mse_final = dense_mse_error_from_dataset(model, dataset, true_func,
                                                        num_samples=dense_samples, device=device)
            # Store the same final dense MSE for all epochs in this grid
            for _ in range(epochs):
                dense_mse_errors.append(dense_mse_final)

            grid_time = time.time() - grid_start_time
            grid_times.append(grid_time)
            print(f"  Dataset {i} ({dataset_name}), grid {grid_size}: {grid_time:.2f}s total, {grid_time/epochs:.3f}s/epoch, {num_params} params")

        total_dataset_time = time.time() - dataset_start_time
        models[i] = model

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
                    'optimizer': 'LM',
                    'dimension': n_var,
                    'architecture': str(kan_width),
                    'architecture_type': architecture
                })

        print(f"  Dataset {i} ({dataset_name}) complete: {total_dataset_time:.2f}s total")

    return pd.DataFrame(rows), models


# ============= Training with different optimizers =============

print("Training KANs with LBFGS optimizer (with dense MSE metrics)...")
lbfgs_results, lbfgs_models = track_time(timers, "KAN LBFGS training",
                                        run_kan_optimizer_tests_custom_arch,
                                        dataset, grids, epochs, device, "LBFGS",
                                        true_function, dataset_name, kan_width)

print("\nTraining KANs with LM optimizer (with dense MSE metrics)...")
lm_results, lm_models = track_time(timers, "KAN LM training",
                                    run_kan_lm_tests_custom_arch,
                                    dataset, grids, epochs, device,
                                    true_function, dataset_name, kan_width)

# Print timing summary
print_timing_summary(timers, f"Section 2.1 High-D ({dim}D, {architecture})", num_datasets=1)

all_results = {'lbfgs': lbfgs_results, 'lm': lm_results}
print(f"\nResults summary:")
for model_type, df in all_results.items():
    print(f"  {model_type}: {df.shape[0]} rows, {df.shape[1]} columns")

# Print optimizer summary table
print_optimizer_summary(all_results, [dataset_name])

# Save results
run_name = f'section2_1_highd_{dim}d_{architecture}'
save_run(all_results, run_name,
         models={'lbfgs': lbfgs_models, 'lm': lm_models},
         epochs=epochs, device=str(device),
         dimension=dim, architecture=architecture, kan_width=kan_width)

print(f"\n{'='*60}")
print(f"Experiment complete! Results saved to: {run_name}")
print(f"{'='*60}")
