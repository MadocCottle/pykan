import sys
from pathlib import Path

# Add pykan to path (parent directory of madoc)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from kan import *
from utils import data_funcs as dfs
from utils import save_run, track_time, print_timing_summary, count_parameters, dense_mse_error_from_dataset
from lm_optimizer import LevenbergMarquardt
import argparse
import time
import pandas as pd

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Section 2.1: Optimizer Comparison')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training (default: 10)')
args = parser.parse_args()

epochs = args.epochs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print(device)
print(f"Running with {epochs} epochs")

# Section 2.1: Optimizer Comparison on 2D Poisson PDE
# ============= Create Datasets =============
datasets = []
true_functions = [dfs.f_poisson_2d_sin, dfs.f_poisson_2d_poly, dfs.f_poisson_2d_highfreq, dfs.f_poisson_2d_spec]
dataset_names = ['poisson_2d_sin', 'poisson_2d_poly', 'poisson_2d_highfreq', 'poisson_2d_spec']

for f in true_functions:
    datasets.append(create_dataset(f, n_var=2, train_num=1000, test_num=1000))

grids = np.array([3, 5, 10, 20, 50, 100])

print("\n" + "="*60)
print("Starting Section 2.1 Optimizer Comparison")
print("="*60 + "\n")

timers = {}


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
        Tuple of (DataFrame, models_dict) where:
        - DataFrame has columns: dataset_idx, dataset_name, grid_size, epoch, train_loss, test_loss,
                                 dense_mse, total_time, time_per_epoch, num_params, optimizer
        - models_dict maps dataset_idx -> trained model
    """
    print(f"kan_{optimizer_name.lower()}")
    rows = []
    models = {}

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

    return pd.DataFrame(rows), models


def run_kan_lm_tests(datasets, grids, epochs, device, true_functions=None, dataset_names=None):
    """Run KAN tests with custom LM optimizer

    Args:
        datasets: List of datasets
        grids: List of grid sizes to test
        epochs: Number of training epochs per grid
        device: Device to run on
        true_functions: List of true functions for each dataset (required)
        dataset_names: List of descriptive names for each dataset (optional)

    Returns:
        Tuple of (DataFrame, models_dict) where:
        - DataFrame has columns: dataset_idx, dataset_name, grid_size, epoch, train_loss, test_loss,
                                 dense_mse, total_time, time_per_epoch, num_params, optimizer
        - models_dict maps dataset_idx -> trained model
    """
    print("kan_lm")
    rows = []
    models = {}

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

            # Train with LM optimizer manually
            optimizer = LevenbergMarquardt(model.parameters(), lr=1.0, damping=1e-3)
            criterion = nn.MSELoss()

            for epoch in range(epochs):
                def closure():
                    optimizer.zero_grad()
                    pred = model(dataset['train_input'])
                    loss = criterion(pred, dataset['train_label'])
                    loss.backward()
                    return loss

                optimizer.step(closure)

                with torch.no_grad():
                    train_pred = model(dataset['train_input'])
                    test_pred = model(dataset['test_input'])
                    train_loss = criterion(train_pred, dataset['train_label']).item()
                    test_loss = criterion(test_pred, dataset['test_label']).item()
                    train_losses.append(train_loss)
                    test_losses.append(test_loss)

                    # Compute dense MSE
                    dense_mse = dense_mse_error_from_dataset(model, dataset, true_func,
                                                            num_samples=10000, device=device)
                    dense_mse_errors.append(dense_mse)

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
                    'optimizer': 'LM'
                })

        print(f"  Dataset {i} ({dataset_name}) complete: {total_dataset_time:.2f}s total")

    return pd.DataFrame(rows), models


print("Training KANs with ADAM optimizer (with dense MSE metrics)...")
adam_results, adam_models = track_time(timers, "KAN ADAM training",
                                        run_kan_optimizer_tests,
                                        datasets, grids, epochs, device, "Adam", true_functions, dataset_names)

print("\nTraining KANs with LM optimizer (with dense MSE metrics)...")
lm_results, lm_models = track_time(timers, "KAN LM training",
                                    run_kan_lm_tests,
                                    datasets, grids, epochs, device, true_functions, dataset_names)

# Print timing summary
print_timing_summary(timers, "Section 2.1", num_datasets=len(datasets))

all_results = {'adam': adam_results, 'lm': lm_results}
print(f"\nResults summary:")
for model_type, df in all_results.items():
    print(f"  {model_type}: {df.shape[0]} rows, {df.shape[1]} columns")

save_run(all_results, 'section2_1',
         models={'adam': adam_models, 'lm': lm_models},
         epochs=epochs, device=str(device))
# Note: Derivable metadata (grids, num_datasets, dataset_names) can be obtained from DataFrames:
# - grids: adam_results['grid_size'].unique()
# - num_datasets: adam_results['dataset_idx'].nunique()
# - dataset_names: adam_results['dataset_name'].unique()
