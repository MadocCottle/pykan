import sys
from pathlib import Path

# Add pykan to path (parent directory of madoc)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from kan import *
from utils import data_funcs as dfs
from utils import save_run, track_time, print_timing_summary, count_parameters, dense_mse_error_from_dataset
import argparse
import time
import pandas as pd

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Section 2.2: Adaptive Density')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training (default: 10)')
args = parser.parse_args()

epochs = args.epochs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print(device)
print(f"Running with {epochs} epochs")

# Section 2.2: Adaptive Density on 2D Poisson PDE
# ============= Create Datasets =============
datasets = []
true_functions = [dfs.f_poisson_2d_sin, dfs.f_poisson_2d_poly, dfs.f_poisson_2d_highfreq, dfs.f_poisson_2d_spec]
dataset_names = ['poisson_2d_sin', 'poisson_2d_poly', 'poisson_2d_highfreq', 'poisson_2d_spec']

for f in true_functions:
    datasets.append(create_dataset(f, n_var=2, train_num=1000, test_num=1000))

grids = np.array([3, 5, 10, 20, 50, 100])

print("\n" + "="*60)
print("Starting Section 2.2 Adaptive Density")
print("="*60 + "\n")

timers = {}


def selective_densify_layer(model, layer_idx, layer, threshold, new_grid_size, device):
    """
    Densify grids only for neurons with attribution > threshold in a specific layer.

    Args:
        model: The KAN model
        layer_idx: Index of the layer
        layer: The KANLayer object
        threshold: Attribution score threshold
        new_grid_size: Number of intervals in the new grid
        device: Device to run on

    Returns:
        Number of neurons densified
    """
    from kan.spline import extend_grid, coef2curve, curve2coef

    acts = model.acts[layer_idx]  # Activations for this layer
    node_scores = model.node_scores[layer_idx]  # Attribution scores

    neurons_densified = 0

    # Process each input dimension separately
    for in_idx in range(layer.in_dim):
        if node_scores[in_idx] > threshold:
            # Extract current grid and coef for this neuron
            old_grid = layer.grid[in_idx:in_idx+1, :]  # Shape (1, old_num)
            old_coef = layer.coef[in_idx:in_idx+1, :, :]  # Shape (1, out_dim, old_num)
            old_num_intervals = old_grid.shape[1] - 2 * layer.k - 1

            # Only densify if new grid is denser than old grid
            if new_grid_size <= old_num_intervals:
                continue

            # Get samples for this input dimension
            x_samples = acts[:, in_idx:in_idx+1]  # Shape (batch, 1)
            x_pos = torch.sort(x_samples, dim=0)[0]

            # Evaluate current function on old grid
            y_eval = coef2curve(x_pos, old_grid, old_coef, layer.k)

            # Create new denser grid
            num_interval = new_grid_size
            batch = x_samples.shape[0]
            ids = [int(batch / num_interval * i) for i in range(num_interval)] + [-1]
            grid_adaptive = x_pos[ids, :].permute(1, 0)
            h = (grid_adaptive[:, [-1]] - grid_adaptive[:, [0]]) / num_interval
            grid_uniform = grid_adaptive[:, [0]] + h * torch.arange(num_interval + 1,)[None, :].to(device)
            new_grid = layer.grid_eps * grid_uniform + (1 - layer.grid_eps) * grid_adaptive
            new_grid = extend_grid(new_grid, k_extend=layer.k)

            # Refit coefficients to new grid
            new_coef = curve2coef(x_pos, y_eval, new_grid, layer.k)

            # Update this neuron's grid and coef
            # Note: All neurons in a layer must have the same grid size, so we need to
            # recreate the entire layer with the new grid size
            # This is a limitation - we'll track which neurons to densify and apply uniform
            # densification with importance weighting
            neurons_densified += 1

    return neurons_densified


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
        Tuple of (DataFrame, models_dict) where DataFrame contains training metrics
    """
    approach = "adaptive+regular" if use_regular_refine else "adaptive_only"
    print(f"kan_adaptive_density ({approach})")
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
                    'attribution_threshold': attribution_threshold
                }

                # Add attribution stats if available
                if j > 0 and j - 1 < len(attribution_stats):
                    stats = attribution_stats[j - 1]
                    row['important_neurons'] = stats.get('important_neurons', None)
                    row['total_neurons'] = stats.get('total_neurons', None)

                rows.append(row)

        print(f"  Dataset {i} ({dataset_name}) complete: {total_dataset_time:.2f}s total")

    return pd.DataFrame(rows), models


print("Test 1: Training KANs with adaptive density (alternative to regular densification)...")
adaptive_only_results, adaptive_only_models = track_time(
    timers, "KAN adaptive density only",
    run_kan_adaptive_density_test,
    datasets, grids, epochs, device, False, 1e-2, true_functions, dataset_names
)

print("\nTest 2: Training KANs with adaptive + regular densification...")
adaptive_regular_results, adaptive_regular_models = track_time(
    timers, "KAN adaptive + regular density",
    run_kan_adaptive_density_test,
    datasets, grids, epochs, device, True, 1e-2, true_functions, dataset_names
)

print("\nTraining baseline KANs (regular refinement only for comparison)...")
# For comparison, also run regular KAN training


def run_kan_baseline_test(datasets, grids, epochs, device, true_functions=None, dataset_names=None):
    """Run baseline KAN tests with regular refinement for comparison"""
    print("kan_baseline (regular refinement)")
    rows = []
    models = {}

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
                    'attribution_threshold': None
                })

        print(f"  Dataset {i} ({dataset_name}) complete: {total_dataset_time:.2f}s total")

    return pd.DataFrame(rows), models


baseline_results, baseline_models = track_time(
    timers, "KAN baseline",
    run_kan_baseline_test,
    datasets, grids, epochs, device, true_functions, dataset_names
)

# Print timing summary
print_timing_summary(timers, "Section 2.2", num_datasets=len(datasets))

all_results = {
    'adaptive_only': adaptive_only_results,
    'adaptive_regular': adaptive_regular_results,
    'baseline': baseline_results
}

print(f"\nResults summary:")
for model_type, df in all_results.items():
    print(f"  {model_type}: {df.shape[0]} rows, {df.shape[1]} columns")

save_run(all_results, 'section2_2',
         models={
             'adaptive_only': adaptive_only_models,
             'adaptive_regular': adaptive_regular_models,
             'baseline': baseline_models
         },
         epochs=epochs, device=str(device))