"""
Plot function fits for best performing models from each class.

This script:
1. Loads results and models from the most recent run
2. Finds the best configuration for each model type (MLP, SIREN, KAN, KAN+Pruning)
3. Loads the saved models and reconstructs them
4. Plots the learned functions vs ground truth for all datasets
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import yaml

# Fix YAML loading for numpy scalars in KAN checkpoints
# The KAN library uses yaml.safe_load() but numpy scalars aren't safe_load compatible
# We'll monkey-patch yaml to use unsafe_load in the KAN context
_original_safe_load = yaml.safe_load

def _patched_safe_load(stream):
    """Use unsafe_load for KAN checkpoints which contain numpy objects"""
    try:
        return _original_safe_load(stream)
    except yaml.constructor.ConstructorError:
        # If safe_load fails, fall back to unsafe_load for numpy objects
        stream.seek(0)  # Reset stream position
        return yaml.unsafe_load(stream)

# Apply the patch
yaml.safe_load = _patched_safe_load

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_run, trad_nn as tnn, data_funcs as dfs
from kan import KAN


def find_latest_timestamp(section='section1_1'):
    """Find the most recent timestamp for a section"""
    sec_num = section.split('_')[-1]
    results_dir = Path(__file__).parent.parent / 'results' / f'sec{sec_num}_results'

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    # Find all timestamps
    timestamps = set()
    for f in results_dir.glob(f'{section}_*_mlp.pkl'):
        # Extract timestamp from filename: section1_1_TIMESTAMP_mlp.pkl
        timestamp = f.stem.replace(f'{section}_', '').replace('_mlp', '')
        timestamps.add(timestamp)

    if not timestamps:
        raise FileNotFoundError(f"No results found for {section}")

    return sorted(timestamps)[-1]  # Return most recent


def get_true_functions_and_names(section='section1_1'):
    """
    Get the list of true functions and dataset names for a section.

    Returns:
        Tuple of (true_functions, dataset_names, n_var)
    """
    if section == 'section1_1':
        # Section 1.1: Function Approximation
        freq = [1, 2, 3, 4, 5]
        true_functions = [dfs.sinusoid_1d(f) for f in freq]
        true_functions.extend([
            dfs.f_piecewise,
            dfs.f_sawtooth,
            dfs.f_polynomial,
            dfs.f_poisson_1d_highfreq
        ])
        dataset_names = [f'sin_freq{f}' for f in freq]
        dataset_names.extend(['piecewise', 'sawtooth', 'polynomial', 'poisson_1d_highfreq'])
        n_var = 1

    elif section == 'section1_2':
        # Section 1.2: 1D Poisson PDE
        true_functions = dfs.sec1_2
        dataset_names = ['poisson_1d_sin', 'poisson_1d_poly', 'poisson_1d_highfreq']
        n_var = 1

    elif section == 'section1_3':
        # Section 1.3: 2D Poisson PDE
        true_functions = [
            dfs.f_poisson_2d_sin,
            dfs.f_poisson_2d_poly,
            dfs.f_poisson_2d_highfreq,
            dfs.f_poisson_2d_spec
        ]
        dataset_names = ['poisson_2d_sin', 'poisson_2d_poly', 'poisson_2d_highfreq', 'poisson_2d_spec']
        n_var = 2

    else:
        raise ValueError(f"Unknown section: {section}")

    return true_functions, dataset_names, n_var


def load_and_reconstruct_mlp(models, results, dataset_idx, n_var=1, device='cpu'):
    """
    Load MLP model and reconstruct it.

    Args:
        models: Models dict from load_run
        results: Results dict from load_run
        dataset_idx: Dataset index
        n_var: Number of input variables
        device: Device to load model on

    Returns:
        Tuple of (model, config_info) or (None, None) if not available
    """
    if 'mlp' not in models or dataset_idx not in models['mlp']:
        return None, None

    # Get state dict
    state_dict = models['mlp'][dataset_idx]

    # Infer architecture from state_dict (robust approach)
    # Count linear layers: network.0.weight, network.2.weight, network.4.weight, etc.
    linear_layers = [k for k in state_dict.keys() if 'weight' in k and 'network.' in k]
    actual_depth = len(linear_layers)

    # Infer activation from the first activation layer in state_dict keys
    # MLP uses network.1 for first activation (Tanh/ReLU/SiLU have no parameters)
    # We need to get this from DataFrame
    mlp_df = results['mlp']
    mlp_dataset = mlp_df[mlp_df['dataset_idx'] == dataset_idx]

    # Get final epoch for each config, filtering out NaN
    final_epoch_rows = mlp_dataset.loc[
        mlp_dataset.groupby(['depth', 'activation'])['epoch'].idxmax()
    ]
    final_epoch_rows = final_epoch_rows[final_epoch_rows['dense_mse'].notna()]

    if len(final_epoch_rows) == 0:
        print(f"  Warning: No valid models found for MLP dataset {dataset_idx} (all diverged)")
        return None, None

    # Find best config from DataFrame
    best_row = final_epoch_rows.loc[final_epoch_rows['dense_mse'].idxmin()]
    df_depth = int(best_row['depth'])
    activation = best_row['activation']
    dense_mse = best_row['dense_mse']

    # Validate: check if actual_depth matches df_depth
    if actual_depth != df_depth:
        print(f"  Warning: MLP depth mismatch for dataset {dataset_idx}!")
        print(f"    DataFrame says depth={df_depth}, but saved model has depth={actual_depth}")
        print(f"    Using actual saved model depth={actual_depth}")
        depth = actual_depth
    else:
        depth = df_depth

    # Reconstruct model with ACTUAL architecture from saved model
    model = tnn.MLP(in_features=n_var, width=5, depth=depth, activation=activation)

    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"  Error loading MLP for dataset {dataset_idx}: {e}")
        return None, None

    model.to(device)
    model.eval()

    config_info = {
        'depth': depth,
        'activation': activation,
        'dense_mse': dense_mse,
        'label': f'MLP (d={depth}, {activation})'
    }

    return model, config_info


def load_and_reconstruct_siren(models, results, dataset_idx, n_var=1, device='cpu'):
    """
    Load SIREN model and reconstruct it.

    Args:
        models: Models dict from load_run
        results: Results dict from load_run
        dataset_idx: Dataset index
        n_var: Number of input variables
        device: Device to load model on

    Returns:
        Tuple of (model, config_info) or (None, None) if not available
    """
    if 'siren' not in models or dataset_idx not in models['siren']:
        return None, None

    # Get state dict
    state_dict = models['siren'][dataset_idx]

    # Infer architecture from state_dict (robust approach)
    # SIREN has: net.0 (input), net.2 (hidden1), net.4 (hidden2), ..., net.N (output)
    # Count linear layers
    linear_layers = [k for k in state_dict.keys() if 'weight' in k and 'net.' in k]
    actual_depth = len(linear_layers)

    # Find the best configuration from results (filtering NaN)
    siren_df = results['siren']
    siren_dataset = siren_df[siren_df['dataset_idx'] == dataset_idx]

    # Get final epoch for each config, filtering out NaN
    final_epoch_rows = siren_dataset.loc[
        siren_dataset.groupby('depth')['epoch'].idxmax()
    ]
    final_epoch_rows = final_epoch_rows[final_epoch_rows['dense_mse'].notna()]

    if len(final_epoch_rows) == 0:
        print(f"  Warning: No valid models found for SIREN dataset {dataset_idx} (all diverged)")
        return None, None

    # Find best config from DataFrame
    best_row = final_epoch_rows.loc[final_epoch_rows['dense_mse'].idxmin()]
    df_depth = int(best_row['depth'])
    dense_mse = best_row['dense_mse']

    # Validate: check if actual_depth matches df_depth
    if actual_depth != df_depth:
        print(f"  Warning: SIREN depth mismatch for dataset {dataset_idx}!")
        print(f"    DataFrame says depth={df_depth}, but saved model has depth={actual_depth}")
        print(f"    Using actual saved model depth={actual_depth}")
        depth = actual_depth
    else:
        depth = df_depth

    # Reconstruct model with ACTUAL architecture from saved model
    # SIREN: depth = total layers, hidden_layers = depth - 2 (input and output)
    model = tnn.SIREN(in_features=n_var, hidden_features=5, hidden_layers=depth-2, out_features=1)

    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"  Error loading SIREN for dataset {dataset_idx}: {e}")
        return None, None

    model.to(device)
    model.eval()

    config_info = {
        'depth': depth,
        'dense_mse': dense_mse,
        'label': f'SIREN (d={depth})'
    }

    return model, config_info


def load_kan_model(models, results, dataset_idx, device='cpu'):
    """
    Load KAN model from checkpoint.

    Args:
        models: Models dict from load_run
        results: Results dict from load_run
        dataset_idx: Dataset index
        device: Device to load model on

    Returns:
        Tuple of (model, config_info) or (None, None) if not available
    """
    if 'kan' not in models or dataset_idx not in models['kan']:
        return None, None

    # Get checkpoint path
    checkpoint_path = models['kan'][dataset_idx]

    # Load using KAN's method
    model = KAN.loadckpt(checkpoint_path)
    model.to(device)
    model.eval()

    # Get final performance from results
    kan_df = results['kan']
    kan_dataset = kan_df[(kan_df['dataset_idx'] == dataset_idx) & (~kan_df['is_pruned'])]

    if len(kan_dataset) > 0:
        final_row = kan_dataset.iloc[-1]
        dense_mse = final_row['dense_mse']
        grid_size = int(final_row['grid_size'])
    else:
        dense_mse = float('nan')
        grid_size = -1

    config_info = {
        'grid_size': grid_size,
        'dense_mse': dense_mse,
        'label': f'KAN (g={grid_size})'
    }

    return model, config_info


def load_kan_pruned_model(models, results, dataset_idx, device='cpu'):
    """
    Load pruned KAN model from checkpoint.

    Args:
        models: Models dict from load_run
        results: Results dict from load_run
        dataset_idx: Dataset index
        device: Device to load model on

    Returns:
        Tuple of (model, config_info) or (None, None) if not available
    """
    if 'kan_pruned' not in models or dataset_idx not in models['kan_pruned']:
        return None, None

    # Get checkpoint path
    checkpoint_path = models['kan_pruned'][dataset_idx]

    # Load using KAN's method
    model = KAN.loadckpt(checkpoint_path)
    model.to(device)
    model.eval()

    # Get final performance from results
    kan_pruning_df = results['kan_pruning']
    kan_dataset = kan_pruning_df[(kan_pruning_df['dataset_idx'] == dataset_idx) & (kan_pruning_df['is_pruned'])]

    if len(kan_dataset) > 0:
        pruned_row = kan_dataset.iloc[-1]
        dense_mse = pruned_row['dense_mse']
        grid_size = int(pruned_row['grid_size'])
    else:
        dense_mse = float('nan')
        grid_size = -1

    config_info = {
        'grid_size': grid_size,
        'dense_mse': dense_mse,
        'label': f'KAN+Pruning (g={grid_size})'
    }

    return model, config_info


def plot_function_fits(section='section1_1', timestamp=None, device='cpu', save_individual=False):
    """
    Plot function fits for all datasets and model types.

    Args:
        section: Section name (e.g., 'section1_1')
        timestamp: Specific timestamp to load, or None for most recent
        device: Device to run on ('cpu' or 'cuda')
        save_individual: If True, save individual plots for each dataset
    """
    # Find latest timestamp if not provided
    if timestamp is None:
        timestamp = find_latest_timestamp(section)
        print(f"Using most recent timestamp: {timestamp}")

    print(f"Loading results and models from {section}_{timestamp}...")

    # Load results and models
    results, meta, models = load_run(section, timestamp, load_models=True)

    # Get true functions and dataset info
    true_functions, dataset_names, n_var = get_true_functions_and_names(section)
    num_datasets = len(true_functions)

    print(f"\nGenerating function fit plots for {num_datasets} datasets...")

    # Determine grid layout
    if num_datasets <= 4:
        nrows, ncols = 2, 2
    elif num_datasets <= 6:
        nrows, ncols = 2, 3
    elif num_datasets <= 9:
        nrows, ncols = 3, 3
    else:
        nrows, ncols = 4, 3

    # Create figure for all datasets
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
    axes = axes.flatten() if num_datasets > 1 else [axes]

    # Generate x values for plotting
    if n_var == 1:
        x_plot = torch.linspace(-1, 1, 1000, device=device).reshape(-1, 1)
    else:
        # For 2D, plot a slice at y=0.5
        x_plot = torch.zeros(1000, 2, device=device)
        x_plot[:, 0] = torch.linspace(0, 1, 1000)
        x_plot[:, 1] = 0.5

    # Plot each dataset
    for dataset_idx in range(num_datasets):
        ax = axes[dataset_idx]
        true_func = true_functions[dataset_idx]
        dataset_name = dataset_names[dataset_idx]

        print(f"\n  Dataset {dataset_idx} ({dataset_name}):")

        # Get ground truth
        with torch.no_grad():
            y_true = true_func(x_plot)
            if y_true.dim() > 1:
                y_true = y_true.squeeze()

        # Plot ground truth
        x_np = x_plot[:, 0].cpu().numpy() if n_var == 2 else x_plot.cpu().numpy().flatten()
        y_true_np = y_true.cpu().numpy()
        ax.plot(x_np, y_true_np, 'k--', label='Ground Truth', linewidth=2.5, alpha=0.7)

        colors = {'mlp': 'C0', 'siren': 'C1', 'kan': 'C2', 'kan_pruned': 'C3'}

        # Load and plot MLP
        mlp_model, mlp_info = load_and_reconstruct_mlp(models, results, dataset_idx, n_var, device)
        if mlp_model is not None:
            with torch.no_grad():
                y_pred = mlp_model(x_plot).squeeze().cpu().numpy()
            ax.plot(x_np, y_pred, label=mlp_info['label'],
                   color=colors['mlp'], linewidth=2, alpha=0.8)
            print(f"    MLP: depth={mlp_info['depth']}, {mlp_info['activation']}, "
                  f"dense_mse={mlp_info['dense_mse']:.6e}")

        # Load and plot SIREN
        siren_model, siren_info = load_and_reconstruct_siren(models, results, dataset_idx, n_var, device)
        if siren_model is not None:
            with torch.no_grad():
                y_pred = siren_model(x_plot).squeeze().cpu().numpy()
            ax.plot(x_np, y_pred, label=siren_info['label'],
                   color=colors['siren'], linewidth=2, alpha=0.8)
            print(f"    SIREN: depth={siren_info['depth']}, "
                  f"dense_mse={siren_info['dense_mse']:.6e}")

        # Load and plot KAN
        kan_model, kan_info = load_kan_model(models, results, dataset_idx, device)
        if kan_model is not None:
            with torch.no_grad():
                y_pred = kan_model(x_plot).squeeze().cpu().numpy()
            ax.plot(x_np, y_pred, label=kan_info['label'],
                   color=colors['kan'], linewidth=2, alpha=0.8)
            print(f"    KAN: grid={kan_info['grid_size']}, "
                  f"dense_mse={kan_info['dense_mse']:.6e}")

        # Load and plot KAN+Pruning
        kan_pruned_model, kan_pruned_info = load_kan_pruned_model(models, results, dataset_idx, device)
        if kan_pruned_model is not None:
            with torch.no_grad():
                y_pred = kan_pruned_model(x_plot).squeeze().cpu().numpy()
            ax.plot(x_np, y_pred, label=kan_pruned_info['label'],
                   color=colors['kan_pruned'], linewidth=2, alpha=0.8, linestyle=':')
            print(f"    KAN+Pruning: grid={kan_pruned_info['grid_size']}, "
                  f"dense_mse={kan_pruned_info['dense_mse']:.6e}")

        # Format subplot
        xlabel = 'x' if n_var == 1 else 'x (at y=0.5)'
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel('y', fontsize=11)
        ax.set_title(f'Dataset {dataset_idx}: {dataset_name}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)

        # Save individual plot if requested
        if save_individual:
            fig_single, ax_single = plt.subplots(figsize=(10, 6))
            ax_single.plot(x_np, y_true_np, 'k--', label='Ground Truth', linewidth=3, alpha=0.7)

            if mlp_model is not None:
                with torch.no_grad():
                    y_pred = mlp_model(x_plot).squeeze().cpu().numpy()
                ax_single.plot(x_np, y_pred, label=mlp_info['label'],
                              color=colors['mlp'], linewidth=2.5, alpha=0.8)

            if siren_model is not None:
                with torch.no_grad():
                    y_pred = siren_model(x_plot).squeeze().cpu().numpy()
                ax_single.plot(x_np, y_pred, label=siren_info['label'],
                              color=colors['siren'], linewidth=2.5, alpha=0.8)

            if kan_model is not None:
                with torch.no_grad():
                    y_pred = kan_model(x_plot).squeeze().cpu().numpy()
                ax_single.plot(x_np, y_pred, label=kan_info['label'],
                              color=colors['kan'], linewidth=2.5, alpha=0.8)

            if kan_pruned_model is not None:
                with torch.no_grad():
                    y_pred = kan_pruned_model(x_plot).squeeze().cpu().numpy()
                ax_single.plot(x_np, y_pred, label=kan_pruned_info['label'],
                              color=colors['kan_pruned'], linewidth=2.5, alpha=0.8, linestyle=':')

            ax_single.set_xlabel(xlabel, fontsize=14)
            ax_single.set_ylabel('y', fontsize=14)
            ax_single.set_title(f'Function Fit: {dataset_name}', fontsize=16, fontweight='bold')
            ax_single.legend(fontsize=12, loc='best')
            ax_single.grid(True, alpha=0.3)

            output_file = Path(__file__).parent / f'function_fit_dataset_{dataset_idx}_{dataset_name}_{timestamp}.png'
            plt.tight_layout()
            fig_single.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig_single)
            print(f"    Saved individual plot to: {output_file}")

    # Hide unused subplots
    for idx in range(num_datasets, len(axes)):
        axes[idx].axis('off')

    # Save combined plot
    output_file = Path(__file__).parent / f'function_fits_all_datasets_{timestamp}.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved combined plot to: {output_file}")

    # Show plot
    plt.show()

    return fig, axes


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Plot function fits for best models from each class'
    )
    parser.add_argument('--section', type=str, default='section1_1',
                       help='Section to load (e.g., section1_1)')
    parser.add_argument('--timestamp', type=str, default=None,
                       help='Specific timestamp to load (default: most recent)')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (cpu or cuda)')
    parser.add_argument('--save-individual', action='store_true',
                       help='Save individual plots for each dataset')

    args = parser.parse_args()

    try:
        plot_function_fits(
            section=args.section,
            timestamp=args.timestamp,
            device=args.device,
            save_individual=args.save_individual
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nMake sure you have run the training script first and that models were saved.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
