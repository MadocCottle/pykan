"""
Plot function fits comparing Adam and LM optimizer results.

This script:
1. Loads results and models from the most recent run
2. Loads KAN models trained with Adam and LM optimizers
3. Plots the learned functions vs ground truth for all datasets
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_run, data_funcs as dfs
from kan import KAN


def find_latest_timestamp(section='section2_1'):
    """Find the most recent timestamp for a section"""
    sec_num = section.split('_')[-1]
    results_dir = Path(__file__).parent.parent / 'results' / f'sec{sec_num}_results'

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    # Find all timestamps
    timestamps = set()
    for f in results_dir.glob(f'{section}_*_adam.pkl'):
        # Extract timestamp from filename: section2_1_TIMESTAMP_adam.pkl
        timestamp = f.stem.replace(f'{section}_', '').replace('_adam', '')
        timestamps.add(timestamp)

    if not timestamps:
        raise FileNotFoundError(f"No results found for {section}")

    return sorted(timestamps)[-1]  # Return most recent


def get_true_functions_and_names(section='section2_1'):
    """
    Get the list of true functions and dataset names for a section.

    Returns:
        Tuple of (true_functions, dataset_names, n_var)
    """
    if section == 'section2_1':
        # Section 2.1: Optimizer Comparison on 2D Poisson PDE
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


def load_kan_model(models, results, optimizer, dataset_idx, device='cpu'):
    """
    Load KAN model from checkpoint.

    Args:
        models: Models dict from load_run
        results: Results dict from load_run
        optimizer: Optimizer name ('adam' or 'lm')
        dataset_idx: Dataset index
        device: Device to load model on

    Returns:
        Tuple of (model, config_info) or (None, None) if not available
    """
    if optimizer not in models or dataset_idx not in models[optimizer]:
        return None, None

    # Get checkpoint path (stored as directory path string for KAN models)
    checkpoint_info = models[optimizer][dataset_idx]

    # Check if it's a path string (KAN checkpoint) or state_dict
    if isinstance(checkpoint_info, str):
        # It's a KAN checkpoint path
        model = KAN.loadckpt(checkpoint_info)
    else:
        print(f"  Warning: Unexpected model format for {optimizer}, dataset {dataset_idx}")
        return None, None

    model.to(device)
    model.eval()

    # Get final performance from results
    df = results[optimizer]
    df_dataset = df[df['dataset_idx'] == dataset_idx]

    if len(df_dataset) > 0:
        final_row = df_dataset.iloc[-1]
        dense_mse = final_row['dense_mse']
        grid_size = int(final_row['grid_size'])
    else:
        dense_mse = float('nan')
        grid_size = -1

    config_info = {
        'grid_size': grid_size,
        'dense_mse': dense_mse,
        'optimizer': optimizer.upper(),
        'label': f'{optimizer.upper()} (g={grid_size})'
    }

    return model, config_info


def plot_function_fits(section='section2_1', timestamp=None, device='cpu', save_individual=False):
    """
    Plot function fits comparing Adam and LM optimizers for all datasets.

    Args:
        section: Section name (e.g., 'section2_1')
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
    print(f"Datasets are {n_var}D functions")

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

    # Generate x values for plotting (2D slice)
    # For 2D, plot a slice at y=0.5
    x_plot = torch.zeros(1000, 2, device=device)
    x_plot[:, 0] = torch.linspace(0, 1, 1000)
    x_plot[:, 1] = 0.5

    colors = {'adam': 'C0', 'lm': 'C1'}
    linestyles = {'adam': '-', 'lm': '--'}

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
        x_np = x_plot[:, 0].cpu().numpy()
        y_true_np = y_true.cpu().numpy()
        ax.plot(x_np, y_true_np, 'k-', label='Ground Truth', linewidth=3, alpha=0.7)

        # Load and plot models for each optimizer
        for optimizer in ['adam', 'lm']:
            if optimizer not in models:
                continue

            model, info = load_kan_model(models, results, optimizer, dataset_idx, device)

            if model is not None:
                with torch.no_grad():
                    y_pred = model(x_plot).squeeze().cpu().numpy()

                ax.plot(x_np, y_pred, label=info['label'],
                       color=colors[optimizer], linestyle=linestyles[optimizer],
                       linewidth=2.5, alpha=0.8)
                print(f"    {info['optimizer']}: grid={info['grid_size']}, "
                      f"dense_mse={info['dense_mse']:.6e}")

        # Format subplot
        xlabel = 'x (at y=0.5)'
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel('f(x, 0.5)', fontsize=11)
        ax.set_title(f'Dataset {dataset_idx}: {dataset_name}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)

        # Save individual plot if requested
        if save_individual:
            fig_single, ax_single = plt.subplots(figsize=(10, 6))
            ax_single.plot(x_np, y_true_np, 'k-', label='Ground Truth', linewidth=3, alpha=0.7)

            for optimizer in ['adam', 'lm']:
                if optimizer not in models:
                    continue

                model, info = load_kan_model(models, results, optimizer, dataset_idx, device)

                if model is not None:
                    with torch.no_grad():
                        y_pred = model(x_plot).squeeze().cpu().numpy()
                    ax_single.plot(x_np, y_pred, label=info['label'],
                                  color=colors[optimizer], linestyle=linestyles[optimizer],
                                  linewidth=2.5, alpha=0.8)

            ax_single.set_xlabel(xlabel, fontsize=14)
            ax_single.set_ylabel('f(x, 0.5)', fontsize=14)
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


def plot_2d_heatmap(section='section2_1', timestamp=None, dataset_idx=0, device='cpu'):
    """
    Plot 2D heatmap comparing optimizer results for a single dataset.

    Args:
        section: Section name (e.g., 'section2_1')
        timestamp: Specific timestamp to load, or None for most recent
        dataset_idx: Which dataset to visualize
        device: Device to run on ('cpu' or 'cuda')
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

    if dataset_idx >= len(true_functions):
        raise ValueError(f"Dataset index {dataset_idx} out of range. Max: {len(true_functions)-1}")

    true_func = true_functions[dataset_idx]
    dataset_name = dataset_names[dataset_idx]

    print(f"\nGenerating 2D heatmap for dataset {dataset_idx} ({dataset_name})...")

    # Create grid for 2D visualization
    resolution = 100
    x = torch.linspace(0, 1, resolution, device=device)
    y = torch.linspace(0, 1, resolution, device=device)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    grid = torch.stack([X.flatten(), Y.flatten()], dim=1)

    # Get ground truth
    with torch.no_grad():
        Z_true = true_func(grid).reshape(resolution, resolution).cpu().numpy()

    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot ground truth
    im0 = axes[0].imshow(Z_true, origin='lower', extent=[0, 1, 0, 1], cmap='viridis')
    axes[0].set_title('Ground Truth', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('x', fontsize=12)
    axes[0].set_ylabel('y', fontsize=12)
    plt.colorbar(im0, ax=axes[0])

    # Plot optimizer results
    for idx, optimizer in enumerate(['adam', 'lm'], start=1):
        if optimizer not in models:
            continue

        model, info = load_kan_model(models, results, optimizer, dataset_idx, device)

        if model is not None:
            with torch.no_grad():
                Z_pred = model(grid).reshape(resolution, resolution).cpu().numpy()

            im = axes[idx].imshow(Z_pred, origin='lower', extent=[0, 1, 0, 1], cmap='viridis')
            axes[idx].set_title(f'{info["optimizer"]} (dense_mse={info["dense_mse"]:.3e})',
                               fontsize=14, fontweight='bold')
            axes[idx].set_xlabel('x', fontsize=12)
            axes[idx].set_ylabel('y', fontsize=12)
            plt.colorbar(im, ax=axes[idx])

    plt.suptitle(f'2D Function Comparison: {dataset_name}', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save plot
    output_file = Path(__file__).parent / f'heatmap_2d_dataset_{dataset_idx}_{dataset_name}_{timestamp}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved heatmap to: {output_file}")

    plt.show()

    return fig, axes


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Plot function fits comparing optimizers'
    )
    parser.add_argument('--section', type=str, default='section2_1',
                       help='Section to load (e.g., section2_1)')
    parser.add_argument('--timestamp', type=str, default=None,
                       help='Specific timestamp to load (default: most recent)')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (cpu or cuda)')
    parser.add_argument('--save-individual', action='store_true',
                       help='Save individual plots for each dataset')
    parser.add_argument('--heatmap', action='store_true',
                       help='Generate 2D heatmap visualization')
    parser.add_argument('--dataset', type=int, default=0,
                       help='Dataset index for heatmap (default: 0)')

    args = parser.parse_args()

    try:
        if args.heatmap:
            plot_2d_heatmap(
                section=args.section,
                timestamp=args.timestamp,
                dataset_idx=args.dataset,
                device=args.device
            )
        else:
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
