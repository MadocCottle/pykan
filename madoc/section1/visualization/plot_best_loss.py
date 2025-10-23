"""
Plot training and test loss curves for the best performing model from each class.

This script:
1. Loads results from the most recent run
2. Finds the best configuration for each model type (MLP, SIREN, KAN, KAN+Pruning)
3. Plots their loss evolution over training epochs for each dataset
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_run, data_funcs as dfs


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


def get_dataset_names(section='section1_1'):
    """
    Get the list of dataset names for a section.

    Returns:
        List of dataset names
    """
    if section == 'section1_1':
        # Section 1.1: Function Approximation
        freq = [1, 2, 3, 4, 5]
        dataset_names = [f'sin_freq{f}' for f in freq]
        dataset_names.extend(['piecewise', 'sawtooth', 'polynomial', 'poisson_1d_highfreq'])

    elif section == 'section1_2':
        # Section 1.2: 1D Poisson PDE
        dataset_names = ['poisson_1d_sin', 'poisson_1d_poly', 'poisson_1d_highfreq']

    elif section == 'section1_3':
        # Section 1.3: 2D Poisson PDE
        dataset_names = ['poisson_2d_sin', 'poisson_2d_poly', 'poisson_2d_highfreq', 'poisson_2d_spec']

    else:
        raise ValueError(f"Unknown section: {section}")

    return dataset_names


def find_best_model_config(df_final, config_cols):
    """
    Find the configuration with the lowest final test loss.

    Args:
        df_final: DataFrame with final epoch results
        config_cols: List of columns that define a unique configuration

    Returns:
        Dictionary with the best configuration
    """
    # Filter out NaN values
    df_clean = df_final[df_final['test_loss'].notna()].copy()

    if len(df_clean) == 0:
        return None

    # Convert numpy arrays to scalar values (test_loss may be stored as 0-d arrays)
    def extract_scalar(x):
        if isinstance(x, np.ndarray):
            return x.item()
        return x

    df_clean['test_loss'] = df_clean['test_loss'].apply(extract_scalar)

    # Ensure test_loss is numeric (convert from object dtype if needed)
    df_clean['test_loss'] = pd.to_numeric(df_clean['test_loss'], errors='coerce')

    # Filter out any NaN values that resulted from conversion
    df_clean = df_clean[df_clean['test_loss'].notna()]

    if len(df_clean) == 0:
        return None

    # Find row with minimum test_loss
    best_idx = df_clean['test_loss'].idxmin()
    best_row = df_clean.loc[best_idx]

    # Return config as dict
    config = {col: best_row[col] for col in config_cols}
    config['final_test_loss'] = best_row['test_loss']
    config['final_dense_mse'] = best_row['dense_mse']  # Keep for reference

    return config


def extract_training_curve(df, config):
    """
    Extract the training curve for a specific configuration.

    Args:
        df: Full DataFrame with all epochs
        config: Dictionary specifying the configuration

    Returns:
        DataFrame sorted by epoch with training metrics
    """
    # Build filter condition
    mask = pd.Series([True] * len(df))
    for key, value in config.items():
        if key not in ['final_test_loss', 'final_dense_mse'] and key in df.columns:
            mask &= (df[key] == value)

    # Extract and sort by epoch
    curve = df[mask].sort_values('epoch').copy()

    # Filter out NaN values in loss columns
    curve = curve[curve['test_loss'].notna() & curve['train_loss'].notna()]

    # Convert numpy arrays to scalar values for loss columns
    def extract_scalar(x):
        if isinstance(x, np.ndarray):
            return x.item()
        return x

    if len(curve) > 0:
        curve['test_loss'] = curve['test_loss'].apply(extract_scalar)
        curve['train_loss'] = curve['train_loss'].apply(extract_scalar)

    return curve


def plot_best_loss_curves(section='section1_1', timestamp=None, loss_type='test', show=False):
    """
    Plot loss curves over epochs for the best model from each class for all datasets.

    Args:
        section: Section name (e.g., 'section1_1')
        timestamp: Specific timestamp to load, or None for most recent
        loss_type: Which loss to plot - 'train', 'test', or 'both' (default: 'test')
        show: If True, display plot in window; otherwise only save to file (default: False)
    """
    # Load results
    if timestamp is None:
        timestamp = find_latest_timestamp(section)
        print(f"Using most recent timestamp: {timestamp}")

    print(f"Loading results from {section}_{timestamp}...")
    results, meta = load_run(section, timestamp)

    # Get dataset names
    dataset_names = get_dataset_names(section)
    num_datasets = len(dataset_names)

    print(f"\nGenerating loss curve plots for {num_datasets} datasets...")

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

    colors = {'MLP': 'C0', 'SIREN': 'C1', 'KAN': 'C2', 'KAN+Pruning': 'C3'}
    markers = {'MLP': 'o', 'SIREN': 's', 'KAN': '^', 'KAN+Pruning': 'D'}

    # Plot each dataset
    for dataset_idx in range(num_datasets):
        ax = axes[dataset_idx]
        dataset_name = dataset_names[dataset_idx]

        print(f"\n  Dataset {dataset_idx} ({dataset_name}):")

        # MLP: Get final epoch for each (dataset_idx, depth, activation)
        mlp_final = results['mlp'].loc[
            results['mlp'].groupby(['dataset_idx', 'depth', 'activation'])['epoch'].idxmax()
        ]
        mlp_final = mlp_final[mlp_final['dataset_idx'] == dataset_idx]
        mlp_best_config = find_best_model_config(mlp_final, ['dataset_idx', 'depth', 'activation'])

        # SIREN: Get final epoch for each (dataset_idx, depth)
        siren_final = results['siren'].loc[
            results['siren'].groupby(['dataset_idx', 'depth'])['epoch'].idxmax()
        ]
        siren_final = siren_final[siren_final['dataset_idx'] == dataset_idx]
        siren_best_config = find_best_model_config(siren_final, ['dataset_idx', 'depth'])

        # KAN: Get final epoch for each (dataset_idx, grid_size), excluding pruned
        kan_unpruned = results['kan'][~results['kan']['is_pruned']]
        kan_final = kan_unpruned.loc[
            kan_unpruned.groupby(['dataset_idx', 'grid_size'])['epoch'].idxmax()
        ]
        kan_final = kan_final[kan_final['dataset_idx'] == dataset_idx]
        kan_best_config = find_best_model_config(kan_final, ['dataset_idx', 'grid_size'])

        # KAN Pruning: Similar to KAN but from the pruning results
        kan_pruning_unpruned = results['kan_pruning'][~results['kan_pruning']['is_pruned']]
        kan_pruning_final = kan_pruning_unpruned.loc[
            kan_pruning_unpruned.groupby(['dataset_idx', 'grid_size'])['epoch'].idxmax()
        ]
        kan_pruning_final = kan_pruning_final[kan_pruning_final['dataset_idx'] == dataset_idx]
        kan_pruning_best_config = find_best_model_config(kan_pruning_final, ['dataset_idx', 'grid_size'])

        # Print best configurations
        if mlp_best_config:
            print(f"    MLP: depth={mlp_best_config['depth']}, "
                  f"activation={mlp_best_config['activation']}, "
                  f"final_test_loss={mlp_best_config['final_test_loss']:.6e}")
        if siren_best_config:
            print(f"    SIREN: depth={siren_best_config['depth']}, "
                  f"final_test_loss={siren_best_config['final_test_loss']:.6e}")
        if kan_best_config:
            print(f"    KAN: grid_size={kan_best_config['grid_size']}, "
                  f"final_test_loss={kan_best_config['final_test_loss']:.6e}")
        if kan_pruning_best_config:
            print(f"    KAN+Pruning: grid_size={kan_pruning_best_config['grid_size']}, "
                  f"final_test_loss={kan_pruning_best_config['final_test_loss']:.6e}")

        # Extract training curves for best configs
        curves = {}
        if mlp_best_config:
            curves['MLP'] = extract_training_curve(results['mlp'], mlp_best_config)
        if siren_best_config:
            curves['SIREN'] = extract_training_curve(results['siren'], siren_best_config)
        if kan_best_config:
            curves['KAN'] = extract_training_curve(results['kan'], kan_best_config)
        if kan_pruning_best_config:
            curves['KAN+Pruning'] = extract_training_curve(results['kan_pruning'], kan_pruning_best_config)

        # Plot curves
        loss_col = f'{loss_type}_loss'
        for model_name, curve in curves.items():
            if len(curve) > 0:
                ax.plot(curve['epoch'], curve[loss_col],
                       label=model_name, color=colors[model_name],
                       marker=markers[model_name], markersize=4,
                       linewidth=2, alpha=0.8)

        # Format subplot
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(f'{loss_type.capitalize()} Loss', fontsize=11)
        ax.set_title(f'Dataset {dataset_idx}: {dataset_name}', fontsize=12, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='best')

    # Hide unused subplots
    for idx in range(num_datasets, len(axes)):
        axes[idx].axis('off')

    # Save combined plot
    output_dir = Path(__file__).parent
    output_file = output_dir / f'best_loss_curves_all_datasets_{timestamp}.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved combined plot to: {output_file}")

    # Show plot only if requested
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, axes


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Plot training/test loss curves for best models from each class across all datasets'
    )
    parser.add_argument('--section', type=str, default='section1_1',
                       help='Section to load (e.g., section1_1)')
    parser.add_argument('--timestamp', type=str, default=None,
                       help='Specific timestamp to load (default: most recent)')
    parser.add_argument('--loss-type', type=str, default='test',
                       choices=['train', 'test'],
                       help='Which loss to plot: train or test (default: test)')
    parser.add_argument('--show', action='store_true',
                       help='Display the plot in a window (default: only save to file)')

    args = parser.parse_args()

    try:
        plot_best_loss_curves(
            section=args.section,
            timestamp=args.timestamp,
            loss_type=args.loss_type,
            show=args.show
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nMake sure you have run the training script first to generate results.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
