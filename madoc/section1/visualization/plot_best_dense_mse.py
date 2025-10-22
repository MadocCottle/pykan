"""
Plot Dense MSE over epochs for the best performing model from each class.

This script:
1. Loads results from the most recent run
2. Finds the best configuration for each model type (MLP, SIREN, KAN, KAN+Pruning)
3. Plots their dense MSE evolution over training epochs
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_run


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


def find_best_model_config(df_final, config_cols):
    """
    Find the configuration with the lowest final dense MSE.

    Args:
        df_final: DataFrame with final epoch results
        config_cols: List of columns that define a unique configuration

    Returns:
        Dictionary with the best configuration
    """
    # Filter out NaN values
    df_clean = df_final[df_final['dense_mse'].notna()]

    if len(df_clean) == 0:
        return None

    # Find row with minimum dense_mse
    best_idx = df_clean['dense_mse'].idxmin()
    best_row = df_clean.loc[best_idx]

    # Return config as dict
    config = {col: best_row[col] for col in config_cols}
    config['final_dense_mse'] = best_row['dense_mse']

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
        if key != 'final_dense_mse' and key in df.columns:
            mask &= (df[key] == value)

    # Extract and sort by epoch
    curve = df[mask].sort_values('epoch').copy()

    # Filter out NaN values
    curve = curve[curve['dense_mse'].notna()]

    return curve


def plot_best_dense_mse(section='section1_1', timestamp=None, dataset_idx=0):
    """
    Plot dense MSE over epochs for the best model from each class.

    Args:
        section: Section name (e.g., 'section1_1')
        timestamp: Specific timestamp to load, or None for most recent
        dataset_idx: Which dataset to analyze (default: 0)
    """
    # Load results
    if timestamp is None:
        timestamp = find_latest_timestamp(section)
        print(f"Using most recent timestamp: {timestamp}")

    print(f"Loading results from {section}_{timestamp}...")
    results, meta = load_run(section, timestamp)

    # Get final epoch for each model type
    print("\nFinding best configurations...")

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
    print(f"\nBest configurations for dataset {dataset_idx}:")
    if mlp_best_config:
        print(f"  MLP: depth={mlp_best_config['depth']}, "
              f"activation={mlp_best_config['activation']}, "
              f"final_dense_mse={mlp_best_config['final_dense_mse']:.6e}")
    if siren_best_config:
        print(f"  SIREN: depth={siren_best_config['depth']}, "
              f"final_dense_mse={siren_best_config['final_dense_mse']:.6e}")
    if kan_best_config:
        print(f"  KAN: grid_size={kan_best_config['grid_size']}, "
              f"final_dense_mse={kan_best_config['final_dense_mse']:.6e}")
    if kan_pruning_best_config:
        print(f"  KAN+Pruning: grid_size={kan_pruning_best_config['grid_size']}, "
              f"final_dense_mse={kan_pruning_best_config['final_dense_mse']:.6e}")

    # Extract training curves for best configs
    print("\nExtracting training curves...")

    curves = {}
    if mlp_best_config:
        curves['MLP'] = extract_training_curve(results['mlp'], mlp_best_config)
    if siren_best_config:
        curves['SIREN'] = extract_training_curve(results['siren'], siren_best_config)
    if kan_best_config:
        curves['KAN'] = extract_training_curve(results['kan'], kan_best_config)
    if kan_pruning_best_config:
        curves['KAN+Pruning'] = extract_training_curve(results['kan_pruning'], kan_pruning_best_config)

    # Create plot
    print("\nGenerating plot...")
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'MLP': 'C0', 'SIREN': 'C1', 'KAN': 'C2', 'KAN+Pruning': 'C3'}
    markers = {'MLP': 'o', 'SIREN': 's', 'KAN': '^', 'KAN+Pruning': 'D'}

    for model_name, curve in curves.items():
        if len(curve) > 0:
            ax.plot(curve['epoch'], curve['dense_mse'],
                   label=model_name, color=colors[model_name],
                   marker=markers[model_name], markersize=4,
                   linewidth=2, alpha=0.8)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Dense MSE', fontsize=12)
    ax.set_title(f'Dense MSE Evolution: Best Models (Dataset {dataset_idx})', fontsize=14)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='best')

    plt.tight_layout()

    # Save plot
    output_dir = Path(__file__).parent
    output_file = output_dir / f'best_dense_mse_dataset_{dataset_idx}_{timestamp}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")

    # Show plot
    plt.show()

    return fig, ax, curves


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Plot Dense MSE evolution for best models from each class'
    )
    parser.add_argument('--section', type=str, default='section1_1',
                       help='Section to load (e.g., section1_1)')
    parser.add_argument('--timestamp', type=str, default=None,
                       help='Specific timestamp to load (default: most recent)')
    parser.add_argument('--dataset', type=int, default=0,
                       help='Dataset index to analyze (default: 0)')

    args = parser.parse_args()

    try:
        plot_best_dense_mse(
            section=args.section,
            timestamp=args.timestamp,
            dataset_idx=args.dataset
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
