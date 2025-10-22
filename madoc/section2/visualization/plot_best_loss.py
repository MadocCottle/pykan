"""
Plot dense MSE curves for optimizer/approach comparisons in Section 2.

This script:
1. Loads results from the most recent run
2. Plots dense MSE evolution over training epochs
3. Section 2.1: Compares optimizers (Adam, LBFGS, LM)
4. Section 2.2: Compares approaches (Adaptive Only, Adaptive+Regular, Baseline)
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_run


def find_latest_timestamp(section='section2_1'):
    """Find the most recent timestamp for a section"""
    sec_num = section.split('_')[-1]
    results_dir = Path(__file__).parent.parent / 'results' / f'sec{sec_num}_results'

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    timestamps = set()

    # Look for different file patterns depending on section
    if section == 'section2_1':
        patterns = [f'{section}_*_adam.pkl', f'{section}_*_lbfgs.pkl', f'{section}_*_lm.pkl']
    elif section == 'section2_2':
        patterns = [f'{section}_*_adaptive_only.pkl', f'{section}_*_baseline.pkl']
    else:
        patterns = [f'{section}_*.pkl']

    for pattern in patterns:
        for f in results_dir.glob(pattern):
            parts = f.stem.split('_')
            if len(parts) >= 4:
                timestamp = parts[2]
                timestamps.add(timestamp)

    if not timestamps:
        raise FileNotFoundError(f"No results found for {section}")

    return sorted(timestamps)[-1]


def get_dataset_names(section='section2_1'):
    """
    Get the list of dataset names for a section.

    Returns:
        List of dataset names
    """
    if section in ['section2_1', 'section2_2']:
        # Both use 2D Poisson PDE datasets
        dataset_names = ['poisson_2d_sin', 'poisson_2d_poly', 'poisson_2d_highfreq', 'poisson_2d_spec']
    else:
        raise ValueError(f"Unknown section: {section}")

    return dataset_names


def plot_best_loss_curves(section='section2_1', timestamp=None, metric='dense_mse'):
    """
    Plot loss curves over epochs for all optimizers/approaches for all datasets.

    Args:
        section: Section name (e.g., 'section2_1', 'section2_2')
        timestamp: Specific timestamp to load, or None for most recent
        metric: Which metric to plot - 'dense_mse', 'test_loss', or 'train_loss' (default: 'dense_mse')
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

    print(f"\nGenerating {metric} curve plots for {num_datasets} datasets...")

    # Determine grid layout
    if num_datasets <= 4:
        nrows, ncols = 2, 2
    elif num_datasets <= 6:
        nrows, ncols = 2, 3
    else:
        nrows, ncols = 3, 3

    # Create figure for all datasets
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
    axes = axes.flatten() if num_datasets > 1 else [axes]

    # Define colors and markers based on section
    if section == 'section2_1':
        # Optimizer comparison
        colors = {'adam': 'C0', 'lbfgs': 'C1', 'lm': 'C2'}
        markers = {'adam': 'o', 'lbfgs': 's', 'lm': '^'}
        labels = {'adam': 'Adam', 'lbfgs': 'LBFGS', 'lm': 'LM'}
    elif section == 'section2_2':
        # Adaptive density comparison
        colors = {'adaptive_only': 'C0', 'adaptive_regular': 'C1', 'adaptive+regular': 'C1', 'baseline': 'C2'}
        markers = {'adaptive_only': 'o', 'adaptive_regular': 's', 'adaptive+regular': 's', 'baseline': '^'}
        labels = {'adaptive_only': 'Adaptive Only', 'adaptive_regular': 'Adaptive+Regular',
                 'adaptive+regular': 'Adaptive+Regular', 'baseline': 'Baseline'}
    else:
        colors = {}
        markers = {}
        labels = {}

    # Plot each dataset
    for dataset_idx in range(num_datasets):
        ax = axes[dataset_idx]
        dataset_name = dataset_names[dataset_idx]

        print(f"\n  Dataset {dataset_idx} ({dataset_name}):")

        # Plot each optimizer/approach
        for opt_name, df in results.items():
            df_dataset = df[df['dataset_idx'] == dataset_idx].copy()

            if len(df_dataset) == 0:
                continue

            # Sort by epoch
            df_dataset = df_dataset.sort_values('epoch')

            # Get color and marker
            color = colors.get(opt_name, f'C{list(results.keys()).index(opt_name)}')
            marker = markers.get(opt_name, 'o')
            label = labels.get(opt_name, opt_name.upper())

            # Plot
            ax.plot(df_dataset['epoch'], df_dataset[metric],
                   label=label, color=color, marker=marker,
                   markersize=3, markevery=max(1, len(df_dataset)//20),
                   linewidth=2, alpha=0.8)

            # Print final performance
            final_metric = df_dataset[metric].iloc[-1]
            print(f"    {label}: final_{metric}={final_metric:.6e}")

        # Format subplot
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
        ax.set_title(f'Dataset {dataset_idx}: {dataset_name}', fontsize=12, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='best')

    # Hide unused subplots
    for idx in range(num_datasets, len(axes)):
        axes[idx].axis('off')

    # Save combined plot
    output_dir = Path(__file__).parent
    output_file = output_dir / f'best_{metric}_curves_all_datasets_{section}_{timestamp}.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved combined plot to: {output_file}")

    # Show plot
    plt.show()

    return fig, axes


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Plot dense MSE/loss curves for optimizers/approaches across all datasets'
    )
    parser.add_argument('--section', type=str, default='section2_1',
                       choices=['section2_1', 'section2_2'],
                       help='Section to load (e.g., section2_1)')
    parser.add_argument('--timestamp', type=str, default=None,
                       help='Specific timestamp to load (default: most recent)')
    parser.add_argument('--metric', type=str, default='dense_mse',
                       choices=['dense_mse', 'test_loss', 'train_loss'],
                       help='Which metric to plot (default: dense_mse)')

    args = parser.parse_args()

    try:
        plot_best_loss_curves(
            section=args.section,
            timestamp=args.timestamp,
            metric=args.metric
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
