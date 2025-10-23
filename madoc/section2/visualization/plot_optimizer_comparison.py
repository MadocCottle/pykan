"""
Plot Dense MSE over epochs comparing Adam and LM optimizers.

This script:
1. Loads results from the most recent run
2. Compares Adam vs LM optimizer performance on KAN models
3. Plots their dense MSE evolution over training epochs for all datasets
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_run
from utils.result_finder import select_run


def plot_optimizer_comparison(section='section2_1', timestamp=None, dataset_idx=None, show=False,
                             strategy='latest', epochs=None, verbose=False):
    """
    Plot dense MSE over epochs comparing Adam and LM optimizers.

    Args:
        section: Section name (e.g., 'section2_1')
        timestamp: Specific timestamp to load, or None for most recent
        dataset_idx: Which dataset to analyze, or None for all datasets
        strategy: Run selection strategy ('latest', 'max_epochs', 'min_epochs', 'exact_epochs')
        epochs: Epoch count for exact_epochs strategy
        verbose: Print run selection details
    """
    # Load results
    if timestamp is None:
        results_base = Path(__file__).parent.parent / 'results'
        timestamp = select_run(section, results_base,
                             strategy=strategy,
                             epochs=epochs,
                             verbose=verbose)
        print(f"Using selected timestamp: {timestamp}")

    print(f"Loading results from {section}_{timestamp}...")
    results, meta = load_run(section, timestamp)

    # Get available optimizers
    optimizers = list(results.keys())
    print(f"Found optimizers: {optimizers}")

    # Get dataset info
    if 'adam' in results:
        adam_df = results['adam']
        dataset_indices = sorted(adam_df['dataset_idx'].unique())
        dataset_names = adam_df.groupby('dataset_idx')['dataset_name'].first().to_dict()
        num_datasets = len(dataset_indices)
    else:
        raise ValueError("No Adam results found")

    # Filter to specific dataset if requested
    if dataset_idx is not None:
        if dataset_idx not in dataset_indices:
            raise ValueError(f"Dataset {dataset_idx} not found. Available: {dataset_indices}")
        dataset_indices = [dataset_idx]
        num_datasets = 1

    print(f"\nGenerating plots for {num_datasets} dataset(s)...")

    # Determine grid layout
    if num_datasets == 1:
        fig, axes = plt.subplots(1, 1, figsize=(10, 6))
        axes = [axes]
    elif num_datasets <= 4:
        nrows, ncols = 2, 2
        fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
        axes = axes.flatten()
    else:
        nrows = (num_datasets + 2) // 3
        ncols = 3
        fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
        axes = axes.flatten()

    colors = {'adam': 'C0', 'Adam': 'C0', 'lm': 'C1', 'LM': 'C1'}
    markers = {'adam': 'o', 'Adam': 'o', 'lm': 's', 'LM': 's'}

    # Plot each dataset
    for plot_idx, ds_idx in enumerate(dataset_indices):
        ax = axes[plot_idx]
        ds_name = dataset_names[ds_idx]

        print(f"\n  Dataset {ds_idx} ({ds_name}):")

        for opt in optimizers:
            df = results[opt]
            df_dataset = df[df['dataset_idx'] == ds_idx].copy()

            if len(df_dataset) == 0:
                print(f"    Warning: No data for {opt}")
                continue

            # Sort by epoch
            df_dataset = df_dataset.sort_values('epoch')

            # Plot
            opt_label = opt.upper() if opt.lower() in ['adam', 'lm'] else opt
            ax.plot(df_dataset['epoch'], df_dataset['dense_mse'],
                   label=opt_label, color=colors.get(opt, f'C{optimizers.index(opt)}'),
                   marker=markers.get(opt, 'o'), markersize=3, markevery=max(1, len(df_dataset)//20),
                   linewidth=2, alpha=0.8)

            # Print final performance
            final_mse = df_dataset['dense_mse'].iloc[-1]
            print(f"    {opt_label}: final_dense_mse={final_mse:.6e}")

        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Dense MSE', fontsize=11)
        ax.set_title(f'Dataset {ds_idx}: {ds_name}', fontsize=12, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='best')

    # Hide unused subplots
    for idx in range(num_datasets, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    # Save plot
    output_dir = Path(__file__).parent
    if dataset_idx is not None:
        output_file = output_dir / f'optimizer_comparison_dataset_{dataset_idx}_{timestamp}.png'
    else:
        output_file = output_dir / f'optimizer_comparison_all_datasets_{timestamp}.png'

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")

    # Show plot
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, axes


def plot_training_loss_comparison(section='section2_1', timestamp=None, show=False):
    """
    Plot training and test loss comparison between optimizers.

    Args:
        section: Section name (e.g., 'section2_1')
        timestamp: Specific timestamp to load, or None for most recent
    """
    # Load results
    if timestamp is None:
        timestamp = find_latest_timestamp(section)
        print(f"Using most recent timestamp: {timestamp}")

    print(f"Loading results from {section}_{timestamp}...")
    results, meta = load_run(section, timestamp)

    # Get dataset info
    adam_df = results['adam']
    dataset_indices = sorted(adam_df['dataset_idx'].unique())
    dataset_names = adam_df.groupby('dataset_idx')['dataset_name'].first().to_dict()
    num_datasets = len(dataset_indices)

    print(f"\nGenerating training loss plots for {num_datasets} datasets...")

    # Create figure
    fig, axes = plt.subplots(num_datasets, 2, figsize=(14, 4*num_datasets))
    if num_datasets == 1:
        axes = axes.reshape(1, -1)

    colors = {'adam': 'C0', 'Adam': 'C0', 'lm': 'C1', 'LM': 'C1'}

    for plot_idx, ds_idx in enumerate(dataset_indices):
        ds_name = dataset_names[ds_idx]

        # Plot train loss
        ax_train = axes[plot_idx, 0]
        # Plot test loss
        ax_test = axes[plot_idx, 1]

        for opt in results.keys():
            df = results[opt]
            df_dataset = df[df['dataset_idx'] == ds_idx].sort_values('epoch')

            opt_label = opt.upper() if opt.lower() in ['adam', 'lm'] else opt

            ax_train.plot(df_dataset['epoch'], df_dataset['train_loss'],
                         label=opt_label, color=colors.get(opt, f'C{list(results.keys()).index(opt)}'),
                         linewidth=2, alpha=0.8)

            ax_test.plot(df_dataset['epoch'], df_dataset['test_loss'],
                        label=opt_label, color=colors.get(opt, f'C{list(results.keys()).index(opt)}'),
                        linewidth=2, alpha=0.8)

        ax_train.set_xlabel('Epoch', fontsize=11)
        ax_train.set_ylabel('Train Loss', fontsize=11)
        ax_train.set_title(f'{ds_name} - Training Loss', fontsize=12, fontweight='bold')
        ax_train.set_yscale('log')
        ax_train.grid(True, alpha=0.3)
        ax_train.legend(fontsize=10)

        ax_test.set_xlabel('Epoch', fontsize=11)
        ax_test.set_ylabel('Test Loss', fontsize=11)
        ax_test.set_title(f'{ds_name} - Test Loss', fontsize=12, fontweight='bold')
        ax_test.set_yscale('log')
        ax_test.grid(True, alpha=0.3)
        ax_test.legend(fontsize=10)

    plt.tight_layout()

    # Save plot
    output_dir = Path(__file__).parent
    output_file = output_dir / f'training_loss_comparison_{timestamp}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")

    if show:

        plt.show()

    else:

        plt.close(fig)
    return fig, axes


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Plot optimizer comparison for Section 2'
    )
    parser.add_argument('--section', type=str, default='section2_1',
                       help='Section to load (e.g., section2_1)')
    parser.add_argument('--timestamp', type=str, default=None,
                       help='Specific timestamp to load (default: most recent)')
    parser.add_argument('--dataset', type=int, default=None,
                       help='Dataset index to analyze (default: all datasets)')
    parser.add_argument('--plot-type', type=str, default='optimizer',
                       choices=['optimizer', 'training', 'both'],
                       help='Type of plot to generate (default: optimizer)')
    parser.add_argument('--show', action='store_true',
                       help='Display the plot in a window (default: only save to file)')
    parser.add_argument('--strategy', type=str, default='latest',
                       choices=['latest', 'max_epochs', 'min_epochs', 'exact_epochs'],
                       help='Run selection strategy (default: latest)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Epoch count for exact_epochs strategy')
    parser.add_argument('--verbose', action='store_true',
                       help='Print run selection details')

    args = parser.parse_args()

    try:
        if args.plot_type in ['optimizer', 'both']:
            plot_optimizer_comparison(
                section=args.section,
                timestamp=args.timestamp,
                dataset_idx=args.dataset,
                show=args.show,
                strategy=args.strategy,
                epochs=args.epochs,
                verbose=args.verbose
            )

        if args.plot_type in ['training', 'both']:
            plot_training_loss_comparison(
                section=args.section,
                timestamp=args.timestamp,
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
