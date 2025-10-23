"""
Visualization for two-checkpoint comparison strategy in Section 2

This script creates bar charts comparing optimizer performance at:
1. LBFGS interpolation threshold (iso-compute comparison)
2. Final training (full budget comparison)
3. Time-to-threshold analysis

For Section 2.1: Compares Adam, LBFGS, and LM optimizers
For Section 2.2: Compares adaptive density approaches
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pickle
import argparse

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import load_run
from utils.result_finder import select_run


def plot_iso_compute_comparison_optimizers(checkpoint_metadata, dataset_names, output_path=None):
    """
    Bar chart comparing dense_mse at iso-compute checkpoint for optimizers

    Shows: When LBFGS reaches threshold, how do Adam and LM compare?
    """
    n_datasets = len(dataset_names)
    optimizers = ['adam', 'lbfgs', 'lm']

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(n_datasets)
    width = 0.25

    # Extract dense_mse at threshold checkpoint
    adam_mse = []
    lbfgs_mse = []
    lm_mse = []

    for dataset_idx in range(n_datasets):
        # Adam at threshold
        if dataset_idx in checkpoint_metadata.get('adam', {}) and 'at_threshold' in checkpoint_metadata['adam'][dataset_idx]:
            adam_mse.append(checkpoint_metadata['adam'][dataset_idx]['at_threshold']['dense_mse'])
        else:
            adam_mse.append(None)

        # LBFGS at threshold
        if dataset_idx in checkpoint_metadata.get('lbfgs', {}) and 'at_threshold' in checkpoint_metadata['lbfgs'][dataset_idx]:
            lbfgs_mse.append(checkpoint_metadata['lbfgs'][dataset_idx]['at_threshold']['dense_mse'])
        else:
            lbfgs_mse.append(None)

        # LM at threshold
        if dataset_idx in checkpoint_metadata.get('lm', {}) and 'at_threshold' in checkpoint_metadata['lm'][dataset_idx]:
            lm_mse.append(checkpoint_metadata['lm'][dataset_idx]['at_threshold']['dense_mse'])
        else:
            lm_mse.append(None)

    # Plot bars (use log scale for y-axis)
    ax.bar(x - width, adam_mse, width, label='Adam', alpha=0.8, color='C0')
    ax.bar(x, lbfgs_mse, width, label='LBFGS', alpha=0.8, color='C1')
    ax.bar(x + width, lm_mse, width, label='LM', alpha=0.8, color='C2')

    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Dense MSE (log scale)', fontsize=12)
    ax.set_title('Iso-Compute Comparison: Optimizer Performance at Interpolation Threshold', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names, rotation=45, ha='right')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved iso-compute comparison to {output_path}")

    return fig


def plot_final_comparison_optimizers(checkpoint_metadata, dataset_names, output_path=None):
    """
    Bar chart comparing final dense_mse for optimizers

    Shows: Given full training budget, which optimizer performs best?
    """
    n_datasets = len(dataset_names)

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(n_datasets)
    width = 0.25

    # Extract dense_mse at final checkpoint
    adam_mse = []
    lbfgs_mse = []
    lm_mse = []

    for dataset_idx in range(n_datasets):
        # Adam final
        if dataset_idx in checkpoint_metadata.get('adam', {}) and 'final' in checkpoint_metadata['adam'][dataset_idx]:
            adam_mse.append(checkpoint_metadata['adam'][dataset_idx]['final']['dense_mse'])
        else:
            adam_mse.append(None)

        # LBFGS final
        if dataset_idx in checkpoint_metadata.get('lbfgs', {}) and 'final' in checkpoint_metadata['lbfgs'][dataset_idx]:
            lbfgs_mse.append(checkpoint_metadata['lbfgs'][dataset_idx]['final']['dense_mse'])
        else:
            lbfgs_mse.append(None)

        # LM final
        if dataset_idx in checkpoint_metadata.get('lm', {}) and 'final' in checkpoint_metadata['lm'][dataset_idx]:
            lm_mse.append(checkpoint_metadata['lm'][dataset_idx]['final']['dense_mse'])
        else:
            lm_mse.append(None)

    # Plot bars
    ax.bar(x - width, adam_mse, width, label='Adam', alpha=0.8, color='C0')
    ax.bar(x, lbfgs_mse, width, label='LBFGS', alpha=0.8, color='C1')
    ax.bar(x + width, lm_mse, width, label='LM', alpha=0.8, color='C2')

    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Dense MSE (log scale)', fontsize=12)
    ax.set_title('Final Performance: Optimizer Comparison with Full Training Budget', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names, rotation=45, ha='right')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved final comparison to {output_path}")

    return fig


def plot_time_to_threshold(checkpoint_metadata, dataset_names, output_path=None):
    """
    Bar chart comparing time to reach interpolation threshold

    Shows: Which optimizer converges fastest?
    """
    n_datasets = len(dataset_names)

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(n_datasets)
    width = 0.25

    # Extract time at threshold checkpoint
    adam_time = []
    lbfgs_time = []
    lm_time = []

    for dataset_idx in range(n_datasets):
        # Adam time to threshold
        if dataset_idx in checkpoint_metadata.get('adam', {}) and 'at_threshold' in checkpoint_metadata['adam'][dataset_idx]:
            adam_time.append(checkpoint_metadata['adam'][dataset_idx]['at_threshold']['time'])
        else:
            adam_time.append(None)

        # LBFGS time to threshold
        if dataset_idx in checkpoint_metadata.get('lbfgs', {}) and 'at_threshold' in checkpoint_metadata['lbfgs'][dataset_idx]:
            lbfgs_time.append(checkpoint_metadata['lbfgs'][dataset_idx]['at_threshold']['time'])
        else:
            lbfgs_time.append(None)

        # LM time to threshold
        if dataset_idx in checkpoint_metadata.get('lm', {}) and 'at_threshold' in checkpoint_metadata['lm'][dataset_idx]:
            lm_time.append(checkpoint_metadata['lm'][dataset_idx]['at_threshold']['time'])
        else:
            lm_time.append(None)

    # Plot bars
    ax.bar(x - width, adam_time, width, label='Adam', alpha=0.8, color='C0')
    ax.bar(x, lbfgs_time, width, label='LBFGS', alpha=0.8, color='C1')
    ax.bar(x + width, lm_time, width, label='LM', alpha=0.8, color='C2')

    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Time to Threshold (seconds)', fontsize=12)
    ax.set_title('Convergence Speed: Time to Reach Interpolation Threshold', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved time-to-threshold comparison to {output_path}")

    return fig


def plot_iso_compute_comparison_approaches(checkpoint_metadata, dataset_names, output_path=None):
    """
    Bar chart comparing dense_mse at iso-compute checkpoint for approaches

    Shows: When baseline reaches threshold, how do adaptive approaches compare?
    """
    n_datasets = len(dataset_names)

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(n_datasets)
    width = 0.25

    # Extract dense_mse at threshold checkpoint
    adaptive_only_mse = []
    adaptive_regular_mse = []
    baseline_mse = []

    for dataset_idx in range(n_datasets):
        # Adaptive only at threshold
        if dataset_idx in checkpoint_metadata.get('adaptive_only', {}) and 'at_threshold' in checkpoint_metadata['adaptive_only'][dataset_idx]:
            adaptive_only_mse.append(checkpoint_metadata['adaptive_only'][dataset_idx]['at_threshold']['dense_mse'])
        else:
            adaptive_only_mse.append(None)

        # Adaptive+regular at threshold
        if dataset_idx in checkpoint_metadata.get('adaptive_regular', {}) and 'at_threshold' in checkpoint_metadata['adaptive_regular'][dataset_idx]:
            adaptive_regular_mse.append(checkpoint_metadata['adaptive_regular'][dataset_idx]['at_threshold']['dense_mse'])
        else:
            adaptive_regular_mse.append(None)

        # Baseline at threshold
        if dataset_idx in checkpoint_metadata.get('baseline', {}) and 'at_threshold' in checkpoint_metadata['baseline'][dataset_idx]:
            baseline_mse.append(checkpoint_metadata['baseline'][dataset_idx]['at_threshold']['dense_mse'])
        else:
            baseline_mse.append(None)

    # Plot bars (use log scale for y-axis)
    ax.bar(x - width, adaptive_only_mse, width, label='Adaptive Only', alpha=0.8, color='C0')
    ax.bar(x, adaptive_regular_mse, width, label='Adaptive+Regular', alpha=0.8, color='C1')
    ax.bar(x + width, baseline_mse, width, label='Baseline', alpha=0.8, color='C2')

    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Dense MSE (log scale)', fontsize=12)
    ax.set_title('Iso-Compute Comparison: Adaptive Density vs Baseline at Threshold', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names, rotation=45, ha='right')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved iso-compute comparison to {output_path}")

    return fig


def plot_final_comparison_approaches(checkpoint_metadata, dataset_names, output_path=None):
    """
    Bar chart comparing final dense_mse for approaches

    Shows: Given full training budget, which approach performs best?
    """
    n_datasets = len(dataset_names)

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(n_datasets)
    width = 0.25

    # Extract dense_mse at final checkpoint
    adaptive_only_mse = []
    adaptive_regular_mse = []
    baseline_mse = []

    for dataset_idx in range(n_datasets):
        # Adaptive only final
        if dataset_idx in checkpoint_metadata.get('adaptive_only', {}) and 'final' in checkpoint_metadata['adaptive_only'][dataset_idx]:
            adaptive_only_mse.append(checkpoint_metadata['adaptive_only'][dataset_idx]['final']['dense_mse'])
        else:
            adaptive_only_mse.append(None)

        # Adaptive+regular final
        if dataset_idx in checkpoint_metadata.get('adaptive_regular', {}) and 'final' in checkpoint_metadata['adaptive_regular'][dataset_idx]:
            adaptive_regular_mse.append(checkpoint_metadata['adaptive_regular'][dataset_idx]['final']['dense_mse'])
        else:
            adaptive_regular_mse.append(None)

        # Baseline final
        if dataset_idx in checkpoint_metadata.get('baseline', {}) and 'final' in checkpoint_metadata['baseline'][dataset_idx]:
            baseline_mse.append(checkpoint_metadata['baseline'][dataset_idx]['final']['dense_mse'])
        else:
            baseline_mse.append(None)

    # Plot bars
    ax.bar(x - width, adaptive_only_mse, width, label='Adaptive Only', alpha=0.8, color='C0')
    ax.bar(x, adaptive_regular_mse, width, label='Adaptive+Regular', alpha=0.8, color='C1')
    ax.bar(x + width, baseline_mse, width, label='Baseline', alpha=0.8, color='C2')

    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Dense MSE (log scale)', fontsize=12)
    ax.set_title('Final Performance: Adaptive Density Comparison with Full Training Budget', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names, rotation=45, ha='right')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved final comparison to {output_path}")

    return fig


def main():
    parser = argparse.ArgumentParser(description='Plot checkpoint comparisons for Section 2 experiments')
    parser.add_argument('section', type=str, help='Section name (e.g., section2_1 or section2_2)')
    parser.add_argument('--timestamp', type=str, default=None, help='Timestamp (default: latest)')
    parser.add_argument('--strategy', type=str, default='latest',
                       choices=['latest', 'max_epochs', 'min_epochs'],
                       help='Strategy for selecting run (default: latest)')
    parser.add_argument('--show', action='store_true', help='Show plots instead of saving')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory (default: section results dir)')

    args = parser.parse_args()

    # Determine section number for result path
    sec_num = args.section.split('_')[1]
    base_dir = Path(__file__).parent.parent
    results_dir = base_dir / 'results' / f'sec{sec_num}_results'

    # Find or select timestamp
    if args.timestamp is None:
        timestamp = select_run(args.section, strategy=args.strategy, return_format='timestamp')
        if timestamp is None:
            print(f"No results found for {args.section}")
            return
        print(f"Using {args.strategy} run: {timestamp}")
    else:
        timestamp = args.timestamp

    # Load results and metadata
    results, meta = load_run(args.section, timestamp)

    # Load checkpoint metadata
    checkpoint_metadata_files = list(results_dir.glob(f'{args.section}_{timestamp}*checkpoint_metadata.pkl'))
    if not checkpoint_metadata_files:
        print(f"No checkpoint metadata found for {args.section} {timestamp}")
        print("Available files:")
        for f in results_dir.glob(f'{args.section}_{timestamp}*'):
            print(f"  {f.name}")
        return

    with open(checkpoint_metadata_files[0], 'rb') as f:
        checkpoint_metadata = pickle.load(f)

    # Get dataset names from results
    first_df = next(iter(results.values()))
    dataset_names = sorted(first_df['dataset_name'].unique())

    print(f"\nLoaded checkpoint metadata for {args.section}")
    print(f"Datasets: {dataset_names}")
    print(f"Checkpoint types: {list(checkpoint_metadata.keys())}")

    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = results_dir

    output_dir.mkdir(exist_ok=True, parents=True)

    # Create plots based on section
    if args.section.startswith('section2_1'):
        print("\nCreating optimizer comparison plots...")

        # Iso-compute comparison
        output_path = None if args.show else output_dir / f'{args.section}_{timestamp}_iso_compute.png'
        plot_iso_compute_comparison_optimizers(checkpoint_metadata, dataset_names, output_path)

        # Final comparison
        output_path = None if args.show else output_dir / f'{args.section}_{timestamp}_final.png'
        plot_final_comparison_optimizers(checkpoint_metadata, dataset_names, output_path)

        # Time to threshold
        output_path = None if args.show else output_dir / f'{args.section}_{timestamp}_time_to_threshold.png'
        plot_time_to_threshold(checkpoint_metadata, dataset_names, output_path)

    elif args.section.startswith('section2_2'):
        print("\nCreating adaptive density comparison plots...")

        # Iso-compute comparison
        output_path = None if args.show else output_dir / f'{args.section}_{timestamp}_iso_compute.png'
        plot_iso_compute_comparison_approaches(checkpoint_metadata, dataset_names, output_path)

        # Final comparison
        output_path = None if args.show else output_dir / f'{args.section}_{timestamp}_final.png'
        plot_final_comparison_approaches(checkpoint_metadata, dataset_names, output_path)

    else:
        print(f"Unknown section: {args.section}")
        return

    if args.show:
        plt.show()

    print("\nCheckpoint comparison plots created successfully!")


if __name__ == '__main__':
    main()
