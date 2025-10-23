"""
Visualization for two-checkpoint comparison strategy

This script creates bar charts comparing model performance at:
1. KAN interpolation threshold (iso-compute comparison)
2. Final training (full budget comparison)
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pickle

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import load_run
from utils.result_finder import select_run


def plot_iso_compute_comparison(checkpoint_metadata, dataset_names, output_path=None):
    """
    Bar chart comparing dense_mse at iso-compute checkpoint

    Shows: When KAN reaches threshold, how do others compare?
    """
    n_datasets = len(dataset_names)
    model_types = ['mlp', 'siren', 'kan']

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(n_datasets)
    width = 0.25

    # Extract dense_mse at threshold checkpoint
    mlp_mse = []
    siren_mse = []
    kan_mse = []

    for dataset_idx in range(n_datasets):
        # MLP at KAN threshold time
        if dataset_idx in checkpoint_metadata.get('mlp', {}):
            if 'at_kan_threshold_time' in checkpoint_metadata['mlp'][dataset_idx]:
                mlp_mse.append(checkpoint_metadata['mlp'][dataset_idx]['at_kan_threshold_time']['dense_mse'])
            else:
                mlp_mse.append(None)
        else:
            mlp_mse.append(None)

        # SIREN at KAN threshold time
        if dataset_idx in checkpoint_metadata.get('siren', {}):
            if 'at_kan_threshold_time' in checkpoint_metadata['siren'][dataset_idx]:
                siren_mse.append(checkpoint_metadata['siren'][dataset_idx]['at_kan_threshold_time']['dense_mse'])
            else:
                siren_mse.append(None)
        else:
            siren_mse.append(None)

        # KAN at threshold
        if dataset_idx in checkpoint_metadata.get('kan', {}):
            if 'at_threshold' in checkpoint_metadata['kan'][dataset_idx]:
                kan_mse.append(checkpoint_metadata['kan'][dataset_idx]['at_threshold']['dense_mse'])
            else:
                kan_mse.append(None)
        else:
            kan_mse.append(None)

    # Plot bars (use log scale for y-axis)
    ax.bar(x - width, mlp_mse, width, label='MLP', alpha=0.8)
    ax.bar(x, siren_mse, width, label='SIREN', alpha=0.8)
    ax.bar(x + width, kan_mse, width, label='KAN', alpha=0.8)

    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Dense MSE (log scale)', fontsize=12)
    ax.set_title('Iso-Compute Comparison: Performance When KAN Reaches Interpolation Threshold', fontsize=14, fontweight='bold')
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


def plot_final_comparison(checkpoint_metadata, dataset_names, output_path=None):
    """
    Bar chart comparing final dense_mse

    Shows: Given unlimited time, who performs best?
    """
    n_datasets = len(dataset_names)

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(n_datasets)
    width = 0.25

    # Extract dense_mse at final checkpoint
    mlp_mse = []
    siren_mse = []
    kan_mse = []

    for dataset_idx in range(n_datasets):
        # MLP final
        if dataset_idx in checkpoint_metadata.get('mlp', {}) and 'final' in checkpoint_metadata['mlp'][dataset_idx]:
            mlp_mse.append(checkpoint_metadata['mlp'][dataset_idx]['final']['dense_mse'])
        else:
            mlp_mse.append(None)

        # SIREN final
        if dataset_idx in checkpoint_metadata.get('siren', {}) and 'final' in checkpoint_metadata['siren'][dataset_idx]:
            siren_mse.append(checkpoint_metadata['siren'][dataset_idx]['final']['dense_mse'])
        else:
            siren_mse.append(None)

        # KAN final
        if dataset_idx in checkpoint_metadata.get('kan', {}) and 'final' in checkpoint_metadata['kan'][dataset_idx]:
            kan_mse.append(checkpoint_metadata['kan'][dataset_idx]['final']['dense_mse'])
        else:
            kan_mse.append(None)

    # Plot bars
    ax.bar(x - width, mlp_mse, width, label='MLP', alpha=0.8)
    ax.bar(x, siren_mse, width, label='SIREN', alpha=0.8)
    ax.bar(x + width, kan_mse, width, label='KAN', alpha=0.8)

    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Dense MSE (log scale)', fontsize=12)
    ax.set_title('Final Performance Comparison: Best Achievable with Full Training Budget', fontsize=14, fontweight='bold')
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
    Bar chart showing how long KAN took to reach interpolation threshold

    Shows: KAN convergence time per dataset
    """
    n_datasets = len(dataset_names)

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(n_datasets)

    # Extract time at threshold
    kan_times = []
    for dataset_idx in range(n_datasets):
        if dataset_idx in checkpoint_metadata.get('kan', {}) and 'at_threshold' in checkpoint_metadata['kan'][dataset_idx]:
            kan_times.append(checkpoint_metadata['kan'][dataset_idx]['at_threshold']['time'])
        else:
            kan_times.append(0)

    # Plot bars
    ax.bar(x, kan_times, alpha=0.8, color='green')

    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Time to Interpolation Threshold (seconds)', fontsize=12)
    ax.set_title('KAN Convergence Speed: Time to Reach Interpolation Threshold', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved time-to-threshold plot to {output_path}")

    return fig


def main():
    """
    Example usage: Load results and create all comparison plots
    """
    import argparse

    parser = argparse.ArgumentParser(description='Create checkpoint comparison plots')
    parser.add_argument('section', type=str, help='Section name (e.g., section1_1)')
    parser.add_argument('timestamp', nargs='?', default=None, type=str,
                       help='Timestamp of the run (default: auto-detect latest)')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory for plots')
    parser.add_argument('--strategy', type=str, default='latest',
                       choices=['latest', 'max_epochs', 'min_epochs', 'exact_epochs'],
                       help='Run selection strategy (default: latest)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Epoch count for exact_epochs strategy')
    parser.add_argument('--verbose', action='store_true',
                       help='Print run selection details')
    args = parser.parse_args()

    # Auto-detect timestamp if not provided
    if args.timestamp is None:
        results_base = Path(__file__).parent.parent / 'results'
        args.timestamp = select_run(args.section, results_base,
                                   strategy=args.strategy,
                                   epochs=args.epochs,
                                   verbose=args.verbose)
        print(f"Using selected timestamp: {args.timestamp}")

    # Load results
    print(f"Loading results from {args.section}, timestamp {args.timestamp}")
    results, meta = load_run(args.section, args.timestamp, load_models=False)

    # Load checkpoint metadata
    sec_num = args.section.split('_')[-1]
    base_dir = Path(__file__).parent.parent
    results_dir = base_dir / 'results' / f'sec{sec_num}_results'
    checkpoint_metadata_path = results_dir / f'{args.section}_{args.timestamp}_checkpoint_metadata.pkl'

    if not checkpoint_metadata_path.exists():
        print(f"Error: Checkpoint metadata not found at {checkpoint_metadata_path}")
        print("Make sure you ran training with the two-checkpoint strategy.")
        return

    with open(checkpoint_metadata_path, 'rb') as f:
        checkpoint_metadata = pickle.load(f)

    # Extract dataset names from results
    if 'mlp' in results:
        dataset_names = results['mlp']['dataset_name'].unique().tolist()
    elif 'kan' in results:
        dataset_names = results['kan']['dataset_name'].unique().tolist()
    else:
        print("Error: No results found")
        return

    print(f"Found {len(dataset_names)} datasets: {dataset_names}")

    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = results_dir
    output_dir.mkdir(exist_ok=True, parents=True)

    # Create plots
    print("\nCreating comparison plots...")

    print("1. Iso-compute comparison...")
    plot_iso_compute_comparison(
        checkpoint_metadata,
        dataset_names,
        output_path=output_dir / f'{args.section}_{args.timestamp}_iso_compute_comparison.png'
    )

    print("2. Final performance comparison...")
    plot_final_comparison(
        checkpoint_metadata,
        dataset_names,
        output_path=output_dir / f'{args.section}_{args.timestamp}_final_comparison.png'
    )

    print("3. Time to threshold...")
    plot_time_to_threshold(
        checkpoint_metadata,
        dataset_names,
        output_path=output_dir / f'{args.section}_{args.timestamp}_time_to_threshold.png'
    )

    print(f"\nAll plots saved to {output_dir}")
    print("Done!")


if __name__ == '__main__':
    main()
