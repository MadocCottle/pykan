"""
Plot scaling laws: Dense MSE vs # Parameters (log-log plots).

This is the PRIMARY visualization for high-dimensional experiments, following
the KAN paper's approach (Figure model_scaling.pdf, model_scale_exp100d.pdf).

Key insights:
- Shows whether KANs beat curse of dimensionality
- Compares shallow vs deep architectures
- Validates theoretical scaling exponents (α=4 vs α=4/d)
- Compares optimizer performance across dimensions

For each dimension, creates a log-log plot showing:
- X-axis: Number of parameters
- Y-axis: Final Dense MSE per grid size
- Lines: Different (architecture, optimizer) combinations
- Reference: Theoretical scaling laws

Usage:
    python plot_scaling_laws.py --dim 3 --timestamp TIMESTAMP
    python plot_scaling_laws.py --dim all  # Compare all dimensions
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_run


def find_latest_highd_timestamp(dim, architecture):
    """Find the most recent timestamp for a high-D experiment"""
    results_dir = Path(__file__).parent.parent / 'results' / 'sec1_results'

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    # Look for files matching pattern: section2_1_highd_3d_shallow_TIMESTAMP_lbfgs.pkl
    pattern = f'section2_1_highd_{dim}d_{architecture}_*_lbfgs.pkl'
    timestamps = set()

    for f in results_dir.glob(pattern):
        # Extract timestamp from filename
        parts = f.stem.split('_')
        # Find timestamp (between architecture and optimizer)
        for i, part in enumerate(parts):
            if part == architecture and i + 1 < len(parts):
                timestamp = parts[i + 1]
                if timestamp != 'lbfgs':  # Make sure it's not the optimizer name
                    timestamps.add(timestamp)

    if not timestamps:
        raise FileNotFoundError(f"No results found for {dim}D {architecture}")

    return sorted(timestamps)[-1]


def compute_theoretical_scaling(params, alpha, reference_param, reference_mse):
    """
    Compute theoretical scaling law: MSE ∝ N^(-alpha)

    Args:
        params: Array of parameter counts
        alpha: Scaling exponent (e.g., 4 for KAN theory, 4/d for classical)
        reference_param: Reference parameter count
        reference_mse: MSE at reference parameter count

    Returns:
        Array of theoretical MSE values
    """
    return reference_mse * (params / reference_param) ** (-alpha)


def plot_scaling_law_single_dim(dim, architecture, timestamp=None, save=True, show=False):
    """
    Plot scaling law for a single dimension.

    Args:
        dim: Dimension (3, 4, 10, or 100)
        architecture: 'shallow' or 'deep'
        timestamp: Specific timestamp, or None for most recent
        save: Whether to save the plot
    """
    # Load results
    if timestamp is None:
        timestamp = find_latest_highd_timestamp(dim, architecture)
        print(f"Using most recent timestamp: {timestamp}")

    section = f'section2_1_highd_{dim}d_{architecture}'
    print(f"Loading results from {section}_{timestamp}...")

    results, meta = load_run(section, timestamp)

    # Get optimizers
    optimizers = list(results.keys())
    print(f"Found optimizers: {optimizers}")

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    colors = {'lbfgs': '#2E86AB', 'LM': '#A23B72', 'LBFGS': '#2E86AB', 'lm': '#A23B72'}
    markers = {'lbfgs': 'o', 'LM': 's', 'LBFGS': 'o', 'lm': 's'}
    linestyles = {architecture: '-'}

    # Track min/max for reference lines
    all_params = []
    all_mses = []

    # Plot each optimizer
    for opt in optimizers:
        df = results[opt]

        # Get final MSE for each grid size (take last epoch of each grid)
        grid_sizes = sorted(df['grid_size'].unique())
        final_results = []

        for grid in grid_sizes:
            grid_df = df[df['grid_size'] == grid]
            if len(grid_df) == 0:
                continue

            # Get last epoch for this grid
            last_epoch_df = grid_df[grid_df['epoch'] == grid_df['epoch'].max()]

            if len(last_epoch_df) > 0:
                final_results.append({
                    'grid_size': grid,
                    'num_params': last_epoch_df['num_params'].iloc[0],
                    'dense_mse': last_epoch_df['dense_mse'].iloc[0]
                })

        if not final_results:
            continue

        final_df = pd.DataFrame(final_results)
        params = final_df['num_params'].values
        mses = final_df['dense_mse'].values

        all_params.extend(params)
        all_mses.extend(mses)

        # Plot
        opt_label = opt.upper() if opt.lower() in ['lbfgs', 'lm'] else opt
        ax.plot(params, mses,
               label=opt_label,
               color=colors.get(opt, f'C{optimizers.index(opt)}'),
               marker=markers.get(opt, 'o'),
               markersize=8,
               linewidth=2.5,
               alpha=0.8)

        # Print final performance
        print(f"  {opt_label}: {len(final_df)} grid sizes, "
              f"params={params[0]}-{params[-1]}, "
              f"MSE={mses[-1]:.2e}")

    # Add theoretical scaling law references
    if len(all_params) > 0:
        param_range = np.logspace(np.log10(min(all_params)), np.log10(max(all_params)), 100)

        # Reference point: median parameter count and corresponding MSE
        ref_idx = len(all_params) // 2
        ref_param = sorted(all_params)[ref_idx]
        ref_mse = sorted(all_mses)[ref_idx]

        # KAN theory: α = 4 (dimension-independent)
        kan_theory = compute_theoretical_scaling(param_range, alpha=4,
                                                  reference_param=ref_param,
                                                  reference_mse=ref_mse)
        ax.plot(param_range, kan_theory,
               '--', color='#E63946', linewidth=2, alpha=0.6,
               label='KAN theory (α=4)')

        # Classical theory: α = 4/d (curse of dimensionality)
        classical_alpha = 4.0 / dim
        classical_theory = compute_theoretical_scaling(param_range, alpha=classical_alpha,
                                                       reference_param=ref_param,
                                                       reference_mse=ref_mse)
        ax.plot(param_range, classical_theory,
               ':', color='#6C757D', linewidth=2, alpha=0.6,
               label=f'Classical (α=4/{dim}={classical_alpha:.2f})')

    # Formatting
    ax.set_xlabel('Number of Parameters', fontsize=14, fontweight='bold')
    ax.set_ylabel('Final Dense MSE', fontsize=14, fontweight='bold')
    ax.set_title(f'{dim}D Poisson PDE - Scaling Laws ({architecture.capitalize()} Architecture)',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both', linestyle='-', linewidth=0.5)
    ax.legend(fontsize=11, loc='best', framealpha=0.9)

    # Add architecture info as text box
    arch_width = meta.get('kan_width', 'Unknown')
    textstr = f'Architecture: {arch_width}\nDimension: {dim}D'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=props)

    plt.tight_layout()

    # Save plot
    if save:
        output_dir = Path(__file__).parent
        output_file = output_dir / f'scaling_laws_{dim}d_{architecture}_{timestamp}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {output_file}")

    return fig, ax


def plot_scaling_law_comparison(dims=[3, 4, 10, 100], architecture='deep', save=True, show=False):
    """
    Compare scaling laws across multiple dimensions.

    Args:
        dims: List of dimensions to compare
        architecture: 'shallow' or 'deep'
        save: Whether to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    for idx, dim in enumerate(dims):
        ax = axes[idx]

        # Find latest timestamp for this dimension
        try:
            timestamp = find_latest_highd_timestamp(dim, architecture)
        except FileNotFoundError:
            print(f"Warning: No results found for {dim}D {architecture}")
            ax.text(0.5, 0.5, f'No data for {dim}D', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14)
            ax.set_title(f'{dim}D Poisson PDE', fontsize=14, fontweight='bold')
            continue

        section = f'section2_1_highd_{dim}d_{architecture}'
        print(f"\nLoading {dim}D results...")

        try:
            results, meta = load_run(section, timestamp)
        except Exception as e:
            print(f"Error loading {dim}D: {e}")
            continue

        optimizers = list(results.keys())
        colors = {'lbfgs': '#2E86AB', 'LM': '#A23B72', 'LBFGS': '#2E86AB', 'lm': '#A23B72'}
        markers = {'lbfgs': 'o', 'LM': 's', 'LBFGS': 'o', 'lm': 's'}

        all_params = []
        all_mses = []

        # Plot each optimizer
        for opt in optimizers:
            df = results[opt]
            grid_sizes = sorted(df['grid_size'].unique())
            final_results = []

            for grid in grid_sizes:
                grid_df = df[df['grid_size'] == grid]
                if len(grid_df) == 0:
                    continue
                last_epoch_df = grid_df[grid_df['epoch'] == grid_df['epoch'].max()]
                if len(last_epoch_df) > 0:
                    final_results.append({
                        'num_params': last_epoch_df['num_params'].iloc[0],
                        'dense_mse': last_epoch_df['dense_mse'].iloc[0]
                    })

            if not final_results:
                continue

            final_df = pd.DataFrame(final_results)
            params = final_df['num_params'].values
            mses = final_df['dense_mse'].values

            all_params.extend(params)
            all_mses.extend(mses)

            opt_label = opt.upper() if opt.lower() in ['lbfgs', 'lm'] else opt
            ax.plot(params, mses, label=opt_label,
                   color=colors.get(opt, f'C{optimizers.index(opt)}'),
                   marker=markers.get(opt, 'o'), markersize=6,
                   linewidth=2, alpha=0.8)

        # Add theoretical scaling laws
        if len(all_params) > 0:
            param_range = np.logspace(np.log10(min(all_params)), np.log10(max(all_params)), 100)
            ref_idx = len(all_params) // 2
            ref_param = sorted(all_params)[ref_idx]
            ref_mse = sorted(all_mses)[ref_idx]

            kan_theory = compute_theoretical_scaling(param_range, 4, ref_param, ref_mse)
            ax.plot(param_range, kan_theory, '--', color='#E63946',
                   linewidth=1.5, alpha=0.5, label='α=4')

            classical_alpha = 4.0 / dim
            classical_theory = compute_theoretical_scaling(param_range, classical_alpha,
                                                           ref_param, ref_mse)
            ax.plot(param_range, classical_theory, ':', color='#6C757D',
                   linewidth=1.5, alpha=0.5, label=f'α=4/{dim}')

        ax.set_xlabel('Parameters', fontsize=12, fontweight='bold')
        ax.set_ylabel('Dense MSE', fontsize=12, fontweight='bold')
        ax.set_title(f'{dim}D Poisson PDE', fontsize=14, fontweight='bold')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, which='both')
        if idx == 0:
            ax.legend(fontsize=9, loc='best')

    plt.suptitle(f'Scaling Laws Comparison ({architecture.capitalize()} Architecture)',
                fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()

    if save:
        output_dir = Path(__file__).parent
        output_file = output_dir / f'scaling_laws_comparison_{architecture}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nComparison plot saved to: {output_file}")

    return fig, axes


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Plot scaling laws for high-dimensional experiments'
    )
    parser.add_argument('--dim', type=str, default='all',
                       help='Dimension to plot (3, 4, 10, 100, or "all")')
    parser.add_argument('--architecture', type=str, default='deep',
                       choices=['shallow', 'deep'],
                       help='Architecture type (default: deep)')
    parser.add_argument('--timestamp', type=str, default=None,
                       help='Specific timestamp (default: most recent)')
    parser.add_argument('--show', action='store_true',
                       help='Display the plot in a window (default: only save to file)')

    args = parser.parse_args()

    try:
        if args.dim == 'all':
            plot_scaling_law_comparison(
                dims=[3, 4, 10, 100],
                architecture=args.architecture,
                save=True
            )
        else:
            dim = int(args.dim)
            if dim not in [3, 4, 10, 100]:
                raise ValueError(f"Invalid dimension: {dim}. Must be 3, 4, 10, or 100")

            plot_scaling_law_single_dim(
                dim=dim,
                architecture=args.architecture,
                timestamp=args.timestamp,
                save=True
            )

        if args.show:
            plt.show()
        else:
            plt.close('all')

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nMake sure you have run the training script first:")
        print(f"  python section2_1_highd.py --dim {args.dim} --architecture {args.architecture}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
