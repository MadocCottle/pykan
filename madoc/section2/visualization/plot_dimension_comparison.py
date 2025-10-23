"""
Plot dimension comparison heatmap showing final Dense MSE across all experiments.

This visualization provides a quick overview of performance across:
- Different dimensions (3D, 4D, 10D, 100D)
- Different architectures (shallow vs deep)
- Different optimizers (LBFGS vs LM)

Creates a heatmap where:
- Rows: Dimensions
- Columns: (Architecture × Optimizer) combinations
- Cell values: Final Dense MSE (log scale)
- Cell colors: Performance (darker = better)

Usage:
    python plot_dimension_comparison.py
    python plot_dimension_comparison.py --section section2_1
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_run


def find_all_highd_experiments():
    """
    Find all completed high-D experiments.

    Returns:
        List of dicts with keys: dim, architecture, timestamp, section
    """
    results_dir = Path(__file__).parent.parent / 'results' / 'sec1_results'

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    experiments = []

    # Pattern: section2_1_highd_3d_shallow_TIMESTAMP_lbfgs.pkl
    for dim in [3, 4, 10, 100]:
        for arch in ['shallow', 'deep']:
            pattern = f'section2_1_highd_{dim}d_{arch}_*_lbfgs.pkl'

            timestamps = set()
            for f in results_dir.glob(pattern):
                # Extract timestamp
                parts = f.stem.split('_')
                for i, part in enumerate(parts):
                    if part == arch and i + 1 < len(parts):
                        timestamp = parts[i + 1]
                        if timestamp not in ['lbfgs', 'lm']:
                            timestamps.add(timestamp)

            if timestamps:
                # Use most recent timestamp
                latest_ts = sorted(timestamps)[-1]
                experiments.append({
                    'dim': dim,
                    'architecture': arch,
                    'timestamp': latest_ts,
                    'section': f'section2_1_highd_{dim}d_{arch}'
                })

    return experiments


def extract_final_mse(results, optimizer):
    """
    Extract final Dense MSE for a given optimizer.

    Args:
        results: Results dictionary
        optimizer: Optimizer name (e.g., 'lbfgs', 'lm')

    Returns:
        Final Dense MSE value (float)
    """
    if optimizer not in results:
        return np.nan

    df = results[optimizer]

    # Get the very last epoch (largest grid, last training step)
    last_epoch = df['epoch'].max()
    final_row = df[df['epoch'] == last_epoch]

    if len(final_row) == 0:
        return np.nan

    return final_row['dense_mse'].iloc[0]


def plot_dimension_comparison_heatmap(section='section2_1', save=True):
    """
    Create heatmap comparing all dimension/architecture/optimizer combinations.

    Args:
        section: Section prefix (e.g., 'section2_1')
        save: Whether to save the plot

    Returns:
        fig, ax: Matplotlib figure and axis
    """
    print("Finding all completed high-D experiments...")
    experiments = find_all_highd_experiments()

    if len(experiments) == 0:
        raise FileNotFoundError("No high-D experiments found. Run training scripts first.")

    print(f"Found {len(experiments)} experiments")

    # Collect data
    data_rows = []

    for exp in experiments:
        print(f"Loading {exp['dim']}D {exp['architecture']}...")

        try:
            results, meta = load_run(exp['section'], exp['timestamp'])
        except Exception as e:
            print(f"  Error loading: {e}")
            continue

        # Extract final MSE for each optimizer
        for opt in results.keys():
            final_mse = extract_final_mse(results, opt)

            data_rows.append({
                'Dimension': f"{exp['dim']}D",
                'Architecture': exp['architecture'].capitalize(),
                'Optimizer': opt.upper(),
                'Config': f"{exp['architecture'].capitalize()}-{opt.upper()}",
                'Final MSE': final_mse,
                'Log MSE': np.log10(final_mse) if not np.isnan(final_mse) else np.nan,
                'dim_num': exp['dim']
            })

            print(f"  {opt.upper()}: {final_mse:.2e}")

    if len(data_rows) == 0:
        raise ValueError("No data could be loaded from experiments")

    df = pd.DataFrame(data_rows)

    # Create pivot table for heatmap
    # Rows: Dimensions, Columns: Configurations
    pivot_data = df.pivot_table(
        index='Dimension',
        columns='Config',
        values='Log MSE',
        aggfunc='first'
    )

    # Sort rows by dimension number
    dim_order = ['3D', '4D', '10D', '100D']
    pivot_data = pivot_data.reindex([d for d in dim_order if d in pivot_data.index])

    # Sort columns: Shallow-LBFGS, Shallow-LM, Deep-LBFGS, Deep-LM
    col_order = []
    for arch in ['Shallow', 'Deep']:
        for opt in ['LBFGS', 'LM']:
            col_name = f"{arch}-{opt}"
            if col_name in pivot_data.columns:
                col_order.append(col_name)
    pivot_data = pivot_data[col_order]

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Create heatmap
    # Use reversed colormap so darker = better (lower MSE)
    sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='RdYlGn_r',
                linewidths=2, linecolor='white', cbar_kws={'label': 'log₁₀(Dense MSE)'},
                ax=ax, vmin=None, vmax=None, center=None)

    # Add actual MSE values as second annotation
    for i, row_label in enumerate(pivot_data.index):
        for j, col_label in enumerate(pivot_data.columns):
            log_mse = pivot_data.loc[row_label, col_label]

            if not np.isnan(log_mse):
                actual_mse = 10 ** log_mse

                # Add small text with actual MSE
                ax.text(j + 0.5, i + 0.75, f'({actual_mse:.1e})',
                       ha='center', va='center', fontsize=8, color='gray', style='italic')

    ax.set_title('High-Dimensional Poisson PDE: Performance Comparison\n'
                'Cell values: log₁₀(Dense MSE) with actual MSE in parentheses',
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Architecture - Optimizer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Dimension', fontsize=12, fontweight='bold')

    # Rotate column labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    # Save plot
    if save:
        output_dir = Path(__file__).parent
        output_file = output_dir / f'dimension_comparison_heatmap.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nHeatmap saved to: {output_file}")

    return fig, ax


def plot_architecture_depth_comparison(save=True):
    """
    Create side-by-side comparison of shallow vs deep architectures.

    Shows:
    - Left: Shallow architecture performance across dimensions
    - Right: Deep architecture performance across dimensions
    - Highlights depth benefit

    Returns:
        fig, axes: Matplotlib figure and axes
    """
    print("Finding all completed high-D experiments...")
    experiments = find_all_highd_experiments()

    if len(experiments) == 0:
        raise FileNotFoundError("No high-D experiments found.")

    # Collect data
    data_rows = []
    for exp in experiments:
        try:
            results, meta = load_run(exp['section'], exp['timestamp'])
        except Exception as e:
            continue

        for opt in results.keys():
            final_mse = extract_final_mse(results, opt)
            data_rows.append({
                'Dimension': exp['dim'],
                'Architecture': exp['architecture'],
                'Optimizer': opt.upper(),
                'Final MSE': final_mse
            })

    df = pd.DataFrame(data_rows)

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, arch in enumerate(['shallow', 'deep']):
        ax = axes[idx]
        arch_df = df[df['Architecture'] == arch]

        # Group by dimension and optimizer
        dims = sorted(arch_df['Dimension'].unique())
        optimizers = sorted(arch_df['Optimizer'].unique())

        x = np.arange(len(dims))
        width = 0.35

        for i, opt in enumerate(optimizers):
            opt_data = []
            for dim in dims:
                dim_opt_df = arch_df[(arch_df['Dimension'] == dim) &
                                     (arch_df['Optimizer'] == opt)]
                if len(dim_opt_df) > 0:
                    opt_data.append(dim_opt_df['Final MSE'].iloc[0])
                else:
                    opt_data.append(np.nan)

            offset = (i - len(optimizers)/2 + 0.5) * width
            bars = ax.bar(x + offset, opt_data, width, label=opt, alpha=0.8)

            # Add value labels on bars
            for bar, val in zip(bars, opt_data):
                if not np.isnan(val):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{val:.1e}', ha='center', va='bottom', fontsize=8, rotation=90)

        ax.set_xlabel('Dimension', fontsize=12, fontweight='bold')
        ax.set_ylabel('Final Dense MSE', fontsize=12, fontweight='bold')
        ax.set_title(f'{arch.capitalize()} Architecture', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{d}D' for d in dims])
        ax.set_yscale('log')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Architecture Comparison: Shallow vs Deep',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save:
        output_dir = Path(__file__).parent
        output_file = output_dir / f'architecture_comparison.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nArchitecture comparison saved to: {output_file}")

    return fig, axes


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Plot dimension comparison heatmap and architecture comparisons'
    )
    parser.add_argument('--section', type=str, default='section2_1',
                       help='Section prefix (default: section2_1)')
    parser.add_argument('--plot-type', type=str, default='both',
                       choices=['heatmap', 'architecture', 'both'],
                       help='Type of plot to generate (default: both)')

    args = parser.parse_args()

    try:
        if args.plot_type in ['heatmap', 'both']:
            plot_dimension_comparison_heatmap(section=args.section, save=True)

        if args.plot_type in ['architecture', 'both']:
            plot_architecture_depth_comparison(save=True)

        plt.show()

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nMake sure you have run the high-D training scripts first:")
        print("  python section2_1_highd.py --dim 3 --architecture shallow")
        print("  python section2_1_highd.py --dim 3 --architecture deep")
        print("  (repeat for dims 4, 10, 100)")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
