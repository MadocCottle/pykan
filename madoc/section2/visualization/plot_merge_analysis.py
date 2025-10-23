"""
Plot Merge_KAN Analysis (Section 2.3)

Creates comprehensive 3-panel visualization showing:
- Panel A: Expert pool performance (bar chart with dependency groups)
- Panel B: Training progression (line plot comparing experts vs merged)
- Panel C: Before/after merge comparison (bar chart)

Usage:
    python plot_merge_analysis.py --dataset 0  # Specific dataset
    python plot_merge_analysis.py              # All datasets
"""

import sys
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_run
from utils.result_finder import select_run


def load_section2_3_data(timestamp=None):
    """Load Section 2.3 results"""
    if timestamp is None:
        timestamp = select_run('section2_3')
        if timestamp is None:
            print("Error: No Section 2.3 results found")
            print("Please run: python ../section2_3.py --n-seeds 5")
            return None

    print(f"Loading Section 2.3 results with timestamp: {timestamp}")

    try:
        results, models, metadata = load_run('section2_3', timestamp, load_models=False)
        return results, metadata, timestamp
    except Exception as e:
        print(f"Error loading results: {e}")
        return None


def plot_expert_pool(ax, experts_df, selected_df, dataset_idx, dataset_name):
    """
    Panel A: Expert pool performance grouped by dependency pattern

    Args:
        ax: Matplotlib axis
        experts_df: DataFrame with all experts
        selected_df: DataFrame with selected experts
        dataset_idx: Index of dataset to plot
        dataset_name: Name of dataset for title
    """
    # Filter for this dataset
    dataset_experts = experts_df[experts_df['dataset_idx'] == dataset_idx].copy()
    dataset_selected = selected_df[selected_df['dataset_idx'] == dataset_idx]

    if dataset_experts.empty:
        ax.text(0.5, 0.5, 'No expert data available',
               ha='center', va='center', transform=ax.transAxes)
        return

    # Group by dependency pattern
    dataset_experts['dep_str'] = dataset_experts['dependencies'].astype(str)
    dataset_experts = dataset_experts.sort_values(['dep_str', 'dense_mse'])

    # Create x-positions
    x_pos = np.arange(len(dataset_experts))

    # Get unique dependency patterns for coloring
    unique_deps = dataset_experts['dep_str'].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_deps)))
    dep_colors = {dep: colors[i] for i, dep in enumerate(unique_deps)}

    # Plot bars
    bar_colors = [dep_colors[dep] for dep in dataset_experts['dep_str']]
    bars = ax.bar(x_pos, dataset_experts['dense_mse'], color=bar_colors, alpha=0.7, edgecolor='black', linewidth=0.5)

    # Highlight selected experts
    selected_indices = dataset_experts.index.isin(dataset_selected.index)
    for i, (idx, is_selected) in enumerate(zip(dataset_experts.index, selected_indices)):
        if is_selected:
            bars[i].set_edgecolor('red')
            bars[i].set_linewidth(2)
            # Add star marker
            ax.plot(i, dataset_experts.loc[idx, 'dense_mse'] * 1.05,
                   marker='*', markersize=12, color='red')

    ax.set_xlabel('Expert Index', fontsize=10)
    ax.set_ylabel('Dense MSE', fontsize=10)
    ax.set_title(f'Panel A: Expert Pool Performance\n{dataset_name}', fontsize=11, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3)

    # Legend for dependency patterns
    legend_elements = []
    for dep in unique_deps:
        legend_elements.append(plt.Rectangle((0,0),1,1, fc=dep_colors[dep], alpha=0.7, edgecolor='black', label=f'Deps: {dep}'))
    legend_elements.append(plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='red',
                                     markersize=10, label='Selected'))

    ax.legend(handles=legend_elements, loc='upper right', fontsize=8, framealpha=0.9)

    # Annotate: number of experts trained vs selected
    n_trained = len(dataset_experts)
    n_selected = len(dataset_selected)
    ax.text(0.02, 0.98, f'Trained: {n_trained}\nSelected: {n_selected}',
           transform=ax.transAxes, fontsize=9, va='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


def plot_training_progression(ax, grid_history_df, dataset_idx, dataset_name):
    """
    Panel B: Training progression showing grid refinement

    Args:
        ax: Matplotlib axis
        grid_history_df: DataFrame with grid refinement history
        dataset_idx: Index of dataset to plot
        dataset_name: Name of dataset for title
    """
    # Filter for this dataset
    dataset_history = grid_history_df[grid_history_df['dataset_idx'] == dataset_idx]

    if dataset_history.empty:
        ax.text(0.5, 0.5, 'No training history available',
               ha='center', va='center', transform=ax.transAxes)
        return

    # Plot train and test loss progression
    grid_sizes = dataset_history['grid_size'].values
    train_losses = dataset_history['final_train_loss'].values
    test_losses = dataset_history['final_test_loss'].values

    x_pos = np.arange(len(grid_sizes))

    ax.plot(x_pos, train_losses, marker='o', label='Train Loss', linewidth=2, markersize=6)
    ax.plot(x_pos, test_losses, marker='s', label='Test Loss', linewidth=2, markersize=6)

    ax.set_xlabel('Refinement Step', fontsize=10)
    ax.set_ylabel('Loss (MSE)', fontsize=10)
    ax.set_title(f'Panel B: Merged KAN Training Progression\n{dataset_name}', fontsize=11, fontweight='bold')
    ax.set_yscale('log')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'Grid {g}' for g in grid_sizes], rotation=45, ha='right')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    # Annotate final performance
    final_test_loss = test_losses[-1]
    ax.text(0.98, 0.98, f'Final Test Loss:\n{final_test_loss:.2e}',
           transform=ax.transAxes, fontsize=9, va='top', ha='right',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))


def plot_merge_comparison(ax, summary_df, experts_df, dataset_idx, dataset_name):
    """
    Panel C: Before/after merge comparison

    Args:
        ax: Matplotlib axis
        summary_df: DataFrame with merge summary
        experts_df: DataFrame with all experts
        dataset_idx: Index of dataset to plot
        dataset_name: Name of dataset for title
    """
    # Get merged KAN performance
    dataset_summary = summary_df[summary_df['dataset_idx'] == dataset_idx]

    if dataset_summary.empty:
        ax.text(0.5, 0.5, 'No summary data available',
               ha='center', va='center', transform=ax.transAxes)
        return

    merged_mse = dataset_summary['merged_dense_mse'].values[0]

    # Get best solo expert and average of selected experts
    dataset_experts = experts_df[experts_df['dataset_idx'] == dataset_idx]

    if not dataset_experts.empty:
        best_solo_mse = dataset_experts['dense_mse'].min()
        selected_experts = dataset_experts[dataset_experts['selected'] == True]

        if not selected_experts.empty:
            avg_selected_mse = selected_experts['dense_mse'].mean()
        else:
            avg_selected_mse = best_solo_mse
    else:
        best_solo_mse = merged_mse
        avg_selected_mse = merged_mse

    # Create bar chart
    categories = ['Best Solo\nExpert', 'Avg Selected\nExperts', 'Merged\nKAN']
    values = [best_solo_mse, avg_selected_mse, merged_mse]
    colors = ['#ff9999', '#ffcc99', '#99cc99']

    bars = ax.bar(categories, values, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)

    ax.set_ylabel('Dense MSE', fontsize=10)
    ax.set_title(f'Panel C: Performance Comparison\n{dataset_name}', fontsize=11, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3)

    # Annotate bars with values
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height * 1.1,
               f'{val:.2e}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Calculate and show improvement
    if best_solo_mse > 0:
        improvement = ((best_solo_mse - merged_mse) / best_solo_mse) * 100
        ax.text(0.98, 0.02, f'Merge Gain:\n{improvement:.1f}%',
               transform=ax.transAxes, fontsize=10, ha='right', va='bottom',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
               fontweight='bold')


def create_merge_analysis_plot(dataset_idx, dataset_name, results, output_dir='outputs'):
    """Create 3-panel merge analysis plot for a single dataset"""

    # Extract DataFrames
    summary_df = results.get('summary', pd.DataFrame())
    experts_df = results.get('experts', pd.DataFrame())
    selected_df = results.get('selected_experts', pd.DataFrame())
    grid_history_df = results.get('grid_history', pd.DataFrame())

    if summary_df.empty or experts_df.empty:
        print(f"Warning: Insufficient data for dataset {dataset_idx}")
        return

    # Create figure with 3 panels
    fig = plt.figure(figsize=(18, 6))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.3)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    # Plot each panel
    plot_expert_pool(ax1, experts_df, selected_df, dataset_idx, dataset_name)
    plot_training_progression(ax2, grid_history_df, dataset_idx, dataset_name)
    plot_merge_comparison(ax3, summary_df, experts_df, dataset_idx, dataset_name)

    # Overall title
    fig.suptitle(f'Merge_KAN Analysis: {dataset_name}',
                fontsize=14, fontweight='bold', y=0.98)

    # Save figure
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    filename = f'merge_analysis_dataset_{dataset_idx}_{dataset_name.replace(" ", "_").lower()}.png'
    filepath = output_path / filename

    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"Saved: {filepath}")

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Plot Merge_KAN analysis for Section 2.3')
    parser.add_argument('--dataset', type=int, help='Specific dataset index to plot (0-3)')
    parser.add_argument('--timestamp', type=str, help='Specific timestamp to load')
    parser.add_argument('--output', type=str, default='outputs', help='Output directory')

    args = parser.parse_args()

    # Load data
    data = load_section2_3_data(args.timestamp)
    if data is None:
        return 1

    results, metadata, timestamp = data

    # Dataset names
    dataset_names = [
        '2D Sin',
        '2D Polynomial',
        '2D High-freq',
        '2D Special'
    ]

    # Determine which datasets to plot
    if args.dataset is not None:
        datasets_to_plot = [args.dataset]
    else:
        # Plot all datasets
        summary_df = results.get('summary', pd.DataFrame())
        if not summary_df.empty:
            datasets_to_plot = summary_df['dataset_idx'].unique().tolist()
        else:
            datasets_to_plot = [0, 1, 2, 3]

    print(f"\nGenerating merge analysis plots for {len(datasets_to_plot)} dataset(s)...")

    for dataset_idx in datasets_to_plot:
        dataset_name = dataset_names[dataset_idx] if dataset_idx < len(dataset_names) else f'Dataset {dataset_idx}'
        print(f"\nProcessing {dataset_name}...")
        create_merge_analysis_plot(dataset_idx, dataset_name, results, args.output)

    print(f"\n{'='*80}")
    print(f"All merge analysis plots saved to: {args.output}/")
    print(f"{'='*80}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
