"""
Table 6: KAN Grid Size Ablation Study

Analyzes how different grid sizes affect KAN performance across datasets.
Shows trade-off between grid resolution and accuracy.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from utils import (load_latest_results, print_table, create_latex_table,
                   save_table, format_scientific)

def create_grid_ablation_table():
    """Generate grid size ablation analysis."""

    all_sections = [
        ('section1_1', ['sin_freq1', 'piecewise', 'polynomial', 'poisson_1d_highfreq']),
        ('section1_2', ['poisson_1d_sin', 'poisson_1d_poly', 'poisson_1d_highfreq']),
        ('section1_3', ['poisson_2d_sin', 'poisson_2d_poly'])
    ]

    all_data = []

    for section_name, dataset_names in all_sections:
        results = load_latest_results(section_name)

        # Analyze both regular KAN and pruned KAN
        for kan_type in ['kan', 'kan_pruning']:
            df = results.get(kan_type, pd.DataFrame())

            if df.empty:
                continue

            for dataset_name in dataset_names:
                dataset_df = df[df['dataset_name'] == dataset_name]

                if dataset_df.empty:
                    continue

                grid_analysis = analyze_grid_sizes(dataset_df, dataset_name, section_name, kan_type)
                all_data.extend(grid_analysis)

    grid_df = pd.DataFrame(all_data)

    # Print full table
    print_table(grid_df, "Table 6: KAN Grid Size Ablation Study")

    # Create summary table
    summary_df = create_grid_summary(grid_df)
    print_table(summary_df, "Grid Size Summary: Best Performing Grids")

    # Create pivot table by grid size
    pivot_df = create_grid_pivot(grid_df)
    print_table(pivot_df, "Average MSE by Grid Size and Section")

    # Save LaTeX table
    latex_str = create_latex_table(
        summary_df,
        caption="Grid size ablation study for KAN models. Shows how different grid resolutions "
                "affect test MSE across problem types. Larger grids generally improve accuracy "
                "but increase parameter count and training time.",
        label="tab:grid_ablation",
        column_format="|l|l|c|c|c|c|c|"
    )
    save_table(latex_str, 'table6_grid_ablation.tex')

    # Save CSV
    grid_df.to_csv(Path(__file__).parent / 'table6_grid_ablation.csv', index=False)
    print("Saved CSV to table6_grid_ablation.csv")

    return grid_df, summary_df

def analyze_grid_sizes(dataset_df: pd.DataFrame, dataset_name: str,
                       section_name: str, kan_type: str) -> list:
    """Analyze performance across different grid sizes for a dataset."""
    results = []

    if 'grid_size' not in dataset_df.columns:
        return results

    for grid_size in dataset_df['grid_size'].unique():
        grid_df = dataset_df[dataset_df['grid_size'] == grid_size]

        if grid_df.empty:
            continue

        # Get best result for this grid size
        best_idx = grid_df['test_mse'].idxmin()
        best_row = grid_df.loc[best_idx]

        results.append({
            'Section': section_name,
            'Dataset': dataset_name,
            'Model': 'KAN Pruned' if kan_type == 'kan_pruning' else 'KAN',
            'Grid Size': int(grid_size),
            'Test MSE': best_row['test_mse'],
            'Num Params': int(best_row.get('num_params', 0)),
            'Time/Epoch (s)': best_row.get('time_per_epoch', np.nan)
        })

    return results

def create_grid_summary(grid_df: pd.DataFrame) -> pd.DataFrame:
    """Create summary showing best performing grid for each dataset."""
    summary_data = []

    for (section, dataset, model), group in grid_df.groupby(['Section', 'Dataset', 'Model']):
        best_idx = group['Test MSE'].idxmin()
        best_row = group.loc[best_idx]

        summary_data.append({
            'Section': section,
            'Dataset': dataset,
            'Model': model,
            'Best Grid': int(best_row['Grid Size']),
            'Best MSE': format_scientific(best_row['Test MSE']),
            'Params at Best': int(best_row['Num Params']),
            'Worst Grid': int(group.loc[group['Test MSE'].idxmax(), 'Grid Size']),
            'Worst MSE': format_scientific(group['Test MSE'].max())
        })

    return pd.DataFrame(summary_data)

def create_grid_pivot(grid_df: pd.DataFrame) -> pd.DataFrame:
    """Create pivot table showing average MSE by grid size across sections."""
    pivot_data = []

    for section in grid_df['Section'].unique():
        section_df = grid_df[grid_df['Section'] == section]

        row = {'Section': section}

        for grid_size in sorted(section_df['Grid Size'].unique()):
            grid_data = section_df[section_df['Grid Size'] == grid_size]
            avg_mse = grid_data['Test MSE'].mean()
            row[f'Grid {int(grid_size)}'] = format_scientific(avg_mse)

        pivot_data.append(row)

    return pd.DataFrame(pivot_data)

def create_param_vs_accuracy_plot_data():
    """Prepare data for plotting parameter count vs accuracy."""
    # This could be used with matplotlib to create visualizations
    print("Note: Use this data with visualization scripts for param vs accuracy plots")
    return None

if __name__ == '__main__':
    grid_df, summary_df = create_grid_ablation_table()
