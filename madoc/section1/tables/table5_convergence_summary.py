"""
Table 5: Training Efficiency and Convergence Summary

Analyzes training time, convergence speed, and computational efficiency
across different model types.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from utils import (load_latest_results, print_table, create_latex_table,
                   save_table, format_scientific)

def create_convergence_summary_table():
    """Generate training efficiency comparison table."""

    all_sections = [
        ('section1_1', 'Function Approximation'),
        ('section1_2', '1D Poisson PDE'),
        ('section1_3', '2D Poisson PDE')
    ]

    all_data = []

    for section_name, section_label in all_sections:
        results = load_latest_results(section_name)

        for model_type in ['mlp', 'siren', 'kan', 'kan_pruning']:
            df = results.get(model_type, pd.DataFrame())

            if df.empty:
                continue

            # Calculate statistics
            stats = calculate_training_stats(df, model_type, section_label)
            if stats:
                all_data.append(stats)

    convergence_df = pd.DataFrame(all_data)

    # Print full table
    print_table(convergence_df, "Table 5: Training Efficiency and Convergence Summary")

    # Create comparison table
    comparison_df = create_speed_comparison(convergence_df)
    print_table(comparison_df, "Training Speed Comparison")

    # Save LaTeX table
    latex_str = create_latex_table(
        convergence_df,
        caption="Training efficiency summary showing average time per epoch, final test MSE, "
                "and parameter counts across different model types and problem sections. "
                "Times measured on the same hardware for fair comparison.",
        label="tab:convergence_summary",
        column_format="|l|l|c|c|c|c|"
    )
    save_table(latex_str, 'table5_convergence_summary.tex')

    # Save CSV
    convergence_df.to_csv(Path(__file__).parent / 'table5_convergence_summary.csv', index=False)
    print("Saved CSV to table5_convergence_summary.csv")

    return convergence_df, comparison_df

def calculate_training_stats(df: pd.DataFrame, model_type: str, section_label: str) -> dict:
    """Calculate training statistics for a model type."""

    stats = {
        'Section': section_label,
        'Model': model_type.upper()
    }

    # Time per epoch
    if 'time_per_epoch' in df.columns:
        stats['Avg Time/Epoch (s)'] = f"{df['time_per_epoch'].mean():.4f}"
        stats['Std Time/Epoch (s)'] = f"{df['time_per_epoch'].std():.4f}"
    else:
        stats['Avg Time/Epoch (s)'] = 'N/A'
        stats['Std Time/Epoch (s)'] = 'N/A'

    # Best test MSE per dataset
    if 'test_mse' in df.columns:
        best_per_dataset = df.groupby('dataset_name')['test_mse'].min()
        stats['Avg Best MSE'] = format_scientific(best_per_dataset.mean())
        stats['Std Best MSE'] = format_scientific(best_per_dataset.std())
    else:
        stats['Avg Best MSE'] = 'N/A'
        stats['Std Best MSE'] = 'N/A'

    # Parameter count
    if 'num_params' in df.columns:
        stats['Avg Params'] = f"{df['num_params'].mean():.0f}"
    else:
        stats['Avg Params'] = 'N/A'

    # Total runs
    stats['Num Configs'] = len(df)

    return stats

def create_speed_comparison(convergence_df: pd.DataFrame) -> pd.DataFrame:
    """Create relative speed comparison."""
    comparison_data = []

    for section in convergence_df['Section'].unique():
        section_df = convergence_df[convergence_df['Section'] == section]

        row = {'Section': section}

        # Extract time per epoch for each model
        times = {}
        for model in ['MLP', 'SIREN', 'KAN', 'KAN_PRUNING']:
            model_row = section_df[section_df['Model'] == model]
            if not model_row.empty:
                time_str = model_row.iloc[0]['Avg Time/Epoch (s)']
                if time_str != 'N/A':
                    times[model] = float(time_str)

        # Calculate relative speeds (normalized to MLP)
        if 'MLP' in times:
            mlp_time = times['MLP']
            for model, time in times.items():
                row[f'{model} Relative Speed'] = f"{mlp_time / time:.2f}x" if time > 0 else 'N/A'

        comparison_data.append(row)

    return pd.DataFrame(comparison_data)

def create_epoch_efficiency_table():
    """Create table showing MSE achieved at different epoch counts."""
    # This requires epoch-by-epoch data which may not be stored
    # Can be implemented if training history is saved
    print("Note: Epoch efficiency analysis requires epoch-by-epoch training history")
    return None

if __name__ == '__main__':
    convergence_df, comparison_df = create_convergence_summary_table()
