"""
Table 1: Function Approximation Comparison (Section 1.1)

This table compares MLP, SIREN, KAN, and KAN with pruning on various 1D function
approximation tasks including sinusoids, piecewise functions, sawtooth, polynomial,
and high-frequency Poisson solutions.

Similar to KAN paper Table 1 (Special Functions) but adapted for Section 1.1 data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from utils import (load_latest_results, print_table, create_latex_table,
                   save_table, format_scientific, compare_models)

def create_function_approximation_table():
    """Generate comprehensive function approximation comparison table."""

    # Load results
    results = load_latest_results('section1_1')

    # Dataset names from section1_1
    dataset_names = [
        'sin_freq1', 'sin_freq2', 'sin_freq3', 'sin_freq4', 'sin_freq5',
        'piecewise', 'sawtooth', 'polynomial', 'poisson_1d_highfreq'
    ]

    # Create main comparison table
    comparison_df = compare_models(results, dataset_names, metric='test_mse')

    # Print to console
    print_table(comparison_df, "Table 1: Function Approximation Comparison (Section 1.1)")

    # Create a simplified version for LaTeX
    latex_data = []
    for _, row in comparison_df.iterrows():
        dataset = row['Dataset']

        # Extract best MSE values
        mlp_mse = row.get('MLP test_mse', 'N/A')
        siren_mse = row.get('SIREN test_mse', 'N/A')
        kan_mse = row.get('KAN test_mse', 'N/A')
        kan_pruning_mse = row.get('KAN_PRUNING test_mse', 'N/A')

        # Extract architectures
        mlp_arch = row.get('MLP arch', 'N/A')
        siren_arch = row.get('SIREN arch', 'N/A')
        kan_arch = row.get('KAN arch', 'N/A')
        kan_pruning_arch = row.get('KAN_PRUNING arch', 'N/A')

        # Extract parameters
        mlp_params = row.get('MLP params', 0)
        siren_params = row.get('SIREN params', 0)
        kan_params = row.get('KAN params', 0)
        kan_pruning_params = row.get('KAN_PRUNING params', 0)

        latex_data.append({
            'Function': dataset,
            'MLP Config': mlp_arch,
            'MLP Test MSE': mlp_mse,
            'MLP Params': mlp_params,
            'SIREN Config': siren_arch,
            'SIREN Test MSE': siren_mse,
            'SIREN Params': siren_params,
            'KAN Config': kan_arch,
            'KAN Test MSE': kan_mse,
            'KAN Params': kan_params,
            'KAN Pruned Config': kan_pruning_arch,
            'KAN Pruned MSE': kan_pruning_mse,
            'KAN Pruned Params': kan_pruning_params,
        })

    latex_df = pd.DataFrame(latex_data)

    # Print LaTeX version
    print_table(latex_df, "LaTeX Format Table", tablefmt='latex_raw')

    # Save LaTeX table
    latex_str = create_latex_table(
        latex_df,
        caption="Function approximation comparison across MLP, SIREN, KAN, and KAN with pruning. "
                "Lower test MSE is better. KAN models typically achieve comparable or better accuracy "
                "with fewer parameters.",
        label="tab:function_approximation",
        column_format="|l|c|c|c|c|c|c|c|c|c|c|c|c|"
    )
    save_table(latex_str, 'table1_function_approximation.tex')

    # Create summary statistics
    summary_stats = create_summary_statistics(results)
    print_table(summary_stats, "Summary Statistics")

    # Save CSV
    comparison_df.to_csv(Path(__file__).parent / 'table1_function_approximation.csv', index=False)
    print("Saved CSV to table1_function_approximation.csv")

    return comparison_df, latex_df

def create_summary_statistics(results: dict) -> pd.DataFrame:
    """Create summary statistics across all datasets."""
    summary_data = []

    for model_type in ['mlp', 'siren', 'kan', 'kan_pruning']:
        df = results.get(model_type, pd.DataFrame())

        if df.empty:
            continue

        # Get best result per dataset
        best_per_dataset = df.groupby('dataset_name')['test_mse'].min()

        summary_data.append({
            'Model': model_type.upper(),
            'Mean Best MSE': format_scientific(best_per_dataset.mean()),
            'Std Best MSE': format_scientific(best_per_dataset.std()),
            'Min Best MSE': format_scientific(best_per_dataset.min()),
            'Max Best MSE': format_scientific(best_per_dataset.max()),
            'Avg Params': f"{df['num_params'].mean():.0f}",
            'Min Params': f"{df['num_params'].min():.0f}",
            'Max Params': f"{df['num_params'].max():.0f}",
        })

    return pd.DataFrame(summary_data)

if __name__ == '__main__':
    comparison_df, latex_df = create_function_approximation_table()