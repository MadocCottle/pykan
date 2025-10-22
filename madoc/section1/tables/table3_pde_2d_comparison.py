"""
Table 3: 2D Poisson PDE Comparison (Section 1.3)

This table compares MLP, SIREN, KAN, and KAN with pruning on 2D Poisson PDE
solutions with different source terms.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from utils import (load_latest_results, print_table, create_latex_table,
                   save_table, compare_models)

def create_pde_2d_comparison_table():
    """Generate 2D PDE comparison table."""

    # Load results
    results = load_latest_results('section1_3')

    # Dataset names from section1_3
    dataset_names = ['poisson_2d_sin', 'poisson_2d_poly', 'poisson_2d_highfreq', 'poisson_2d_spec']

    # Create main comparison table
    comparison_df = compare_models(results, dataset_names, metric='test_mse')

    # Print to console
    print_table(comparison_df, "Table 3: 2D Poisson PDE Comparison (Section 1.3)")

    # Create simplified LaTeX version
    latex_data = []
    for _, row in comparison_df.iterrows():
        dataset = row['Dataset'].replace('poisson_2d_', '')

        latex_data.append({
            'PDE Type': dataset,
            'MLP MSE': row.get('MLP test_mse', 'N/A'),
            'MLP Params': row.get('MLP params', 0),
            'SIREN MSE': row.get('SIREN test_mse', 'N/A'),
            'SIREN Params': row.get('SIREN params', 0),
            'KAN MSE': row.get('KAN test_mse', 'N/A'),
            'KAN Params': row.get('KAN params', 0),
            'KAN Pruned MSE': row.get('KAN_PRUNING test_mse', 'N/A'),
            'KAN Pruned Params': row.get('KAN_PRUNING params', 0),
        })

    latex_df = pd.DataFrame(latex_data)

    # Print LaTeX format
    print_table(latex_df, "LaTeX Format Table", tablefmt='latex_raw')

    # Save LaTeX table
    latex_str = create_latex_table(
        latex_df,
        caption="2D Poisson PDE solution comparison. Results show test MSE and parameter counts "
                "for different 2D source terms. The 2D problem is more challenging than 1D, "
                "requiring more parameters across all model types.",
        label="tab:pde_2d_comparison",
        column_format="|l|c|c|c|c|c|c|c|c|"
    )
    save_table(latex_str, 'table3_pde_2d_comparison.tex')

    # Calculate improvement ratios
    improvement_df = calculate_improvements(results, dataset_names)
    print_table(improvement_df, "KAN vs MLP Improvement Ratios (2D PDEs)")

    # Save CSV
    comparison_df.to_csv(Path(__file__).parent / 'table3_pde_2d_comparison.csv', index=False)
    print("Saved CSV to table3_pde_2d_comparison.csv")

    return comparison_df, latex_df

def calculate_improvements(results: dict, dataset_names: list) -> pd.DataFrame:
    """Calculate improvement ratios of KAN vs baseline models."""
    improvement_data = []

    for dataset_name in dataset_names:
        row_data = {'Dataset': dataset_name.replace('poisson_2d_', '')}

        # Get MSE values
        mse_values = {}
        for model_type in ['mlp', 'siren', 'kan', 'kan_pruning']:
            df = results.get(model_type, pd.DataFrame())

            if not df.empty:
                dataset_df = df[df['dataset_name'] == dataset_name]
                if not dataset_df.empty:
                    mse_values[model_type] = dataset_df['test_mse'].min()

        # Calculate improvements
        if 'mlp' in mse_values and 'kan' in mse_values:
            row_data['KAN vs MLP MSE Ratio'] = f"{mse_values['kan'] / mse_values['mlp']:.3f}"
        else:
            row_data['KAN vs MLP MSE Ratio'] = 'N/A'

        if 'siren' in mse_values and 'kan' in mse_values:
            row_data['KAN vs SIREN MSE Ratio'] = f"{mse_values['kan'] / mse_values['siren']:.3f}"
        else:
            row_data['KAN vs SIREN MSE Ratio'] = 'N/A'

        if 'kan' in mse_values and 'kan_pruning' in mse_values:
            row_data['Pruned vs Regular KAN'] = f"{mse_values['kan_pruning'] / mse_values['kan']:.3f}"
        else:
            row_data['Pruned vs Regular KAN'] = 'N/A'

        improvement_data.append(row_data)

    return pd.DataFrame(improvement_data)

if __name__ == '__main__':
    comparison_df, latex_df = create_pde_2d_comparison_table()
