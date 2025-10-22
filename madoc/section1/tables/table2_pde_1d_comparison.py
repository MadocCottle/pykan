"""
Table 2: 1D Poisson PDE Comparison (Section 1.2)

This table compares MLP, SIREN, KAN, and KAN with pruning on 1D Poisson PDE
solutions with different source terms (sinusoidal, polynomial, high-frequency).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from utils import (load_latest_results, print_table, create_latex_table,
                   save_table, compare_models, format_scientific)

def create_pde_1d_comparison_table():
    """Generate 1D PDE comparison table."""

    # Load results
    results = load_latest_results('section1_2')

    # Dataset names from section1_2
    dataset_names = ['poisson_1d_sin', 'poisson_1d_poly', 'poisson_1d_highfreq']

    # Create main comparison table
    comparison_df = compare_models(results, dataset_names, metric='test_mse')

    # Print to console
    print_table(comparison_df, "Table 2: 1D Poisson PDE Comparison (Section 1.2)")

    # Create simplified LaTeX version
    latex_data = []
    for _, row in comparison_df.iterrows():
        dataset = row['Dataset'].replace('poisson_1d_', '')

        latex_data.append({
            'PDE Type': dataset,
            'MLP MSE': row.get('MLP test_mse', 'N/A'),
            'MLP Config': row.get('MLP arch', 'N/A'),
            'SIREN MSE': row.get('SIREN test_mse', 'N/A'),
            'SIREN Config': row.get('SIREN arch', 'N/A'),
            'KAN MSE': row.get('KAN test_mse', 'N/A'),
            'KAN Config': row.get('KAN arch', 'N/A'),
            'KAN Pruned MSE': row.get('KAN_PRUNING test_mse', 'N/A'),
            'KAN Pruned Config': row.get('KAN_PRUNING arch', 'N/A'),
        })

    latex_df = pd.DataFrame(latex_data)

    # Print LaTeX format
    print_table(latex_df, "LaTeX Format Table", tablefmt='latex_raw')

    # Save LaTeX table
    latex_str = create_latex_table(
        latex_df,
        caption="1D Poisson PDE solution comparison. Results show test MSE for different "
                "source terms: sinusoidal, polynomial, and high-frequency. KAN models "
                "demonstrate strong performance on PDE solving tasks.",
        label="tab:pde_1d_comparison",
        column_format="|l|c|c|c|c|c|c|c|c|"
    )
    save_table(latex_str, 'table2_pde_1d_comparison.tex')

    # Create parameter efficiency comparison
    param_comparison = create_parameter_comparison(results, dataset_names)
    print_table(param_comparison, "Parameter Efficiency for 1D PDEs")

    # Save CSV
    comparison_df.to_csv(Path(__file__).parent / 'table2_pde_1d_comparison.csv', index=False)
    print("Saved CSV to table2_pde_1d_comparison.csv")

    return comparison_df, latex_df

def create_parameter_comparison(results: dict, dataset_names: list) -> pd.DataFrame:
    """Compare parameter counts across models for best-performing configurations."""
    param_data = []

    for dataset_name in dataset_names:
        row_data = {'Dataset': dataset_name.replace('poisson_1d_', '')}

        for model_type in ['mlp', 'siren', 'kan', 'kan_pruning']:
            df = results.get(model_type, pd.DataFrame())

            if df.empty:
                row_data[f'{model_type.upper()} Params'] = 'N/A'
                continue

            # Filter for this dataset
            dataset_df = df[df['dataset_name'] == dataset_name]

            if dataset_df.empty:
                row_data[f'{model_type.upper()} Params'] = 'N/A'
                continue

            # Get best result
            best_idx = dataset_df['test_mse'].idxmin()
            best_row = dataset_df.loc[best_idx]

            row_data[f'{model_type.upper()} Params'] = int(best_row.get('num_params', 0))

        param_data.append(row_data)

    return pd.DataFrame(param_data)

if __name__ == '__main__':
    comparison_df, latex_df = create_pde_1d_comparison_table()