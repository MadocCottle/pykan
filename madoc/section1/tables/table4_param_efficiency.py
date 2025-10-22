"""
Table 4: Parameter Efficiency Analysis

This table demonstrates KAN's parameter efficiency compared to MLPs and SIRENs.
Shows that KANs can achieve similar or better accuracy with significantly fewer parameters.

Inspired by KAN paper Table 3 (signature classification comparison).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from utils import (load_latest_results, print_table, create_latex_table,
                   save_table, format_scientific)

def create_param_efficiency_table():
    """Generate parameter efficiency comparison across all sections."""

    all_sections = [
        ('section1_1', ['sin_freq1', 'sin_freq2', 'sin_freq3', 'sin_freq4', 'sin_freq5',
                        'piecewise', 'sawtooth', 'polynomial', 'poisson_1d_highfreq']),
        ('section1_2', ['poisson_1d_sin', 'poisson_1d_poly', 'poisson_1d_highfreq']),
        ('section1_3', ['poisson_2d_sin', 'poisson_2d_poly', 'poisson_2d_highfreq', 'poisson_2d_spec'])
    ]

    all_data = []

    for section_name, dataset_names in all_sections:
        results = load_latest_results(section_name)

        for dataset_name in dataset_names:
            row_data = analyze_dataset_efficiency(results, dataset_name, section_name)
            if row_data:
                all_data.append(row_data)

    efficiency_df = pd.DataFrame(all_data)

    # Print full table
    print_table(efficiency_df, "Table 4: Parameter Efficiency Analysis")

    # Create summary by section
    summary_df = create_efficiency_summary(efficiency_df)
    print_table(summary_df, "Parameter Efficiency Summary by Section")

    # Create best-case comparison
    best_case_df = create_best_case_comparison(efficiency_df)
    print_table(best_case_df, "Best Case: KAN vs MLP Parameter Reduction")

    # Save LaTeX table
    latex_str = create_latex_table(
        summary_df,
        caption="Parameter efficiency summary. Shows average parameter counts and test MSE "
                "for each model type across different problem sections. KAN models achieve "
                "competitive accuracy with significantly fewer parameters than MLPs.",
        label="tab:param_efficiency",
        column_format="|l|c|c|c|c|c|c|c|c|"
    )
    save_table(latex_str, 'table4_param_efficiency.tex')

    # Save CSV
    efficiency_df.to_csv(Path(__file__).parent / 'table4_param_efficiency.csv', index=False)
    print("Saved CSV to table4_param_efficiency.csv")

    return efficiency_df, summary_df

def analyze_dataset_efficiency(results: dict, dataset_name: str, section_name: str) -> dict:
    """Analyze parameter efficiency for a single dataset."""
    row_data = {
        'Section': section_name,
        'Dataset': dataset_name
    }

    for model_type in ['mlp', 'siren', 'kan', 'kan_pruning']:
        df = results.get(model_type, pd.DataFrame())

        if df.empty:
            row_data[f'{model_type.upper()} Params'] = np.nan
            row_data[f'{model_type.upper()} MSE'] = np.nan
            continue

        # Filter for this dataset
        dataset_df = df[df['dataset_name'] == dataset_name]

        if dataset_df.empty:
            row_data[f'{model_type.upper()} Params'] = np.nan
            row_data[f'{model_type.upper()} MSE'] = np.nan
            continue

        # Get best result
        best_idx = dataset_df['test_mse'].idxmin()
        best_row = dataset_df.loc[best_idx]

        row_data[f'{model_type.upper()} Params'] = int(best_row.get('num_params', 0))
        row_data[f'{model_type.upper()} MSE'] = best_row['test_mse']

    # Calculate efficiency ratios
    if pd.notna(row_data.get('MLP Params')) and pd.notna(row_data.get('KAN Params')):
        if row_data['KAN Params'] > 0:
            row_data['Param Reduction (MLP/KAN)'] = row_data['MLP Params'] / row_data['KAN Params']
        else:
            row_data['Param Reduction (MLP/KAN)'] = np.nan
    else:
        row_data['Param Reduction (MLP/KAN)'] = np.nan

    return row_data if any(pd.notna(row_data.get(f'{m.upper()} Params')) for m in ['mlp', 'siren', 'kan']) else None

def create_efficiency_summary(efficiency_df: pd.DataFrame) -> pd.DataFrame:
    """Create summary statistics by section."""
    summary_data = []

    for section in efficiency_df['Section'].unique():
        section_df = efficiency_df[efficiency_df['Section'] == section]

        summary_row = {'Section': section}

        for model_type in ['MLP', 'SIREN', 'KAN', 'KAN_PRUNING']:
            param_col = f'{model_type} Params'
            mse_col = f'{model_type} MSE'

            if param_col in section_df.columns:
                summary_row[f'{model_type} Avg Params'] = f"{section_df[param_col].mean():.0f}"
                summary_row[f'{model_type} Avg MSE'] = format_scientific(section_df[mse_col].mean())

        # Average parameter reduction
        if 'Param Reduction (MLP/KAN)' in section_df.columns:
            avg_reduction = section_df['Param Reduction (MLP/KAN)'].mean()
            summary_row['Avg MLP/KAN Ratio'] = f"{avg_reduction:.2f}x"

        summary_data.append(summary_row)

    return pd.DataFrame(summary_data)

def create_best_case_comparison(efficiency_df: pd.DataFrame) -> pd.DataFrame:
    """Find best cases of parameter efficiency."""
    # Filter rows where KAN achieves better or similar MSE with fewer params
    best_cases = []

    for _, row in efficiency_df.iterrows():
        if (pd.notna(row.get('MLP MSE')) and pd.notna(row.get('KAN MSE')) and
            pd.notna(row.get('MLP Params')) and pd.notna(row.get('KAN Params'))):

            # KAN achieves better or within 10% MSE with fewer params
            if row['KAN MSE'] <= row['MLP MSE'] * 1.1 and row['KAN Params'] < row['MLP Params']:
                best_cases.append({
                    'Dataset': row['Dataset'],
                    'MLP Params': int(row['MLP Params']),
                    'MLP MSE': format_scientific(row['MLP MSE']),
                    'KAN Params': int(row['KAN Params']),
                    'KAN MSE': format_scientific(row['KAN MSE']),
                    'Param Reduction': f"{row['Param Reduction (MLP/KAN)']:.2f}x",
                    'MSE Improvement': f"{(1 - row['KAN MSE'] / row['MLP MSE']) * 100:.1f}%"
                })

    return pd.DataFrame(best_cases)

if __name__ == '__main__':
    efficiency_df, summary_df = create_param_efficiency_table()
