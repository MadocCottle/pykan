"""
Table 7: Depth Ablation Study for MLP and SIREN

Analyzes how network depth affects performance for MLP and SIREN models.
Compares different depths and activation functions.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from utils import (load_latest_results, print_table, create_latex_table,
                   save_table, format_scientific)

def create_depth_ablation_table():
    """Generate depth ablation analysis for MLP and SIREN."""

    all_sections = [
        ('section1_1', ['sin_freq1', 'piecewise', 'polynomial', 'poisson_1d_highfreq']),
        ('section1_2', ['poisson_1d_sin', 'poisson_1d_poly', 'poisson_1d_highfreq']),
        ('section1_3', ['poisson_2d_sin', 'poisson_2d_poly'])
    ]

    all_data = []

    for section_name, dataset_names in all_sections:
        results = load_latest_results(section_name)

        # Analyze MLP with different activations and depths
        mlp_df = results.get('mlp', pd.DataFrame())
        if not mlp_df.empty:
            for dataset_name in dataset_names:
                dataset_df = mlp_df[mlp_df['dataset_name'] == dataset_name]
                if not dataset_df.empty:
                    depth_analysis = analyze_depths(dataset_df, dataset_name, section_name, 'MLP')
                    all_data.extend(depth_analysis)

        # Analyze SIREN depths
        siren_df = results.get('siren', pd.DataFrame())
        if not siren_df.empty:
            for dataset_name in dataset_names:
                dataset_df = siren_df[siren_df['dataset_name'] == dataset_name]
                if not dataset_df.empty:
                    depth_analysis = analyze_depths(dataset_df, dataset_name, section_name, 'SIREN')
                    all_data.extend(depth_analysis)

    depth_df = pd.DataFrame(all_data)

    # Print full table
    print_table(depth_df, "Table 7: Depth Ablation Study")

    # Create summary by depth
    summary_df = create_depth_summary(depth_df)
    print_table(summary_df, "Depth Summary: Average Performance by Depth")

    # Create activation comparison for MLP
    activation_df = create_activation_comparison(depth_df)
    print_table(activation_df, "MLP Activation Function Comparison")

    # Create best depth table
    best_depth_df = create_best_depth_table(depth_df)
    print_table(best_depth_df, "Best Performing Depth per Dataset")

    # Save LaTeX table
    latex_str = create_latex_table(
        summary_df,
        caption="Depth ablation study for MLP and SIREN models. Shows how network depth "
                "affects test MSE and parameter count. Deeper networks have more parameters "
                "but may not always achieve better accuracy.",
        label="tab:depth_ablation",
        column_format="|l|l|c|c|c|c|"
    )
    save_table(latex_str, 'table7_depth_ablation.tex')

    # Save CSV
    depth_df.to_csv(Path(__file__).parent / 'table7_depth_ablation.csv', index=False)
    print("Saved CSV to table7_depth_ablation.csv")

    return depth_df, summary_df

def analyze_depths(dataset_df: pd.DataFrame, dataset_name: str,
                   section_name: str, model_type: str) -> list:
    """Analyze performance across different depths for a dataset."""
    results = []

    if 'depth' not in dataset_df.columns:
        return results

    for depth in dataset_df['depth'].unique():
        depth_data = dataset_df[dataset_df['depth'] == depth]

        if depth_data.empty:
            continue

        # For MLP, also group by activation
        if model_type == 'MLP' and 'activation' in depth_data.columns:
            for activation in depth_data['activation'].unique():
                act_data = depth_data[depth_data['activation'] == activation]

                if act_data.empty:
                    continue

                best_idx = act_data['test_mse'].idxmin()
                best_row = act_data.loc[best_idx]

                results.append({
                    'Section': section_name,
                    'Dataset': dataset_name,
                    'Model': model_type,
                    'Depth': int(depth),
                    'Activation': activation,
                    'Test MSE': best_row['test_mse'],
                    'Num Params': int(best_row.get('num_params', 0)),
                    'Time/Epoch (s)': best_row.get('time_per_epoch', np.nan)
                })
        else:
            # SIREN doesn't have activation variation
            best_idx = depth_data['test_mse'].idxmin()
            best_row = depth_data.loc[best_idx]

            results.append({
                'Section': section_name,
                'Dataset': dataset_name,
                'Model': model_type,
                'Depth': int(depth),
                'Activation': 'SIREN' if model_type == 'SIREN' else 'N/A',
                'Test MSE': best_row['test_mse'],
                'Num Params': int(best_row.get('num_params', 0)),
                'Time/Epoch (s)': best_row.get('time_per_epoch', np.nan)
            })

    return results

def create_depth_summary(depth_df: pd.DataFrame) -> pd.DataFrame:
    """Create summary statistics by depth."""
    summary_data = []

    for (model, depth), group in depth_df.groupby(['Model', 'Depth']):
        summary_data.append({
            'Model': model,
            'Depth': int(depth),
            'Avg MSE': format_scientific(group['Test MSE'].mean()),
            'Std MSE': format_scientific(group['Test MSE'].std()),
            'Avg Params': f"{group['Num Params'].mean():.0f}",
            'Count': len(group)
        })

    return pd.DataFrame(summary_data).sort_values(['Model', 'Depth'])

def create_activation_comparison(depth_df: pd.DataFrame) -> pd.DataFrame:
    """Compare activation functions for MLP."""
    mlp_df = depth_df[depth_df['Model'] == 'MLP']

    if mlp_df.empty or 'Activation' not in mlp_df.columns:
        return pd.DataFrame()

    comparison_data = []

    for activation in mlp_df['Activation'].unique():
        if activation == 'N/A':
            continue

        act_data = mlp_df[mlp_df['Activation'] == activation]

        comparison_data.append({
            'Activation': activation,
            'Avg MSE': format_scientific(act_data['Test MSE'].mean()),
            'Best MSE': format_scientific(act_data['Test MSE'].min()),
            'Worst MSE': format_scientific(act_data['Test MSE'].max()),
            'Avg Params': f"{act_data['Num Params'].mean():.0f}",
            'Num Configs': len(act_data)
        })

    return pd.DataFrame(comparison_data)

def create_best_depth_table(depth_df: pd.DataFrame) -> pd.DataFrame:
    """Find best performing depth for each dataset and model."""
    best_data = []

    for (section, dataset, model), group in depth_df.groupby(['Section', 'Dataset', 'Model']):
        best_idx = group['Test MSE'].idxmin()
        best_row = group.loc[best_idx]

        best_data.append({
            'Section': section,
            'Dataset': dataset,
            'Model': model,
            'Best Depth': int(best_row['Depth']),
            'Best MSE': format_scientific(best_row['Test MSE']),
            'Activation': best_row.get('Activation', 'N/A'),
            'Params': int(best_row['Num Params'])
        })

    return pd.DataFrame(best_data)

if __name__ == '__main__':
    depth_df, summary_df = create_depth_ablation_table()
