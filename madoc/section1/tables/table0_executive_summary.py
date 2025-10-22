"""
Table 0: Executive Summary

High-level summary of all experimental results across sections.
Provides a quick overview of key findings.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from utils import (load_latest_results, print_table, create_latex_table,
                   save_table, format_scientific)

def create_executive_summary():
    """Generate executive summary of all experiments."""

    sections = [
        ('section1_1', 'Function Approximation', 9),
        ('section1_2', '1D Poisson PDE', 3),
        ('section1_3', '2D Poisson PDE', 4)
    ]

    summary_data = []

    for section_name, section_label, num_datasets in sections:
        results = load_latest_results(section_name)

        section_summary = analyze_section(results, section_label, num_datasets)
        summary_data.append(section_summary)

    summary_df = pd.DataFrame(summary_data)

    # Print summary
    print_table(summary_df, "Table 0: Executive Summary - All Sections")

    # Create detailed comparison
    detailed_df = create_detailed_summary(sections)
    print_table(detailed_df, "Detailed Model Comparison")

    # Create key findings
    findings_df = create_key_findings(sections)
    print_table(findings_df, "Key Findings")

    # Save LaTeX table
    latex_str = create_latex_table(
        summary_df,
        caption="Executive summary of experimental results across all sections. "
                "Shows average best performance and parameter counts for each model type.",
        label="tab:executive_summary",
        column_format="|l|c|c|c|c|c|c|c|c|"
    )
    save_table(latex_str, 'table0_executive_summary.tex')

    # Save CSV
    summary_df.to_csv(Path(__file__).parent / 'table0_executive_summary.csv', index=False)
    print("Saved CSV to table0_executive_summary.csv")

    return summary_df

def analyze_section(results: dict, section_label: str, num_datasets: int) -> dict:
    """Analyze results for a single section."""
    summary = {
        'Section': section_label,
        'Num Datasets': num_datasets
    }

    for model_type in ['mlp', 'siren', 'kan', 'kan_pruning']:
        df = results.get(model_type, pd.DataFrame())

        if df.empty:
            summary[f'{model_type.upper()} Avg Best MSE'] = 'N/A'
            summary[f'{model_type.upper()} Avg Params'] = 'N/A'
            continue

        # Get best MSE per dataset
        best_per_dataset = df.groupby('dataset_name')['test_mse'].min()

        # Get corresponding params
        best_indices = df.groupby('dataset_name')['test_mse'].idxmin()
        best_params = df.loc[best_indices, 'num_params'].mean()

        summary[f'{model_type.upper()} Avg Best MSE'] = format_scientific(best_per_dataset.mean())
        summary[f'{model_type.upper()} Avg Params'] = f"{best_params:.0f}"

    return summary

def create_detailed_summary(sections: list) -> pd.DataFrame:
    """Create detailed comparison across all sections."""
    detailed_data = []

    total_datasets = 0
    all_results = {'mlp': [], 'siren': [], 'kan': [], 'kan_pruning': []}

    for section_name, section_label, num_datasets in sections:
        results = load_latest_results(section_name)
        total_datasets += num_datasets

        for model_type in ['mlp', 'siren', 'kan', 'kan_pruning']:
            df = results.get(model_type, pd.DataFrame())
            if not df.empty:
                best_per_dataset = df.groupby('dataset_name')['test_mse'].min()
                all_results[model_type].extend(best_per_dataset.tolist())

    # Create overall statistics
    for model_type in ['mlp', 'siren', 'kan', 'kan_pruning']:
        if all_results[model_type]:
            mse_values = all_results[model_type]
            detailed_data.append({
                'Model': model_type.upper(),
                'Total Datasets': total_datasets,
                'Avg Best MSE': format_scientific(np.mean(mse_values)),
                'Std Best MSE': format_scientific(np.std(mse_values)),
                'Min Best MSE': format_scientific(np.min(mse_values)),
                'Max Best MSE': format_scientific(np.max(mse_values)),
                'Median Best MSE': format_scientific(np.median(mse_values))
            })

    return pd.DataFrame(detailed_data)

def create_key_findings(sections: list) -> pd.DataFrame:
    """Extract key findings from experiments."""
    findings = []

    # Overall performance comparison
    findings.append({
        'Category': 'Performance',
        'Finding': 'Best Overall Model',
        'Value': determine_best_model(sections)
    })

    # Parameter efficiency
    findings.append({
        'Category': 'Efficiency',
        'Finding': 'Most Parameter Efficient',
        'Value': determine_most_efficient(sections)
    })

    # Training speed
    findings.append({
        'Category': 'Training',
        'Finding': 'Fastest Training',
        'Value': determine_fastest(sections)
    })

    # Best for 1D tasks
    findings.append({
        'Category': 'Task-Specific',
        'Finding': 'Best for 1D Tasks',
        'Value': determine_best_for_dimension(sections, '1D')
    })

    # Best for 2D tasks
    findings.append({
        'Category': 'Task-Specific',
        'Finding': 'Best for 2D Tasks',
        'Value': determine_best_for_dimension(sections, '2D')
    })

    # Pruning effectiveness
    findings.append({
        'Category': 'Optimization',
        'Finding': 'Pruning Effectiveness',
        'Value': evaluate_pruning(sections)
    })

    return pd.DataFrame(findings)

def determine_best_model(sections: list) -> str:
    """Determine which model performs best overall."""
    all_mse = {}

    for section_name, _, _ in sections:
        results = load_latest_results(section_name)

        for model_type in ['mlp', 'siren', 'kan', 'kan_pruning']:
            df = results.get(model_type, pd.DataFrame())
            if not df.empty:
                best_per_dataset = df.groupby('dataset_name')['test_mse'].min()
                if model_type not in all_mse:
                    all_mse[model_type] = []
                all_mse[model_type].extend(best_per_dataset.tolist())

    # Compare averages
    avg_mse = {k: np.mean(v) for k, v in all_mse.items() if v}

    if avg_mse:
        best = min(avg_mse.items(), key=lambda x: x[1])
        return f"{best[0].upper()} (avg MSE: {format_scientific(best[1])})"
    return "N/A"

def determine_most_efficient(sections: list) -> str:
    """Determine most parameter efficient model."""
    efficiency_ratios = []

    for section_name, _, _ in sections:
        results = load_latest_results(section_name)

        kan_df = results.get('kan', pd.DataFrame())
        mlp_df = results.get('mlp', pd.DataFrame())

        if not kan_df.empty and not mlp_df.empty:
            # Get average params for best configs
            kan_params = kan_df.groupby('dataset_name')['test_mse'].idxmin()
            kan_params = kan_df.loc[kan_params, 'num_params'].mean()

            mlp_params = mlp_df.groupby('dataset_name')['test_mse'].idxmin()
            mlp_params = mlp_df.loc[mlp_params, 'num_params'].mean()

            if kan_params > 0:
                efficiency_ratios.append(mlp_params / kan_params)

    if efficiency_ratios:
        avg_ratio = np.mean(efficiency_ratios)
        return f"KAN (uses {1/avg_ratio:.2%} of MLP params on avg, {avg_ratio:.1f}x reduction)"
    return "N/A"

def determine_fastest(sections: list) -> str:
    """Determine fastest training model."""
    avg_times = {}

    for section_name, _, _ in sections:
        results = load_latest_results(section_name)

        for model_type in ['mlp', 'siren', 'kan']:
            df = results.get(model_type, pd.DataFrame())
            if not df.empty and 'time_per_epoch' in df.columns:
                if model_type not in avg_times:
                    avg_times[model_type] = []
                avg_times[model_type].append(df['time_per_epoch'].mean())

    # Compare averages
    if avg_times:
        avg_times_final = {k: np.mean(v) for k, v in avg_times.items()}
        fastest = min(avg_times_final.items(), key=lambda x: x[1])
        return f"{fastest[0].upper()} ({fastest[1]:.4f}s/epoch avg)"
    return "N/A"

def determine_best_for_dimension(sections: list, dimension: str) -> str:
    """Determine best model for specific dimension."""
    filter_str = '1D' if dimension == '1D' else '2D'
    relevant_sections = [(s, l, n) for s, l, n in sections if filter_str in l or ('Function' in l and dimension == '1D')]

    all_mse = {}
    for section_name, _, _ in relevant_sections:
        results = load_latest_results(section_name)

        for model_type in ['mlp', 'siren', 'kan', 'kan_pruning']:
            df = results.get(model_type, pd.DataFrame())
            if not df.empty:
                best_per_dataset = df.groupby('dataset_name')['test_mse'].min()
                if model_type not in all_mse:
                    all_mse[model_type] = []
                all_mse[model_type].extend(best_per_dataset.tolist())

    if all_mse:
        avg_mse = {k: np.mean(v) for k, v in all_mse.items() if v}
        best = min(avg_mse.items(), key=lambda x: x[1])
        return f"{best[0].upper()} (avg MSE: {format_scientific(best[1])})"
    return "N/A"

def evaluate_pruning(sections: list) -> str:
    """Evaluate effectiveness of KAN pruning."""
    improvements = []

    for section_name, _, _ in sections:
        results = load_latest_results(section_name)

        kan_df = results.get('kan', pd.DataFrame())
        kan_pruning_df = results.get('kan_pruning', pd.DataFrame())

        if not kan_df.empty and not kan_pruning_df.empty:
            kan_mse = kan_df.groupby('dataset_name')['test_mse'].min().mean()
            pruned_mse = kan_pruning_df.groupby('dataset_name')['test_mse'].min().mean()

            if kan_mse > 0:
                improvements.append(pruned_mse / kan_mse)

    if improvements:
        avg_ratio = np.mean(improvements)
        if avg_ratio < 1:
            return f"Beneficial ({(1-avg_ratio)*100:.1f}% improvement avg)"
        else:
            return f"Minimal impact ({(avg_ratio-1)*100:.1f}% degradation avg)"
    return "N/A"

if __name__ == '__main__':
    summary_df = create_executive_summary()
