"""
Table 3: Merge_KAN Expert Analysis (Section 2.3)

Shows expert pool statistics, selection process, and merge effectiveness.
Demonstrates if merging multiple experts improves over best individual expert.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from utils import (
    load_section2_results,
    format_scientific,
    create_latex_table,
    save_table,
    print_table,
    get_dataset_names,
    compute_improvement
)


def create_merge_kan_table():
    """Generate Merge_KAN analysis table for Section 2.3"""

    # Load results
    print("Loading Section 2.3 results...")
    results = load_section2_results('section2_3')

    if not results or all(df.empty for df in results.values()):
        print("Error: No results found for section2_3")
        print("Please run: python ../section2_3.py --n-seeds 5")
        return

    # Extract key DataFrames
    summary_df = results.get('summary', pd.DataFrame())
    experts_df = results.get('experts', pd.DataFrame())
    selected_df = results.get('selected_experts', pd.DataFrame())

    if summary_df.empty:
        print("Error: Summary DataFrame is empty")
        return

    # Get dataset names
    dataset_names = get_dataset_names()

    # Create comparison table
    table_rows = []

    for _, summary_row in summary_df.iterrows():
        dataset_idx = summary_row['dataset_idx']
        dataset_name = dataset_names[dataset_idx] if dataset_idx < len(dataset_names) else f'Dataset {dataset_idx}'

        row = {'Dataset': dataset_name}

        # Expert pool statistics
        row['Experts Trained'] = int(summary_row['n_experts_trained'])
        row['Experts Selected'] = int(summary_row['n_experts_selected'])

        # Get unique dependency patterns
        dataset_experts = experts_df[experts_df['dataset_idx'] == dataset_idx]
        if not dataset_experts.empty:
            unique_patterns = dataset_experts['dependencies'].nunique()
            row['Unique Patterns'] = int(unique_patterns)
        else:
            row['Unique Patterns'] = 0

        # Merged KAN performance
        merged_mse = summary_row['merged_dense_mse']
        row['Merged MSE'] = format_scientific(merged_mse, 2)

        # Best individual expert performance
        if not dataset_experts.empty:
            best_solo_mse = dataset_experts['dense_mse'].min()
            row['Best Solo MSE'] = format_scientific(best_solo_mse, 2)

            # Compute improvement from merging
            improvement = compute_improvement(best_solo_mse, merged_mse)
            row['Merge Gain'] = f"{improvement:.1f}%"
        else:
            row['Best Solo MSE'] = 'N/A'
            row['Merge Gain'] = 'N/A'

        # Parameters
        row['Params'] = int(summary_row['merged_num_params'])

        table_rows.append(row)

    # Add average/summary row
    avg_row = {'Dataset': 'Average'}

    # Average numeric columns
    avg_row['Experts Trained'] = int(np.mean([r['Experts Trained'] for r in table_rows]))
    avg_row['Experts Selected'] = int(np.mean([r['Experts Selected'] for r in table_rows]))
    avg_row['Unique Patterns'] = int(np.mean([r['Unique Patterns'] for r in table_rows]))

    # Average MSE values
    merged_mses = []
    solo_mses = []
    for row in table_rows:
        try:
            merged_mses.append(float(row['Merged MSE']))
        except:
            pass
        try:
            solo_mses.append(float(row['Best Solo MSE']))
        except:
            pass

    if merged_mses:
        avg_row['Merged MSE'] = format_scientific(np.mean(merged_mses), 2)
    else:
        avg_row['Merged MSE'] = 'N/A'

    if solo_mses:
        avg_row['Best Solo MSE'] = format_scientific(np.mean(solo_mses), 2)
    else:
        avg_row['Best Solo MSE'] = 'N/A'

    # Average improvement
    improvements = []
    for row in table_rows:
        imp_str = row.get('Merge Gain', 'N/A')
        if imp_str != 'N/A' and '%' in imp_str:
            try:
                improvements.append(float(imp_str.replace('%', '')))
            except:
                pass

    if improvements:
        avg_row['Merge Gain'] = f"{np.mean(improvements):.1f}%"
    else:
        avg_row['Merge Gain'] = 'N/A'

    # Average parameters
    avg_row['Params'] = int(np.mean([r['Params'] for r in table_rows]))

    table_rows.append(avg_row)

    # Create DataFrame
    df = pd.DataFrame(table_rows)

    # Print to console
    print_table(df, "Table 3: Merge_KAN Expert Analysis (Section 2.3)")

    # Save as CSV
    csv_path = Path(__file__).parent / 'table3_merge_kan.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV to {csv_path}")

    # Create LaTeX table
    caption = (
        "Merge\\_KAN expert analysis on 2D Poisson PDE problems. "
        "Shows expert pool statistics, unique dependency patterns discovered, "
        "and performance improvement from merging. "
        "Merge Gain shows percentage improvement of merged model over best individual expert."
    )
    label = "tab:section2_merge_kan"

    # Adjust column format for readability
    column_format = '|l|c|c|c|c|c|c|c|'

    latex_str = create_latex_table(df, caption, label, column_format=column_format)

    # Save LaTeX
    tex_path = Path(__file__).parent / 'table3_merge_kan.tex'
    save_table(latex_str, 'table3_merge_kan.tex', output_dir=Path(__file__).parent)

    # Print detailed findings
    print("\n" + "="*80)
    print("Key Findings:")
    print(f"  Average experts trained per dataset: {avg_row['Experts Trained']}")
    print(f"  Average unique dependency patterns: {avg_row['Unique Patterns']}")
    print(f"  Average merge improvement: {avg_row['Merge Gain']}")
    print(f"  Merge typically improves over best solo expert by learning")
    print(f"    complementary representations from different dependency patterns")
    print("="*80)

    # Additional: Print dependency pattern distribution
    if not selected_df.empty:
        print("\n" + "="*80)
        print("Dependency Pattern Distribution:")
        print("="*80)

        for dataset_idx in range(len(table_rows) - 1):  # Exclude average row
            dataset_name = dataset_names[dataset_idx] if dataset_idx < len(dataset_names) else f'Dataset {dataset_idx}'
            dataset_selected = selected_df[selected_df['dataset_idx'] == dataset_idx]

            if not dataset_selected.empty:
                print(f"\n{dataset_name}:")
                for _, expert in dataset_selected.iterrows():
                    deps = expert['dependencies']
                    mse = format_scientific(expert['dense_mse'], 2)
                    depth = expert['depth']
                    k = expert['k']
                    print(f"  Pattern {deps}: MSE={mse}, depth={depth}, k={k}")

    return df


if __name__ == '__main__':
    create_merge_kan_table()
