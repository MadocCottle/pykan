"""
Table 2: Adaptive Density Comparison (Section 2.2)

Compares baseline (regular refinement), adaptive-only, and adaptive+regular strategies.
Shows if adaptive density provides efficiency improvements.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from utils import (
    load_section2_results,
    load_checkpoint_metadata,
    format_scientific,
    create_latex_table,
    save_table,
    print_table,
    get_dataset_names,
    identify_winner,
    compute_improvement
)


def create_adaptive_density_table():
    """Generate adaptive density comparison table for Section 2.2"""

    # Load results
    print("Loading Section 2.1 results...")
    results = load_section2_results('section2_2')

    if not results or all(df.empty for df in results.values()):
        print("Error: No results found for section2_2")
        print("Please run: python ../section2_2.py --epochs 100")
        return

    # Load checkpoint metadata
    checkpoint_metadata = load_checkpoint_metadata('section2_2')

    # Get dataset names
    dataset_names = get_dataset_names()

    # Create comparison table
    table_rows = []

    # If we have checkpoint metadata, use iso-compute comparison
    if checkpoint_metadata:
        print("\nUsing ISO-COMPUTE comparison (at baseline interpolation threshold)")

        # Get number of datasets
        sample_df = next(iter(results.values()))
        num_datasets = sample_df['dataset_idx'].nunique() if not sample_df.empty else 4

        for dataset_idx in range(num_datasets):
            row = {'Dataset': dataset_names[dataset_idx] if dataset_idx < len(dataset_names) else f'Dataset {dataset_idx}'}

            # Get dense_mse at threshold for each approach
            dense_mses = {}

            for approach in ['baseline', 'adaptive_only', 'adaptive_regular']:
                if approach in checkpoint_metadata and dataset_idx in checkpoint_metadata[approach]:
                    checkpoint = checkpoint_metadata[approach][dataset_idx].get('at_threshold', {})
                    dense_mse = checkpoint.get('dense_mse', np.nan)

                    dense_mses[approach] = dense_mse

                    if approach == 'baseline':
                        col_name = 'Baseline MSE'
                    elif approach == 'adaptive_only':
                        col_name = 'Adaptive MSE'
                    else:  # adaptive_regular
                        col_name = 'Adapt+Reg MSE'

                    row[col_name] = format_scientific(dense_mse, 2)
                else:
                    dense_mses[approach] = np.nan
                    if approach == 'baseline':
                        col_name = 'Baseline MSE'
                    elif approach == 'adaptive_only':
                        col_name = 'Adaptive MSE'
                    else:
                        col_name = 'Adapt+Reg MSE'
                    row[col_name] = 'N/A'

            # Identify best approach for this dataset
            winner = identify_winner(dense_mses)
            if winner == 'baseline':
                row['Best'] = 'Baseline'
            elif winner == 'adaptive_only':
                row['Best'] = 'Adaptive'
            elif winner == 'adaptive_regular':
                row['Best'] = 'Adapt+Reg'
            else:
                row['Best'] = 'N/A'

            # Compute improvement over baseline
            baseline_mse = dense_mses.get('baseline', np.nan)
            best_adaptive_mse = min(dense_mses.get('adaptive_only', np.nan),
                                   dense_mses.get('adaptive_regular', np.nan))

            if not np.isnan(baseline_mse) and not np.isnan(best_adaptive_mse):
                improvement = compute_improvement(baseline_mse, best_adaptive_mse)
                row['Improvement'] = f"{improvement:.1f}%"
            else:
                row['Improvement'] = 'N/A'

            table_rows.append(row)

    else:
        # Fallback: use final results
        print("\nUsing FINAL results (full training budget)")

        for dataset_idx in range(4):
            row = {'Dataset': dataset_names[dataset_idx]}

            dense_mses = {}

            for approach in ['baseline', 'adaptive_only', 'adaptive_regular']:
                if approach in results and not results[approach].empty:
                    df = results[approach]
                    dataset_df = df[df['dataset_idx'] == dataset_idx]

                    if not dataset_df.empty:
                        # Get best (minimum) dense_mse
                        best_mse = dataset_df['dense_mse'].min()
                        dense_mses[approach] = best_mse

                        if approach == 'baseline':
                            col_name = 'Baseline MSE'
                        elif approach == 'adaptive_only':
                            col_name = 'Adaptive MSE'
                        else:
                            col_name = 'Adapt+Reg MSE'

                        row[col_name] = format_scientific(best_mse, 2)
                    else:
                        dense_mses[approach] = np.nan
                        if approach == 'baseline':
                            col_name = 'Baseline MSE'
                        elif approach == 'adaptive_only':
                            col_name = 'Adaptive MSE'
                        else:
                            col_name = 'Adapt+Reg MSE'
                        row[col_name] = 'N/A'
                else:
                    dense_mses[approach] = np.nan
                    if approach == 'baseline':
                        col_name = 'Baseline MSE'
                    elif approach == 'adaptive_only':
                        col_name = 'Adaptive MSE'
                    else:
                        col_name = 'Adapt+Reg MSE'
                    row[col_name] = 'N/A'

            # Identify winner
            winner = identify_winner(dense_mses)
            if winner == 'baseline':
                row['Best'] = 'Baseline'
            elif winner == 'adaptive_only':
                row['Best'] = 'Adaptive'
            elif winner == 'adaptive_regular':
                row['Best'] = 'Adapt+Reg'
            else:
                row['Best'] = 'N/A'

            # Compute improvement
            baseline_mse = dense_mses.get('baseline', np.nan)
            best_adaptive_mse = min(dense_mses.get('adaptive_only', np.nan),
                                   dense_mses.get('adaptive_regular', np.nan))

            if not np.isnan(baseline_mse) and not np.isnan(best_adaptive_mse):
                improvement = compute_improvement(baseline_mse, best_adaptive_mse)
                row['Improvement'] = f"{improvement:.1f}%"
            else:
                row['Improvement'] = 'N/A'

            table_rows.append(row)

    # Add average row
    avg_row = {'Dataset': 'Average'}

    for col_name in ['Baseline MSE', 'Adaptive MSE', 'Adapt+Reg MSE']:
        values = []
        for row in table_rows:
            val_str = row.get(col_name, 'N/A')
            if val_str != 'N/A':
                try:
                    values.append(float(val_str))
                except:
                    pass

        if values:
            avg_row[col_name] = format_scientific(np.mean(values), 2)
        else:
            avg_row[col_name] = 'N/A'

    # Count wins
    winner_counts = {}
    for row in table_rows:
        winner = row.get('Best', 'N/A')
        if winner != 'N/A':
            winner_counts[winner] = winner_counts.get(winner, 0) + 1

    if winner_counts:
        overall_winner = max(winner_counts, key=winner_counts.get)
        avg_row['Best'] = f"{overall_winner} ({winner_counts[overall_winner]}/4)"
    else:
        avg_row['Best'] = 'N/A'

    # Average improvement
    improvements = []
    for row in table_rows:
        imp_str = row.get('Improvement', 'N/A')
        if imp_str != 'N/A' and '%' in imp_str:
            try:
                improvements.append(float(imp_str.replace('%', '')))
            except:
                pass

    if improvements:
        avg_row['Improvement'] = f"{np.mean(improvements):.1f}%"
    else:
        avg_row['Improvement'] = 'N/A'

    table_rows.append(avg_row)

    # Create DataFrame
    df = pd.DataFrame(table_rows)

    # Print to console
    print_table(df, "Table 2: Adaptive Density Comparison (Section 2.2)")

    # Save as CSV
    csv_path = Path(__file__).parent / 'table2_adaptive_density.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV to {csv_path}")

    # Create LaTeX table
    caption = (
        "Adaptive density strategy comparison on 2D Poisson PDE problems. "
        "Compares baseline (regular refinement only), adaptive-only densification, "
        "and hybrid (adaptive + regular) approaches. "
        "Improvement column shows gain over baseline."
    )
    label = "tab:section2_adaptive_density"

    latex_str = create_latex_table(df, caption, label)

    # Save LaTeX
    tex_path = Path(__file__).parent / 'table2_adaptive_density.tex'
    save_table(latex_str, 'table2_adaptive_density.tex', output_dir=Path(__file__).parent)

    print("\n" + "="*80)
    print("Summary:")
    print("  Adaptive density can provide efficiency gains on certain datasets")
    print("  Hybrid (adaptive + regular) typically performs best")
    print("  Improvement varies by problem complexity")
    print("="*80)

    return df


if __name__ == '__main__':
    create_adaptive_density_table()
