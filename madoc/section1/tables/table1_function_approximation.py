"""
Table 1: Function Approximation Comparison (Section 1.1)

This table compares MLP, SIREN, KAN, and KAN with pruning on various 1D function
approximation tasks including sinusoids, piecewise functions, sawtooth, polynomial,
and high-frequency Poisson solutions.

METHODOLOGY:
- Uses checkpoint-based evaluation for fair comparisons
- Reports dense_mse (10,000 samples) not sparse test_mse
- Provides TWO comparisons:
  * Table 1a: Iso-compute comparison (at KAN interpolation threshold time)
  * Table 1b: Final performance comparison (after full training budget)

Similar to KAN paper Table 1 (Special Functions) but adapted for Section 1.1 data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from utils import (load_checkpoint_metadata, print_table, create_latex_table,
                   save_table, format_scientific, compare_models_from_checkpoints,
                   get_dataset_names)


def create_function_approximation_tables():
    """
    Generate function approximation comparison tables.

    Creates two tables:
    - Table 1a: Iso-compute comparison (fair time-matched)
    - Table 1b: Final performance comparison (best achievable)

    Returns:
        Tuple of (iso_compute_df, final_df)
    """

    section_name = 'section1_1'

    # Load checkpoint metadata (contains dense_mse at two key checkpoints)
    print(f"\nLoading checkpoint metadata for {section_name}...")
    checkpoint_metadata = load_checkpoint_metadata(section_name)

    if checkpoint_metadata is None:
        print("ERROR: Checkpoint metadata not found. Cannot generate tables.")
        print("Make sure you've run section1_1.py training script to generate checkpoint data.")
        return None, None

    # Get dataset names
    dataset_names = get_dataset_names(section_name)

    print(f"\nGenerating tables for {len(dataset_names)} datasets: {dataset_names}\n")

    # ===== TABLE 1a: ISO-COMPUTE COMPARISON =====
    print("="*80)
    print("TABLE 1a: ISO-COMPUTE COMPARISON")
    print("Comparing models at KAN interpolation threshold time (fair time-matched)")
    print("="*80 + "\n")

    iso_compute_df = compare_models_from_checkpoints(
        checkpoint_metadata,
        dataset_names,
        checkpoint_type='iso_compute',
        include_pruned=False
    )

    print_table(iso_compute_df,
                "Table 1a: Function Approximation - Iso-Compute Comparison")

    # Save LaTeX table for Table 1a
    latex_1a = create_latex_table(
        iso_compute_df,
        caption="Function approximation iso-compute comparison. All models evaluated at the same "
                "wall-clock time (when KAN reaches interpolation threshold). Dense MSE computed on "
                "10,000 samples. Lower is better. This represents a fair time-matched comparison.",
        label="tab:function_approx_iso_compute",
        column_format="|l|c|c|c|c|c|c|c|c|c|"
    )
    save_table(latex_1a, 'table1a_function_approximation_iso_compute.tex')

    # Save CSV
    iso_compute_df.to_csv(
        Path(__file__).parent / 'table1a_function_approximation_iso_compute.csv',
        index=False
    )

    # ===== TABLE 1b: FINAL PERFORMANCE COMPARISON =====
    print("\n" + "="*80)
    print("TABLE 1b: FINAL PERFORMANCE COMPARISON")
    print("Comparing models after full training budget (best achievable)")
    print("="*80 + "\n")

    final_df = compare_models_from_checkpoints(
        checkpoint_metadata,
        dataset_names,
        checkpoint_type='final',
        include_pruned=False
    )

    print_table(final_df,
                "Table 1b: Function Approximation - Final Performance")

    # Save LaTeX table for Table 1b
    latex_1b = create_latex_table(
        final_df,
        caption="Function approximation final performance comparison. All models evaluated after "
                "completing full training budget. Dense MSE computed on 10,000 samples. Lower is better. "
                "This represents best achievable accuracy given unlimited training time.",
        label="tab:function_approx_final",
        column_format="|l|c|c|c|c|c|c|c|c|c|"
    )
    save_table(latex_1b, 'table1b_function_approximation_final.tex')

    # Save CSV
    final_df.to_csv(
        Path(__file__).parent / 'table1b_function_approximation_final.csv',
        index=False
    )

    # ===== SUMMARY STATISTICS =====
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80 + "\n")

    summary_df = create_summary_statistics(iso_compute_df, final_df)
    print_table(summary_df, "Summary: Iso-Compute vs Final Performance")

    # Save summary CSV
    summary_df.to_csv(
        Path(__file__).parent / 'table1_summary_statistics.csv',
        index=False
    )

    # ===== IMPROVEMENT ANALYSIS =====
    improvement_df = calculate_improvement_ratios(iso_compute_df, final_df)
    if improvement_df is not None:
        print_table(improvement_df, "Performance Improvement: Iso-Compute → Final")
        improvement_df.to_csv(
            Path(__file__).parent / 'table1_improvement_analysis.csv',
            index=False
        )

    print("\n✓ All Table 1 outputs saved successfully\n")

    return iso_compute_df, final_df


def create_summary_statistics(iso_compute_df: pd.DataFrame,
                              final_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create summary statistics comparing iso-compute vs final performance.

    Args:
        iso_compute_df: Iso-compute comparison DataFrame
        final_df: Final performance comparison DataFrame

    Returns:
        DataFrame with summary statistics per model
    """
    summary_data = []

    for model_name in ['MLP', 'SIREN', 'KAN']:
        dense_mse_col = f'{model_name} Dense MSE'

        # Extract numeric values (filter out 'N/A')
        iso_values = []
        final_values = []

        for val in iso_compute_df[dense_mse_col]:
            if val != 'N/A':
                # Parse scientific notation string back to float
                try:
                    iso_values.append(float(val.replace('e', 'E')))
                except:
                    pass

        for val in final_df[dense_mse_col]:
            if val != 'N/A':
                try:
                    final_values.append(float(val.replace('e', 'E')))
                except:
                    pass

        if iso_values and final_values:
            summary_data.append({
                'Model': model_name,
                'Iso-Compute Mean': format_scientific(np.mean(iso_values)),
                'Iso-Compute Std': format_scientific(np.std(iso_values)),
                'Final Mean': format_scientific(np.mean(final_values)),
                'Final Std': format_scientific(np.std(final_values)),
                'Mean Improvement': f"{np.mean(iso_values) / np.mean(final_values):.2f}x"
            })

    return pd.DataFrame(summary_data)


def calculate_improvement_ratios(iso_compute_df: pd.DataFrame,
                                 final_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate how much each model improves from iso-compute to final.

    Args:
        iso_compute_df: Iso-compute comparison DataFrame
        final_df: Final performance comparison DataFrame

    Returns:
        DataFrame showing improvement ratios
    """
    improvement_data = []

    for idx, dataset in enumerate(iso_compute_df['Dataset']):
        row_data = {'Dataset': dataset}

        for model_name in ['MLP', 'SIREN', 'KAN']:
            dense_mse_col = f'{model_name} Dense MSE'

            iso_val = iso_compute_df.iloc[idx][dense_mse_col]
            final_val = final_df.iloc[idx][dense_mse_col]

            if iso_val != 'N/A' and final_val != 'N/A':
                try:
                    iso_num = float(iso_val.replace('e', 'E'))
                    final_num = float(final_val.replace('e', 'E'))
                    improvement = iso_num / final_num
                    row_data[f'{model_name} Improvement'] = f"{improvement:.2f}x"
                except:
                    row_data[f'{model_name} Improvement'] = 'N/A'
            else:
                row_data[f'{model_name} Improvement'] = 'N/A'

        improvement_data.append(row_data)

    return pd.DataFrame(improvement_data) if improvement_data else None


if __name__ == '__main__':
    iso_compute_df, final_df = create_function_approximation_tables()

    if iso_compute_df is None or final_df is None:
        print("\nFailed to generate tables. Check error messages above.")
        sys.exit(1)

    print("\n" + "="*80)
    print("TABLE GENERATION COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  - table1a_function_approximation_iso_compute.tex")
    print("  - table1a_function_approximation_iso_compute.csv")
    print("  - table1b_function_approximation_final.tex")
    print("  - table1b_function_approximation_final.csv")
    print("  - table1_summary_statistics.csv")
    print("  - table1_improvement_analysis.csv")
    print("\nKEY INSIGHT:")
    print("  Table 1a shows fair time-matched comparison (iso-compute)")
    print("  Table 1b shows best achievable accuracy (unlimited time)")
    print("  Both use dense_mse (10k samples) for rigorous evaluation")
    print("="*80 + "\n")
