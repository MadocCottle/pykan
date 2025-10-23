"""
Table 2: 1D Poisson PDE Comparison (Section 1.2)

This table compares MLP, SIREN, KAN, and KAN with pruning on 1D Poisson PDE
solutions with different source terms (sinusoidal, polynomial, high-frequency).

METHODOLOGY:
- Uses checkpoint-based evaluation for fair comparisons
- Reports dense_mse (10,000 samples) not sparse test_mse
- Provides TWO comparisons:
  * Table 2a: Iso-compute comparison (at KAN interpolation threshold time)
  * Table 2b: Final performance comparison (after full training budget)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from utils import (load_checkpoint_metadata, print_table, create_latex_table,
                   save_table, format_scientific, compare_models_from_checkpoints,
                   get_dataset_names)


def create_pde_1d_comparison_tables():
    """
    Generate 1D PDE comparison tables.

    Creates two tables:
    - Table 2a: Iso-compute comparison (fair time-matched)
    - Table 2b: Final performance comparison (best achievable)

    Returns:
        Tuple of (iso_compute_df, final_df)
    """

    section_name = 'section1_2'

    # Load checkpoint metadata
    print(f"\nLoading checkpoint metadata for {section_name}...")
    checkpoint_metadata = load_checkpoint_metadata(section_name)

    if checkpoint_metadata is None:
        print("ERROR: Checkpoint metadata not found. Cannot generate tables.")
        print("Make sure you've run section1_2.py training script to generate checkpoint data.")
        return None, None

    # Get dataset names
    dataset_names = get_dataset_names(section_name)

    print(f"\nGenerating tables for {len(dataset_names)} datasets: {dataset_names}\n")

    # ===== TABLE 2a: ISO-COMPUTE COMPARISON =====
    print("="*80)
    print("TABLE 2a: ISO-COMPUTE COMPARISON (1D PDEs)")
    print("Comparing models at KAN interpolation threshold time (fair time-matched)")
    print("="*80 + "\n")

    iso_compute_df = compare_models_from_checkpoints(
        checkpoint_metadata,
        dataset_names,
        checkpoint_type='iso_compute',
        include_pruned=False
    )

    print_table(iso_compute_df,
                "Table 2a: 1D PDE - Iso-Compute Comparison")

    # Save LaTeX table for Table 2a
    latex_2a = create_latex_table(
        iso_compute_df,
        caption="1D Poisson PDE iso-compute comparison. All models evaluated at the same "
                "wall-clock time (when KAN reaches interpolation threshold). Dense MSE computed on "
                "10,000 samples. Lower is better.",
        label="tab:pde_1d_iso_compute",
        column_format="|l|c|c|c|c|c|c|c|c|c|"
    )
    save_table(latex_2a, 'table2a_pde_1d_comparison_iso_compute.tex')

    iso_compute_df.to_csv(
        Path(__file__).parent / 'table2a_pde_1d_comparison_iso_compute.csv',
        index=False
    )

    # ===== TABLE 2b: FINAL PERFORMANCE COMPARISON =====
    print("\n" + "="*80)
    print("TABLE 2b: FINAL PERFORMANCE COMPARISON (1D PDEs)")
    print("Comparing models after full training budget (best achievable)")
    print("="*80 + "\n")

    final_df = compare_models_from_checkpoints(
        checkpoint_metadata,
        dataset_names,
        checkpoint_type='final',
        include_pruned=False
    )

    print_table(final_df,
                "Table 2b: 1D PDE - Final Performance")

    # Save LaTeX table for Table 2b
    latex_2b = create_latex_table(
        final_df,
        caption="1D Poisson PDE final performance comparison. All models evaluated after "
                "completing full training budget. Dense MSE computed on 10,000 samples. Lower is better.",
        label="tab:pde_1d_final",
        column_format="|l|c|c|c|c|c|c|c|c|c|"
    )
    save_table(latex_2b, 'table2b_pde_1d_comparison_final.tex')

    final_df.to_csv(
        Path(__file__).parent / 'table2b_pde_1d_comparison_final.csv',
        index=False
    )

    print("\n✓ All Table 2 outputs saved successfully\n")

    return iso_compute_df, final_df


if __name__ == '__main__':
    iso_compute_df, final_df = create_pde_1d_comparison_tables()

    if iso_compute_df is None or final_df is None:
        print("\nFailed to generate tables. Check error messages above.")
        sys.exit(1)

    print("\n" + "="*80)
    print("TABLE GENERATION COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  - table2a_pde_1d_comparison_iso_compute.tex/csv")
    print("  - table2b_pde_1d_comparison_final.tex/csv")
    print("="*80 + "\n")
