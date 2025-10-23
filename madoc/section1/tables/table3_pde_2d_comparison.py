"""
Table 3: 2D Poisson PDE Comparison (Section 1.3)

This table compares MLP, SIREN, KAN, and KAN with pruning on 2D Poisson PDE
solutions with different source terms (2d sinusoidal, polynomial, high-frequency, special).

METHODOLOGY:
- Uses checkpoint-based evaluation for fair comparisons
- Reports dense_mse (10,000 samples) not sparse test_mse
- Provides TWO comparisons:
  * Table 3a: Iso-compute comparison (at KAN interpolation threshold time)
  * Table 3b: Final performance comparison (after full training budget)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from utils import (load_checkpoint_metadata, print_table, create_latex_table,
                   save_table, format_scientific, compare_models_from_checkpoints,
                   get_dataset_names)


def create_pde_2d_comparison_tables():
    """
    Generate 2D PDE comparison tables.

    Creates two tables:
    - Table 3a: Iso-compute comparison (fair time-matched)
    - Table 3b: Final performance comparison (best achievable)

    Returns:
        Tuple of (iso_compute_df, final_df)
    """

    section_name = 'section1_3'

    # Load checkpoint metadata
    print(f"\nLoading checkpoint metadata for {section_name}...")
    checkpoint_metadata = load_checkpoint_metadata(section_name)

    if checkpoint_metadata is None:
        print("ERROR: Checkpoint metadata not found. Cannot generate tables.")
        print("Make sure you've run section1_3.py training script to generate checkpoint data.")
        return None, None

    # Get dataset names
    dataset_names = get_dataset_names(section_name)

    print(f"\nGenerating tables for {len(dataset_names)} datasets: {dataset_names}\n")

    # ===== TABLE 3a: ISO-COMPUTE COMPARISON =====
    print("="*80)
    print("TABLE 3a: ISO-COMPUTE COMPARISON (2D PDEs)")
    print("Comparing models at KAN interpolation threshold time (fair time-matched)")
    print("="*80 + "\n")

    iso_compute_df = compare_models_from_checkpoints(
        checkpoint_metadata,
        dataset_names,
        checkpoint_type='iso_compute',
        include_pruned=False
    )

    print_table(iso_compute_df,
                "Table 3a: 2D PDE - Iso-Compute Comparison")

    # Save LaTeX table for Table 3a
    latex_3a = create_latex_table(
        iso_compute_df,
        caption="2D Poisson PDE iso-compute comparison. All models evaluated at the same "
                "wall-clock time (when KAN reaches interpolation threshold). Dense MSE computed on "
                "10,000 samples. Lower is better.",
        label="tab:pde_2d_iso_compute",
        column_format="|l|c|c|c|c|c|c|c|c|c|"
    )
    save_table(latex_3a, 'table3a_pde_2d_comparison_iso_compute.tex')

    iso_compute_df.to_csv(
        Path(__file__).parent / 'table3a_pde_2d_comparison_iso_compute.csv',
        index=False
    )

    # ===== TABLE 3b: FINAL PERFORMANCE COMPARISON =====
    print("\n" + "="*80)
    print("TABLE 3b: FINAL PERFORMANCE COMPARISON (2D PDEs)")
    print("Comparing models after full training budget (best achievable)")
    print("="*80 + "\n")

    final_df = compare_models_from_checkpoints(
        checkpoint_metadata,
        dataset_names,
        checkpoint_type='final',
        include_pruned=False
    )

    print_table(final_df,
                "Table 3b: 2D PDE - Final Performance")

    # Save LaTeX table for Table 3b
    latex_3b = create_latex_table(
        final_df,
        caption="2D Poisson PDE final performance comparison. All models evaluated after "
                "completing full training budget. Dense MSE computed on 10,000 samples. Lower is better.",
        label="tab:pde_2d_final",
        column_format="|l|c|c|c|c|c|c|c|c|c|"
    )
    save_table(latex_3b, 'table3b_pde_2d_comparison_final.tex')

    final_df.to_csv(
        Path(__file__).parent / 'table3b_pde_2d_comparison_final.csv',
        index=False
    )

    print("\n✓ All Table 3 outputs saved successfully\n")

    return iso_compute_df, final_df


if __name__ == '__main__':
    iso_compute_df, final_df = create_pde_2d_comparison_tables()

    if iso_compute_df is None or final_df is None:
        print("\nFailed to generate tables. Check error messages above.")
        sys.exit(1)

    print("\n" + "="*80)
    print("TABLE GENERATION COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  - table3a_pde_2d_comparison_iso_compute.tex/csv")
    print("  - table3b_pde_2d_comparison_final.tex/csv")
    print("="*80 + "\n")
