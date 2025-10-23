#!/usr/bin/env python3
"""
Table 1: Function Approximation Comparison (Section 1.1)

Compares MLP, SIREN, and KAN on 9 function approximation tasks:
- Sinusoids (frequencies 1-5)
- Piecewise, Sawtooth, Polynomial
- High-frequency 1D Poisson solution

Generates TWO tables:
- Table 1a: Iso-compute comparison (fair time-matched at KAN threshold)
- Table 1b: Final performance comparison (best achievable after full training)

Usage:
    python table1_function_approximation.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from utils import (load_checkpoint_metadata, compare_models_from_checkpoints,
                   print_table, create_latex_table, save_table, get_dataset_names)


def main():
    section_name = 'section1_1'

    print("\n" + "="*80)
    print("TABLE 1: FUNCTION APPROXIMATION COMPARISON (Section 1.1)")
    print("="*80)

    # Load checkpoint metadata
    print(f"\nLoading checkpoint metadata for {section_name}...")
    checkpoint_metadata = load_checkpoint_metadata(section_name)

    if checkpoint_metadata is None:
        print("\nERROR: No checkpoint metadata found.")
        print("Please run: python section1/section1_1.py --epochs 100")
        return 1

    # Get dataset names
    dataset_names = get_dataset_names(section_name)
    print(f"Datasets ({len(dataset_names)}): {', '.join(dataset_names)}\n")

    # ===== TABLE 1a: ISO-COMPUTE COMPARISON =====
    print("\n" + "="*80)
    print("TABLE 1a: ISO-COMPUTE COMPARISON")
    print("Models evaluated at the same wall-clock time (KAN interpolation threshold)")
    print("="*80)

    iso_df = compare_models_from_checkpoints(
        checkpoint_metadata,
        dataset_names,
        checkpoint_type='iso_compute',
        include_pruned=False
    )

    print_table(iso_df, "Table 1a: Function Approximation - Iso-Compute")

    # Save outputs
    latex_1a = create_latex_table(
        iso_df,
        caption="Function approximation iso-compute comparison. All models evaluated at the same "
                "wall-clock time (when KAN reaches interpolation threshold). Dense MSE computed on "
                "10,000 samples. Lower is better.",
        label="tab:function_approx_iso"
    )
    save_table(latex_1a, 'table1a_function_approx_iso.tex')
    iso_df.to_csv(Path(__file__).parent / 'table1a_function_approx_iso.csv', index=False)

    # ===== TABLE 1b: FINAL PERFORMANCE COMPARISON =====
    print("\n" + "="*80)
    print("TABLE 1b: FINAL PERFORMANCE COMPARISON")
    print("Models evaluated after full training budget (best achievable)")
    print("="*80)

    final_df = compare_models_from_checkpoints(
        checkpoint_metadata,
        dataset_names,
        checkpoint_type='final',
        include_pruned=False
    )

    print_table(final_df, "Table 1b: Function Approximation - Final Performance")

    # Save outputs
    latex_1b = create_latex_table(
        final_df,
        caption="Function approximation final performance comparison. All models evaluated "
                "after completing their full training budget. Dense MSE computed on 10,000 samples. "
                "Lower is better.",
        label="tab:function_approx_final"
    )
    save_table(latex_1b, 'table1b_function_approx_final.tex')
    final_df.to_csv(Path(__file__).parent / 'table1b_function_approx_final.csv', index=False)

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("Generated 2 tables (4 files):")
    print("  - table1a_function_approx_iso.tex / .csv")
    print("  - table1b_function_approx_final.tex / .csv")
    print("="*80 + "\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
