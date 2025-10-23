#!/usr/bin/env python3
"""
Table 2: 1D Poisson PDE Comparison (Section 1.2)

Compares MLP, SIREN, and KAN on 3 1D Poisson PDE tasks:
- Sinusoidal source term
- Polynomial source term
- High-frequency source term

NOTE: Section 1.2 does NOT train KAN first, so MLP/SIREN do not have valid
'at_kan_threshold_time' checkpoints. This table shows FINAL performance only.

Usage:
    python table2_pde_1d_comparison.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from utils import (load_checkpoint_metadata, compare_models_from_checkpoints,
                   print_table, create_latex_table, save_table, get_dataset_names)


def main():
    section_name = 'section1_2'

    print("\n" + "="*80)
    print("TABLE 2: 1D POISSON PDE COMPARISON (Section 1.2)")
    print("="*80)

    # Load checkpoint metadata
    print(f"\nLoading checkpoint metadata for {section_name}...")
    checkpoint_metadata = load_checkpoint_metadata(section_name)

    if checkpoint_metadata is None:
        print("\nERROR: No checkpoint metadata found.")
        print("Please run: python section1/section1_2.py --epochs 100")
        return 1

    # Get dataset names
    dataset_names = get_dataset_names(section_name)
    print(f"Datasets ({len(dataset_names)}): {', '.join(dataset_names)}\n")

    # ===== TABLE 2: FINAL PERFORMANCE COMPARISON =====
    print("\n" + "="*80)
    print("TABLE 2: FINAL PERFORMANCE COMPARISON")
    print("Models evaluated after full training budget")
    print("="*80)
    print("\nNOTE: Section 1.2 training does NOT use iso-compute methodology.")
    print("MLP/SIREN/KAN are all trained independently and compared at final epoch.")
    print("="*80)

    final_df = compare_models_from_checkpoints(
        checkpoint_metadata,
        dataset_names,
        checkpoint_type='final',
        include_pruned=False
    )

    print_table(final_df, "Table 2: 1D Poisson PDE - Final Performance")

    # Save outputs
    latex_2 = create_latex_table(
        final_df,
        caption="1D Poisson PDE final performance comparison. All models evaluated "
                "after completing their full training budget. Dense MSE computed on 10,000 samples. "
                "Lower is better.",
        label="tab:pde_1d_final"
    )
    save_table(latex_2, 'table2_pde_1d_final.tex')
    final_df.to_csv(Path(__file__).parent / 'table2_pde_1d_final.csv', index=False)

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("Generated 1 table (2 files):")
    print("  - table2_pde_1d_final.tex / .csv")
    print("="*80 + "\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
