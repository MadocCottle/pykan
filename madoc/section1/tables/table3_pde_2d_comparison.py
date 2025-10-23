#!/usr/bin/env python3
"""
Table 3: 2D Poisson PDE Comparison (Section 1.3)

Compares MLP, SIREN, and KAN on 4 2D Poisson PDE tasks:
- Sinusoidal source term
- Polynomial source term
- High-frequency source term
- Special source term

NOTE: Section 1.3 does NOT train KAN first, so MLP/SIREN do not have valid
'at_kan_threshold_time' checkpoints. This table shows FINAL performance only.

Usage:
    python table3_pde_2d_comparison.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from utils import (load_checkpoint_metadata, compare_models_from_checkpoints,
                   print_table, create_latex_table, save_table, get_dataset_names)


def main():
    section_name = 'section1_3'

    print("\n" + "="*80)
    print("TABLE 3: 2D POISSON PDE COMPARISON (Section 1.3)")
    print("="*80)

    # Load checkpoint metadata
    print(f"\nLoading checkpoint metadata for {section_name}...")
    checkpoint_metadata = load_checkpoint_metadata(section_name)

    if checkpoint_metadata is None:
        print("\nERROR: No checkpoint metadata found.")
        print("Please run: python section1/section1_3.py --epochs 100")
        return 1

    # Get dataset names
    dataset_names = get_dataset_names(section_name)
    print(f"Datasets ({len(dataset_names)}): {', '.join(dataset_names)}\n")

    # ===== TABLE 3: FINAL PERFORMANCE COMPARISON =====
    print("\n" + "="*80)
    print("TABLE 3: FINAL PERFORMANCE COMPARISON")
    print("Models evaluated after full training budget")
    print("="*80)
    print("\nNOTE: Section 1.3 training does NOT use iso-compute methodology.")
    print("MLP/SIREN/KAN are all trained independently and compared at final epoch.")
    print("="*80)

    final_df = compare_models_from_checkpoints(
        checkpoint_metadata,
        dataset_names,
        checkpoint_type='final',
        include_pruned=False
    )

    print_table(final_df, "Table 3: 2D Poisson PDE - Final Performance")

    # Save outputs
    latex_3 = create_latex_table(
        final_df,
        caption="2D Poisson PDE final performance comparison. All models evaluated "
                "after completing their full training budget. Dense MSE computed on 10,000 samples. "
                "Lower is better.",
        label="tab:pde_2d_final"
    )
    save_table(latex_3, 'table3_pde_2d_final.tex')
    final_df.to_csv(Path(__file__).parent / 'table3_pde_2d_final.csv', index=False)

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("Generated 1 table (2 files):")
    print("  - table3_pde_2d_final.tex / .csv")
    print("="*80 + "\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
