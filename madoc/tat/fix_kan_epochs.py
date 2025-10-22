#!/usr/bin/env python3
"""
Fix epoch numbering in saved KAN results.

This script fixes the issue where KAN epoch numbers were stored as global
increments across grid sizes instead of resetting for each grid size.

For example:
- Before: grid_size=3 has epochs 0-4, grid_size=5 has epochs 5-9, etc.
- After:  grid_size=3 has epochs 0-4, grid_size=5 has epochs 0-4, etc.
"""

import sys
from pathlib import Path
import pandas as pd
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))
from section1.utils import load_run, save_run


def fix_kan_epochs(section, timestamp, num_grids, epochs_per_grid):
    """
    Fix epoch numbering for KAN results.

    Args:
        section: Section name (e.g., 'section1_1', 'section1_2', 'section1_3')
        timestamp: Timestamp of the run to fix
        num_grids: Number of grid sizes used in training
        epochs_per_grid: Number of epochs per grid size
    """
    print(f"Loading results from {section}_{timestamp}...")
    results, meta = load_run(section, timestamp)

    # Fix KAN epochs
    print("\nFixing KAN epoch numbering...")
    kan_df = results['kan'].copy()

    # For each dataset, fix epochs
    for dataset_idx in kan_df['dataset_idx'].unique():
        dataset_mask = kan_df['dataset_idx'] == dataset_idx

        # For each grid size
        for grid_idx, grid_size in enumerate(sorted(kan_df[dataset_mask]['grid_size'].unique())):
            grid_mask = (kan_df['dataset_idx'] == dataset_idx) & (kan_df['grid_size'] == grid_size)

            # Calculate expected global epochs for this grid
            expected_global_start = grid_idx * epochs_per_grid
            expected_global_end = (grid_idx + 1) * epochs_per_grid

            # Get rows for this grid
            grid_rows = kan_df[grid_mask]

            # Check if epochs need fixing (are they global epochs?)
            if len(grid_rows) > 0:
                min_epoch = grid_rows['epoch'].min()
                max_epoch = grid_rows['epoch'].max()

                # If epochs are already in the correct range (0 to epochs_per_grid-1), skip
                if min_epoch == 0 and max_epoch < epochs_per_grid:
                    print(f"  Dataset {dataset_idx}, grid_size={grid_size}: Already correct (epochs {min_epoch}-{max_epoch})")
                    continue

                # If epochs are in the global range, fix them
                if min_epoch >= expected_global_start and max_epoch < expected_global_end:
                    # Convert global epochs to per-grid epochs
                    kan_df.loc[grid_mask, 'epoch'] = kan_df.loc[grid_mask, 'epoch'] - expected_global_start
                    print(f"  Dataset {dataset_idx}, grid_size={grid_size}: Fixed epochs from {min_epoch}-{max_epoch} to 0-{epochs_per_grid-1}")
                else:
                    print(f"  Dataset {dataset_idx}, grid_size={grid_size}: WARNING - Unexpected epoch range {min_epoch}-{max_epoch}")

    results['kan'] = kan_df

    # Fix KAN+Pruning epochs
    print("\nFixing KAN+Pruning epoch numbering...")
    kan_pruning_df = results['kan_pruning'].copy()

    # For each dataset, fix epochs (only for unpruned rows)
    for dataset_idx in kan_pruning_df['dataset_idx'].unique():
        dataset_mask = (kan_pruning_df['dataset_idx'] == dataset_idx) & (~kan_pruning_df['is_pruned'])

        # For each grid size
        for grid_idx, grid_size in enumerate(sorted(kan_pruning_df[dataset_mask]['grid_size'].unique())):
            grid_mask = (kan_pruning_df['dataset_idx'] == dataset_idx) & \
                       (kan_pruning_df['grid_size'] == grid_size) & \
                       (~kan_pruning_df['is_pruned'])

            # Calculate expected global epochs for this grid
            expected_global_start = grid_idx * epochs_per_grid
            expected_global_end = (grid_idx + 1) * epochs_per_grid

            # Get rows for this grid
            grid_rows = kan_pruning_df[grid_mask]

            # Check if epochs need fixing
            if len(grid_rows) > 0:
                min_epoch = grid_rows['epoch'].min()
                max_epoch = grid_rows['epoch'].max()

                # If epochs are already correct, skip
                if min_epoch == 0 and max_epoch < epochs_per_grid:
                    print(f"  Dataset {dataset_idx}, grid_size={grid_size}: Already correct (epochs {min_epoch}-{max_epoch})")
                    continue

                # If epochs are in the global range, fix them
                if min_epoch >= expected_global_start and max_epoch < expected_global_end:
                    kan_pruning_df.loc[grid_mask, 'epoch'] = kan_pruning_df.loc[grid_mask, 'epoch'] - expected_global_start
                    print(f"  Dataset {dataset_idx}, grid_size={grid_size}: Fixed epochs from {min_epoch}-{max_epoch} to 0-{epochs_per_grid-1}")
                else:
                    print(f"  Dataset {dataset_idx}, grid_size={grid_size}: WARNING - Unexpected epoch range {min_epoch}-{max_epoch}")

    results['kan_pruning'] = kan_pruning_df

    # Save fixed results
    print(f"\nSaving fixed results...")
    save_run(results, section, epochs=meta.get('epochs'), device=meta.get('device'))

    print("Done! Fixed results saved with new timestamp.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Fix epoch numbering in saved KAN results'
    )
    parser.add_argument('--section', type=str, required=True,
                       help='Section to fix (e.g., section1_1, section1_2, section1_3)')
    parser.add_argument('--timestamp', type=str, required=True,
                       help='Timestamp of the run to fix')
    parser.add_argument('--num-grids', type=int, default=6,
                       help='Number of grid sizes (default: 6)')
    parser.add_argument('--epochs-per-grid', type=int, required=True,
                       help='Number of epochs per grid size')

    args = parser.parse_args()

    try:
        fix_kan_epochs(
            section=args.section,
            timestamp=args.timestamp,
            num_grids=args.num_grids,
            epochs_per_grid=args.epochs_per_grid
        )
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
