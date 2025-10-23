#!/usr/bin/env python3
"""
Table Validation Script

This script validates that table generation is using the correct methodology:
- Loads checkpoint metadata (not just DataFrames)
- Uses dense_mse (not test_mse)
- Iso-compute timestamps match across models
- All required checkpoint data exists

Usage:
    python validate_tables.py
    python validate_tables.py --section section1_1
    python validate_tables.py --verbose
"""

import sys
from pathlib import Path
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from utils import load_checkpoint_metadata, get_dataset_names


def validate_section(section_name, verbose=False):
    """
    Validate checkpoint metadata for a section.

    Args:
        section_name: Section name (e.g., 'section1_1')
        verbose: Print detailed information

    Returns:
        Tuple of (is_valid, issues)
    """
    print(f"\n{'='*80}")
    print(f"VALIDATING: {section_name}")
    print('='*80)

    issues = []

    # Load checkpoint metadata
    print(f"\n1. Loading checkpoint metadata...")
    try:
        checkpoint_metadata = load_checkpoint_metadata(section_name)
        if checkpoint_metadata is None:
            issues.append("Checkpoint metadata not found")
            print("  ✗ Checkpoint metadata not found")
            return False, issues
        print("  ✓ Checkpoint metadata loaded")
    except Exception as e:
        issues.append(f"Failed to load checkpoint metadata: {e}")
        print(f"  ✗ {issues[-1]}")
        return False, issues

    # Get dataset names
    try:
        dataset_names = get_dataset_names(section_name)
        print(f"  ✓ Found {len(dataset_names)} datasets")
    except Exception as e:
        issues.append(f"Failed to get dataset names: {e}")
        print(f"  ✗ {issues[-1]}")
        return False, issues

    # Validate each model type
    print(f"\n2. Validating model types...")
    model_types = ['mlp', 'siren', 'kan', 'kan_pruning']
    found_models = []

    for model_type in model_types:
        if model_type in checkpoint_metadata:
            found_models.append(model_type)
            print(f"  ✓ {model_type} checkpoints found")
        else:
            print(f"  ⚠ {model_type} checkpoints not found")

    if len(found_models) < 3:
        issues.append(f"Only {len(found_models)}/4 model types have checkpoints")

    # Validate checkpoints for each dataset
    print(f"\n3. Validating checkpoints per dataset...")
    for dataset_idx, dataset_name in enumerate(dataset_names):
        if verbose:
            print(f"\n  Dataset {dataset_idx}: {dataset_name}")

        for model_type in found_models:
            if dataset_idx not in checkpoint_metadata[model_type]:
                issues.append(f"{model_type} missing checkpoint for dataset {dataset_idx} ({dataset_name})")
                print(f"    ✗ {model_type}: missing checkpoint")
                continue

            # Check for required checkpoint types
            checkpoints = checkpoint_metadata[model_type][dataset_idx]

            # MLP/SIREN use different keys than KAN
            if model_type in ['mlp', 'siren']:
                iso_key = 'at_kan_threshold_time'
            else:
                iso_key = 'at_threshold'

            # Validate iso-compute checkpoint
            if iso_key not in checkpoints:
                issues.append(f"{model_type} missing '{iso_key}' checkpoint for {dataset_name}")
                print(f"    ✗ {model_type}: missing '{iso_key}' checkpoint")
            elif 'dense_mse' not in checkpoints[iso_key]:
                issues.append(f"{model_type} '{iso_key}' checkpoint missing dense_mse for {dataset_name}")
                print(f"    ✗ {model_type}: '{iso_key}' missing dense_mse")
            elif verbose:
                dense_mse = checkpoints[iso_key]['dense_mse']
                time = checkpoints[iso_key]['time']
                print(f"    ✓ {model_type}: iso-compute dense_mse={dense_mse:.6e}, time={time:.2f}s")

            # Validate final checkpoint
            if 'final' not in checkpoints:
                issues.append(f"{model_type} missing 'final' checkpoint for {dataset_name}")
                print(f"    ✗ {model_type}: missing 'final' checkpoint")
            elif 'dense_mse' not in checkpoints['final']:
                issues.append(f"{model_type} 'final' checkpoint missing dense_mse for {dataset_name}")
                print(f"    ✗ {model_type}: 'final' missing dense_mse")
            elif verbose:
                dense_mse = checkpoints['final']['dense_mse']
                time = checkpoints['final']['time']
                print(f"    ✓ {model_type}: final dense_mse={dense_mse:.6e}, time={time:.2f}s")

    # Validate timestamp matching (iso-compute comparison)
    print(f"\n4. Validating iso-compute timestamp matching...")
    for dataset_idx, dataset_name in enumerate(dataset_names):
        times = {}

        # Collect timestamps
        for model_type in found_models:
            if dataset_idx not in checkpoint_metadata[model_type]:
                continue

            checkpoints = checkpoint_metadata[model_type][dataset_idx]
            iso_key = 'at_kan_threshold_time' if model_type in ['mlp', 'siren'] else 'at_threshold'

            if iso_key in checkpoints and 'time' in checkpoints[iso_key]:
                times[model_type] = checkpoints[iso_key]['time']

        # Check if timestamps are close (should be within 1 second)
        if 'kan' in times:
            kan_time = times['kan']

            for model_type, time in times.items():
                if model_type == 'kan':
                    continue

                time_diff = abs(time - kan_time)
                if time_diff > 1.0:
                    issues.append(
                        f"Dataset {dataset_idx}: {model_type} timestamp differs from KAN by {time_diff:.2f}s"
                    )
                    print(f"  ⚠ Dataset {dataset_idx}: {model_type} time diff = {time_diff:.2f}s")
                elif verbose:
                    print(f"  ✓ Dataset {dataset_idx}: {model_type} time matches KAN (diff={time_diff:.3f}s)")

    # Summary
    print(f"\n{'='*80}")
    print("VALIDATION SUMMARY")
    print('='*80)

    if not issues:
        print("✓ All validations passed!")
        print(f"✓ Checkpoint methodology is correctly implemented for {section_name}")
        return True, []
    else:
        print(f"✗ Found {len(issues)} issue(s):")
        for issue in issues:
            print(f"  • {issue}")
        return False, issues


def main():
    parser = argparse.ArgumentParser(
        description='Validate table generation methodology'
    )
    parser.add_argument('--section', type=str, default=None,
                        help='Section to validate (e.g., section1_1, section1_2)')
    parser.add_argument('--all', action='store_true',
                        help='Validate all sections')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed validation information')
    args = parser.parse_args()

    # Determine sections to validate
    if args.all or args.section is None:
        sections = ['section1_1', 'section1_2', 'section1_3']
    else:
        sections = [args.section]

    print("\n" + "="*80)
    print("TABLE METHODOLOGY VALIDATION")
    print("="*80)
    print(f"Validating {len(sections)} section(s)")
    print("="*80)

    all_valid = True
    all_issues = []

    for section in sections:
        is_valid, issues = validate_section(section, verbose=args.verbose)
        all_valid = all_valid and is_valid
        all_issues.extend(issues)

    # Final summary
    print(f"\n{'='*80}")
    print("OVERALL VALIDATION RESULT")
    print('='*80)

    if all_valid:
        print("✓ ALL VALIDATIONS PASSED")
        print("\nThe checkpoint-based methodology is correctly implemented.")
        print("Tables can be generated with confidence!")
        return 0
    else:
        print(f"✗ VALIDATION FAILED ({len(all_issues)} issues)")
        print("\nPlease address the issues above before generating tables.")
        print("\nCommon fixes:")
        print("  • Re-run training scripts to regenerate checkpoint metadata")
        print("  • Ensure training completed successfully without errors")
        print("  • Check that checkpoint metadata files exist in results/sec1_results/")
        return 1


if __name__ == '__main__':
    exit_code = main()
    print()
    sys.exit(exit_code)
