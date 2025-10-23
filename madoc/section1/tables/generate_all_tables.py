#!/usr/bin/env python3
"""
Generate All Tables - Updated for Checkpoint-Based Methodology

This script runs all table generation scripts in sequence using the new
checkpoint-based evaluation methodology.

METHODOLOGY:
- All tables now use checkpoint metadata (not DataFrame rows)
- All metrics use dense_mse (10,000 samples, not sparse test_mse)
- Primary tables split into iso-compute (fair time-matched) and final (best achievable)

Usage:
    python generate_all_tables.py
    python generate_all_tables.py --sections 1_1 1_2   # Only specific sections
    python generate_all_tables.py --skip 6 7            # Skip ablation studies
"""

import sys
from pathlib import Path
import argparse
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(
        description='Generate all Section 1 comparison tables using checkpoint methodology'
    )
    parser.add_argument('--sections', nargs='+', default=['1_1', '1_2', '1_3'],
                        choices=['1_1', '1_2', '1_3'],
                        help='Sections to generate tables for (default: all)')
    parser.add_argument('--skip', nargs='+', type=int, default=[],
                        help='Table numbers to skip (e.g., --skip 4 5 6 7)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed progress information')
    args = parser.parse_args()

    print("\n" + "="*80)
    print("SECTION 1 TABLE GENERATION - CHECKPOINT-BASED METHODOLOGY")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Sections: {', '.join(args.sections)}")
    if args.skip:
        print(f"Skipping tables: {', '.join(map(str, args.skip))}")
    print("="*80 + "\n")

    # Define tables per section
    section_tables = {
        '1_1': [
            (1, 'table1_function_approximation', 'create_function_approximation_tables',
             'Function Approximation (Section 1.1)')
        ],
        '1_2': [
            (2, 'table2_pde_1d_comparison', 'create_pde_1d_comparison_tables',
             '1D Poisson PDE (Section 1.2)')
        ],
        '1_3': [
            (3, 'table3_pde_2d_comparison', 'create_pde_2d_comparison_tables',
             '2D Poisson PDE (Section 1.3)')
        ]
    }

    results = []
    errors = []
    generated_files = []

    # Process each section
    for section in args.sections:
        tables = section_tables.get(section, [])

        for table_num, module_name, function_name, description in tables:
            if table_num in args.skip:
                print(f"⊘ Skipping Table {table_num}: {description}")
                continue

            print(f"\n{'='*80}")
            print(f"TABLE {table_num}: {description}")
            print('='*80 + "\n")

            try:
                # Import module
                module = __import__(module_name)

                # Call the generation function
                if hasattr(module, function_name):
                    func = getattr(module, function_name)
                    result = func()

                    # Check if function returned data (indicating success)
                    if result is not None:
                        results.append((table_num, description, 'SUCCESS'))
                        print(f"\n✓ Table {table_num} generated successfully")

                        # Track generated files
                        output_dir = Path(__file__).parent
                        generated_files.extend([
                            f"table{table_num}a_*_iso_compute.tex",
                            f"table{table_num}a_*_iso_compute.csv",
                            f"table{table_num}b_*_final.tex",
                            f"table{table_num}b_*_final.csv",
                        ])
                    else:
                        raise RuntimeError("Function returned None (generation failed)")
                else:
                    raise AttributeError(f"Module {module_name} has no function {function_name}")

            except FileNotFoundError as e:
                error_msg = f"Table {table_num}: Checkpoint metadata not found - {str(e)}"
                errors.append(error_msg)
                results.append((table_num, description, 'MISSING DATA'))
                print(f"\n⚠ {error_msg}")
                print(f"   Run training script first: python section{section}.py")

            except Exception as e:
                error_msg = f"Table {table_num}: {type(e).__name__}: {str(e)}"
                errors.append(error_msg)
                results.append((table_num, description, 'FAILED'))
                print(f"\n✗ {error_msg}")

                if args.verbose:
                    import traceback
                    traceback.print_exc()

    # Print summary
    print("\n" + "="*80)
    print("GENERATION SUMMARY")
    print("="*80 + "\n")

    if not results:
        print("No tables were attempted. Check your --sections and --skip arguments.")
    else:
        for table_num, description, status in results:
            if status == 'SUCCESS':
                symbol = "✓"
            elif status == 'MISSING DATA':
                symbol = "⚠"
            else:
                symbol = "✗"
            print(f"{symbol} Table {table_num}: {description} - {status}")

    # Show errors if any
    if errors:
        print("\n" + "="*80)
        print("ERRORS / WARNINGS")
        print("="*80)
        for error in errors:
            print(f"  • {error}")

    # Statistics
    success_count = sum(1 for _, _, s in results if s == 'SUCCESS')
    missing_count = sum(1 for _, _, s in results if s == 'MISSING DATA')
    failed_count = sum(1 for _, _, s in results if s == 'FAILED')

    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    print(f"Total tables attempted: {len(results)}")
    print(f"  ✓ Successful: {success_count}")
    print(f"  ⚠ Missing data: {missing_count}")
    print(f"  ✗ Failed: {failed_count}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    # List generated files
    if success_count > 0:
        output_dir = Path(__file__).parent
        tex_files = sorted(output_dir.glob('table*.tex'))
        csv_files = sorted(output_dir.glob('table*.csv'))

        print("\n" + "="*80)
        print("GENERATED FILES")
        print("="*80)

        if tex_files:
            print(f"\nLaTeX files ({len(tex_files)}):")
            for f in tex_files:
                print(f"  • {f.name}")

        if csv_files:
            print(f"\nCSV files ({len(csv_files)}):")
            for f in csv_files:
                print(f"  • {f.name}")

        print("\n" + "-"*80)
        print("NOTE: Each table now generates TWO versions:")
        print("  - tableXa_*_iso_compute.*  (fair time-matched comparison)")
        print("  - tableXb_*_final.*         (best achievable performance)")
        print("="*80)

    # Next steps
    if missing_count > 0:
        print("\n" + "="*80)
        print("NEXT STEPS")
        print("="*80)
        print("Some tables have missing checkpoint metadata. To fix:")
        print()
        for table_num, description, status in results:
            if status == 'MISSING DATA':
                if table_num == 1:
                    print(f"  python section1/section1_1.py --epochs 100")
                elif table_num == 2:
                    print(f"  python section1/section1_2.py --epochs 100")
                elif table_num == 3:
                    print(f"  python section1/section1_3.py --epochs 100")
        print("\nThen re-run this script.")
        print("="*80)

    print()

    # Return exit code
    return 0 if (failed_count == 0 and missing_count == 0) else 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
