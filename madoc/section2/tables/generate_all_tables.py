#!/usr/bin/env python3
"""
Generate all Section 2 tables

Usage:
    python generate_all_tables.py              # Generate all tables
    python generate_all_tables.py --tables 1 3 # Generate specific tables
    python generate_all_tables.py --help       # Show help
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import table generation functions
from table1_optimizer_comparison import create_optimizer_comparison_table
from table2_adaptive_density import create_adaptive_density_table
from table3_merge_kan import create_merge_kan_table
from table4_section2_summary import create_section2_summary


def main():
    parser = argparse.ArgumentParser(
        description='Generate Section 2 tables',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python generate_all_tables.py              # Generate all 4 tables
    python generate_all_tables.py --tables 1   # Generate only table 1
    python generate_all_tables.py --tables 1 3 # Generate tables 1 and 3

Tables:
    1. Optimizer Comparison (Section 2.1)
    2. Adaptive Density (Section 2.2)
    3. Merge_KAN Analysis (Section 2.3)
    4. Section 2 Executive Summary
        """
    )

    parser.add_argument(
        '--tables',
        type=int,
        nargs='+',
        choices=[1, 2, 3, 4],
        help='Specific table numbers to generate (default: all)'
    )

    args = parser.parse_args()

    # Determine which tables to generate
    if args.tables:
        tables_to_generate = args.tables
    else:
        tables_to_generate = [1, 2, 3, 4]

    print("="*80)
    print("Section 2 Table Generation")
    print("="*80)
    print(f"Generating tables: {tables_to_generate}\n")

    # Track success/failure
    results = {}

    # Generate requested tables
    if 1 in tables_to_generate:
        print("\n" + "="*80)
        print("Generating Table 1: Optimizer Comparison")
        print("="*80)
        try:
            create_optimizer_comparison_table()
            results[1] = 'Success'
        except Exception as e:
            print(f"Error generating Table 1: {e}")
            results[1] = f'Failed: {str(e)}'

    if 2 in tables_to_generate:
        print("\n" + "="*80)
        print("Generating Table 2: Adaptive Density")
        print("="*80)
        try:
            create_adaptive_density_table()
            results[2] = 'Success'
        except Exception as e:
            print(f"Error generating Table 2: {e}")
            results[2] = f'Failed: {str(e)}'

    if 3 in tables_to_generate:
        print("\n" + "="*80)
        print("Generating Table 3: Merge_KAN Analysis")
        print("="*80)
        try:
            create_merge_kan_table()
            results[3] = 'Success'
        except Exception as e:
            print(f"Error generating Table 3: {e}")
            results[3] = f'Failed: {str(e)}'

    if 4 in tables_to_generate:
        print("\n" + "="*80)
        print("Generating Table 4: Section 2 Summary")
        print("="*80)
        try:
            create_section2_summary()
            results[4] = 'Success'
        except Exception as e:
            print(f"Error generating Table 4: {e}")
            results[4] = f'Failed: {str(e)}'

    # Print summary
    print("\n" + "="*80)
    print("Generation Summary")
    print("="*80)

    for table_num, status in sorted(results.items()):
        status_symbol = "✓" if status == 'Success' else "✗"
        print(f"{status_symbol} Table {table_num}: {status}")

    print("\n" + "="*80)
    print("Output files:")
    print("  CSV:   table{1-4}_*.csv")
    print("  LaTeX: table{1-4}_*.tex")
    print("="*80)

    # Return exit code
    if all(status == 'Success' for status in results.values()):
        print("\nAll tables generated successfully!")
        return 0
    else:
        print("\nSome tables failed to generate. Check errors above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
