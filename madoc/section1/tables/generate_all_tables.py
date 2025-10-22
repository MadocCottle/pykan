#!/usr/bin/env python3
"""
Generate All Tables

This script runs all table generation scripts in sequence and creates
a comprehensive summary report.

Usage:
    python generate_all_tables.py
    python generate_all_tables.py --output-dir ./output
"""

import sys
from pathlib import Path
import argparse
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    parser = argparse.ArgumentParser(description='Generate all Section 1 tables')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Directory for output files (default: current directory)')
    parser.add_argument('--skip', nargs='+', default=[],
                        help='Table numbers to skip (e.g., --skip 1 3 5)')
    args = parser.parse_args()

    print("\n" + "="*80)
    print("GENERATING ALL SECTION 1 TABLES")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {args.output_dir}")
    print("="*80 + "\n")

    # Define all table modules
    tables = [
        ('0', 'table0_executive_summary', 'Executive Summary'),
        ('1', 'table1_function_approximation', 'Function Approximation Comparison'),
        ('2', 'table2_pde_1d_comparison', '1D Poisson PDE Comparison'),
        ('3', 'table3_pde_2d_comparison', '2D Poisson PDE Comparison'),
        ('4', 'table4_param_efficiency', 'Parameter Efficiency Analysis'),
        ('5', 'table5_convergence_summary', 'Training Efficiency Summary'),
        ('6', 'table6_grid_ablation', 'KAN Grid Size Ablation'),
        ('7', 'table7_depth_ablation', 'Depth Ablation Study'),
    ]

    results = []
    errors = []

    for table_num, module_name, description in tables:
        if table_num in args.skip:
            print(f"Skipping Table {table_num}: {description}")
            continue

        print(f"\n{'='*80}")
        print(f"TABLE {table_num}: {description}")
        print('='*80)

        try:
            # Import and run the module
            module = __import__(module_name)

            # Call the main function (assumes each module has a main creation function)
            if table_num == '0':
                module.create_executive_summary()
            elif table_num == '1':
                module.create_function_approximation_table()
            elif table_num == '2':
                module.create_pde_1d_comparison_table()
            elif table_num == '3':
                module.create_pde_2d_comparison_table()
            elif table_num == '4':
                module.create_param_efficiency_table()
            elif table_num == '5':
                module.create_convergence_summary_table()
            elif table_num == '6':
                module.create_grid_ablation_table()
            elif table_num == '7':
                module.create_depth_ablation_table()

            results.append((table_num, description, 'SUCCESS'))
            print(f"\n✓ Table {table_num} completed successfully")

        except Exception as e:
            error_msg = f"Error in Table {table_num}: {str(e)}"
            errors.append(error_msg)
            results.append((table_num, description, 'FAILED'))
            print(f"\n✗ {error_msg}")

    # Print summary
    print("\n" + "="*80)
    print("GENERATION SUMMARY")
    print("="*80)

    for table_num, description, status in results:
        status_symbol = "✓" if status == "SUCCESS" else "✗"
        print(f"{status_symbol} Table {table_num}: {description} - {status}")

    if errors:
        print("\n" + "="*80)
        print("ERRORS")
        print("="*80)
        for error in errors:
            print(f"  • {error}")

    print("\n" + "="*80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total tables attempted: {len(results)}")
    print(f"Successful: {sum(1 for _, _, s in results if s == 'SUCCESS')}")
    print(f"Failed: {sum(1 for _, _, s in results if s == 'FAILED')}")
    print("="*80 + "\n")

    # List generated files
    output_dir = Path(args.output_dir)
    tex_files = list(output_dir.glob('table*.tex'))
    csv_files = list(output_dir.glob('table*.csv'))

    if tex_files or csv_files:
        print("Generated files:")
        print("-" * 80)
        if tex_files:
            print(f"LaTeX files ({len(tex_files)}):")
            for f in sorted(tex_files):
                print(f"  • {f.name}")
        if csv_files:
            print(f"\nCSV files ({len(csv_files)}):")
            for f in sorted(csv_files):
                print(f"  • {f.name}")
        print("-" * 80 + "\n")

    # Return exit code
    return 0 if not errors else 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
