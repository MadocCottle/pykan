#!/usr/bin/env python3
"""
Generate All Section 1 Tables

Runs all 3 core comparison table scripts:
- Table 1: Function Approximation (Section 1.1) - iso-compute + final
- Table 2: 1D Poisson PDEs (Section 1.2) - final only
- Table 3: 2D Poisson PDEs (Section 1.3) - final only

Usage:
    python generate_all_tables.py
    python generate_all_tables.py --skip 2 3  # Only generate Table 1
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate all Section 1 comparison tables')
    parser.add_argument('--skip', nargs='*', type=int, default=[],
                        help='Table numbers to skip (e.g., --skip 2 3)')
    args = parser.parse_args()

    print("\n" + "="*80)
    print("SECTION 1 TABLE GENERATION")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if args.skip:
        print(f"Skipping tables: {', '.join(map(str, args.skip))}")
    print("="*80 + "\n")

    tables = [
        (1, 'table1_function_approximation.py', 'Function Approximation (Section 1.1)'),
        (2, 'table2_pde_1d_comparison.py', '1D Poisson PDEs (Section 1.2)'),
        (3, 'table3_pde_2d_comparison.py', '2D Poisson PDEs (Section 1.3)')
    ]

    results = []
    script_dir = Path(__file__).parent

    for table_num, script_name, description in tables:
        if table_num in args.skip:
            print(f"Skipping Table {table_num}: {description}\n")
            continue

        print(f"{'='*80}")
        print(f"TABLE {table_num}: {description}")
        print(f"{'='*80}\n")

        script_path = script_dir / script_name

        try:
            # Run the table script
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                cwd=str(script_dir)
            )

            # Print the output
            print(result.stdout)

            if result.returncode == 0:
                results.append((table_num, description, 'SUCCESS'))
                print(f"✓ Table {table_num} generated successfully\n")
            else:
                results.append((table_num, description, 'FAILED'))
                print(f"✗ Table {table_num} failed\n")
                if result.stderr:
                    print(f"Error output:\n{result.stderr}\n")

        except Exception as e:
            results.append((table_num, description, 'ERROR'))
            print(f"✗ Table {table_num} error: {e}\n")

    # Print summary
    print("\n" + "="*80)
    print("GENERATION SUMMARY")
    print("="*80 + "\n")

    for table_num, description, status in results:
        symbol = "✓" if status == 'SUCCESS' else "✗"
        print(f"{symbol} Table {table_num}: {description} - {status}")

    # Count successes
    success_count = sum(1 for _, _, s in results if s == 'SUCCESS')
    failed_count = len(results) - success_count

    print("\n" + "="*80)
    print(f"Total: {len(results)} attempted, {success_count} succeeded, {failed_count} failed")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    # List generated files
    if success_count > 0:
        output_files = sorted(script_dir.glob('table*.tex')) + sorted(script_dir.glob('table*.csv'))
        if output_files:
            print("\n" + "="*80)
            print(f"GENERATED FILES ({len(output_files)})")
            print("="*80)
            for f in output_files:
                print(f"  {f.name}")
            print("="*80)

    print()
    return 0 if failed_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
