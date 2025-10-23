#!/usr/bin/env python3
"""
Cleanup old results from Section 1 directories

This script helps manage result accumulation by cleaning up old training runs,
checkpoints, and generated outputs.

Usage:
    python cleanup_results.py --keep-latest 3
    python cleanup_results.py --before 20251020
    python cleanup_results.py --epochs 100 --keep-latest 1
    python cleanup_results.py --dry-run

Examples:
    # Keep only the 3 most recent runs
    python cleanup_results.py --keep-latest 3

    # Delete runs before Oct 20, 2025
    python cleanup_results.py --before 20251020

    # Keep only the latest 100-epoch run, delete all others
    python cleanup_results.py --epochs 100 --keep-latest 1

    # Preview what would be deleted
    python cleanup_results.py --keep-latest 2 --dry-run
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import re


class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'=' * 70}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{text}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'=' * 70}{Colors.ENDC}\n")


def print_success(text):
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")


def print_warning(text):
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")


def print_error(text):
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")


def print_info(text):
    print(f"{Colors.OKCYAN}ℹ {text}{Colors.ENDC}")


def parse_filename(filename):
    """
    Parse result filename to extract metadata.

    Supports both formats:
    - New: section1_1_20251024_123456_e100_mlp.pkl
    - Old: section1_1_20251024_123456_mlp.pkl

    Returns:
        dict with keys: section, timestamp, epochs (None if not present), model_type, extension
    """
    stem = filename.stem if hasattr(filename, 'stem') else Path(filename).stem

    # Try new format with epochs
    pattern_new = r'(section\d+_\d+)_(\d{8}_\d{6})_e(\d+)_(\w+)'
    match = re.match(pattern_new, stem)
    if match:
        return {
            'section': match.group(1),
            'timestamp': match.group(2),
            'epochs': int(match.group(3)),
            'model_type': match.group(4),
            'extension': Path(filename).suffix if hasattr(filename, 'suffix') else ''
        }

    # Try old format without epochs
    pattern_old = r'(section\d+_\d+)_(\d{8}_\d{6})_(\w+)'
    match = re.match(pattern_old, stem)
    if match:
        return {
            'section': match.group(1),
            'timestamp': match.group(2),
            'epochs': None,
            'model_type': match.group(3),
            'extension': Path(filename).suffix if hasattr(filename, 'suffix') else ''
        }

    return None


def find_all_runs(results_dir, section_filter=None, epochs_filter=None):
    """
    Find all training runs in the results directory.

    Returns:
        dict mapping timestamp -> list of files for that run
    """
    runs = defaultdict(list)

    # Find all result files
    for pkl_file in results_dir.rglob('*.pkl'):
        metadata = parse_filename(pkl_file)
        if not metadata:
            continue

        # Apply filters
        if section_filter and metadata['section'] != section_filter:
            continue

        if epochs_filter is not None and metadata['epochs'] != epochs_filter:
            continue

        runs[metadata['timestamp']].append(pkl_file)

    # Also find associated checkpoint files, model files, etc
    for timestamp in list(runs.keys()):
        # Find all files with this timestamp
        for results_subdir in results_dir.iterdir():
            if not results_subdir.is_dir():
                continue

            # Find all files matching this timestamp
            patterns = [
                f'*_{timestamp}_*.pth',
                f'*_{timestamp}_*_config.yml',
                f'*_{timestamp}_*_state',
                f'*_{timestamp}_*_cache_data',
                f'*_{timestamp}_*.parquet',
            ]

            # Also support new format with epochs
            patterns.extend([
                f'*_{timestamp}_e*_*.pth',
                f'*_{timestamp}_e*_*_config.yml',
                f'*_{timestamp}_e*_*_state',
                f'*_{timestamp}_e*_*_cache_data',
                f'*_{timestamp}_e*_*.parquet',
            ])

            for pattern in patterns:
                for file in results_subdir.glob(pattern):
                    if file not in runs[timestamp]:
                        runs[timestamp].append(file)

    return runs


def get_run_size(files):
    """Calculate total size of files in bytes"""
    total_size = 0
    for file in files:
        try:
            total_size += file.stat().st_size
        except OSError:
            pass
    return total_size


def format_size(size_bytes):
    """Format bytes to human-readable string"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def delete_run(files, dry_run=False):
    """Delete all files associated with a run"""
    deleted_count = 0
    for file in files:
        if dry_run:
            print_info(f"  Would delete: {file.name}")
        else:
            try:
                file.unlink()
                deleted_count += 1
            except OSError as e:
                print_error(f"  Failed to delete {file.name}: {e}")
    return deleted_count


def main():
    parser = argparse.ArgumentParser(
        description='Cleanup old results from Section 1',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--section',
        type=str,
        help='Filter by section (e.g., section1_1, section1_2)'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        help='Filter by epoch count (e.g., 100, 1000)'
    )

    parser.add_argument(
        '--keep-latest',
        type=int,
        help='Keep N most recent runs, delete the rest'
    )

    parser.add_argument(
        '--before',
        type=str,
        help='Delete runs before date (format: YYYYMMDD)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview what would be deleted without actually deleting'
    )

    parser.add_argument(
        '--results-dir',
        type=Path,
        default=Path(__file__).parent / 'results',
        help='Results directory to clean (default: ./results)'
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.keep_latest and not args.before:
        print_error("Error: Must specify either --keep-latest or --before")
        parser.print_help()
        return 1

    if args.before:
        try:
            datetime.strptime(args.before, '%Y%m%d')
        except ValueError:
            print_error(f"Error: Invalid date format '{args.before}'. Use YYYYMMDD")
            return 1

    # Find all runs
    print_header("Section 1 Results Cleanup")

    if args.dry_run:
        print_warning("DRY RUN MODE - No files will be deleted\n")

    print_info(f"Scanning: {args.results_dir}")
    if args.section:
        print_info(f"Filtering by section: {args.section}")
    if args.epochs:
        print_info(f"Filtering by epochs: {args.epochs}")

    runs = find_all_runs(args.results_dir, args.section, args.epochs)

    if not runs:
        print_warning("No runs found")
        return 0

    print_success(f"Found {len(runs)} run(s)\n")

    # Sort runs by timestamp (newest first)
    sorted_runs = sorted(runs.items(), key=lambda x: x[0], reverse=True)

    # Determine which runs to delete
    runs_to_delete = []

    if args.keep_latest:
        # Delete all but the N most recent
        runs_to_delete = sorted_runs[args.keep_latest:]
        print_info(f"Keeping {args.keep_latest} most recent run(s), deleting {len(runs_to_delete)}\n")

    elif args.before:
        # Delete runs before specified date
        cutoff_date = args.before
        for timestamp, files in sorted_runs:
            run_date = timestamp.split('_')[0]  # Extract YYYYMMDD
            if run_date < cutoff_date:
                runs_to_delete.append((timestamp, files))
        print_info(f"Deleting {len(runs_to_delete)} run(s) before {args.before}\n")

    if not runs_to_delete:
        print_success("No runs to delete")
        return 0

    # Show what will be deleted
    print_header("Runs to Delete")

    total_size = 0
    total_files = 0

    for timestamp, files in runs_to_delete:
        run_size = get_run_size(files)
        total_size += run_size
        total_files += len(files)

        print(f"\n{Colors.BOLD}Timestamp: {timestamp}{Colors.ENDC}")
        print(f"  Files: {len(files)}")
        print(f"  Size: {format_size(run_size)}")

        # Show sample files
        if len(files) <= 5:
            for file in files:
                print(f"    - {file.name}")
        else:
            for file in files[:3]:
                print(f"    - {file.name}")
            print(f"    ... and {len(files) - 3} more files")

    # Summary
    print_header("Cleanup Summary")
    print(f"Runs to delete: {len(runs_to_delete)}")
    print(f"Total files: {total_files}")
    print(f"Total space: {format_size(total_size)}")

    # Confirm deletion
    if not args.dry_run:
        print()
        response = input(f"{Colors.WARNING}Proceed with deletion? (yes/no): {Colors.ENDC}")
        if response.lower() not in ['yes', 'y']:
            print_info("Cleanup cancelled")
            return 0

    # Delete runs
    print()
    deleted_files = 0
    for timestamp, files in runs_to_delete:
        if args.dry_run:
            print(f"{Colors.BOLD}Would delete run: {timestamp}{Colors.ENDC}")
        else:
            print(f"{Colors.BOLD}Deleting run: {timestamp}{Colors.ENDC}")

        deleted_files += delete_run(files, args.dry_run)

    # Final summary
    print_header("Cleanup Complete")
    if args.dry_run:
        print(f"Would delete {deleted_files} files freeing {format_size(total_size)}")
        print_info("\nThis was a dry run. Use without --dry-run to actually delete files.")
    else:
        print_success(f"Deleted {deleted_files} files")
        print_success(f"Freed {format_size(total_size)}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
