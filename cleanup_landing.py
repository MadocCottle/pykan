#!/usr/bin/env python3
"""
Cleanup landing directory (fetched Gadi results)

This script helps manage the landing directory by cleaning up old job_results
directories and PBS log files (.o* files).

Usage:
    python cleanup_landing.py --keep-latest 5
    python cleanup_landing.py --archive-all
    python cleanup_landing.py --clean-pbs-logs
    python cleanup_landing.py --dry-run

Examples:
    # Keep only the 5 most recent job results
    python cleanup_landing.py --keep-latest 5

    # Archive all job_results before deletion
    python cleanup_landing.py --keep-latest 3 --archive

    # Clean up PBS log files only
    python cleanup_landing.py --clean-pbs-logs

    # Preview what would be deleted
    python cleanup_landing.py --keep-latest 3 --dry-run
"""

import argparse
import sys
import shutil
from pathlib import Path
from datetime import datetime


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


def get_dir_size(directory):
    """Calculate total size of directory in bytes"""
    total_size = 0
    try:
        for item in directory.rglob('*'):
            if item.is_file():
                total_size += item.stat().st_size
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


def find_job_results(landing_dir):
    """Find all job_results directories, sorted by timestamp"""
    job_dirs = []
    for path in landing_dir.glob('job_results_*'):
        if path.is_dir():
            # Extract timestamp from directory name
            timestamp = path.name.replace('job_results_', '')
            job_dirs.append((timestamp, path))

    # Sort by timestamp (oldest first)
    job_dirs.sort(key=lambda x: x[0])
    return job_dirs


def find_pbs_logs(landing_dir):
    """Find all PBS log files (.o*)"""
    pbs_logs = []
    for path in landing_dir.glob('*.o*'):
        if path.is_file():
            pbs_logs.append(path)
    return sorted(pbs_logs, key=lambda x: x.stat().st_mtime)


def archive_directory(src_dir, archive_dir, dry_run=False):
    """Archive a directory before deletion"""
    archive_path = archive_dir / src_dir.name

    if dry_run:
        print_info(f"  Would archive to: {archive_path}")
        return True

    try:
        if archive_path.exists():
            shutil.rmtree(archive_path)
        shutil.move(str(src_dir), str(archive_path))
        return True
    except OSError as e:
        print_error(f"  Failed to archive {src_dir.name}: {e}")
        return False


def delete_directory(directory, dry_run=False):
    """Delete a directory and all its contents"""
    if dry_run:
        print_info(f"  Would delete: {directory.name}")
        return True

    try:
        shutil.rmtree(directory)
        return True
    except OSError as e:
        print_error(f"  Failed to delete {directory.name}: {e}")
        return False


def delete_file(file_path, dry_run=False):
    """Delete a file"""
    if dry_run:
        print_info(f"  Would delete: {file_path.name}")
        return True

    try:
        file_path.unlink()
        return True
    except OSError as e:
        print_error(f"  Failed to delete {file_path.name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Cleanup landing directory',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--landing-dir',
        type=Path,
        default=Path.home() / 'Desktop' / 'landing',
        help='Landing directory path (default: ~/Desktop/landing)'
    )

    parser.add_argument(
        '--keep-latest',
        type=int,
        help='Keep N most recent job_results, delete the rest'
    )

    parser.add_argument(
        '--archive',
        action='store_true',
        help='Archive job_results to archived/ subdirectory before deletion'
    )

    parser.add_argument(
        '--clean-pbs-logs',
        action='store_true',
        help='Clean up PBS log files (*.o*)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview what would be deleted without actually deleting'
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.keep_latest and not args.clean_pbs_logs:
        print_error("Error: Must specify either --keep-latest or --clean-pbs-logs")
        parser.print_help()
        return 1

    if not args.landing_dir.exists():
        print_error(f"Error: Landing directory not found: {args.landing_dir}")
        return 1

    # Start cleanup
    print_header("Landing Directory Cleanup")

    if args.dry_run:
        print_warning("DRY RUN MODE - No files will be deleted\n")

    print_info(f"Landing directory: {args.landing_dir}")

    # Handle job_results cleanup
    if args.keep_latest:
        print_info(f"Keeping {args.keep_latest} most recent job_results\n")

        job_dirs = find_job_results(args.landing_dir)

        if not job_dirs:
            print_warning("No job_results directories found")
        else:
            print_success(f"Found {len(job_dirs)} job_results director{'y' if len(job_dirs) == 1 else 'ies'}")

            # Determine which to delete (keep most recent N)
            if len(job_dirs) > args.keep_latest:
                to_delete = job_dirs[:-args.keep_latest]  # Delete oldest
                to_keep = job_dirs[-args.keep_latest:]    # Keep newest

                print_header("Job Results to Keep")
                for timestamp, path in to_keep:
                    size = get_dir_size(path)
                    print(f"  {path.name} ({format_size(size)})")

                print_header("Job Results to Delete")
                total_size = 0
                for timestamp, path in to_delete:
                    size = get_dir_size(path)
                    total_size += size
                    print(f"  {path.name} ({format_size(size)})")

                print(f"\nTotal to delete: {len(to_delete)} director{'y' if len(to_delete) == 1 else 'ies'}, {format_size(total_size)}")

                # Confirm deletion
                if not args.dry_run:
                    print()
                    response = input(f"{Colors.WARNING}Proceed with deletion? (yes/no): {Colors.ENDC}")
                    if response.lower() not in ['yes', 'y']:
                        print_info("Cleanup cancelled")
                        return 0

                # Archive if requested
                archive_dir = None
                if args.archive:
                    archive_dir = args.landing_dir / 'archived'
                    if not args.dry_run:
                        archive_dir.mkdir(exist_ok=True)
                    print()
                    print_info("Archiving job_results before deletion...")

                # Delete/archive
                print()
                deleted_count = 0
                for timestamp, path in to_delete:
                    if args.archive and archive_dir:
                        if archive_directory(path, archive_dir, args.dry_run):
                            deleted_count += 1
                            if not args.dry_run:
                                print_success(f"Archived: {path.name}")
                    else:
                        if delete_directory(path, args.dry_run):
                            deleted_count += 1
                            if not args.dry_run:
                                print_success(f"Deleted: {path.name}")

                print_header("Job Results Cleanup Complete")
                if args.dry_run:
                    print(f"Would delete {deleted_count} director{'y' if deleted_count == 1 else 'ies'}")
                else:
                    print_success(f"Deleted {deleted_count} director{'y' if deleted_count == 1 else 'ies'}")
                    print_success(f"Freed {format_size(total_size)}")

            else:
                print_info(f"\nOnly {len(job_dirs)} job_results found, keeping all (threshold: {args.keep_latest})")

    # Handle PBS logs cleanup
    if args.clean_pbs_logs:
        print_header("PBS Logs Cleanup")

        pbs_logs = find_pbs_logs(args.landing_dir)

        if not pbs_logs:
            print_warning("No PBS log files found")
        else:
            print_success(f"Found {len(pbs_logs)} PBS log file(s)")

            total_size = sum(f.stat().st_size for f in pbs_logs)
            print(f"Total size: {format_size(total_size)}\n")

            # Show files to delete
            for log_file in pbs_logs[:10]:  # Show first 10
                size = log_file.stat().st_size
                print(f"  {log_file.name} ({format_size(size)})")

            if len(pbs_logs) > 10:
                print(f"  ... and {len(pbs_logs) - 10} more files")

            # Confirm deletion
            if not args.dry_run:
                print()
                response = input(f"{Colors.WARNING}Delete all PBS log files? (yes/no): {Colors.ENDC}")
                if response.lower() not in ['yes', 'y']:
                    print_info("PBS log cleanup cancelled")
                    return 0

            # Delete
            print()
            deleted_count = 0
            for log_file in pbs_logs:
                if delete_file(log_file, args.dry_run):
                    deleted_count += 1

            print_header("PBS Logs Cleanup Complete")
            if args.dry_run:
                print(f"Would delete {deleted_count} file(s), freeing {format_size(total_size)}")
            else:
                print_success(f"Deleted {deleted_count} file(s)")
                print_success(f"Freed {format_size(total_size)}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
