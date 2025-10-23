"""
Centralized result discovery and selection utilities.

This module provides unified result finding logic for both visualization
and table generation scripts, eliminating code duplication and providing
consistent behavior across all scripts.

Supports:
- Finding runs by timestamp (latest, specific)
- Finding runs by epoch count (max, min, exact)
- Filtering by section
- Backward compatible with old naming format (without epochs)

Usage:
    from utils.result_finder import select_run

    # Get most recent run
    timestamp = select_run('section1_1', results_base_dir)

    # Get run with most epochs
    timestamp = select_run('section1_1', results_base_dir, strategy='max_epochs')

    # Get specific epoch count
    timestamp = select_run('section1_1', results_base_dir,
                          strategy='exact_epochs', epochs=100)
"""

from pathlib import Path
from typing import Dict, List, Optional
import re


def find_all_runs(section: str, results_base_dir: Path) -> List[Dict]:
    """
    Find all runs for a section with metadata.

    Args:
        section: Section name (e.g., 'section1_1', 'section1_2')
        results_base_dir: Base results directory containing sec*_results/

    Returns:
        List of dicts with keys:
            - timestamp: Run timestamp (YYYYMMDD_HHMMSS)
            - epochs: Epoch count (int or None for old format)
            - has_epoch_info: Boolean indicating if epoch was in filename

    Example:
        [
            {'timestamp': '20251024_123456', 'epochs': 100, 'has_epoch_info': True},
            {'timestamp': '20251024_000203', 'epochs': None, 'has_epoch_info': False},
        ]
    """
    sec_num = section.split('_')[-1]
    results_dir = results_base_dir / f'sec{sec_num}_results'

    if not results_dir.exists():
        return []

    runs = {}

    # Find all pkl files for this section (try both new and old patterns)
    for pattern in [f'{section}_*_e*_mlp.pkl', f'{section}_*_mlp.pkl']:
        for f in results_dir.glob(pattern):
            metadata = _parse_filename(f.stem, section)
            if metadata:
                ts = metadata['timestamp']
                # Only add if not already found (prefer new format with epochs)
                if ts not in runs or metadata['has_epoch_info']:
                    runs[ts] = metadata

    return list(runs.values())


def _parse_filename(stem: str, section: str) -> Optional[Dict]:
    """
    Extract metadata from result filename.

    Args:
        stem: Filename without extension
        section: Section name to match

    Returns:
        Dict with timestamp, epochs, has_epoch_info, or None if no match
    """
    # New format: section1_1_20251024_123456_e100_mlp
    pattern_new = rf'{re.escape(section)}_(\d{{8}}_\d{{6}})_e(\d+)_'
    match = re.match(pattern_new, stem)
    if match:
        return {
            'timestamp': match.group(1),
            'epochs': int(match.group(2)),
            'has_epoch_info': True,
        }

    # Old format: section1_1_20251024_123456_mlp
    pattern_old = rf'{re.escape(section)}_(\d{{8}}_\d{{6}})_'
    match = re.match(pattern_old, stem)
    if match:
        return {
            'timestamp': match.group(1),
            'epochs': None,
            'has_epoch_info': False,
        }

    return None


def select_run(section: str,
               results_base_dir: Path,
               strategy: str = 'latest',
               epochs: Optional[int] = None,
               verbose: bool = False) -> str:
    """
    Select a run based on strategy.

    Args:
        section: Section name (e.g., 'section1_1')
        results_base_dir: Base results directory (e.g., Path('madoc/section1/results'))
        strategy: Selection strategy:
            - 'latest': Most recent timestamp (default)
            - 'max_epochs': Run with most epochs
            - 'min_epochs': Run with least epochs
            - 'exact_epochs': Run with exact epoch count (requires epochs param)
        epochs: Epoch count for 'exact_epochs' strategy (optional)
        verbose: Print selection details (default: False)

    Returns:
        Timestamp string of selected run (YYYYMMDD_HHMMSS)

    Raises:
        FileNotFoundError: If no runs found matching criteria
        ValueError: If invalid strategy or missing required parameters

    Examples:
        # Get most recent run
        >>> timestamp = select_run('section1_1', Path('results'))

        # Get run with highest epoch count
        >>> timestamp = select_run('section1_1', Path('results'), strategy='max_epochs')

        # Get 100-epoch run (latest if multiple)
        >>> timestamp = select_run('section1_1', Path('results'),
        ...                       strategy='exact_epochs', epochs=100)
    """
    runs = find_all_runs(section, results_base_dir)

    if not runs:
        raise FileNotFoundError(
            f"No runs found for {section} in {results_base_dir}"
        )

    if verbose:
        print(f"Found {len(runs)} run(s) for {section}")
        for run in sorted(runs, key=lambda r: r['timestamp']):
            epoch_str = f"{run['epochs']} epochs" if run['has_epoch_info'] else "unknown epochs"
            print(f"  {run['timestamp']}: {epoch_str}")

    # Select run based on strategy
    if strategy == 'latest':
        # Sort by timestamp (chronological), return most recent
        selected = sorted(runs, key=lambda r: r['timestamp'])[-1]
        if verbose:
            epoch_str = f" ({selected['epochs']} epochs)" if selected['has_epoch_info'] else ""
            print(f"Selected: {selected['timestamp']}{epoch_str} (latest)")

    elif strategy == 'max_epochs':
        # Filter out runs without epoch info (old format)
        runs_with_epochs = [r for r in runs if r['has_epoch_info']]
        if not runs_with_epochs:
            raise FileNotFoundError(
                f"No runs with epoch information found for {section}. "
                f"All {len(runs)} run(s) use old naming format without epochs."
            )
        # Sort by epoch count descending, then by timestamp
        selected = sorted(runs_with_epochs,
                         key=lambda r: (r['epochs'], r['timestamp']))[-1]
        if verbose:
            print(f"Selected: {selected['timestamp']} ({selected['epochs']} epochs - maximum)")

    elif strategy == 'min_epochs':
        # Filter out runs without epoch info
        runs_with_epochs = [r for r in runs if r['has_epoch_info']]
        if not runs_with_epochs:
            raise FileNotFoundError(
                f"No runs with epoch information found for {section}. "
                f"All {len(runs)} run(s) use old naming format without epochs."
            )
        # Sort by epoch count ascending, then by timestamp
        selected = sorted(runs_with_epochs,
                         key=lambda r: (r['epochs'], r['timestamp']))[0]
        if verbose:
            print(f"Selected: {selected['timestamp']} ({selected['epochs']} epochs - minimum)")

    elif strategy == 'exact_epochs':
        if epochs is None:
            raise ValueError(
                "epochs parameter is required for 'exact_epochs' strategy. "
                "Usage: select_run(..., strategy='exact_epochs', epochs=100)"
            )

        # Filter to exact epoch match
        matching = [r for r in runs if r.get('epochs') == epochs]
        if not matching:
            available_epochs = [r['epochs'] for r in runs if r['has_epoch_info']]
            raise FileNotFoundError(
                f"No runs found with {epochs} epochs for {section}. "
                f"Available epoch counts: {sorted(set(available_epochs))}"
            )

        # If multiple matches, take latest timestamp
        selected = sorted(matching, key=lambda r: r['timestamp'])[-1]
        if verbose:
            if len(matching) > 1:
                print(f"Found {len(matching)} runs with {epochs} epochs, using latest")
            print(f"Selected: {selected['timestamp']} ({epochs} epochs - exact match)")

    else:
        raise ValueError(
            f"Unknown strategy: '{strategy}'. "
            f"Valid options: 'latest', 'max_epochs', 'min_epochs', 'exact_epochs'"
        )

    return selected['timestamp']


def list_available_runs(section: str, results_base_dir: Path) -> None:
    """
    Print a formatted list of all available runs for a section.

    Useful for debugging or manual run selection.

    Args:
        section: Section name
        results_base_dir: Base results directory
    """
    runs = find_all_runs(section, results_base_dir)

    if not runs:
        print(f"No runs found for {section} in {results_base_dir}")
        return

    print(f"\nAvailable runs for {section}:")
    print(f"{'Timestamp':<20} {'Epochs':<10} {'Format'}")
    print("-" * 45)

    for run in sorted(runs, key=lambda r: r['timestamp'], reverse=True):
        epoch_str = str(run['epochs']) if run['has_epoch_info'] else "unknown"
        format_str = "new" if run['has_epoch_info'] else "old"
        print(f"{run['timestamp']:<20} {epoch_str:<10} {format_str}")

    print(f"\nTotal: {len(runs)} run(s)")
