"""
Centralized I/O Module for Analysis Package

This module provides utilities for loading experimental results, metadata, and models
with intelligent path resolution and default directory handling.

Features:
- Automatic discovery of latest results
- Support for both explicit paths and section IDs
- Default directory handling (sec1_results, sec2_results, sec3_results)
- Format-agnostic loading (.pkl and .json)
- Metadata and model directory discovery
- Comprehensive error handling

Usage Examples:
    # Load latest results for a section
    results, metadata = load_section_results('section1_1')

    # Load specific results file
    results, metadata = load_results('/path/to/results.pkl')

    # Find models directory
    models_dir = find_models_dir(results_path)

    # Find latest results with timestamp
    info = find_latest_results('section1_1')
"""

import json
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
from datetime import datetime


# Section ID to results directory mapping
SECTION_DIRS = {
    'section1_1': 'sec1_results',
    'section1_2': 'sec2_results',
    'section1_3': 'sec3_results',
}


class ResultsNotFoundError(Exception):
    """Raised when results files cannot be found"""
    pass


class InvalidSectionError(Exception):
    """Raised when an invalid section ID is provided"""
    pass


def get_analysis_base_dir() -> Path:
    """
    Get the base directory for analysis (section1 directory)

    Returns:
        Path to section1 directory
    """
    # This file is in section1/analysis/io.py, so go up two levels
    return Path(__file__).parent.parent


def get_section_results_dir(section_id: str, base_dir: Optional[Path] = None) -> Path:
    """
    Get the results directory for a given section ID

    Args:
        section_id: Section identifier (e.g., 'section1_1', 'section1_2', 'section1_3')
        base_dir: Optional base directory (defaults to section1/)

    Returns:
        Path to results directory

    Raises:
        InvalidSectionError: If section_id is not recognized
    """
    if section_id not in SECTION_DIRS:
        raise InvalidSectionError(
            f"Invalid section ID: {section_id}. "
            f"Valid options: {list(SECTION_DIRS.keys())}"
        )

    if base_dir is None:
        base_dir = get_analysis_base_dir()

    return base_dir / SECTION_DIRS[section_id]


def find_latest_results(
    section_id: str,
    base_dir: Optional[Path] = None,
    timestamp: Optional[str] = None
) -> Dict[str, Optional[Path]]:
    """
    Find latest results file and associated files for a section

    Args:
        section_id: Section identifier (e.g., 'section1_1')
        base_dir: Optional base directory (defaults to section1/)
        timestamp: Optional specific timestamp to look for (instead of latest)

    Returns:
        Dictionary with keys:
            - 'results_file': Path to results .pkl file
            - 'metadata_file': Path to metadata .json file (if exists)
            - 'models_dir': Path to KAN models directory (if exists)
            - 'pruned_models_dir': Path to pruned KAN models directory (if exists)
            - 'timestamp': Timestamp string

    Raises:
        ResultsNotFoundError: If no results found for the section
    """
    results_dir = get_section_results_dir(section_id, base_dir)

    if not results_dir.exists():
        raise ResultsNotFoundError(
            f"Results directory does not exist: {results_dir}\n"
            f"Have you run the {section_id} experiments yet?"
        )

    # Find results files
    if timestamp:
        # Look for specific timestamp
        results_file = results_dir / f'{section_id}_results_{timestamp}.pkl'
        if not results_file.exists():
            raise ResultsNotFoundError(
                f"No results found for timestamp {timestamp}\n"
                f"Looking for: {results_file}"
            )
        results_files = [results_file]
    else:
        # Find all results files and get latest
        results_files = list(results_dir.glob(f'{section_id}_results_*.pkl'))

        if not results_files:
            raise ResultsNotFoundError(
                f"No results files found in {results_dir}\n"
                f"Expected files matching: {section_id}_results_YYYYMMDD_HHMMSS.pkl"
            )

        # Sort by timestamp (embedded in filename) and get latest
        results_files = sorted(results_files, key=lambda p: p.stem.split('_')[-1])

    results_file = results_files[-1]
    # Extract full timestamp: section1_X_results_YYYYMMDD_HHMMSS.pkl
    # Split by '_' and get last two parts to form: YYYYMMDD_HHMMSS
    parts = results_file.stem.split('_')
    found_timestamp = '_'.join(parts[-2:])

    # Find associated files
    metadata_file = results_dir / f'{section_id}_metadata_{found_timestamp}.json'
    models_dir = results_dir / f'kan_models_{found_timestamp}'
    pruned_models_dir = results_dir / f'kan_pruned_models_{found_timestamp}'

    return {
        'results_file': results_file,
        'metadata_file': metadata_file if metadata_file.exists() else None,
        'models_dir': models_dir if models_dir.exists() else None,
        'pruned_models_dir': pruned_models_dir if pruned_models_dir.exists() else None,
        'timestamp': found_timestamp
    }


def load_results_file(file_path: Union[str, Path]) -> Dict:
    """
    Load results from a .pkl or .json file

    Args:
        file_path: Path to results file (.pkl or .json)

    Returns:
        Dictionary containing results data

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is not supported
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Results file not found: {file_path}")

    if file_path.suffix == '.pkl':
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    elif file_path.suffix == '.json':
        with open(file_path, 'r') as f:
            return json.load(f)
    else:
        raise ValueError(
            f"Unsupported file format: {file_path.suffix}\n"
            f"Supported formats: .pkl, .json"
        )


def load_metadata_file(file_path: Union[str, Path]) -> Optional[Dict]:
    """
    Load metadata from a .json file

    Args:
        file_path: Path to metadata .json file

    Returns:
        Dictionary containing metadata, or None if file doesn't exist
    """
    file_path = Path(file_path)

    if not file_path.exists():
        return None

    with open(file_path, 'r') as f:
        return json.load(f)


def infer_metadata_path(results_path: Union[str, Path]) -> Path:
    """
    Infer metadata file path from results file path

    Args:
        results_path: Path to results file

    Returns:
        Path to corresponding metadata file
    """
    results_path = Path(results_path)

    # Extract timestamp and section from results filename
    # Format: section1_X_results_YYYYMMDD_HHMMSS.pkl
    parts = results_path.stem.split('_')
    timestamp = '_'.join(parts[-2:])  # Get YYYYMMDD_HHMMSS
    section = '_'.join(parts[:-3])  # Get section1_X (everything before 'results')

    return results_path.parent / f"{section}_metadata_{timestamp}.json"


def find_models_dir(
    results_path: Union[str, Path],
    pruned: bool = False
) -> Optional[Path]:
    """
    Find KAN models directory corresponding to results file

    Args:
        results_path: Path to results file
        pruned: If True, look for pruned models directory

    Returns:
        Path to models directory, or None if not found
    """
    results_path = Path(results_path)

    # Extract timestamp from results filename (YYYYMMDD_HHMMSS)
    parts = results_path.stem.split('_')
    timestamp = '_'.join(parts[-2:])

    if pruned:
        models_dir = results_path.parent / f'kan_pruned_models_{timestamp}'
    else:
        models_dir = results_path.parent / f'kan_models_{timestamp}'

    return models_dir if models_dir.exists() else None


def load_results(
    path_or_section: Union[str, Path],
    timestamp: Optional[str] = None,
    base_dir: Optional[Path] = None
) -> Tuple[Dict, Optional[Dict]]:
    """
    Load results and metadata from path or section ID

    This is the main convenience function that handles both:
    1. Explicit file paths: '/path/to/results.pkl'
    2. Section IDs: 'section1_1' (finds latest or specified timestamp)

    Args:
        path_or_section: Either a file path or section ID
        timestamp: Optional timestamp (only used if path_or_section is a section ID)
        base_dir: Optional base directory (only used if path_or_section is a section ID)

    Returns:
        Tuple of (results_dict, metadata_dict)
        metadata_dict will be None if metadata file not found

    Raises:
        ResultsNotFoundError: If results cannot be found
    """
    path_or_section = str(path_or_section)

    # Check if it's a section ID or a file path
    if path_or_section in SECTION_DIRS:
        # It's a section ID - find latest results
        info = find_latest_results(path_or_section, base_dir, timestamp)
        results_file = info['results_file']
        metadata_file = info['metadata_file']
    else:
        # It's a file path
        results_file = Path(path_or_section)
        if not results_file.exists():
            raise ResultsNotFoundError(f"Results file not found: {results_file}")

        # Try to find metadata
        metadata_file = infer_metadata_path(results_file)

    # Load results
    results = load_results_file(results_file)

    # Load metadata if available
    metadata = load_metadata_file(metadata_file) if metadata_file else None

    return results, metadata


def load_section_results(
    section_id: str,
    timestamp: Optional[str] = None,
    base_dir: Optional[Path] = None,
    load_models_info: bool = True
) -> Dict:
    """
    Load all data for a section (results, metadata, model paths)

    Args:
        section_id: Section identifier (e.g., 'section1_1')
        timestamp: Optional specific timestamp (defaults to latest)
        base_dir: Optional base directory
        load_models_info: If True, include model directory paths in output

    Returns:
        Dictionary containing:
            - 'results': Results data dictionary
            - 'metadata': Metadata dictionary (or None)
            - 'results_file': Path to results file
            - 'metadata_file': Path to metadata file (or None)
            - 'models_dir': Path to models directory (or None) [if load_models_info]
            - 'pruned_models_dir': Path to pruned models directory (or None) [if load_models_info]
            - 'timestamp': Timestamp string

    Raises:
        ResultsNotFoundError: If results cannot be found
    """
    # Find all associated files
    info = find_latest_results(section_id, base_dir, timestamp)

    # Load results and metadata
    results = load_results_file(info['results_file'])
    metadata = load_metadata_file(info['metadata_file']) if info['metadata_file'] else None

    output = {
        'results': results,
        'metadata': metadata,
        'results_file': info['results_file'],
        'metadata_file': info['metadata_file'],
        'timestamp': info['timestamp']
    }

    if load_models_info:
        output['models_dir'] = info['models_dir']
        output['pruned_models_dir'] = info['pruned_models_dir']

    return output


def get_all_available_results(base_dir: Optional[Path] = None) -> Dict[str, Dict]:
    """
    Get information about all available results across all sections

    Args:
        base_dir: Optional base directory

    Returns:
        Dictionary mapping section_id to result info (or None if no results)
    """
    all_results = {}

    for section_id in SECTION_DIRS.keys():
        try:
            info = find_latest_results(section_id, base_dir)
            all_results[section_id] = info
        except (ResultsNotFoundError, InvalidSectionError):
            all_results[section_id] = None

    return all_results


def print_available_results(base_dir: Optional[Path] = None):
    """
    Print summary of available results for all sections

    Args:
        base_dir: Optional base directory
    """
    print("Available Results Summary")
    print("=" * 70)

    all_results = get_all_available_results(base_dir)

    for section_id, info in all_results.items():
        print(f"\n{section_id}:")
        if info is None:
            print("  ✗ No results found")
        else:
            print(f"  ✓ Results found")
            print(f"    Timestamp: {info['timestamp']}")
            print(f"    Results: {info['results_file'].name}")
            if info['metadata_file']:
                print(f"    Metadata: {info['metadata_file'].name}")
            if info['models_dir']:
                print(f"    Models: {info['models_dir'].name}")
            if info['pruned_models_dir']:
                print(f"    Pruned Models: {info['pruned_models_dir'].name}")

    print("\n" + "=" * 70)


# Convenience exports
__all__ = [
    'load_results',
    'load_section_results',
    'load_results_file',
    'load_metadata_file',
    'find_latest_results',
    'find_models_dir',
    'get_section_results_dir',
    'get_all_available_results',
    'print_available_results',
    'ResultsNotFoundError',
    'InvalidSectionError',
]
