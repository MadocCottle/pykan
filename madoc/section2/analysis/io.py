"""Concise loading for Section 2 analysis"""
import json
import pickle
from pathlib import Path
from typing import Tuple, Optional, Dict


SECTIONS = ['section2_1', 'section2_2', 'section2_3', 'section2_4', 'section2_5']
SECTION_DIRS = {
    'section2_1': 'sec1_results',
    'section2_2': 'sec2_results',
    'section2_3': 'sec3_results',
    'section2_4': 'sec4_results',
    'section2_5': 'sec5_results'
}


def load_run(section: str, timestamp: Optional[str] = None, base_dir: Optional[Path] = None) -> Tuple[Dict, Dict, Optional[str]]:
    """Load experiment run (latest or specific timestamp)

    Args:
        section: Section name ('section2_1', 'section2_2', etc.)
        timestamp: Specific timestamp (YYYYMMDD_HHMMSS) or None for latest
        base_dir: Base directory (defaults to ../../section2 from this file)

    Returns:
        (results_dict, metadata_dict, models_dir_or_none)

        results_dict: Experiment results (structure varies by section)
        metadata_dict: {epochs, n_experts, device, ...}
        models_dir: Path to models directory if exists, None otherwise

    Example:
        results, meta, models_dir = load_run('section2_1')
        mse = results['ensemble_mse']
    """
    if section not in SECTION_DIRS:
        raise ValueError(f"Unknown section: {section}. Use one of {SECTIONS}")

    # Find results directory
    if base_dir is None:
        base_dir = Path(__file__).parent.parent
    else:
        base_dir = Path(base_dir)

    results_dir = base_dir / 'results' / SECTION_DIRS[section]

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    # Find pkl file
    if timestamp:
        pkl_file = results_dir / f'{section}_{timestamp}.pkl'
        if not pkl_file.exists():
            raise FileNotFoundError(f"No results for timestamp {timestamp}: {pkl_file}")
    else:
        pkl_files = sorted(results_dir.glob(f'{section}_*.pkl'))
        if not pkl_files:
            raise FileNotFoundError(f"No results found in {results_dir}")
        pkl_file = pkl_files[-1]  # Latest
        timestamp = '_'.join(pkl_file.stem.split('_')[2:])

    # Load results
    with open(pkl_file, 'rb') as f:
        results = pickle.load(f)

    # Load metadata
    json_file = pkl_file.with_suffix('.json')
    if json_file.exists():
        with open(json_file, 'r') as f:
            data = json.load(f)
            metadata = data.get('meta', {})
    else:
        metadata = {}

    # Find models directory
    models_dir = results_dir / f'{section}_{timestamp}_models'
    models_dir = str(models_dir) if models_dir.exists() else None

    return results, metadata, models_dir


def list_runs(section: str, base_dir: Optional[Path] = None) -> list:
    """List all available runs for a section

    Args:
        section: Section name
        base_dir: Base directory (optional)

    Returns:
        List of timestamp strings
    """
    if section not in SECTION_DIRS:
        raise ValueError(f"Unknown section: {section}. Use one of {SECTIONS}")

    if base_dir is None:
        base_dir = Path(__file__).parent.parent
    else:
        base_dir = Path(base_dir)

    results_dir = base_dir / 'results' / SECTION_DIRS[section]

    if not results_dir.exists():
        return []

    pkl_files = sorted(results_dir.glob(f'{section}_*.pkl'))
    timestamps = ['_'.join(f.stem.split('_')[2:]) for f in pkl_files]

    return timestamps
