"""Concise loading for Section 1 analysis"""
import json
import pickle
from pathlib import Path
from typing import Tuple, Optional, Dict


SECTIONS = ['section1_1', 'section1_2', 'section1_3']
SECTION_DIRS = {
    'section1_1': 'sec1_results',
    'section1_2': 'sec2_results',
    'section1_3': 'sec3_results'
}
SECTION_IS_2D = {
    'section1_1': False,
    'section1_2': False,
    'section1_3': True
}


def load_run(section: str, timestamp: Optional[str] = None, base_dir: Optional[Path] = None) -> Tuple[Dict, Dict, Optional[str]]:
    """Load experiment run (latest or specific timestamp)

    Args:
        section: Section name ('section1_1', 'section1_2', or 'section1_3')
        timestamp: Specific timestamp (YYYYMMDD_HHMMSS) or None for latest
        base_dir: Base directory (defaults to ../../section1 from this file)

    Returns:
        (results_dict, metadata_dict, models_dir_or_none)

        results_dict: {model_type: {dataset_idx: {config: {subconfig: metrics}}}}
        metadata_dict: {epochs, grids, depths, activations, device, ...}
        models_dir: Path to models directory if exists, None otherwise

    Example:
        results, meta, models_dir = load_run('section1_1')
        mse = results['mlp'][0][2]['tanh']['test'][-1]  # Final test MSE
    """
    if section not in SECTION_DIRS:
        raise ValueError(f"Unknown section: {section}. Use one of {SECTIONS}")

    # Find results directory
    if base_dir is None:
        # This file is in section1/analysis/io.py, go up one level to section1/
        base_dir = Path(__file__).parent.parent
    else:
        base_dir = Path(base_dir)

    # Look in section1/results/sec{N}_results/
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
        timestamp = pkl_file.stem.split('_')[-1]

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
    model_files = list(results_dir.glob(f'{section}_{timestamp}_kan_*'))
    models_dir = str(results_dir) if model_files else None

    return results, metadata, models_dir


def is_2d(section: str) -> bool:
    """Check if section has 2D data"""
    return SECTION_IS_2D.get(section, False)


def get_model_path(results_dir: str, section: str, timestamp: str, model_type: str, dataset_idx: int) -> Optional[Path]:
    """Get path to specific model checkpoint

    Args:
        results_dir: Results directory path
        section: Section name
        timestamp: Timestamp string
        model_type: 'kan' or 'pruned'
        dataset_idx: Dataset index

    Returns:
        Path to model checkpoint directory or None
    """
    results_path = Path(results_dir)

    if model_type == 'kan':
        pattern = f'{section}_{timestamp}_kan_{dataset_idx}'
    elif model_type == 'pruned':
        pattern = f'{section}_{timestamp}_pruned_{dataset_idx}'
    else:
        return None

    model_dir = results_path / pattern
    return model_dir if model_dir.exists() else None
