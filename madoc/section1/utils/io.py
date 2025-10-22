"""Concise I/O for Section 1 experiment results"""
import json
import pickle
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import math


def clean(obj):
    """Replace NaN/Inf with None for JSON and convert numpy types to Python native types"""
    if isinstance(obj, dict):
        # Convert both keys and values, handling numpy types in keys
        return {
            (int(k) if isinstance(k, np.integer) else
             float(k) if isinstance(k, np.floating) else k): clean(v)
            for k, v in obj.items()
        }
    elif isinstance(obj, (list, tuple)):
        return [clean(x) for x in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, (float, np.floating)):
        return None if (math.isnan(obj) or math.isinf(obj)) else float(obj)
    elif isinstance(obj, np.ndarray):
        return clean(obj.tolist()) if obj.ndim > 0 else clean(obj.item())
    return obj


def save_run(results, section, models=None, **meta):
    """Save experiment run with timestamp

    Args:
        results: Dict of DataFrames from training {'mlp': df, 'siren': df, 'kan': df, 'kan_pruning': df}
        section: Section name (e.g., 'section1_1')
        models: Dict of {'kan': {idx: model}, 'kan_pruned': {idx: model}} or None
        **meta: Minimal metadata (epochs, device, etc.)
               Note: Derivable metadata (depths, activations, grids) should not be passed
               as they can be computed from the DataFrames themselves.

    Returns:
        Timestamp string
    """
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    sec_num = section.split('_')[-1]

    # Save in section1/results/sec{N}_results/
    base_dir = Path(__file__).parent.parent  # Go to section1/
    results_dir = base_dir / 'results' / f'sec{sec_num}_results'
    results_dir.mkdir(exist_ok=True, parents=True)
    p = results_dir

    # Create minimal metadata that applies to all DataFrames
    minimal_meta = {
        'section': section,
        'timestamp': ts,
        'epochs': meta.get('epochs'),
        'device': meta.get('device'),
    }

    # Save each DataFrame with metadata as attributes
    for model_type, df in results.items():
        if isinstance(df, pd.DataFrame):
            # Attach metadata to DataFrame attributes
            # This metadata travels with the DataFrame and is preserved in pickle format
            df.attrs.update(minimal_meta)
            df.attrs['model_type'] = model_type

            # Save as pickle (preserves attrs)
            df.to_pickle(p / f'{section}_{ts}_{model_type}.pkl')

            # Save as parquet if available (note: parquet doesn't preserve attrs in all versions)
            try:
                df.to_parquet(p / f'{section}_{ts}_{model_type}.parquet')
            except ImportError:
                pass

    # Save models
    if models:
        if 'kan' in models:
            for idx, model in models['kan'].items():
                model.saveckpt(str(p / f'{section}_{ts}_kan_{idx}'))
        if 'kan_pruned' in models:
            for idx, model in models['kan_pruned'].items():
                model.saveckpt(str(p / f'{section}_{ts}_pruned_{idx}'))

    print(f"Saved to {p}/{section}_{ts}.*")
    print(f"Metadata stored in DataFrame attributes (access via df.attrs)")
    return ts


def load_run(section, timestamp):
    """Load experiment run by timestamp

    Args:
        section: Section name (e.g., 'section1_1')
        timestamp: Timestamp string from save_run

    Returns:
        Tuple of (results_dict, meta_dict) where:
        - results_dict: Dict of DataFrames {'mlp': df, 'siren': df, ...}
        - meta_dict: Combined metadata from all DataFrames. Access individual
                     DataFrame metadata via df.attrs (e.g., mlp_df.attrs['epochs'])

    Note:
        Metadata is now stored in DataFrame.attrs. The returned meta_dict is
        consolidated from the first available DataFrame for backward compatibility.
        Derivable metadata (depths, activations, grids) can be obtained from the
        DataFrames themselves:
        - depths: mlp_df['depth'].unique()
        - activations: mlp_df['activation'].unique()
        - grids: kan_df['grid_size'].unique()
    """
    sec_num = section.split('_')[-1]
    base_dir = Path(__file__).parent.parent
    results_dir = base_dir / 'results' / f'sec{sec_num}_results'
    p = results_dir

    # Load DataFrames (prefer pickle to preserve attrs)
    results = {}
    meta = {}

    for model_type in ['mlp', 'siren', 'kan', 'kan_pruning']:
        pkl_file = p / f'{section}_{timestamp}_{model_type}.pkl'
        parquet_file = p / f'{section}_{timestamp}_{model_type}.parquet'

        # Prefer pickle as it preserves DataFrame.attrs
        if pkl_file.exists():
            df = pd.read_pickle(pkl_file)
            results[model_type] = df

            # Extract metadata from first available DataFrame (for backward compatibility)
            if not meta and hasattr(df, 'attrs') and df.attrs:
                meta = dict(df.attrs)

        elif parquet_file.exists():
            # Parquet doesn't preserve attrs in older pandas versions
            df = pd.read_parquet(parquet_file)
            results[model_type] = df

    # Try to load legacy JSON metadata if no attrs found and JSON exists
    json_file = p / f'{section}_{timestamp}_meta.json'
    if not meta and json_file.exists():
        with open(json_file, 'r') as f:
            json_data = json.load(f)
            meta = json_data.get('meta', {})

    return results, meta
