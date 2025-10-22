"""Concise I/O for Section 1 experiment results"""
import json
import pickle
from pathlib import Path
from datetime import datetime
import numpy as np
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
        results: Results dict from training (mlp/siren/kan/kan_pruning)
        section: Section name (e.g., 'section1_1')
        models: Dict of {'kan': {idx: model}, 'kan_pruned': {idx: model}} or None
        **meta: Metadata (epochs, grids, depths, activations, device, etc.)

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

    # Save results
    with open(p / f'{section}_{ts}.pkl', 'wb') as f:
        pickle.dump(results, f)

    # Save JSON with metadata
    json_data = {'results': clean(results), 'meta': meta}
    with open(p / f'{section}_{ts}.json', 'w') as f:
        json.dump(json_data, f, indent=2)

    # Save models
    if models:
        if 'kan' in models:
            for idx, model in models['kan'].items():
                model.saveckpt(str(p / f'{section}_{ts}_kan_{idx}'))
        if 'kan_pruned' in models:
            for idx, model in models['kan_pruned'].items():
                model.saveckpt(str(p / f'{section}_{ts}_pruned_{idx}'))

    print(f"Saved to {p}/{section}_{ts}.*")
    return ts
