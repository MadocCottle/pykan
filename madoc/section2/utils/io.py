"""Concise I/O for Section 2 experiment results"""
import json
import pickle
from pathlib import Path
from datetime import datetime
import numpy as np
import math


def clean(obj):
    """Replace NaN/Inf with None for JSON and convert numpy types to Python native types"""
    if isinstance(obj, dict):
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
        results: Results dict from training (ensemble/adaptive/evolutionary)
        section: Section name (e.g., 'section2_1')
        models: Dict of models or None
        **meta: Metadata (epochs, n_experts, device, etc.)

    Returns:
        Timestamp string
    """
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    sec_num = section.split('_')[-1]

    # Save in section2/results/sec{N}_results/
    base_dir = Path(__file__).parent.parent
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

    # Save models if provided
    if models:
        models_dir = p / f'{section}_{ts}_models'
        models_dir.mkdir(exist_ok=True)

        # Save model states (support both dict and list)
        if isinstance(models, dict):
            for name, model_dict in models.items():
                if hasattr(model_dict, 'items'):
                    for idx, model in model_dict.items():
                        if hasattr(model, 'saveckpt'):
                            model.saveckpt(str(models_dir / f'{name}_{idx}'))
                else:
                    if hasattr(model_dict, 'saveckpt'):
                        model_dict.saveckpt(str(models_dir / name))
        elif isinstance(models, list):
            for idx, model in enumerate(models):
                if hasattr(model, 'saveckpt'):
                    model.saveckpt(str(models_dir / f'expert_{idx}'))

    print(f"Saved to {p}/{section}_{ts}.*")
    return ts
