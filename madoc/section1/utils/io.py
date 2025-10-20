"""I/O utilities for saving and loading experiment results"""
import json
import pickle
from pathlib import Path
from datetime import datetime
import numpy as np
import math


def clean_nan_inf(obj):
    """Recursively replace NaN and Inf values with None for JSON compatibility"""
    if isinstance(obj, dict):
        return {k: clean_nan_inf(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [clean_nan_inf(item) for item in obj]
    elif isinstance(obj, (float, np.floating)):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        if obj.ndim == 0:  # Scalar array
            return clean_nan_inf(obj.item())
        return [clean_nan_inf(item) for item in obj.tolist()]
    else:
        return obj


def save_results(results, section_name, output_dir='sec1_results',
                 epochs=None, device=None, grids=None, depths=None,
                 activations=None, frequencies=None, num_datasets=None,
                 kan_models=None, pruned_models=None):
    """Save experiment results to JSON/pickle and optionally save KAN model checkpoints"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Clean results to make them JSON-serializable
    json_results = clean_nan_inf(results)

    # Convert all keys to strings for JSON compatibility
    def stringify_keys(obj):
        if isinstance(obj, dict):
            return {str(k): stringify_keys(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [stringify_keys(item) for item in obj]
        else:
            return obj

    json_results = stringify_keys(json_results)

    with open(output_path / f'{section_name}_results_{timestamp}.json', 'w') as f:
        json.dump(json_results, f, indent=2, allow_nan=False)
    with open(output_path / f'{section_name}_results_{timestamp}.pkl', 'wb') as f:
        pickle.dump(results, f)

    if kan_models is not None:
        models_dir = output_path / f'kan_models_{timestamp}'
        models_dir.mkdir(exist_ok=True)
        for dataset_idx, model in kan_models.items():
            model.saveckpt(str(models_dir / f'kan_dataset_{dataset_idx}'))
        with open(models_dir / 'models_metadata.json', 'w') as f:
            json.dump({'timestamp': timestamp, 'num_models': len(kan_models),
                      'dataset_indices': list(kan_models.keys())}, f, indent=2)

    if pruned_models is not None:
        pruned_dir = output_path / f'kan_pruned_models_{timestamp}'
        pruned_dir.mkdir(exist_ok=True)
        for dataset_idx, model in pruned_models.items():
            model.saveckpt(str(pruned_dir / f'kan_pruned_dataset_{dataset_idx}'))
        with open(pruned_dir / 'pruned_models_metadata.json', 'w') as f:
            json.dump({'timestamp': timestamp, 'num_models': len(pruned_models),
                      'dataset_indices': list(pruned_models.keys()),
                      'pruning_params': {'node_th': 1e-2, 'edge_th': 3e-2}}, f, indent=2)

    metadata = {'timestamp': timestamp}
    if epochs is not None:
        metadata['epochs'] = epochs
    if device is not None:
        metadata['device'] = str(device)
    if grids is not None:
        metadata['grids'] = grids.tolist() if hasattr(grids, 'tolist') else grids
    if depths is not None:
        metadata['depths'] = depths
    if activations is not None:
        metadata['activations'] = activations
    if frequencies is not None:
        metadata['frequencies'] = frequencies
    if num_datasets is not None:
        metadata['num_datasets'] = num_datasets
    metadata['kan_models_saved'] = kan_models is not None
    metadata['pruned_models_saved'] = pruned_models is not None

    with open(output_path / f'{section_name}_metadata_{timestamp}.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    return timestamp
