"""Concise I/O for Section 2 experiment results"""
import pickle
from pathlib import Path
from datetime import datetime
import pandas as pd
import torch


def save_run(results, section, models=None, **meta):
    """Save experiment run with timestamp

    Args:
        results: Dict of DataFrames from training {'mlp': df, 'siren': df, 'kan': df, 'kan_pruning': df}
        section: Section name (e.g., 'section1_1' or 'section2_1_highd_3d_shallow')
        models: Dict of {'mlp': {idx: model}, 'siren': {idx: model},
                        'kan': {idx: model}, 'kan_pruned': {idx: model}} or None
        **meta: Minimal metadata (epochs, device, etc.)
               Note: Derivable metadata (depths, activations, grids) should not be passed
               as they can be computed from the DataFrames themselves.

    Returns:
        Timestamp string
    """
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Extract section number intelligently
    # Format: section{N}_{M} OR section{N}_{M}_highd_{dim}d_{arch}
    # We need the second part (M) which is the subsection number
    parts = section.split('_')

    if len(parts) >= 2 and parts[1].isdigit():
        # Standard: section2_1 -> sec_num = '1'
        # OR composite: section2_1_highd_3d_shallow -> sec_num = '1' (ignore rest)
        sec_num = parts[1]
    else:
        # Fallback (shouldn't happen with valid section names)
        sec_num = parts[-1]

    # Save in section2/results/sec{N}_results/
    base_dir = Path(__file__).parent.parent  # Go to section2/
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
        for model_type, model_dict in models.items():
            # Check if model_dict contains PyTorch models or KAN models
            if not model_dict:
                continue

            # Get a sample model to determine type
            sample_model = next(iter(model_dict.values()))

            # Check if it's a KAN model (has saveckpt method)
            if hasattr(sample_model, 'saveckpt'):
                # Save KAN models using KAN's saveckpt method
                for idx, model in model_dict.items():
                    model.saveckpt(str(p / f'{section}_{ts}_{model_type}_{idx}'))
            else:
                # Save as PyTorch state_dict
                import torch
                for idx, model in model_dict.items():
                    torch.save(model.state_dict(), p / f'{section}_{ts}_{model_type}_{idx}.pth')

    print(f"Saved to {p}/{section}_{ts}.*")
    print(f"Metadata stored in DataFrame attributes (access via df.attrs)")
    return ts


def load_run(section, timestamp, load_models=False):
    """Load experiment run by timestamp

    Args:
        section: Section name (e.g., 'section1_1' or 'section2_1_highd_3d_shallow')
        timestamp: Timestamp string from save_run
        load_models: If True, also load saved model state_dicts. Default: False
                    Note: To use loaded models, you must reconstruct the model architecture
                    and then load the state_dict.

    Returns:
        If load_models=False:
            Tuple of (results_dict, meta_dict)
        If load_models=True:
            Tuple of (results_dict, meta_dict, models_dict)

        Where:
        - results_dict: Dict of DataFrames {'mlp': df, 'siren': df, ...}
        - meta_dict: Combined metadata from all DataFrames. Access individual
                     DataFrame metadata via df.attrs (e.g., mlp_df.attrs['epochs'])
        - models_dict: Dict of {'mlp': {idx: state_dict}, 'siren': {idx: state_dict},
                                'kan': {idx: checkpoint_path}, 'kan_pruned': {idx: checkpoint_path}}

    Note:
        Metadata is stored in DataFrame.attrs. The returned meta_dict is
        consolidated from the first available DataFrame for convenience.
        Derivable metadata (depths, activations, grids) can be obtained from the
        DataFrames themselves:
        - depths: mlp_df['depth'].unique()
        - activations: mlp_df['activation'].unique()
        - grids: kan_df['grid_size'].unique()
    """
    # Extract section number using same logic as save_run
    parts = section.split('_')
    if len(parts) >= 2 and parts[1].isdigit():
        sec_num = parts[1]
    else:
        sec_num = parts[-1]

    base_dir = Path(__file__).parent.parent
    results_dir = base_dir / 'results' / f'sec{sec_num}_results'
    p = results_dir

    # Load DataFrames (prefer pickle to preserve attrs)
    results = {}
    meta = {}

    # Auto-detect available pickle files
    for pkl_file in p.glob(f'{section}_{timestamp}_*.pkl'):
        # Extract model type from filename
        model_type = pkl_file.stem.split('_')[-1]

        df = pd.read_pickle(pkl_file)
        results[model_type] = df

        # Extract metadata from first available DataFrame
        if not meta and hasattr(df, 'attrs') and df.attrs:
            meta = dict(df.attrs)

    # If no pickle files found, try parquet
    if not results:
        for parquet_file in p.glob(f'{section}_{timestamp}_*.parquet'):
            model_type = parquet_file.stem.split('_')[-1]
            df = pd.read_parquet(parquet_file)
            results[model_type] = df

    # Load models if requested
    if load_models:
        import torch
        models = {}

        # Auto-detect all model types by finding saved files
        # First, find all .pth files (PyTorch state_dicts)
        for model_file in p.glob(f'{section}_{timestamp}_*.pth'):
            # Extract model type and idx: section_timestamp_MODELTYPE_IDX.pth
            parts = model_file.stem.replace(f'{section}_{timestamp}_', '').rsplit('_', 1)
            if len(parts) == 2:
                model_type, idx_str = parts
                try:
                    idx = int(idx_str)
                    if model_type not in models:
                        models[model_type] = {}
                    models[model_type][idx] = torch.load(model_file, map_location='cpu')
                except ValueError:
                    pass  # Skip files that don't match pattern

        # Second, find all KAN checkpoint files (they create _config.yml, _state, _cache_data files)
        # Look for config files as indicators of KAN checkpoints
        for config_file in p.glob(f'{section}_{timestamp}_*_config.yml'):
            # Extract checkpoint path by removing _config.yml suffix
            ckpt_path = str(config_file).replace('_config.yml', '')
            # Extract model type and idx: section_timestamp_MODELTYPE_IDX
            base_name = config_file.stem.replace('_config', '')  # Remove _config from filename
            parts = base_name.replace(f'{section}_{timestamp}_', '').rsplit('_', 1)
            if len(parts) == 2:
                model_type, idx_str = parts
                try:
                    idx = int(idx_str)
                    if model_type not in models:
                        models[model_type] = {}
                    # Store the path without extensions (KAN.loadckpt expects base path)
                    models[model_type][idx] = ckpt_path
                except ValueError:
                    pass  # Skip files that don't match pattern

        return results, meta, models

    return results, meta
