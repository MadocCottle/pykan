"""Concise I/O for Section 1 experiment results"""
import pickle
from pathlib import Path
from datetime import datetime
import pandas as pd
import torch


def save_run(results, section, models=None, checkpoints=None, **meta):
    """Save experiment run with timestamp

    Args:
        results: Dict of DataFrames from training {'mlp': df, 'siren': df, 'kan': df, 'kan_pruning': df}
        section: Section name (e.g., 'section1_1')
        models: DEPRECATED - use checkpoints instead
        checkpoints: Dict of checkpoints from training. Structure:
                    {'mlp': {dataset_idx: {'at_kan_threshold_time': {...}, 'final': {...}}}, ...}
                    Each checkpoint dict contains: 'model', 'epoch', 'time', 'dense_mse', etc.
        **meta: Minimal metadata (epochs, device, kan_threshold_time, etc.)
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
        'kan_threshold_time': meta.get('kan_threshold_time'),
    }

    # Format filename with epochs if provided
    epochs = meta.get('epochs')
    if epochs is not None:
        filename_base = f'{section}_{ts}_e{epochs}'
    else:
        filename_base = f'{section}_{ts}'

    # Save each DataFrame with metadata as attributes
    for model_type, df in results.items():
        if isinstance(df, pd.DataFrame):
            # Attach metadata to DataFrame attributes
            # This metadata travels with the DataFrame and is preserved in pickle format
            df.attrs.update(minimal_meta)
            df.attrs['model_type'] = model_type

            # Save as pickle (preserves attrs)
            df.to_pickle(p / f'{filename_base}_{model_type}.pkl')

            # Save as parquet if available (note: parquet doesn't preserve attrs in all versions)
            try:
                df.to_parquet(p / f'{filename_base}_{model_type}.parquet')
            except ImportError:
                pass

    # Save checkpoints (NEW two-checkpoint strategy)
    if checkpoints:
        for model_type, dataset_checkpoints in checkpoints.items():
            for dataset_idx, checkpoint_dict in dataset_checkpoints.items():
                for checkpoint_name, checkpoint_data in checkpoint_dict.items():
                    model = checkpoint_data['model']

                    # Save based on model type
                    if model_type in ['mlp', 'siren']:
                        # PyTorch models: save state_dict
                        save_path = p / f'{filename_base}_{model_type}_{dataset_idx}_{checkpoint_name}.pth'
                        torch.save(model.state_dict(), save_path)
                    elif model_type in ['kan', 'kan_pruning']:
                        # KAN models: use saveckpt method
                        save_path_base = str(p / f'{filename_base}_{model_type}_{dataset_idx}_{checkpoint_name}')
                        model.saveckpt(save_path_base)

        # Also save checkpoint metadata as pickle for easy loading
        checkpoint_metadata = {}
        for model_type, dataset_checkpoints in checkpoints.items():
            checkpoint_metadata[model_type] = {}
            for dataset_idx, checkpoint_dict in dataset_checkpoints.items():
                checkpoint_metadata[model_type][dataset_idx] = {}
                for checkpoint_name, checkpoint_data in checkpoint_dict.items():
                    # Save everything except the model itself
                    metadata = {k: v for k, v in checkpoint_data.items() if k != 'model'}
                    checkpoint_metadata[model_type][dataset_idx][checkpoint_name] = metadata

        with open(p / f'{filename_base}_checkpoint_metadata.pkl', 'wb') as f:
            pickle.dump(checkpoint_metadata, f)

    # Backward compatibility: save models if provided (DEPRECATED)
    if models:
        print("Warning: 'models' parameter is deprecated. Use 'checkpoints' instead.")
        # Save MLP models (PyTorch state_dicts)
        if 'mlp' in models:
            for idx, model in models['mlp'].items():
                torch.save(model.state_dict(), p / f'{filename_base}_mlp_{idx}.pth')

        # Save SIREN models (PyTorch state_dicts)
        if 'siren' in models:
            for idx, model in models['siren'].items():
                torch.save(model.state_dict(), p / f'{filename_base}_siren_{idx}.pth')

        # Save KAN models (using KAN's saveckpt method)
        if 'kan' in models:
            for idx, model in models['kan'].items():
                model.saveckpt(str(p / f'{filename_base}_kan_{idx}'))

        # Save pruned KAN models
        if 'kan_pruned' in models:
            for idx, model in models['kan_pruned'].items():
                model.saveckpt(str(p / f'{filename_base}_pruned_{idx}'))

    print(f"\nSaved to {p}/{filename_base}.*")
    print(f"Metadata stored in DataFrame attributes (access via df.attrs)")
    if checkpoints:
        print(f"Checkpoints saved:")
        for model_type in checkpoints.keys():
            print(f"  - {model_type}: {len(checkpoints[model_type])} datasets x 2 checkpoints (at_threshold + final)")
    return ts


def load_run(section, timestamp, load_models=False):
    """Load experiment run by timestamp

    Args:
        section: Section name (e.g., 'section1_1')
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
    sec_num = section.split('_')[-1]
    base_dir = Path(__file__).parent.parent
    results_dir = base_dir / 'results' / f'sec{sec_num}_results'
    p = results_dir

    # Load DataFrames (prefer pickle to preserve attrs)
    results = {}
    meta = {}

    for model_type in ['mlp', 'siren', 'kan', 'kan_pruning']:
        # Try new format with epochs first: section1_1_TIMESTAMP_e100_mlp.pkl
        pkl_files = list(p.glob(f'{section}_{timestamp}_e*_{model_type}.pkl'))
        parquet_files = list(p.glob(f'{section}_{timestamp}_e*_{model_type}.parquet'))

        # Fallback to old format without epochs: section1_1_TIMESTAMP_mlp.pkl
        if not pkl_files:
            pkl_files = [p / f'{section}_{timestamp}_{model_type}.pkl']
        if not parquet_files:
            parquet_files = [p / f'{section}_{timestamp}_{model_type}.parquet']

        pkl_file = pkl_files[0] if pkl_files else None
        parquet_file = parquet_files[0] if parquet_files else None

        # Prefer pickle as it preserves DataFrame.attrs
        if pkl_file and pkl_file.exists():
            df = pd.read_pickle(pkl_file)
            results[model_type] = df

            # Extract metadata from first available DataFrame
            if not meta and hasattr(df, 'attrs') and df.attrs:
                meta = dict(df.attrs)

        elif parquet_file and parquet_file.exists():
            # Parquet doesn't preserve attrs in older pandas versions
            df = pd.read_parquet(parquet_file)
            results[model_type] = df

    # Load models if requested
    if load_models:
        import torch
        models = {}

        # Load MLP models
        # Try new checkpoint-based naming first (e.g., mlp_0_final.pth)
        # Prefer 'final' checkpoint for visualizations
        mlp_models = {}
        # Try with epochs: section1_1_TIMESTAMP_e100_mlp_0_final.pth
        for model_file in p.glob(f'{section}_{timestamp}_e*_mlp_*_final.pth'):
            parts = model_file.stem.split('_')
            for i, part in enumerate(parts):
                if part == 'final' and i > 0:
                    idx = int(parts[i-1])
                    mlp_models[idx] = torch.load(model_file, map_location='cpu')
                    break

        # Fallback: without epochs: section1_1_TIMESTAMP_mlp_0_final.pth
        if not mlp_models:
            for model_file in p.glob(f'{section}_{timestamp}_mlp_*_final.pth'):
                parts = model_file.stem.split('_')
                for i, part in enumerate(parts):
                    if part == 'final' and i > 0:
                        idx = int(parts[i-1])
                        mlp_models[idx] = torch.load(model_file, map_location='cpu')
                        break

        # Fallback: try old naming convention (no checkpoints)
        if not mlp_models:
            for model_file in p.glob(f'{section}_{timestamp}*_mlp_*.pth'):
                if '_final' in model_file.stem or '_at_threshold' in model_file.stem:
                    continue
                idx = int(model_file.stem.split('_')[-1])
                mlp_models[idx] = torch.load(model_file, map_location='cpu')

        if mlp_models:
            models['mlp'] = mlp_models

        # Load SIREN models
        siren_models = {}
        # Try with epochs: section1_1_TIMESTAMP_e100_siren_0_final.pth
        for model_file in p.glob(f'{section}_{timestamp}_e*_siren_*_final.pth'):
            parts = model_file.stem.split('_')
            for i, part in enumerate(parts):
                if part == 'final' and i > 0:
                    idx = int(parts[i-1])
                    siren_models[idx] = torch.load(model_file, map_location='cpu')
                    break

        # Fallback: without epochs
        if not siren_models:
            for model_file in p.glob(f'{section}_{timestamp}_siren_*_final.pth'):
                parts = model_file.stem.split('_')
                for i, part in enumerate(parts):
                    if part == 'final' and i > 0:
                        idx = int(parts[i-1])
                        siren_models[idx] = torch.load(model_file, map_location='cpu')
                        break

        # Fallback: old naming convention
        if not siren_models:
            for model_file in p.glob(f'{section}_{timestamp}*_siren_*.pth'):
                if '_final' in model_file.stem or '_at_threshold' in model_file.stem:
                    continue
                idx = int(model_file.stem.split('_')[-1])
                siren_models[idx] = torch.load(model_file, map_location='cpu')

        if siren_models:
            models['siren'] = siren_models

        # Load KAN models (store checkpoint paths)
        kan_models = {}
        # Try with epochs: section1_1_TIMESTAMP_e100_kan_0_final_state
        for state_file in p.glob(f'{section}_{timestamp}_e*_kan_*_final_state'):
            base_path = str(state_file)[:-6]  # Remove '_state' suffix
            parts = base_path.split('_')
            for i, part in enumerate(parts):
                if part == 'final' and i > 0:
                    idx = int(parts[i-1])
                    kan_models[idx] = base_path
                    break

        # Fallback: without epochs
        if not kan_models:
            for state_file in p.glob(f'{section}_{timestamp}_kan_*_final_state'):
                base_path = str(state_file)[:-6]
                parts = base_path.split('_')
                for i, part in enumerate(parts):
                    if part == 'final' and i > 0:
                        idx = int(parts[i-1])
                        kan_models[idx] = base_path
                        break

        # Fallback: old naming convention
        if not kan_models:
            for state_file in p.glob(f'{section}_{timestamp}*_kan_*_state'):
                if '_final_state' in str(state_file) or '_at_threshold_state' in str(state_file):
                    continue
                base_path = str(state_file)[:-6]
                idx = int(base_path.split('_kan_')[-1])
                kan_models[idx] = base_path

        if kan_models:
            models['kan'] = kan_models

        # Load pruned KAN models (store checkpoint paths)
        # Try new checkpoint-based naming first
        kan_pruned_models = {}

        # NEW: Try format with epochs first (e.g., section1_3_20251024_044352_e30_kan_pruning_0_final_state)
        for state_file in p.glob(f'{section}_{timestamp}_e*_kan_pruning_*_final_state'):
            # Extract the base path by removing '_state' suffix
            base_path = str(state_file)[:-6]
            # Extract index from path like: .../section1_1_20251024_044352_e30_kan_pruning_0_final
            parts = base_path.split('_')
            for i, part in enumerate(parts):
                if part == 'final' and i > 0:
                    idx = int(parts[i-1])
                    kan_pruned_models[idx] = base_path
                    break

        # Fallback: Try without epochs (backward compatibility for old format)
        if not kan_pruned_models:
            for state_file in p.glob(f'{section}_{timestamp}_kan_pruning_*_final_state'):
                # Extract the base path by removing '_state' suffix
                base_path = str(state_file)[:-6]
                # Extract index from path like: .../section1_1_20251023_215658_kan_pruning_0_final
                parts = base_path.split('_')
                for i, part in enumerate(parts):
                    if part == 'final' and i > 0:
                        idx = int(parts[i-1])
                        kan_pruned_models[idx] = base_path
                        break

        # Fallback: try old naming with 'pruned' prefix (very old format)
        if not kan_pruned_models:
            for state_file in p.glob(f'{section}_{timestamp}_pruned_*_state'):
                if '_final_state' in str(state_file) or '_at_threshold_state' in str(state_file):
                    continue
                base_path = str(state_file)[:-6]
                idx = int(base_path.split('_pruned_')[-1])
                kan_pruned_models[idx] = base_path

        if kan_pruned_models:
            models['kan_pruned'] = kan_pruned_models

        return results, meta, models

    return results, meta
