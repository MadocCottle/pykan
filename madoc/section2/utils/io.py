"""Concise I/O for Section 2 experiment results"""
import pickle
from pathlib import Path
from datetime import datetime
import pandas as pd
import torch


def save_run(results, section, models=None, checkpoints=None, **meta):
    """Save experiment run with timestamp

    Args:
        results: Dict of DataFrames from training {'mlp': df, 'siren': df, 'kan': df, 'kan_pruning': df}
        section: Section name (e.g., 'section1_1' or 'section2_1_highd_3d_shallow')
        models: DEPRECATED - use checkpoints instead. Dict of {'mlp': {idx: model}, ...} or None
        checkpoints: Dict of checkpoints from training. Structure:
                    {'optimizer_name': {dataset_idx: {'at_threshold': {...}, 'final': {...}}}, ...}
                    Each checkpoint dict contains: 'model', 'epoch', 'time', 'dense_mse', etc.
        **meta: Minimal metadata (epochs, device, lbfgs_threshold_time, etc.)
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
        'lbfgs_threshold_time': meta.get('lbfgs_threshold_time'),  # Section2_1 uses LBFGS as reference
        'baseline_threshold_time': meta.get('baseline_threshold_time'),  # Section2_2 uses baseline as reference
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
        import pickle
        for optimizer_name, dataset_checkpoints in checkpoints.items():
            for dataset_idx, checkpoint_dict in dataset_checkpoints.items():
                for checkpoint_name, checkpoint_data in checkpoint_dict.items():
                    model = checkpoint_data['model']

                    # KAN models: use saveckpt method
                    if hasattr(model, 'saveckpt'):
                        save_path_base = str(p / f'{filename_base}_{optimizer_name}_{dataset_idx}_{checkpoint_name}')
                        model.saveckpt(save_path_base)
                    else:
                        # PyTorch models: save state_dict
                        save_path = p / f'{filename_base}_{optimizer_name}_{dataset_idx}_{checkpoint_name}.pth'
                        torch.save(model.state_dict(), save_path)

        # Also save checkpoint metadata as pickle for easy loading
        checkpoint_metadata = {}
        for optimizer_name, dataset_checkpoints in checkpoints.items():
            checkpoint_metadata[optimizer_name] = {}
            for dataset_idx, checkpoint_dict in dataset_checkpoints.items():
                checkpoint_metadata[optimizer_name][dataset_idx] = {}
                for checkpoint_name, checkpoint_data in checkpoint_dict.items():
                    # Save everything except the model itself
                    metadata = {k: v for k, v in checkpoint_data.items() if k != 'model'}
                    checkpoint_metadata[optimizer_name][dataset_idx][checkpoint_name] = metadata

        with open(p / f'{filename_base}_checkpoint_metadata.pkl', 'wb') as f:
            pickle.dump(checkpoint_metadata, f)

    # Backward compatibility: save models if provided (DEPRECATED)
    if models:
        print("Warning: 'models' parameter is deprecated. Use 'checkpoints' instead.")
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
                    model.saveckpt(str(p / f'{filename_base}_{model_type}_{idx}'))
            else:
                # Save as PyTorch state_dict
                import torch
                for idx, model in model_dict.items():
                    torch.save(model.state_dict(), p / f'{filename_base}_{model_type}_{idx}.pth')

    print(f"\nSaved to {p}/{filename_base}.*")
    print(f"Metadata stored in DataFrame attributes (access via df.attrs)")
    if checkpoints:
        print(f"Checkpoints saved:")
        for optimizer_name in checkpoints.keys():
            print(f"  - {optimizer_name}: {len(checkpoints[optimizer_name])} datasets x 2 checkpoints (at_threshold + final)")
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

    # Try new format with epochs first: section2_1_TIMESTAMP_e10_adam.pkl
    pkl_files = list(p.glob(f'{section}_{timestamp}_e*_*.pkl'))

    # Fall back to old format: section2_1_TIMESTAMP_adam.pkl
    if not pkl_files:
        pkl_files = list(p.glob(f'{section}_{timestamp}_*.pkl'))

    # Auto-detect available pickle files
    for pkl_file in pkl_files:
        # Extract model type from filename
        model_type = pkl_file.stem.split('_')[-1]

        df = pd.read_pickle(pkl_file)
        results[model_type] = df

        # Extract metadata from first available DataFrame
        if not meta and hasattr(df, 'attrs') and df.attrs:
            meta = dict(df.attrs)

    # If no pickle files found, try parquet
    if not results:
        # Try with epochs first
        parquet_files = list(p.glob(f'{section}_{timestamp}_e*_*.parquet'))
        if not parquet_files:
            parquet_files = list(p.glob(f'{section}_{timestamp}_*.parquet'))

        for parquet_file in parquet_files:
            model_type = parquet_file.stem.split('_')[-1]
            df = pd.read_parquet(parquet_file)
            results[model_type] = df

    # Load models if requested
    if load_models:
        import torch
        models = {}

        # Auto-detect all model types by finding saved files
        # Try new format with epochs first, then fall back to old format
        # First, find all .pth files (PyTorch state_dicts)
        pth_files = list(p.glob(f'{section}_{timestamp}_e*_*.pth'))
        if not pth_files:
            pth_files = list(p.glob(f'{section}_{timestamp}_*.pth'))

        for model_file in pth_files:
            # Extract model type and idx
            # New format: section_timestamp_e100_MODELTYPE_IDX.pth
            # Old format: section_timestamp_MODELTYPE_IDX.pth
            filename = model_file.stem
            # Remove section and timestamp prefix
            if f'{section}_{timestamp}_e' in filename:
                # New format with epochs
                parts_str = filename.split(f'{section}_{timestamp}_e')[-1]
                # Remove epoch number: "100_adam_0" -> skip "100_", then parse "adam_0"
                parts_after_e = parts_str.split('_', 1)  # Split after epoch number
                if len(parts_after_e) > 1:
                    parts = parts_after_e[1].rsplit('_', 1)  # Split model_type and idx
                else:
                    continue
            else:
                # Old format
                parts = filename.replace(f'{section}_{timestamp}_', '').rsplit('_', 1)

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
        config_files = list(p.glob(f'{section}_{timestamp}_e*_*_config.yml'))
        if not config_files:
            config_files = list(p.glob(f'{section}_{timestamp}_*_config.yml'))

        for config_file in config_files:
            # Extract checkpoint path by removing _config.yml suffix
            ckpt_path = str(config_file).replace('_config.yml', '')
            # Extract model type and idx
            base_name = config_file.stem.replace('_config', '')

            # Parse filename similar to above
            if f'{section}_{timestamp}_e' in base_name:
                parts_str = base_name.split(f'{section}_{timestamp}_e')[-1]
                parts_after_e = parts_str.split('_', 1)
                if len(parts_after_e) > 1:
                    parts = parts_after_e[1].rsplit('_', 1)
                else:
                    continue
            else:
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
