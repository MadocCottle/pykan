"""
Utility functions for table generation
"""
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tabulate import tabulate

def load_latest_results(section_name: str, results_dir: str = '../results') -> Dict[str, pd.DataFrame]:
    """
    Load the latest results for a given section.

    Args:
        section_name: Name of section (e.g., 'section1_1', 'section1_2')
        results_dir: Path to results directory

    Returns:
        Dictionary with keys: 'mlp', 'siren', 'kan', 'kan_pruning'
    """
    results_path = Path(__file__).parent / results_dir / 'sec1_results'

    # Find latest files for each model type
    model_types = ['mlp', 'siren', 'kan', 'kan_pruning']
    results = {}

    for model_type in model_types:
        # Try new format with epochs first: section1_1_*_e*_mlp.pkl
        pattern_with_epochs = f"{section_name}_*_e*_{model_type}.pkl"
        files = sorted(results_path.glob(pattern_with_epochs))

        # Fall back to old format: section1_1_*_mlp.pkl
        if not files:
            pattern_old = f"{section_name}_*_{model_type}.pkl"
            files = sorted(results_path.glob(pattern_old))

        if files:
            latest_file = files[-1]  # Get most recent (sorts by timestamp)
            with open(latest_file, 'rb') as f:
                results[model_type] = pickle.load(f)
            print(f"Loaded {model_type}: {latest_file.name}")
        else:
            print(f"Warning: No files found for {model_type} in {section_name}")
            results[model_type] = pd.DataFrame()

    return results


def load_checkpoint_metadata(section_name: str, timestamp: Optional[str] = None,
                            results_dir: str = '../results') -> Optional[Dict]:
    """
    Load checkpoint metadata for a given section.

    Checkpoint metadata contains the two-checkpoint evaluation strategy data:
    - 'at_kan_threshold_time' / 'at_threshold': Model state when KAN reaches interpolation threshold
    - 'final': Model state after full training budget

    Each checkpoint contains: model, epoch, time, train_loss, test_loss, dense_mse, num_params

    Args:
        section_name: Name of section (e.g., 'section1_1', 'section1_2')
        timestamp: Specific timestamp to load, or None for latest
        results_dir: Path to results directory

    Returns:
        Dictionary with structure:
        {
            'mlp': {dataset_idx: {'at_kan_threshold_time': {...}, 'final': {...}}},
            'siren': {dataset_idx: {'at_kan_threshold_time': {...}, 'final': {...}}},
            'kan': {dataset_idx: {'at_threshold': {...}, 'final': {...}}},
            'kan_pruning': {dataset_idx: {'at_threshold': {...}, 'final': {...}}}
        }
        Returns None if checkpoint metadata not found
    """
    results_path = Path(__file__).parent / results_dir / 'sec1_results'

    if timestamp is None:
        # Find latest checkpoint metadata file
        pattern = f"{section_name}_*_checkpoint_metadata.pkl"
        files = sorted(results_path.glob(pattern))
        if not files:
            print(f"Warning: No checkpoint metadata files found for {section_name}")
            return None
        metadata_file = files[-1]
    else:
        # Try with epochs: section1_1_TIMESTAMP_e100_checkpoint_metadata.pkl
        metadata_file = results_path / f"{section_name}_{timestamp}_e*_checkpoint_metadata.pkl"
        files = list(results_path.glob(str(metadata_file)))

        # Fallback without epochs
        if not files:
            metadata_file = results_path / f"{section_name}_{timestamp}_checkpoint_metadata.pkl"
            if not metadata_file.exists():
                print(f"Warning: Checkpoint metadata not found for {section_name}_{timestamp}")
                return None
        else:
            metadata_file = files[0]

    with open(metadata_file, 'rb') as f:
        checkpoint_metadata = pickle.load(f)

    print(f"Loaded checkpoint metadata: {metadata_file.name}")
    return checkpoint_metadata

def get_best_result_per_dataset(df: pd.DataFrame, metric: str = 'test_mse') -> pd.DataFrame:
    """
    Get the best result for each dataset based on a metric.

    Args:
        df: DataFrame with results
        metric: Metric to minimize (default: 'test_mse')

    Returns:
        DataFrame with best result per dataset
    """
    if df.empty:
        return df

    # Group by dataset and find minimum metric
    idx = df.groupby('dataset_name')[metric].idxmin()
    return df.loc[idx]

def format_architecture(row: pd.Series, model_type: str) -> str:
    """
    Format architecture string based on model type.

    Args:
        row: DataFrame row
        model_type: 'mlp', 'siren', or 'kan'

    Returns:
        Formatted architecture string
    """
    if model_type == 'mlp':
        depth = row.get('depth', 'N/A')
        activation = row.get('activation', 'N/A')
        return f"depth={depth}, {activation}"
    elif model_type == 'siren':
        depth = row.get('depth', 'N/A')
        return f"depth={depth}, SIREN"
    elif model_type in ['kan', 'kan_pruning']:
        grid = row.get('grid_size', 'N/A')
        is_pruned = row.get('is_pruned', False)
        pruned_str = ", pruned" if is_pruned else ""
        return f"grid={grid}{pruned_str}"
    return "Unknown"

def format_scientific(value: float, precision: int = 2) -> str:
    """Format number in scientific notation."""
    if pd.isna(value):
        return "N/A"
    if value == 0:
        return "0.00"
    if abs(value) >= 0.01:
        return f"{value:.{precision}f}"
    return f"{value:.{precision}e}"

def create_latex_table(df: pd.DataFrame, caption: str, label: str,
                       column_format: Optional[str] = None) -> str:
    """
    Convert DataFrame to LaTeX table format.

    Args:
        df: DataFrame to convert
        caption: Table caption
        label: LaTeX label
        column_format: Optional column format (e.g., '|c|c|c|')

    Returns:
        LaTeX table string
    """
    if column_format is None:
        num_cols = len(df.columns)
        column_format = '|' + 'c|' * num_cols

    latex_str = df.to_latex(
        index=False,
        escape=False,
        column_format=column_format,
        caption=caption,
        label=label
    )

    return latex_str

def save_table(table_str: str, filename: str, output_dir: str = '.') -> None:
    """Save table string to file."""
    output_path = Path(output_dir) / filename
    with open(output_path, 'w') as f:
        f.write(table_str)
    print(f"Saved table to {output_path}")

def print_table(df: pd.DataFrame, title: str, tablefmt: str = 'grid') -> None:
    """
    Print formatted table to console.

    Args:
        df: DataFrame to print
        title: Table title
        tablefmt: Format for tabulate (default: 'grid')
    """
    print("\n" + "="*80)
    print(title)
    print("="*80)
    print(tabulate(df, headers='keys', tablefmt=tablefmt, showindex=False))
    print("="*80 + "\n")

def get_param_count(row: pd.Series) -> int:
    """Extract parameter count from row."""
    return row.get('num_params', 0)

def compare_models(results: Dict[str, pd.DataFrame],
                   dataset_names: List[str],
                   metric: str = 'test_mse') -> pd.DataFrame:
    """
    DEPRECATED: Use compare_models_from_checkpoints() for thesis-grade comparisons.

    Create comparison table across all model types using DataFrame data.
    This function compares "best overall" configurations across all training,
    which is NOT a fair iso-compute comparison.

    Args:
        results: Dictionary of results DataFrames
        dataset_names: List of dataset names to compare
        metric: Metric to compare

    Returns:
        Comparison DataFrame
    """
    comparison_data = []

    for dataset_name in dataset_names:
        row_data = {'Dataset': dataset_name}

        for model_type in ['mlp', 'siren', 'kan', 'kan_pruning']:
            df = results.get(model_type, pd.DataFrame())

            if df.empty:
                row_data[f'{model_type.upper()} {metric}'] = 'N/A'
                row_data[f'{model_type.upper()} arch'] = 'N/A'
                continue

            # Filter for this dataset
            dataset_df = df[df['dataset_name'] == dataset_name]

            if dataset_df.empty:
                row_data[f'{model_type.upper()} {metric}'] = 'N/A'
                row_data[f'{model_type.upper()} arch'] = 'N/A'
                continue

            # Get best result
            best_idx = dataset_df[metric].idxmin()
            best_row = dataset_df.loc[best_idx]

            row_data[f'{model_type.upper()} {metric}'] = format_scientific(best_row[metric])
            row_data[f'{model_type.upper()} arch'] = format_architecture(best_row, model_type)
            row_data[f'{model_type.upper()} params'] = int(best_row.get('num_params', 0))

        comparison_data.append(row_data)

    return pd.DataFrame(comparison_data)


def compare_models_from_checkpoints(checkpoint_metadata: Dict,
                                    dataset_names: List[str],
                                    checkpoint_type: str = 'final',
                                    include_pruned: bool = False) -> pd.DataFrame:
    """
    Create model comparison table using checkpoint metadata (RECOMMENDED FOR THESIS).

    This function provides scientifically rigorous comparisons:
    - 'iso_compute': Compare models at KAN interpolation threshold time (fair time-matched comparison)
    - 'final': Compare models after full training budget (best achievable performance)

    All metrics use dense_mse (evaluation on 10,000 samples), not sparse test set.

    Args:
        checkpoint_metadata: Dictionary from load_checkpoint_metadata()
        dataset_names: List of dataset names to compare
        checkpoint_type: 'iso_compute' (at threshold) or 'final' (after full training)
        include_pruned: Whether to include pruned KAN results

    Returns:
        DataFrame with columns: Dataset, MLP Dense MSE, MLP Arch, MLP Params, etc.
    """
    comparison_data = []

    # Map checkpoint type to actual checkpoint keys
    if checkpoint_type == 'iso_compute':
        mlp_siren_key = 'at_kan_threshold_time'
        kan_key = 'at_threshold'
    elif checkpoint_type == 'final':
        mlp_siren_key = 'final'
        kan_key = 'final'
    else:
        raise ValueError(f"Invalid checkpoint_type: {checkpoint_type}. Use 'iso_compute' or 'final'")

    for dataset_idx, dataset_name in enumerate(dataset_names):
        row_data = {'Dataset': dataset_name}

        # MLP
        if 'mlp' in checkpoint_metadata and dataset_idx in checkpoint_metadata['mlp']:
            if mlp_siren_key in checkpoint_metadata['mlp'][dataset_idx]:
                checkpoint = checkpoint_metadata['mlp'][dataset_idx][mlp_siren_key]
                row_data['MLP Dense MSE'] = format_scientific(checkpoint['dense_mse'])
                row_data['MLP Arch'] = f"depth={checkpoint['depth']}, {checkpoint['activation']}"
                row_data['MLP Params'] = int(checkpoint['num_params'])
            else:
                row_data['MLP Dense MSE'] = 'N/A'
                row_data['MLP Arch'] = 'N/A'
                row_data['MLP Params'] = 'N/A'
        else:
            row_data['MLP Dense MSE'] = 'N/A'
            row_data['MLP Arch'] = 'N/A'
            row_data['MLP Params'] = 'N/A'

        # SIREN
        if 'siren' in checkpoint_metadata and dataset_idx in checkpoint_metadata['siren']:
            if mlp_siren_key in checkpoint_metadata['siren'][dataset_idx]:
                checkpoint = checkpoint_metadata['siren'][dataset_idx][mlp_siren_key]
                row_data['SIREN Dense MSE'] = format_scientific(checkpoint['dense_mse'])
                row_data['SIREN Arch'] = f"depth={checkpoint['depth']}"
                row_data['SIREN Params'] = int(checkpoint['num_params'])
            else:
                row_data['SIREN Dense MSE'] = 'N/A'
                row_data['SIREN Arch'] = 'N/A'
                row_data['SIREN Params'] = 'N/A'
        else:
            row_data['SIREN Dense MSE'] = 'N/A'
            row_data['SIREN Arch'] = 'N/A'
            row_data['SIREN Params'] = 'N/A'

        # KAN (unpruned)
        if 'kan' in checkpoint_metadata and dataset_idx in checkpoint_metadata['kan']:
            if kan_key in checkpoint_metadata['kan'][dataset_idx]:
                checkpoint = checkpoint_metadata['kan'][dataset_idx][kan_key]
                row_data['KAN Dense MSE'] = format_scientific(checkpoint['dense_mse'])
                row_data['KAN Arch'] = f"grid={checkpoint['grid_size']}"
                row_data['KAN Params'] = int(checkpoint['num_params'])
            else:
                row_data['KAN Dense MSE'] = 'N/A'
                row_data['KAN Arch'] = 'N/A'
                row_data['KAN Params'] = 'N/A'
        else:
            row_data['KAN Dense MSE'] = 'N/A'
            row_data['KAN Arch'] = 'N/A'
            row_data['KAN Params'] = 'N/A'

        # KAN Pruning (optional)
        if include_pruned:
            if 'kan_pruning' in checkpoint_metadata and dataset_idx in checkpoint_metadata['kan_pruning']:
                if kan_key in checkpoint_metadata['kan_pruning'][dataset_idx]:
                    checkpoint = checkpoint_metadata['kan_pruning'][dataset_idx][kan_key]
                    row_data['KAN+Pruning Dense MSE'] = format_scientific(checkpoint['dense_mse'])
                    row_data['KAN+Pruning Arch'] = f"grid={checkpoint['grid_size']}"
                    row_data['KAN+Pruning Params'] = int(checkpoint['num_params'])
                else:
                    row_data['KAN+Pruning Dense MSE'] = 'N/A'
                    row_data['KAN+Pruning Arch'] = 'N/A'
                    row_data['KAN+Pruning Params'] = 'N/A'
            else:
                row_data['KAN+Pruning Dense MSE'] = 'N/A'
                row_data['KAN+Pruning Arch'] = 'N/A'
                row_data['KAN+Pruning Params'] = 'N/A'

        comparison_data.append(row_data)

    return pd.DataFrame(comparison_data)


def get_dataset_names(section_name: str) -> List[str]:
    """
    Get list of dataset names for a section.

    Args:
        section_name: Section name (e.g., 'section1_1', 'section1_2', 'section1_3')

    Returns:
        List of dataset names
    """
    if section_name == 'section1_1':
        return ['sin_freq1', 'sin_freq2', 'sin_freq3', 'sin_freq4', 'sin_freq5',
                'piecewise', 'sawtooth', 'polynomial', 'poisson_1d_highfreq']
    elif section_name == 'section1_2':
        return ['poisson_1d_sin', 'poisson_1d_poly', 'poisson_1d_highfreq']
    elif section_name == 'section1_3':
        return ['poisson_2d_sin', 'poisson_2d_poly', 'poisson_2d_highfreq', 'poisson_2d_spec']
    else:
        raise ValueError(f"Unknown section: {section_name}")
