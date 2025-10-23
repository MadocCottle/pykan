"""
Utility functions for table generation - Simplified for actual section1_x.py outputs
"""
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from tabulate import tabulate


def load_checkpoint_metadata(section_name: str, timestamp: Optional[str] = None,
                            results_dir: str = '../results') -> Optional[Dict]:
    """
    Load checkpoint metadata for a given section.

    Checkpoint metadata structure:
    {
        'mlp': {dataset_idx: {'final': {...}, 'at_kan_threshold_time': {...}}},
        'siren': {dataset_idx: {'final': {...}, 'at_kan_threshold_time': {...}}},
        'kan': {dataset_idx: {'final': {...}, 'at_threshold': {...}}},
        'kan_pruning': {dataset_idx: {'final': {...}, 'at_threshold': {...}}}
    }

    Note: 'at_kan_threshold_time' may not exist in sections 1_2 and 1_3

    Args:
        section_name: Section name (e.g., 'section1_1', 'section1_2', 'section1_3')
        timestamp: Specific timestamp to load, or None for latest
        results_dir: Path to results directory

    Returns:
        Dictionary of checkpoint metadata, or None if not found
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
        pattern_with_epochs = f"{section_name}_{timestamp}_e*_checkpoint_metadata.pkl"
        files = list(results_path.glob(pattern_with_epochs))

        if not files:
            # Fallback without epochs
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


def format_scientific(value: float, precision: int = 2) -> str:
    """Format number in scientific notation."""
    if pd.isna(value) or value is None:
        return "N/A"
    if value == 0:
        return "0.00"
    if abs(value) >= 0.01:
        return f"{value:.{precision}f}"
    return f"{value:.{precision}e}"


def compare_models_from_checkpoints(checkpoint_metadata: Dict,
                                    dataset_names: List[str],
                                    checkpoint_type: str = 'final',
                                    include_pruned: bool = False) -> pd.DataFrame:
    """
    Create model comparison table using checkpoint metadata.

    Args:
        checkpoint_metadata: Dictionary from load_checkpoint_metadata()
        dataset_names: List of dataset names to compare
        checkpoint_type: 'iso_compute' or 'final'
        include_pruned: Whether to include pruned KAN results

    Returns:
        DataFrame with comparison results
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
                cp = checkpoint_metadata['mlp'][dataset_idx][mlp_siren_key]
                row_data['MLP Dense MSE'] = format_scientific(cp['dense_mse'])
                row_data['MLP Config'] = f"depth={cp['depth']}, {cp['activation']}"
                row_data['MLP Params'] = int(cp['num_params'])
            else:
                row_data['MLP Dense MSE'] = 'N/A'
                row_data['MLP Config'] = 'N/A'
                row_data['MLP Params'] = 'N/A'
        else:
            row_data['MLP Dense MSE'] = 'N/A'
            row_data['MLP Config'] = 'N/A'
            row_data['MLP Params'] = 'N/A'

        # SIREN
        if 'siren' in checkpoint_metadata and dataset_idx in checkpoint_metadata['siren']:
            if mlp_siren_key in checkpoint_metadata['siren'][dataset_idx]:
                cp = checkpoint_metadata['siren'][dataset_idx][mlp_siren_key]
                row_data['SIREN Dense MSE'] = format_scientific(cp['dense_mse'])
                row_data['SIREN Config'] = f"depth={cp['depth']}"
                row_data['SIREN Params'] = int(cp['num_params'])
            else:
                row_data['SIREN Dense MSE'] = 'N/A'
                row_data['SIREN Config'] = 'N/A'
                row_data['SIREN Params'] = 'N/A'
        else:
            row_data['SIREN Dense MSE'] = 'N/A'
            row_data['SIREN Config'] = 'N/A'
            row_data['SIREN Params'] = 'N/A'

        # KAN
        if 'kan' in checkpoint_metadata and dataset_idx in checkpoint_metadata['kan']:
            if kan_key in checkpoint_metadata['kan'][dataset_idx]:
                cp = checkpoint_metadata['kan'][dataset_idx][kan_key]
                row_data['KAN Dense MSE'] = format_scientific(cp['dense_mse'])
                row_data['KAN Config'] = f"grid={cp['grid_size']}"
                row_data['KAN Params'] = int(cp['num_params'])
            else:
                row_data['KAN Dense MSE'] = 'N/A'
                row_data['KAN Config'] = 'N/A'
                row_data['KAN Params'] = 'N/A'
        else:
            row_data['KAN Dense MSE'] = 'N/A'
            row_data['KAN Config'] = 'N/A'
            row_data['KAN Params'] = 'N/A'

        # KAN Pruning (optional)
        if include_pruned:
            if 'kan_pruning' in checkpoint_metadata and dataset_idx in checkpoint_metadata['kan_pruning']:
                if kan_key in checkpoint_metadata['kan_pruning'][dataset_idx]:
                    cp = checkpoint_metadata['kan_pruning'][dataset_idx][kan_key]
                    row_data['KAN+Prune Dense MSE'] = format_scientific(cp['dense_mse'])
                    row_data['KAN+Prune Config'] = f"grid={cp['grid_size']}"
                    row_data['KAN+Prune Params'] = int(cp['num_params'])
                else:
                    row_data['KAN+Prune Dense MSE'] = 'N/A'
                    row_data['KAN+Prune Config'] = 'N/A'
                    row_data['KAN+Prune Params'] = 'N/A'
            else:
                row_data['KAN+Prune Dense MSE'] = 'N/A'
                row_data['KAN+Prune Config'] = 'N/A'
                row_data['KAN+Prune Params'] = 'N/A'

        comparison_data.append(row_data)

    return pd.DataFrame(comparison_data)


def print_table(df: pd.DataFrame, title: str, tablefmt: str = 'grid') -> None:
    """Print formatted table to console."""
    print("\n" + "="*80)
    print(title)
    print("="*80)
    print(tabulate(df, headers='keys', tablefmt=tablefmt, showindex=False))
    print("="*80 + "\n")


def create_latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    """Convert DataFrame to LaTeX table format."""
    num_cols = len(df.columns)
    column_format = '|' + 'l|' + 'c|' * (num_cols - 1)

    latex_str = df.to_latex(
        index=False,
        escape=False,
        column_format=column_format,
        caption=caption,
        label=label
    )

    return latex_str


def save_table(table_str: str, filename: str) -> None:
    """Save table string to file in current directory."""
    output_path = Path(__file__).parent / filename
    with open(output_path, 'w') as f:
        f.write(table_str)
    print(f"Saved: {filename}")
