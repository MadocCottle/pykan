"""
Utility functions for Section 2 table generation

Adapted for Section 2's data structure:
- Section 2.1: Optimizer comparison (adam, lbfgs, lm)
- Section 2.2: Adaptive density (baseline, adaptive_only, adaptive_regular)
- Section 2.3: Merge_KAN (summary, experts, selected_experts, grid_history)
"""
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tabulate import tabulate


def load_section2_results(section_name: str, timestamp: Optional[str] = None,
                          results_dir: str = '../results') -> Dict[str, pd.DataFrame]:
    """
    Load Section 2 results for a given section.

    Args:
        section_name: Name of section (e.g., 'section2_1', 'section2_2', 'section2_3')
        timestamp: Specific timestamp to load, or None for latest
        results_dir: Path to results directory

    Returns:
        Dictionary with keys depending on section:
        - Section 2.1: {'adam', 'lbfgs', 'lm'}
        - Section 2.2: {'baseline', 'adaptive_only', 'adaptive_regular'}
        - Section 2.3: {'summary', 'experts', 'selected_experts', 'grid_history'}
    """
    # Extract section number (2_1 -> 1, 2_2 -> 2, etc.)
    parts = section_name.split('_')
    if len(parts) >= 2:
        sec_num = parts[1]
    else:
        sec_num = '1'

    results_path = Path(__file__).parent / results_dir / f'sec{sec_num}_results'

    if not results_path.exists():
        print(f"Warning: Results directory not found: {results_path}")
        return {}

    # Determine model types based on section
    if 'section2_1' in section_name:
        model_types = ['adam', 'lbfgs', 'lm']
    elif 'section2_2' in section_name:
        model_types = ['baseline', 'adaptive_only', 'adaptive_regular']
    elif 'section2_3' in section_name:
        model_types = ['summary', 'experts', 'selected_experts', 'grid_history']
    else:
        print(f"Warning: Unknown section {section_name}")
        return {}

    results = {}

    # If timestamp provided, use it; otherwise find latest
    if timestamp is None:
        # Find latest timestamp
        for model_type in model_types:
            pattern = f"{section_name}_*_{model_type}.pkl"
            files = sorted(results_path.glob(pattern))

            if files:
                latest_file = files[-1]
                with open(latest_file, 'rb') as f:
                    results[model_type] = pickle.load(f)
                print(f"Loaded {model_type}: {latest_file.name}")
            else:
                print(f"Warning: No files found for {model_type} in {section_name}")
                results[model_type] = pd.DataFrame()
    else:
        # Load specific timestamp
        for model_type in model_types:
            # Try with epochs first
            pattern = f"{section_name}_{timestamp}_e*_{model_type}.pkl"
            files = list(results_path.glob(pattern))

            if not files:
                # Try without epochs
                file_path = results_path / f"{section_name}_{timestamp}_{model_type}.pkl"
                if file_path.exists():
                    files = [file_path]

            if files:
                with open(files[0], 'rb') as f:
                    results[model_type] = pickle.load(f)
                print(f"Loaded {model_type}: {files[0].name}")
            else:
                print(f"Warning: No file found for {model_type} with timestamp {timestamp}")
                results[model_type] = pd.DataFrame()

    return results


def load_checkpoint_metadata(section_name: str, timestamp: Optional[str] = None,
                             results_dir: str = '../results') -> Optional[Dict]:
    """
    Load checkpoint metadata for Section 2.

    For Section 2.1: {'adam', 'lbfgs', 'lm'} -> {dataset_idx: {'at_threshold': {...}, 'final': {...}}}
    For Section 2.2: {'baseline', 'adaptive_only', 'adaptive_regular'} -> similar structure

    Args:
        section_name: Name of section (e.g., 'section2_1', 'section2_2')
        timestamp: Specific timestamp to load, or None for latest
        results_dir: Path to results directory

    Returns:
        Dictionary with checkpoint metadata, or None if not found
    """
    parts = section_name.split('_')
    if len(parts) >= 2:
        sec_num = parts[1]
    else:
        sec_num = '1'

    results_path = Path(__file__).parent / results_dir / f'sec{sec_num}_results'

    if timestamp is None:
        # Find latest checkpoint metadata file
        pattern = f"{section_name}_*_checkpoint_metadata.pkl"
        files = sorted(results_path.glob(pattern))
        if not files:
            print(f"Warning: No checkpoint metadata files found for {section_name}")
            return None
        metadata_file = files[-1]
    else:
        # Try with epochs
        pattern = f"{section_name}_{timestamp}_e*_checkpoint_metadata.pkl"
        files = list(results_path.glob(pattern))

        if not files:
            # Try without epochs
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


def get_best_per_dataset(df: pd.DataFrame, metric: str = 'dense_mse') -> pd.DataFrame:
    """
    Get the best (minimum) metric value for each dataset.

    Args:
        df: DataFrame with results
        metric: Metric to minimize (default: 'dense_mse')

    Returns:
        DataFrame with best result per dataset
    """
    if df.empty:
        return df

    # Group by dataset and find minimum metric
    if 'dataset_idx' in df.columns:
        idx = df.groupby('dataset_idx')[metric].idxmin()
        return df.loc[idx].sort_values('dataset_idx')
    elif 'dataset_name' in df.columns:
        idx = df.groupby('dataset_name')[metric].idxmin()
        return df.loc[idx]
    else:
        print("Warning: No dataset column found")
        return df


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
        column_format: Optional column format (e.g., '|l|c|c|c|')

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
        tablefmt: Table format (default: 'grid')
    """
    print(f"\n{'=' * 80}")
    print(f"{title:^80}")
    print(f"{'=' * 80}\n")
    print(tabulate(df, headers='keys', tablefmt=tablefmt, showindex=False))
    print()


def get_dataset_names() -> List[str]:
    """Get standard dataset names for Section 2 (2D Poisson PDE)."""
    return [
        '2D Sin',
        '2D Polynomial',
        '2D High-freq',
        '2D Special'
    ]


def compute_improvement(baseline: float, improved: float) -> float:
    """
    Compute percentage improvement.

    Args:
        baseline: Baseline value
        improved: Improved value

    Returns:
        Percentage improvement (positive if improved < baseline)
    """
    if baseline == 0 or pd.isna(baseline) or pd.isna(improved):
        return 0.0
    return ((baseline - improved) / baseline) * 100.0


def identify_winner(results_dict: Dict[str, float]) -> str:
    """
    Identify the winner (minimum value) from a dictionary of results.

    Args:
        results_dict: Dictionary mapping approach names to metric values

    Returns:
        Name of winning approach
    """
    if not results_dict:
        return "N/A"

    # Filter out None/NaN values
    valid_results = {k: v for k, v in results_dict.items()
                    if v is not None and not pd.isna(v)}

    if not valid_results:
        return "N/A"

    return min(valid_results, key=valid_results.get)
