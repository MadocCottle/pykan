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
    Create comparison table across all model types.

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
