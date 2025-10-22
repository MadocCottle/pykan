"""
Pareto Frontier Analysis for Model Comparison

Implements methodology from Liu et al. (2024) KAN paper:
- Test RMSE vs. parameter count visualization
- Pareto frontier identification
- Best model selection per architecture
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple, Optional
import seaborn as sns
import importlib.util

# Import IO module
try:
    from . import io
except ImportError:
    spec = importlib.util.spec_from_file_location('io', Path(__file__).parent / 'io.py')
    io = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(io)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 7)
plt.rcParams['font.size'] = 11


def count_parameters(results: Dict, model_type: str, config_key: str, dataset_idx: int = 0) -> int:
    """
    Estimate parameter count for a model configuration

    For now, returns placeholder values. In production, this should:
    1. Load actual model checkpoints and count parameters
    2. Or compute from architecture specs (depth, width, grid)

    Args:
        results: Results dictionary
        model_type: 'mlp', 'siren', 'kan', or 'kan_pruning'
        config_key: Configuration identifier (e.g., 'depth_3', 'grid_20')
        dataset_idx: Dataset index

    Returns:
        Parameter count (integer)
    """
    # TODO: Implement actual parameter counting
    # For now, return estimates based on architecture

    # Convert config_key to string if it's not
    config_str = str(config_key)

    if model_type == 'mlp' or model_type == 'siren':
        # Parse depth from config_key
        if 'depth_' in config_str:
            try:
                depth = int(config_str.split('_')[1])
                width = 64  # Default width assumption
                # Simple estimate: input_dim + (width * width) * (depth-1) + width * output_dim
                # Assuming 1D input/output
                params = 1 * width + (width * width) * (depth - 1) + width * 1
                return params
            except:
                pass
        return 5000  # Fallback

    elif model_type == 'kan':
        # Parse grid from config_key
        if 'grid_' in config_str:
            try:
                grid = int(config_str.split('_')[1])
                # KAN params scale with grid size and number of edges
                # Rough estimate: grid * num_edges * spline_order
                params = grid * 10 * 3  # Placeholder
                return params
            except:
                pass
        return 10000  # Fallback

    elif model_type == 'kan_pruning':
        if 'pruned' in config_str:
            # Pruned model - much smaller
            return 3000  # Estimate
        else:
            # Same as unpruned KAN
            if 'grid_' in config_str:
                try:
                    grid = int(config_str.split('_')[1])
                    return grid * 10 * 3
                except:
                    pass
            return 10000

    return 1000  # Ultimate fallback


class ParetoAnalyzer:
    """Analyzes Pareto frontiers for model comparison"""

    def __init__(self, results: Dict, metadata: Dict):
        """
        Initialize analyzer

        Args:
            results: Results dictionary from training
            metadata: Metadata dictionary
        """
        self.results = results
        self.metadata = metadata
        self.model_types = ['mlp', 'siren', 'kan', 'kan_pruning']

    def extract_model_metrics(self, dataset_idx: int) -> pd.DataFrame:
        """
        Extract test RMSE and parameter counts for all models

        Args:
            dataset_idx: Dataset index

        Returns:
            DataFrame with columns: [architecture, config, test_rmse, params]
        """
        rows = []

        for model_type in self.model_types:
            if model_type not in self.results:
                continue

            model_results = self.results[model_type]
            if dataset_idx not in model_results:
                continue

            configs = model_results[dataset_idx]

            for config_key, config_data in configs.items():
                # Skip if not a dict (could be intermediate results)
                if not isinstance(config_data, dict):
                    continue

                # Extract test RMSE
                test_rmse = None
                if isinstance(config_data, dict):
                    # Try different possible structures
                    if 'test' in config_data and isinstance(config_data['test'], list):
                        test_rmse = config_data['test'][-1]  # Final test MSE
                    elif isinstance(config_data, dict):
                        # Might have sub-configs (e.g., activation types)
                        for subconfig_key, subconfig_data in config_data.items():
                            if isinstance(subconfig_data, dict) and 'test' in subconfig_data:
                                if isinstance(subconfig_data['test'], list):
                                    test_val = subconfig_data['test'][-1]
                                    full_config = f"{config_key}_{subconfig_key}"
                                    params = count_parameters(self.results, model_type, full_config, dataset_idx)
                                    rows.append({
                                        'architecture': model_type.upper().replace('_', '+'),
                                        'config': full_config,
                                        'test_rmse': np.sqrt(test_val) if test_val > 0 else test_val,
                                        'params': params
                                    })
                        continue

                if test_rmse is not None:
                    params = count_parameters(self.results, model_type, config_key, dataset_idx)
                    rows.append({
                        'architecture': model_type.upper().replace('_', '+'),
                        'config': config_key,
                        'test_rmse': np.sqrt(test_rmse) if test_rmse > 0 else test_rmse,
                        'params': params
                    })

        return pd.DataFrame(rows)

    def identify_pareto_frontier(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify Pareto-optimal models (minimize both RMSE and parameters)

        Args:
            df: DataFrame with test_rmse and params columns

        Returns:
            DataFrame with additional 'pareto_optimal' column
        """
        df = df.copy()
        df['pareto_optimal'] = False

        # For each point, check if any other point dominates it
        for idx in df.index:
            rmse = df.loc[idx, 'test_rmse']
            params = df.loc[idx, 'params']

            # Check if this point is dominated by any other
            dominated = False
            for other_idx in df.index:
                if other_idx == idx:
                    continue
                other_rmse = df.loc[other_idx, 'test_rmse']
                other_params = df.loc[other_idx, 'params']

                # Other point dominates if it's better or equal on both dimensions
                # and strictly better on at least one
                if (other_rmse <= rmse and other_params <= params and
                    (other_rmse < rmse or other_params < params)):
                    dominated = True
                    break

            if not dominated:
                df.loc[idx, 'pareto_optimal'] = True

        return df

    def plot_pareto_frontier(self, dataset_idx: int, output_path: str, dataset_name: str = None):
        """
        Create Pareto frontier plot (log-log scale)

        Args:
            dataset_idx: Dataset index
            output_path: Path to save plot
            dataset_name: Optional dataset name for title
        """
        df = self.extract_model_metrics(dataset_idx)
        if df.empty:
            print(f"  Warning: No data for dataset {dataset_idx}")
            return

        df = self.identify_pareto_frontier(df)

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 7))

        # Color map for architectures
        colors = {
            'MLP': '#1f77b4',
            'SIREN': '#ff7f0e',
            'KAN': '#2ca02c',
            'KAN+PRUNING': '#d62728'
        }

        markers = {
            'MLP': 'o',
            'SIREN': 's',
            'KAN': '^',
            'KAN+PRUNING': 'D'
        }

        # Plot each architecture
        for arch in df['architecture'].unique():
            arch_df = df[df['architecture'] == arch]
            pareto_df = arch_df[arch_df['pareto_optimal']]
            non_pareto_df = arch_df[~arch_df['pareto_optimal']]

            # Plot non-Pareto points (faded)
            if len(non_pareto_df) > 0:
                ax.scatter(non_pareto_df['params'], non_pareto_df['test_rmse'],
                          c=colors.get(arch, 'gray'), marker=markers.get(arch, 'o'),
                          s=100, alpha=0.3, label=None)

            # Plot Pareto points (highlighted)
            if len(pareto_df) > 0:
                ax.scatter(pareto_df['params'], pareto_df['test_rmse'],
                          c=colors.get(arch, 'gray'), marker=markers.get(arch, 'o'),
                          s=150, alpha=0.9, edgecolors='black', linewidths=2,
                          label=f"{arch} (Pareto)" if len(pareto_df) > 0 else arch,
                          zorder=10)

                # Connect Pareto points within architecture
                pareto_sorted = pareto_df.sort_values('params')
                if len(pareto_sorted) > 1:
                    ax.plot(pareto_sorted['params'], pareto_sorted['test_rmse'],
                           c=colors.get(arch, 'gray'), alpha=0.5, linestyle='--',
                           linewidth=1.5, zorder=5)

        # Set log scale
        ax.set_xscale('log')
        ax.set_yscale('log')

        # Labels and title
        ax.set_xlabel('Parameters', fontsize=13, fontweight='bold')
        ax.set_ylabel('Test RMSE', fontsize=13, fontweight='bold')
        title = f'Pareto Frontier: {dataset_name}' if dataset_name else f'Pareto Frontier (Dataset {dataset_idx})'
        ax.set_title(title, fontsize=14, fontweight='bold')

        # Legend
        ax.legend(loc='best', frameon=True, shadow=True, fontsize=10)

        # Grid
        ax.grid(True, which='both', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Saved Pareto frontier: {Path(output_path).name}")

    def generate_best_models_table(self, dataset_idx: int, output_path: str) -> pd.DataFrame:
        """
        Generate table of best (Pareto-optimal) models

        Args:
            dataset_idx: Dataset index
            output_path: Path to save CSV

        Returns:
            DataFrame of best models
        """
        df = self.extract_model_metrics(dataset_idx)
        if df.empty:
            return pd.DataFrame()

        df = self.identify_pareto_frontier(df)

        # Get Pareto-optimal models
        best_df = df[df['pareto_optimal']].copy()

        # If no Pareto models, take top 3 per architecture
        if len(best_df) == 0:
            best_df = df.groupby('architecture').apply(
                lambda x: x.nsmallest(3, 'test_rmse')
            ).reset_index(drop=True)

        # Sort by RMSE
        best_df = best_df.sort_values('test_rmse')

        # Format for display
        best_df['test_rmse_fmt'] = best_df['test_rmse'].apply(lambda x: f"{x:.2e}")

        # Save
        display_cols = ['architecture', 'config', 'test_rmse_fmt', 'params', 'pareto_optimal']
        best_df[display_cols].to_csv(output_path, index=False)

        print(f"  Saved best models table: {Path(output_path).name}")
        return best_df

    def analyze_all_datasets(self, output_dir: str):
        """
        Run Pareto analysis for all datasets

        Args:
            output_dir: Output directory path
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        # Determine number of datasets
        num_datasets = 0
        for model_type in self.model_types:
            if model_type in self.results:
                num_datasets = max(num_datasets, len(self.results[model_type]))

        print(f"\nGenerating Pareto analysis for {num_datasets} datasets...")

        for idx in range(num_datasets):
            print(f"\nDataset {idx}:")

            # Pareto frontier plot
            plot_path = output_path / f"pareto_frontier_{idx}.png"
            self.plot_pareto_frontier(idx, str(plot_path))

            # Best models table
            table_path = output_path / f"best_models_{idx}.csv"
            self.generate_best_models_table(idx, str(table_path))
