"""
Scaling Law Analysis for Neural Network Architectures

Implements power-law fitting from Liu et al. (2024) KAN paper:
- Fit RMSE = A × N^(-α) where N = parameter count
- Compare α exponents across architectures
- Visualize empirical data vs. fitted curves
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Tuple, Optional
from scipy.optimize import curve_fit
import seaborn as sns
import importlib.util

# Import IO and Pareto modules
try:
    from . import io
    from .pareto_analysis import ParetoAnalyzer
except ImportError:
    spec = importlib.util.spec_from_file_location('io', Path(__file__).parent / 'io.py')
    io = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(io)
    from pareto_analysis import ParetoAnalyzer

# Set plotting style
sns.set_style("whitegrid")


def power_law(N: np.ndarray, A: float, alpha: float) -> np.ndarray:
    """
    Power law function: RMSE = A × N^(-α)

    Args:
        N: Parameter counts
        A: Scaling constant
        alpha: Exponent (higher = faster scaling)

    Returns:
        Predicted RMSE values
    """
    return A * np.power(N, -alpha)


class ScalingLawAnalyzer:
    """Analyzes and visualizes scaling laws"""

    def __init__(self, results: Dict, metadata: Dict):
        """
        Initialize analyzer

        Args:
            results: Results dictionary
            metadata: Metadata dictionary
        """
        self.results = results
        self.metadata = metadata
        self.pareto_analyzer = ParetoAnalyzer(results, metadata)

    def fit_scaling_law(self, params: np.ndarray, rmse: np.ndarray) -> Tuple[float, float, np.ndarray]:
        """
        Fit power law to data

        Args:
            params: Parameter counts (N)
            rmse: Test RMSE values

        Returns:
            (A, alpha, fitted_rmse)
        """
        # Filter out zeros and invalid values
        valid_mask = (params > 0) & (rmse > 0) & np.isfinite(rmse)
        params_valid = params[valid_mask]
        rmse_valid = rmse[valid_mask]

        if len(params_valid) < 2:
            return None, None, None

        try:
            # Fit power law
            popt, _ = curve_fit(
                power_law,
                params_valid,
                rmse_valid,
                p0=[1.0, 0.5],  # Initial guess
                bounds=([0, 0], [np.inf, 2])  # Reasonable bounds
            )
            A, alpha = popt

            # Generate fitted curve
            fitted_rmse = power_law(params_valid, A, alpha)

            return A, alpha, fitted_rmse

        except (RuntimeError, ValueError) as e:
            print(f"  Warning: Could not fit scaling law: {e}")
            return None, None, None

    def plot_scaling_laws(self, dataset_idx: int, output_path: str, dataset_name: str = None):
        """
        Create scaling law plot with fitted curves

        Args:
            dataset_idx: Dataset index
            output_path: Path to save plot
            dataset_name: Optional dataset name
        """
        # Get metrics
        df = self.pareto_analyzer.extract_model_metrics(dataset_idx)
        if df.empty:
            print(f"  Warning: No data for dataset {dataset_idx}")
            return

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 7))

        # Colors for architectures
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

        # Process each architecture
        for arch in df['architecture'].unique():
            arch_df = df[df['architecture'] == arch].sort_values('params')

            if len(arch_df) < 2:
                continue

            params = arch_df['params'].values
            rmse = arch_df['test_rmse'].values

            # Plot empirical data
            ax.scatter(params, rmse,
                      c=colors.get(arch, 'gray'),
                      marker=markers.get(arch, 'o'),
                      s=100, alpha=0.7,
                      label=f'{arch} (data)',
                      zorder=10)

            # Fit and plot scaling law
            A, alpha, fitted = self.fit_scaling_law(params, rmse)

            if A is not None and alpha is not None:
                # Plot fitted curve
                params_sorted = np.sort(params[params > 0])
                fitted_curve = power_law(params_sorted, A, alpha)

                ax.plot(params_sorted, fitted_curve,
                       c=colors.get(arch, 'gray'),
                       linestyle='--',
                       linewidth=2,
                       alpha=0.8,
                       label=f'{arch} (α={alpha:.3f})',
                       zorder=5)

        # Set log scale
        ax.set_xscale('log')
        ax.set_yscale('log')

        # Labels
        ax.set_xlabel('Parameters (N)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Test RMSE', fontsize=13, fontweight='bold')
        title = f'Scaling Laws: {dataset_name}' if dataset_name else f'Scaling Laws (Dataset {dataset_idx})'
        ax.set_title(title, fontsize=14, fontweight='bold')

        # Legend
        ax.legend(loc='best', frameon=True, shadow=True, fontsize=9)

        # Grid
        ax.grid(True, which='both', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Saved scaling laws: {Path(output_path).name}")

    def generate_scaling_summary(self, dataset_idx: int, output_path: str) -> pd.DataFrame:
        """
        Generate summary table of scaling law parameters

        Args:
            dataset_idx: Dataset index
            output_path: Path to save CSV

        Returns:
            DataFrame with scaling law parameters
        """
        df = self.pareto_analyzer.extract_model_metrics(dataset_idx)
        if df.empty:
            return pd.DataFrame()

        summary_rows = []

        for arch in df['architecture'].unique():
            arch_df = df[df['architecture'] == arch]
            params = arch_df['params'].values
            rmse = arch_df['test_rmse'].values

            A, alpha, _ = self.fit_scaling_law(params, rmse)

            if A is not None:
                summary_rows.append({
                    'Architecture': arch,
                    'A (constant)': f"{A:.4e}",
                    'α (exponent)': f"{alpha:.4f}",
                    'Interpretation': 'Higher α = faster scaling with parameters',
                    'Num configs': len(arch_df)
                })

        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(output_path, index=False)

        print(f"  Saved scaling summary: {Path(output_path).name}")
        return summary_df

    def analyze_all_datasets(self, output_dir: str):
        """
        Run scaling law analysis for all datasets

        Args:
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        # Determine number of datasets
        num_datasets = 0
        model_types = ['mlp', 'siren', 'kan', 'kan_pruning']
        for model_type in model_types:
            if model_type in self.results:
                num_datasets = max(num_datasets, len(self.results[model_type]))

        print(f"\nGenerating scaling law analysis for {num_datasets} datasets...")

        for idx in range(num_datasets):
            print(f"\nDataset {idx}:")

            # Scaling law plot
            plot_path = output_path / f"scaling_laws_{idx}.png"
            self.plot_scaling_laws(idx, str(plot_path))

            # Scaling summary table
            table_path = output_path / f"scaling_summary_{idx}.csv"
            self.generate_scaling_summary(idx, str(table_path))
