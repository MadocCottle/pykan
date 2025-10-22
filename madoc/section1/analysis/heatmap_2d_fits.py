"""
2D Function Heatmap Comparison Script

This script creates detailed heatmap visualizations for 2D equations:
- Side-by-side heatmaps of true function, NN prediction, and error
- Difference/error heatmaps showing where models perform poorly
- Cross-section comparisons at various points
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import torch
from typing import Dict, Optional, Callable, Union

# Add section1 directory to path for utils imports
sys.path.insert(0, str(Path(__file__).parent.parent))
# Add pykan root to path for KAN imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from kan import KAN
from utils import data_funcs as dfs

# Import IO module
try:
    from . import io
except ImportError:
    # Allow running as script (not as package)
    import io as io_module
    io = io_module

# Set style
sns.set_style("white")


class Heatmap2DAnalyzer:
    """Analyzes 2D function fitting with detailed heatmaps"""

    def __init__(self, results_path: Union[str, Path], models_dir: Optional[str] = None):
        """
        Initialize analyzer

        Args:
            results_path: Either:
                - Path to results file (.pkl or .json)
                - Section ID (e.g., 'section1_3') to auto-load latest results
            models_dir: Optional path to saved models. If None, will auto-discover.
        """
        # Load results
        results_path_str = str(results_path)
        if results_path_str in io.SECTIONS:
            # Section ID - load latest
            self.results, _, auto_models_dir = io.load_run(results_path_str)
            self.results_path = results_path_str
            self.models_dir = Path(models_dir) if models_dir else (Path(auto_models_dir) if auto_models_dir else None)
        else:
            # Direct path - manual load
            import pickle
            self.results_path = Path(results_path)
            with open(self.results_path, 'rb') as f:
                self.results = pickle.load(f)
            self.models_dir = Path(models_dir) if models_dir else None

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 2D function definitions
        self.function_map = {
            0: ('2D Sin (π²)', dfs.f_poisson_2d_sin),
            1: ('2D Polynomial', dfs.f_poisson_2d_poly),
            2: ('2D High-freq', dfs.f_poisson_2d_highfreq),
            3: ('2D Special', dfs.f_poisson_2d_spec),
        }

    def create_comparison_heatmaps(self, dataset_idx: int, true_func: Callable,
                                  func_name: str, model_type: str = 'kan',
                                  resolution: int = 100, output_path: Optional[str] = None):
        """
        Create detailed heatmap comparison for a 2D function

        Args:
            dataset_idx: Dataset index
            true_func: True function
            func_name: Function name
            model_type: Model type to analyze
            resolution: Grid resolution
            output_path: Optional save path
        """
        # Generate grid
        x1 = np.linspace(0, 1, resolution)
        x2 = np.linspace(0, 1, resolution)
        X1, X2 = np.meshgrid(x1, x2)

        x_test = torch.tensor(
            np.stack([X1.flatten(), X2.flatten()], axis=1),
            dtype=torch.float32
        ).to(self.device)

        # Get true values
        y_true = true_func(x_test).cpu().numpy().reshape(resolution, resolution)

        # Get predictions
        # Try both integer and string keys for compatibility (PKL uses int, JSON uses str)
        dataset_key = dataset_idx if dataset_idx in self.results.get('kan', {}) else str(dataset_idx)
        y_pred = None

        if model_type in self.results and dataset_key in self.results[model_type]:
            predictions = self._get_predictions(model_type, dataset_key, x_test)
            if predictions is not None:
                y_pred = predictions.reshape(resolution, resolution)

        # Create figure
        if y_pred is not None:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))

            # Row 1: Heatmaps
            # True function
            im1 = axes[0, 0].imshow(y_true, cmap='viridis', aspect='auto',
                                    extent=[0, 1, 0, 1], origin='lower')
            axes[0, 0].set_title('True Function')
            axes[0, 0].set_xlabel('x₁')
            axes[0, 0].set_ylabel('x₂')
            plt.colorbar(im1, ax=axes[0, 0])

            # Prediction
            im2 = axes[0, 1].imshow(y_pred, cmap='viridis', aspect='auto',
                                    extent=[0, 1, 0, 1], origin='lower')
            axes[0, 1].set_title(f'{model_type.upper()} Prediction')
            axes[0, 1].set_xlabel('x₁')
            axes[0, 1].set_ylabel('x₂')
            plt.colorbar(im2, ax=axes[0, 1])

            # Absolute error
            error = np.abs(y_true - y_pred)
            im3 = axes[0, 2].imshow(error, cmap='hot', aspect='auto',
                                    extent=[0, 1, 0, 1], origin='lower')
            axes[0, 2].set_title('Absolute Error')
            axes[0, 2].set_xlabel('x₁')
            axes[0, 2].set_ylabel('x₂')
            plt.colorbar(im3, ax=axes[0, 2])

            # Row 2: Analysis plots
            # Signed error
            signed_error = y_true - y_pred
            im4 = axes[1, 0].imshow(signed_error, cmap='RdBu_r', aspect='auto',
                                    extent=[0, 1, 0, 1], origin='lower',
                                    vmin=-np.max(np.abs(signed_error)),
                                    vmax=np.max(np.abs(signed_error)))
            axes[1, 0].set_title('Signed Error (True - Pred)')
            axes[1, 0].set_xlabel('x₁')
            axes[1, 0].set_ylabel('x₂')
            plt.colorbar(im4, ax=axes[1, 0])

            # Relative error (%)
            relative_error = np.abs(signed_error) / (np.abs(y_true) + 1e-10) * 100
            relative_error = np.clip(relative_error, 0, 100)  # Cap at 100%
            im5 = axes[1, 1].imshow(relative_error, cmap='YlOrRd', aspect='auto',
                                    extent=[0, 1, 0, 1], origin='lower')
            axes[1, 1].set_title('Relative Error (%)')
            axes[1, 1].set_xlabel('x₁')
            axes[1, 1].set_ylabel('x₂')
            plt.colorbar(im5, ax=axes[1, 1])

            # Error histogram
            axes[1, 2].hist(error.flatten(), bins=50, alpha=0.7, edgecolor='black')
            axes[1, 2].set_xlabel('Absolute Error')
            axes[1, 2].set_ylabel('Frequency')
            axes[1, 2].set_title('Error Distribution')
            axes[1, 2].axvline(error.mean(), color='red', linestyle='--',
                             label=f'Mean: {error.mean():.6f}')
            axes[1, 2].axvline(np.median(error), color='blue', linestyle='--',
                             label=f'Median: {np.median(error):.6f}')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)

            # Calculate metrics
            mse = np.mean(error ** 2)
            mae = np.mean(error)
            max_error = np.max(error)

            fig.suptitle(f'{func_name} - {model_type.upper()}\n'
                        f'MSE: {mse:.6f} | MAE: {mae:.6f} | Max Error: {max_error:.6f}',
                        fontsize=14)

            plt.tight_layout()

            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')

            return fig
        else:
            print(f"No predictions available for {model_type} on dataset {dataset_idx}")
            return None

    def create_cross_section_comparison(self, dataset_idx: int, true_func: Callable,
                                       func_name: str, output_path: Optional[str] = None):
        """
        Create cross-section comparisons at x₁=0.25, 0.5, 0.75 and x₂=0.25, 0.5, 0.75

        Args:
            dataset_idx: Dataset index
            true_func: True function
            func_name: Function name
            output_path: Optional save path
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        cross_sections_x1 = [0.25, 0.5, 0.75]
        cross_sections_x2 = [0.25, 0.5, 0.75]

        model_types = ['mlp', 'siren', 'kan', 'kan_pruning']
        colors = {'mlp': 'C0', 'siren': 'C1', 'kan': 'C2', 'kan_pruning': 'C3'}
        labels = {'mlp': 'MLP', 'siren': 'SIREN', 'kan': 'KAN', 'kan_pruning': 'KAN Pruning'}

        resolution = 200

        # Row 1: Fix x₁, vary x₂
        for idx, x1_fixed in enumerate(cross_sections_x1):
            ax = axes[0, idx]

            x2_range = np.linspace(0, 1, resolution)
            x1_vals = np.full_like(x2_range, x1_fixed)
            x_test = torch.tensor(
                np.stack([x1_vals, x2_range], axis=1),
                dtype=torch.float32
            ).to(self.device)

            # True function
            y_true = true_func(x_test).cpu().numpy()
            ax.plot(x2_range, y_true, 'k--', linewidth=2, label='True', alpha=0.8)

            # Model predictions
            for model_type in model_types:
                # Try both integer and string keys for compatibility (PKL uses int, JSON uses str)
                dataset_key = dataset_idx if dataset_idx in self.results.get('kan', {}) else str(dataset_idx)
                if model_type in self.results and dataset_key in self.results[model_type]:
                    y_pred = self._get_predictions(model_type, dataset_key, x_test)
                    if y_pred is not None:
                        ax.plot(x2_range, y_pred, color=colors[model_type],
                               label=labels[model_type], alpha=0.7, linewidth=1.5)

            ax.set_xlabel('x₂')
            ax.set_ylabel('y')
            ax.set_title(f'x₁ = {x1_fixed}')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Row 2: Fix x₂, vary x₁
        for idx, x2_fixed in enumerate(cross_sections_x2):
            ax = axes[1, idx]

            x1_range = np.linspace(0, 1, resolution)
            x2_vals = np.full_like(x1_range, x2_fixed)
            x_test = torch.tensor(
                np.stack([x1_range, x2_vals], axis=1),
                dtype=torch.float32
            ).to(self.device)

            # True function
            y_true = true_func(x_test).cpu().numpy()
            ax.plot(x1_range, y_true, 'k--', linewidth=2, label='True', alpha=0.8)

            # Model predictions
            for model_type in model_types:
                # Try both integer and string keys for compatibility (PKL uses int, JSON uses str)
                dataset_key = dataset_idx if dataset_idx in self.results.get('kan', {}) else str(dataset_idx)
                if model_type in self.results and dataset_key in self.results[model_type]:
                    y_pred = self._get_predictions(model_type, dataset_key, x_test)
                    if y_pred is not None:
                        ax.plot(x1_range, y_pred, color=colors[model_type],
                               label=labels[model_type], alpha=0.7, linewidth=1.5)

            ax.set_xlabel('x₁')
            ax.set_ylabel('y')
            ax.set_title(f'x₂ = {x2_fixed}')
            ax.legend()
            ax.grid(True, alpha=0.3)

        fig.suptitle(f'Cross-Section Comparison: {func_name}', fontsize=16)
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')

        return fig

    def create_error_quantile_map(self, dataset_idx: int, true_func: Callable,
                                 func_name: str, model_type: str = 'kan',
                                 resolution: int = 100, output_path: Optional[str] = None):
        """
        Create heatmap showing error quantiles to identify problematic regions

        Args:
            dataset_idx: Dataset index
            true_func: True function
            func_name: Function name
            model_type: Model type
            resolution: Grid resolution
            output_path: Optional save path
        """
        # Generate grid
        x1 = np.linspace(0, 1, resolution)
        x2 = np.linspace(0, 1, resolution)
        X1, X2 = np.meshgrid(x1, x2)

        x_test = torch.tensor(
            np.stack([X1.flatten(), X2.flatten()], axis=1),
            dtype=torch.float32
        ).to(self.device)

        # Get values
        y_true = true_func(x_test).cpu().numpy().reshape(resolution, resolution)
        # Try both integer and string keys for compatibility (PKL uses int, JSON uses str)
        dataset_key = dataset_idx if dataset_idx in self.results.get('kan', {}) else str(dataset_idx)

        if model_type not in self.results or dataset_key not in self.results[model_type]:
            print(f"No results for {model_type} on dataset {dataset_idx}")
            return None

        y_pred = self._get_predictions(model_type, dataset_key, x_test)
        if y_pred is None:
            return None

        y_pred = y_pred.reshape(resolution, resolution)
        error = np.abs(y_true - y_pred)

        # Create quantile-based error map
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Quantile boundaries
        q25 = np.percentile(error, 25)
        q50 = np.percentile(error, 50)
        q75 = np.percentile(error, 75)
        q90 = np.percentile(error, 90)

        # Categorize errors
        error_categories = np.zeros_like(error)
        error_categories[error <= q25] = 1
        error_categories[(error > q25) & (error <= q50)] = 2
        error_categories[(error > q50) & (error <= q75)] = 3
        error_categories[(error > q75) & (error <= q90)] = 4
        error_categories[error > q90] = 5

        # Plot error categories
        im1 = axes[0].imshow(error_categories, cmap='RdYlGn_r', aspect='auto',
                            extent=[0, 1, 0, 1], origin='lower')
        axes[0].set_title('Error Quantile Regions')
        axes[0].set_xlabel('x₁')
        axes[0].set_ylabel('x₂')
        cbar1 = plt.colorbar(im1, ax=axes[0], ticks=[1, 2, 3, 4, 5])
        cbar1.set_ticklabels(['Q1 (0-25%)', 'Q2 (25-50%)', 'Q3 (50-75%)',
                             'Q4 (75-90%)', 'Q5 (>90%)'])

        # Plot actual error
        im2 = axes[1].imshow(error, cmap='hot', aspect='auto',
                            extent=[0, 1, 0, 1], origin='lower')
        axes[1].set_title('Absolute Error')
        axes[1].set_xlabel('x₁')
        axes[1].set_ylabel('x₂')
        plt.colorbar(im2, ax=axes[1])

        # Mark high-error regions
        high_error_mask = error > q90
        axes[1].contour(X1, X2, high_error_mask, levels=[0.5], colors='cyan',
                       linewidths=2, linestyles='--')

        # Statistics by region
        regions = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
        region_stats = []

        for i in range(1, 6):
            mask = error_categories == i
            if mask.any():
                region_errors = error[mask]
                region_stats.append({
                    'Region': regions[i-1],
                    'Mean Error': region_errors.mean(),
                    'Max Error': region_errors.max(),
                    'Coverage (%)': (mask.sum() / mask.size) * 100
                })

        # Plot statistics
        df_stats = pd.DataFrame(region_stats)
        axes[2].axis('off')
        table = axes[2].table(cellText=df_stats.values,
                             colLabels=df_stats.columns,
                             cellLoc='center',
                             loc='center',
                             bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        axes[2].set_title('Error Statistics by Region')

        fig.suptitle(f'{func_name} - {model_type.upper()} Error Analysis', fontsize=14)
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')

        return fig

    def _get_predictions(self, model_type: str, dataset_key: str,
                        x_test: torch.Tensor) -> Optional[np.ndarray]:
        """Get predictions from model"""
        # Try to load KAN model
        if model_type in ['kan', 'kan_pruning'] and self.models_dir:
            try:
                model = self._load_kan_model(model_type, dataset_key)
                if model:
                    with torch.no_grad():
                        y_pred = model(x_test)
                    return y_pred.cpu().numpy().flatten()
            except Exception as e:
                print(f"Could not load model: {e}")

        return None

    def _load_kan_model(self, model_type: str, dataset_key: str) -> Optional[KAN]:
        """Load KAN model"""
        if not self.models_dir:
            return None

        suffix = 'pruned_' if model_type == 'kan_pruning' else ''
        model_path = self.models_dir / f'kan_{suffix}dataset_{dataset_key}'

        # This is simplified - actual implementation would need proper config loading
        try:
            model = KAN(width=[2, 5, 1], grid=5, k=3, device=self.device)
            model.loadckpt(str(model_path))
            return model
        except:
            return None

    def generate_all_heatmaps(self, output_dir: str):
        """Generate all heatmap visualizations"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        print("Generating 2D heatmap visualizations...")

        for dataset_idx, (func_name, true_func) in self.function_map.items():
            print(f"\nProcessing: {func_name}")

            # Comparison heatmaps for each model
            for model_type in ['mlp', 'siren', 'kan', 'kan_pruning']:
                # Try both integer and string keys for compatibility (PKL uses int, JSON uses str)
                dataset_key = dataset_idx if dataset_idx in self.results.get('kan', {}) else str(dataset_idx)
                if model_type in self.results and dataset_key in self.results[model_type]:
                    print(f"  - {model_type.upper()} comparison heatmap")
                    self.create_comparison_heatmaps(
                        dataset_idx, true_func, func_name, model_type,
                        output_path=output_path / f'heatmap_{dataset_idx}_{func_name}_{model_type}.png'
                    )
                    plt.close()

            # Cross-section comparison
            print(f"  - Cross-section comparison")
            self.create_cross_section_comparison(
                dataset_idx, true_func, func_name,
                output_path=output_path / f'cross_section_{dataset_idx}_{func_name}.png'
            )
            plt.close()

            # Error quantile maps
            for model_type in ['kan', 'kan_pruning']:
                # Try both integer and string keys for compatibility (PKL uses int, JSON uses str)
                dataset_key = dataset_idx if dataset_idx in self.results.get('kan', {}) else str(dataset_idx)
                if model_type in self.results and dataset_key in self.results[model_type]:
                    print(f"  - {model_type.upper()} error quantile map")
                    self.create_error_quantile_map(
                        dataset_idx, true_func, func_name, model_type,
                        output_path=output_path / f'error_quantile_{dataset_idx}_{func_name}_{model_type}.png'
                    )
                    plt.close()

        print(f"\nAll heatmaps saved to: {output_path}")


def main():
    """Main function"""
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(description='Generate 2D heatmap analysis')
    parser.add_argument('results_file', type=str, help='Path to results file')
    parser.add_argument('--models-dir', type=str, help='Path to saved models')
    parser.add_argument('--output-dir', type=str, default='heatmap_analysis',
                       help='Output directory')

    args = parser.parse_args()

    analyzer = Heatmap2DAnalyzer(args.results_file, args.models_dir)
    analyzer.generate_all_heatmaps(args.output_dir)


if __name__ == '__main__':
    main()
