"""
Function Fitting Visualization Script

This script creates visualizations comparing the ground truth functions
with neural network predictions across the test domain.
- 1D functions: line plots showing NN output vs true function
- 2D functions: surface plots and contour plots
"""

import sys
from pathlib import Path
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import torch
from typing import Dict, List, Tuple, Optional, Callable

# Add section1 directory to path for utils imports
sys.path.insert(0, str(Path(__file__).parent.parent))
# Add pykan root to path for KAN imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from kan import KAN, create_dataset
from utils import data_funcs as dfs


class FunctionFittingVisualizer:
    """Visualizes how well models fit the underlying functions"""

    def __init__(self, results_path: str, models_dir: Optional[str] = None):
        """
        Initialize visualizer

        Args:
            results_path: Path to results file
            models_dir: Path to saved models directory (for KAN models)
        """
        self.results_path = Path(results_path)
        self.models_dir = Path(models_dir) if models_dir else None
        self.results = self._load_results()
        self.metadata = self._load_metadata()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Define function mappings
        self.function_map_1d = {
            0: ('sin(2πx)', dfs.sinusoid_1d(1)),
            1: ('sin(4πx)', dfs.sinusoid_1d(2)),
            2: ('sin(6πx)', dfs.sinusoid_1d(3)),
            3: ('sin(8πx)', dfs.sinusoid_1d(4)),
            4: ('sin(10πx)', dfs.sinusoid_1d(5)),
            5: ('Piecewise', dfs.f_piecewise),
            6: ('Sawtooth', dfs.f_sawtooth),
            7: ('Polynomial', dfs.f_polynomial),
            8: ('Poisson 1D High-freq', dfs.f_poisson_1d_highfreq),
        }

        self.function_map_2d = {
            0: ('2D Sin (π²)', dfs.f_poisson_2d_sin),
            1: ('2D Polynomial', dfs.f_poisson_2d_poly),
            2: ('2D High-freq', dfs.f_poisson_2d_highfreq),
            3: ('2D Special', dfs.f_poisson_2d_spec),
        }

    def _load_results(self) -> Dict:
        """Load results from file"""
        if self.results_path.suffix == '.pkl':
            with open(self.results_path, 'rb') as f:
                return pickle.load(f)
        elif self.results_path.suffix == '.json':
            with open(self.results_path, 'r') as f:
                return json.load(f)

    def _load_metadata(self) -> Optional[Dict]:
        """Load metadata file"""
        timestamp = self.results_path.stem.split('_')[-1]
        section = '_'.join(self.results_path.stem.split('_')[:-2])
        metadata_path = self.results_path.parent / f"{section}_metadata_{timestamp}.json"

        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return None

    def plot_1d_function_fit(self, dataset_idx: int, true_func: Callable,
                            func_name: str, output_path: Optional[str] = None):
        """
        Plot 1D function fit for all model types

        Args:
            dataset_idx: Index of dataset
            true_func: True function to plot
            func_name: Name of function for title
            output_path: Optional save path
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        # Generate dense test points
        x_test = torch.linspace(0, 1, 1000).reshape(-1, 1).to(self.device)
        y_true = true_func(x_test).cpu().numpy().flatten()
        x_plot = x_test.cpu().numpy().flatten()

        model_types = ['mlp', 'siren', 'kan', 'kan_pruning']
        titles = ['MLP (Best)', 'SIREN (Best)', 'KAN (Best)', 'KAN with Pruning (Best)']

        for idx, (model_type, title) in enumerate(zip(model_types, titles)):
            ax = axes[idx]

            if model_type not in self.results:
                ax.text(0.5, 0.5, f'No {model_type.upper()} results',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(title)
                continue

            # Plot true function
            ax.plot(x_plot, y_true, 'k--', linewidth=2, label='True Function', alpha=0.7)

            # Get predictions from best model
            dataset_key = str(dataset_idx)
            if dataset_key in self.results[model_type]:
                predictions = self._get_best_predictions_1d(
                    model_type, dataset_key, x_test
                )

                if predictions is not None:
                    ax.plot(x_plot, predictions, 'r-', linewidth=1.5,
                           label='NN Prediction', alpha=0.8)

                    # Calculate and display error
                    mse = np.mean((predictions - y_true) ** 2)
                    ax.text(0.02, 0.98, f'MSE: {mse:.6f}',
                           transform=ax.transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)

        fig.suptitle(f'Function Fit Comparison: {func_name}', fontsize=16, y=1.00)
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')

        return fig

    def _get_best_predictions_1d(self, model_type: str, dataset_key: str,
                                 x_test: torch.Tensor) -> Optional[np.ndarray]:
        """Get predictions from best model configuration"""

        # For KAN models, try to load saved model
        if model_type in ['kan', 'kan_pruning'] and self.models_dir:
            try:
                model = self._load_kan_model(model_type, dataset_key)
                if model:
                    with torch.no_grad():
                        y_pred = model(x_test)
                    return y_pred.cpu().numpy().flatten()
            except Exception as e:
                print(f"Could not load KAN model: {e}")

        # If we can't load model, return None
        # (In a full implementation, we could recreate and retrain models)
        return None

    def _load_kan_model(self, model_type: str, dataset_key: str) -> Optional[KAN]:
        """Load saved KAN model"""
        if not self.models_dir:
            return None

        # Find the model checkpoint
        suffix = 'pruned_' if model_type == 'kan_pruning' else ''
        model_path = self.models_dir / f'kan_{suffix}dataset_{dataset_key}'

        if not model_path.exists():
            # Try with additional directory structure
            model_path = self.models_dir / f'kan_{suffix}dataset_{dataset_key}'

        if model_path.with_suffix('.pth').exists() or (model_path.parent / f'{model_path.name}_state').exists():
            try:
                # Load config
                config_file = list(model_path.parent.glob(f'{model_path.stem}_config.yml'))
                if config_file:
                    # Create KAN with saved configuration
                    # This is simplified - you'd need to parse the config properly
                    model = KAN(width=[1, 5, 1], grid=5, k=3, device=self.device)
                    model.loadckpt(str(model_path))
                    return model
            except Exception as e:
                print(f"Error loading model: {e}")

        return None

    def plot_2d_function_fit(self, dataset_idx: int, true_func: Callable,
                            func_name: str, output_path: Optional[str] = None):
        """
        Plot 2D function fit with surface and contour plots

        Args:
            dataset_idx: Index of dataset
            true_func: True function (expects 2D input)
            func_name: Name of function
            output_path: Optional save path
        """
        # Generate 2D grid
        n_points = 50
        x1 = np.linspace(0, 1, n_points)
        x2 = np.linspace(0, 1, n_points)
        X1, X2 = np.meshgrid(x1, x2)

        # Prepare input tensor
        x_test = torch.tensor(
            np.stack([X1.flatten(), X2.flatten()], axis=1),
            dtype=torch.float32
        ).to(self.device)

        # Get true values
        y_true = true_func(x_test).cpu().numpy().reshape(n_points, n_points)

        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

        model_types = ['mlp', 'siren', 'kan', 'kan_pruning']
        titles = ['MLP', 'SIREN', 'KAN', 'KAN Pruning']

        # Top row: True function
        ax_true_3d = fig.add_subplot(gs[0, 1], projection='3d')
        ax_true_contour = fig.add_subplot(gs[0, 2])

        # Plot true function
        surf = ax_true_3d.plot_surface(X1, X2, y_true, cmap=cm.viridis, alpha=0.8)
        ax_true_3d.set_title(f'True Function: {func_name}')
        ax_true_3d.set_xlabel('x₁')
        ax_true_3d.set_ylabel('x₂')
        ax_true_3d.set_zlabel('y')

        contour = ax_true_contour.contourf(X1, X2, y_true, levels=20, cmap=cm.viridis)
        ax_true_contour.set_title(f'True Function (Contour)')
        ax_true_contour.set_xlabel('x₁')
        ax_true_contour.set_ylabel('x₂')
        plt.colorbar(contour, ax=ax_true_contour)

        # Rows 2-3: Model predictions
        for idx, (model_type, title) in enumerate(zip(model_types, titles)):
            row = 1 + idx // 2
            col = (idx % 2) * 2

            ax_3d = fig.add_subplot(gs[row, col], projection='3d')
            ax_contour = fig.add_subplot(gs[row, col + 1])

            dataset_key = str(dataset_idx)
            if model_type in self.results and dataset_key in self.results[model_type]:
                predictions = self._get_best_predictions_2d(model_type, dataset_key, x_test)

                if predictions is not None:
                    y_pred = predictions.reshape(n_points, n_points)

                    # 3D surface
                    surf = ax_3d.plot_surface(X1, X2, y_pred, cmap=cm.plasma, alpha=0.8)
                    ax_3d.set_title(f'{title} Prediction')
                    ax_3d.set_xlabel('x₁')
                    ax_3d.set_ylabel('x₂')
                    ax_3d.set_zlabel('y')

                    # Contour
                    contour = ax_contour.contourf(X1, X2, y_pred, levels=20, cmap=cm.plasma)
                    ax_contour.set_title(f'{title} (Contour)')
                    ax_contour.set_xlabel('x₁')
                    ax_contour.set_ylabel('x₂')
                    plt.colorbar(contour, ax=ax_contour)

                    # Calculate MSE
                    mse = np.mean((y_pred - y_true) ** 2)
                    ax_contour.text(0.02, 0.98, f'MSE: {mse:.6f}',
                                  transform=ax_contour.transAxes,
                                  verticalalignment='top',
                                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                else:
                    ax_3d.text2D(0.5, 0.5, 'Model not available',
                               transform=ax_3d.transAxes, ha='center')
                    ax_contour.text(0.5, 0.5, 'Model not available',
                                  transform=ax_contour.transAxes, ha='center')
            else:
                ax_3d.text2D(0.5, 0.5, f'No {title} results',
                           transform=ax_3d.transAxes, ha='center')
                ax_contour.text(0.5, 0.5, f'No {title} results',
                              transform=ax_contour.transAxes, ha='center')

            ax_3d.set_title(f'{title}')
            ax_contour.set_title(f'{title} (Contour)')

        fig.suptitle(f'2D Function Fit: {func_name}', fontsize=16)

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')

        return fig

    def _get_best_predictions_2d(self, model_type: str, dataset_key: str,
                                 x_test: torch.Tensor) -> Optional[np.ndarray]:
        """Get 2D predictions from best model"""
        # Similar to 1D version
        if model_type in ['kan', 'kan_pruning'] and self.models_dir:
            try:
                model = self._load_kan_model(model_type, dataset_key)
                if model:
                    with torch.no_grad():
                        y_pred = model(x_test)
                    return y_pred.cpu().numpy().flatten()
            except Exception as e:
                print(f"Could not load KAN model: {e}")

        return None

    def generate_all_function_fits(self, output_dir: str, is_2d: bool = False):
        """
        Generate all function fit visualizations

        Args:
            output_dir: Output directory
            is_2d: Whether to generate 2D plots (otherwise 1D)
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        function_map = self.function_map_2d if is_2d else self.function_map_1d
        plot_func = self.plot_2d_function_fit if is_2d else self.plot_1d_function_fit

        print(f"Generating {'2D' if is_2d else '1D'} function fit visualizations...")

        for dataset_idx, (func_name, true_func) in function_map.items():
            print(f"  Processing: {func_name}")

            try:
                plot_func(
                    dataset_idx,
                    true_func,
                    func_name,
                    output_path / f'function_fit_dataset_{dataset_idx}_{func_name.replace(" ", "_")}.png'
                )
                plt.close()
            except Exception as e:
                print(f"    Error: {e}")

        print(f"Function fit visualizations saved to: {output_path}")


def main():
    """Main function for command-line usage"""
    import argparse

    parser = argparse.ArgumentParser(description='Visualize function fitting quality')
    parser.add_argument('results_file', type=str, help='Path to results file')
    parser.add_argument('--models-dir', type=str, help='Path to saved models directory')
    parser.add_argument('--output-dir', type=str, default='function_fits',
                       help='Output directory')
    parser.add_argument('--2d', action='store_true', help='Generate 2D plots')

    args = parser.parse_args()

    visualizer = FunctionFittingVisualizer(args.results_file, args.models_dir)
    visualizer.generate_all_function_fits(args.output_dir, is_2d=args.__dict__['2d'])


if __name__ == '__main__':
    main()
