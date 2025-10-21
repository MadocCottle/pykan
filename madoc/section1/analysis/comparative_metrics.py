"""
Comparative Metrics Visualization Script

This script creates tables and graphs comparing different models across:
- Training epochs
- Training times
- Test/train MSE
- Dense MSE metrics
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Optional, Union
import seaborn as sns

# Add pykan to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Import centralized IO module
from . import data_io

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class MetricsAnalyzer:
    """Analyzes and visualizes comparative metrics across models"""

    def __init__(self, results_path: Union[str, Path]):
        """
        Initialize analyzer with results file or section ID

        Args:
            results_path: Either:
                - Path to the .pkl or .json results file (e.g., '/path/to/results.pkl')
                - Section ID (e.g., 'section1_1') to auto-load latest results
        """
        # Load results and metadata using centralized IO
        self.results, self.metadata = data_io.load_results(results_path)

        # Store the results path for reference
        results_path_str = str(results_path)
        if results_path_str in data_io.SECTION_DIRS:
            # If section ID was provided, get the actual path
            info = data_io.find_latest_results(results_path_str)
            self.results_path = info['results_file']
        else:
            self.results_path = Path(results_path)

    def create_comparison_table(self, dataset_idx: int = 0, output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Create comparison table for a specific dataset

        Args:
            dataset_idx: Index of dataset to analyze
            output_path: Optional path to save CSV

        Returns:
            DataFrame with comparison metrics
        """
        rows = []

        for model_type in ['mlp', 'siren', 'kan', 'kan_pruning']:
            if model_type not in self.results:
                continue

            data = self.results[model_type]
            # Try both integer and string keys for compatibility (PKL uses int, JSON uses str)
            dataset_key = dataset_idx if dataset_idx in data else str(dataset_idx)

            if dataset_key not in data:
                continue

            # For MLP/SIREN, iterate through configurations
            if model_type in ['mlp', 'siren']:
                for depth_key, depth_data in data[dataset_key].items():
                    if model_type == 'mlp':
                        for activation, metrics in depth_data.items():
                            rows.append(self._extract_metrics(
                                model_type, f"depth_{depth_key}_{activation}", metrics
                            ))
                    else:  # SIREN
                        rows.append(self._extract_metrics(
                            model_type, f"depth_{depth_key}", depth_data
                        ))

            # For KAN, iterate through grid sizes
            else:
                for grid_key, metrics in data[dataset_key].items():
                    rows.append(self._extract_metrics(
                        model_type, f"grid_{grid_key}", metrics
                    ))

        df = pd.DataFrame(rows)

        if output_path:
            df.to_csv(output_path, index=False)

        return df

    def _extract_metrics(self, model_type: str, config: str, metrics: Dict) -> Dict:
        """Extract key metrics from results"""
        final_train = metrics['train'][-1] if isinstance(metrics['train'], list) else metrics['train']
        final_test = metrics['test'][-1] if isinstance(metrics['test'], list) else metrics['test']
        final_dense = metrics.get('dense_mse', [None])[-1] if isinstance(metrics.get('dense_mse', []), list) else metrics.get('dense_mse')

        return {
            'Model': model_type.upper(),
            'Configuration': config,
            'Final Train MSE': final_train,
            'Final Test MSE': final_test,
            'Final Dense MSE': final_dense,
            'Total Time (s)': metrics.get('total_time'),
            'Time per Epoch (s)': metrics.get('time_per_epoch'),
            'Epochs': len(metrics['train']) if isinstance(metrics['train'], list) else 1
        }

    def plot_learning_curves(self, dataset_idx: int = 0, metric: str = 'test',
                            output_path: Optional[str] = None):
        """
        Plot learning curves comparing all models

        Args:
            dataset_idx: Index of dataset to analyze
            metric: Which metric to plot ('train', 'test', 'dense_mse')
            output_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=(14, 8))

        for model_type in ['mlp', 'siren', 'kan', 'kan_pruning']:
            if model_type not in self.results:
                continue

            data = self.results[model_type]
            # Try both integer and string keys for compatibility (PKL uses int, JSON uses str)
            dataset_key = dataset_idx if dataset_idx in data else str(dataset_idx)

            if dataset_key not in data:
                continue

            self._plot_model_curves(ax, model_type, data[dataset_key], metric)

        ax.set_xlabel('Epoch')
        ax.set_ylabel(f'{metric.replace("_", " ").title()} MSE')
        ax.set_title(f'Learning Curves - Dataset {dataset_idx}')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')

        return fig

    def _plot_model_curves(self, ax, model_type: str, data: Dict, metric: str):
        """Plot learning curves for a specific model type"""
        colors = {'mlp': 'C0', 'siren': 'C1', 'kan': 'C2', 'kan_pruning': 'C3'}

        if model_type in ['mlp', 'siren']:
            # Plot best configuration for each depth
            for depth_key, depth_data in data.items():
                if model_type == 'mlp':
                    # Find best activation
                    best_config = min(depth_data.items(),
                                    key=lambda x: x[1][metric][-1] if isinstance(x[1][metric], list) else x[1][metric])
                    activation, metrics = best_config
                    label = f"{model_type.upper()} depth={depth_key} ({activation})"
                else:
                    metrics = depth_data
                    label = f"{model_type.upper()} depth={depth_key}"

                if isinstance(metrics[metric], list):
                    ax.plot(metrics[metric], label=label, alpha=0.7,
                           color=colors[model_type], linestyle='--' if model_type == 'mlp' else '-.')

        else:  # KAN
            # Plot best grid size
            best_grid = min(data.items(),
                          key=lambda x: x[1][metric][-1] if isinstance(x[1][metric], list) else x[1][metric])
            grid_key, metrics = best_grid

            if isinstance(metrics[metric], list):
                label = f"{model_type.upper().replace('_', ' ')} grid={grid_key}"
                ax.plot(metrics[metric], label=label, linewidth=2,
                       color=colors[model_type])

    def plot_training_time_comparison(self, dataset_idx: int = 0, output_path: Optional[str] = None):
        """
        Create bar plot comparing training times across models

        Args:
            dataset_idx: Index of dataset to analyze
            output_path: Optional path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        models = []
        total_times = []
        per_epoch_times = []

        for model_type in ['mlp', 'siren', 'kan', 'kan_pruning']:
            if model_type not in self.results:
                continue

            data = self.results[model_type]
            # Try both integer and string keys for compatibility (PKL uses int, JSON uses str)
            dataset_key = dataset_idx if dataset_idx in data else str(dataset_idx)

            if dataset_key not in data:
                continue

            # Get best configuration times
            times = self._get_best_times(model_type, data[dataset_key])
            if times:
                models.append(model_type.upper().replace('_', ' '))
                total_times.append(times['total'])
                per_epoch_times.append(times['per_epoch'])

        # Plot total times
        bars1 = ax1.bar(models, total_times, alpha=0.7, edgecolor='black')
        ax1.set_ylabel('Total Training Time (s)')
        ax1.set_title(f'Total Training Time - Dataset {dataset_idx}')
        ax1.tick_params(axis='x', rotation=45)

        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}s', ha='center', va='bottom')

        # Plot per-epoch times
        bars2 = ax2.bar(models, per_epoch_times, alpha=0.7, edgecolor='black', color='coral')
        ax2.set_ylabel('Time per Epoch (s)')
        ax2.set_title(f'Time per Epoch - Dataset {dataset_idx}')
        ax2.tick_params(axis='x', rotation=45)

        # Add value labels on bars
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}s', ha='center', va='bottom')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')

        return fig

    def _get_best_times(self, model_type: str, data: Dict) -> Optional[Dict]:
        """Get timing data for best configuration"""
        if model_type in ['mlp', 'siren']:
            # Average across all configurations
            total_times = []
            per_epoch_times = []

            for depth_key, depth_data in data.items():
                if model_type == 'mlp':
                    for activation, metrics in depth_data.items():
                        if 'total_time' in metrics:
                            total_times.append(metrics['total_time'])
                            per_epoch_times.append(metrics.get('time_per_epoch', 0))
                else:
                    if 'total_time' in depth_data:
                        total_times.append(depth_data['total_time'])
                        per_epoch_times.append(depth_data.get('time_per_epoch', 0))

            if total_times:
                return {
                    'total': np.mean(total_times),
                    'per_epoch': np.mean(per_epoch_times)
                }
        else:
            # Get best grid configuration
            best_config = None
            best_test = float('inf')

            for grid_key, metrics in data.items():
                final_test = metrics['test'][-1] if isinstance(metrics['test'], list) else metrics['test']
                if final_test < best_test:
                    best_test = final_test
                    best_config = metrics

            if best_config and 'total_time' in best_config:
                return {
                    'total': best_config['total_time'],
                    'per_epoch': best_config.get('time_per_epoch', 0)
                }

        return None

    def create_final_performance_heatmap(self, metric: str = 'test', output_path: Optional[str] = None):
        """
        Create heatmap showing final performance across all datasets and models

        Args:
            metric: Which metric to visualize ('test', 'train', 'dense_mse')
            output_path: Optional path to save figure
        """
        # Collect data
        model_names = []
        dataset_scores = []

        for model_type in ['mlp', 'siren', 'kan', 'kan_pruning']:
            if model_type not in self.results:
                continue

            scores = []
            for dataset_idx in range(self._get_num_datasets()):
                # Try both integer and string keys for compatibility (PKL uses int, JSON uses str)
                dataset_key = dataset_idx if dataset_idx in self.results[model_type] else str(dataset_idx)
                if dataset_key in self.results[model_type]:
                    best_score = self._get_best_score(
                        model_type, self.results[model_type][dataset_key], metric
                    )
                    scores.append(best_score if best_score is not None else np.nan)
                else:
                    scores.append(np.nan)

            model_names.append(model_type.upper().replace('_', ' '))
            dataset_scores.append(scores)

        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 6))

        df = pd.DataFrame(dataset_scores, index=model_names,
                         columns=[f'Dataset {i}' for i in range(len(dataset_scores[0]))])

        sns.heatmap(df, annot=True, fmt='.4f', cmap='RdYlGn_r', ax=ax,
                   cbar_kws={'label': f'{metric.replace("_", " ").title()} MSE'},
                   vmin=0, vmax=df.max().max() * 0.5)  # Cap color scale for better contrast

        ax.set_title(f'Final {metric.replace("_", " ").title()} MSE Across Datasets and Models')
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Model')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')

        return fig

    def _get_num_datasets(self) -> int:
        """Get number of datasets in results"""
        for model_type in ['mlp', 'siren', 'kan', 'kan_pruning']:
            if model_type in self.results:
                return len(self.results[model_type])
        return 0

    def _get_best_score(self, model_type: str, data: Dict, metric: str) -> Optional[float]:
        """Get best score for a model type on a dataset"""
        scores = []

        if model_type in ['mlp', 'siren']:
            for depth_key, depth_data in data.items():
                if model_type == 'mlp':
                    for activation, metrics in depth_data.items():
                        if metric in metrics:
                            val = metrics[metric]
                            final_val = val[-1] if isinstance(val, list) else val
                            if final_val is not None:
                                scores.append(final_val)
                else:
                    if metric in depth_data:
                        val = depth_data[metric]
                        final_val = val[-1] if isinstance(val, list) else val
                        if final_val is not None:
                            scores.append(final_val)
        else:
            for grid_key, metrics in data.items():
                if metric in metrics:
                    val = metrics[metric]
                    final_val = val[-1] if isinstance(val, list) else val
                    if final_val is not None:
                        scores.append(final_val)

        return min(scores) if scores else None

    def generate_all_visualizations(self, output_dir: str):
        """
        Generate all visualizations and save to output directory

        Args:
            output_dir: Directory to save all outputs
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        num_datasets = self._get_num_datasets()

        print(f"Generating visualizations for {num_datasets} datasets...")

        # Generate for each dataset
        for dataset_idx in range(num_datasets):
            print(f"\nDataset {dataset_idx}:")

            # Comparison table
            print("  - Creating comparison table...")
            df = self.create_comparison_table(
                dataset_idx,
                output_path / f'dataset_{dataset_idx}_comparison_table.csv'
            )
            print(f"    Saved to: dataset_{dataset_idx}_comparison_table.csv")

            # Learning curves
            for metric in ['train', 'test', 'dense_mse']:
                print(f"  - Creating {metric} learning curves...")
                self.plot_learning_curves(
                    dataset_idx,
                    metric,
                    output_path / f'dataset_{dataset_idx}_learning_curves_{metric}.png'
                )
                plt.close()

            # Training time comparison
            print("  - Creating training time comparison...")
            self.plot_training_time_comparison(
                dataset_idx,
                output_path / f'dataset_{dataset_idx}_training_times.png'
            )
            plt.close()

        # Overall heatmaps
        print("\nGenerating overall heatmaps...")
        for metric in ['train', 'test', 'dense_mse']:
            print(f"  - Creating {metric} heatmap...")
            self.create_final_performance_heatmap(
                metric,
                output_path / f'all_datasets_heatmap_{metric}.png'
            )
            plt.close()

        print(f"\nAll visualizations saved to: {output_path}")


def main():
    """Main function for command-line usage"""
    import argparse

    parser = argparse.ArgumentParser(description='Generate comparative metrics visualizations')
    parser.add_argument('results_file', type=str, help='Path to results .pkl or .json file')
    parser.add_argument('--output-dir', type=str, default='analysis_output',
                       help='Output directory for visualizations (default: analysis_output)')

    args = parser.parse_args()

    analyzer = MetricsAnalyzer(args.results_file)
    analyzer.generate_all_visualizations(args.output_dir)


if __name__ == '__main__':
    main()
