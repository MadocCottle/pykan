"""
Plot 2D function fits as heatmaps (3D surface + contour plots) for Section 2.

This script creates visualizations for 2D function fitting tasks for KAN optimizer comparisons.

For each dataset, it creates:
- 3D surface plot showing the true function
- Contour plot of the true function
- 3D surface and contour plots for each optimizer variant (LBFGS, LM, Adam) for section2_1
- 3D surface and contour plots for adaptive density variants for section2_2
- MSE values displayed on contour plots for quantitative comparison

Layout for Section 2.1 (Optimizer Comparison):
- Row 0: True function (3D surface + contour, centered)
- Row 1: LBFGS and LM predictions (each with 3D + contour)
- Row 2: Adam predictions (3D + contour)

Layout for Section 2.2 (Adaptive Density):
- Row 0: True function (3D surface + contour, centered)
- Row 1: Adaptive Only and Adaptive+Regular predictions (each with 3D + contour)
- Row 2: Baseline predictions (3D + contour)
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import torch
import yaml

# Fix YAML loading for numpy scalars in KAN checkpoints
_original_safe_load = yaml.safe_load

def _patched_safe_load(stream):
    """Use unsafe_load for KAN checkpoints which contain numpy objects"""
    try:
        return _original_safe_load(stream)
    except yaml.constructor.ConstructorError:
        stream.seek(0)
        return yaml.unsafe_load(stream)

yaml.safe_load = _patched_safe_load

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_run, data_funcs as dfs
from utils.result_finder import select_run
from kan import KAN


def get_2d_functions():
    """Get 2D function definitions for section2 (uses 2D Poisson PDE)"""
    true_functions = [
        dfs.f_poisson_2d_sin,
        dfs.f_poisson_2d_poly,
        dfs.f_poisson_2d_highfreq,
        dfs.f_poisson_2d_spec
    ]
    dataset_names = [
        '2D Sin (π²)',
        '2D Polynomial',
        '2D High-freq',
        '2D Special'
    ]
    return true_functions, dataset_names


def load_kan_model(models, results, model_key, dataset_idx, device='cpu'):
    """
    Load KAN model from checkpoint for section2.

    Args:
        models: Models dict from load_run
        results: Results dict from load_run
        model_key: Key for the model type ('lbfgs', 'lm', 'adam', 'adaptive_only', etc.)
        dataset_idx: Dataset index
        device: Device to load model on

    Returns:
        Tuple of (model, config_info) or (None, None) if not available
    """
    if model_key not in models or dataset_idx not in models[model_key]:
        return None, None

    checkpoint_path = models[model_key][dataset_idx]
    model = KAN.loadckpt(checkpoint_path)
    model.to(device)
    model.eval()

    # Get final performance from results
    df = results[model_key]
    df_dataset = df[df['dataset_idx'] == dataset_idx]

    if len(df_dataset) > 0:
        final_row = df_dataset.iloc[-1]
        dense_mse = final_row['dense_mse']
        grid_size = int(final_row['grid_size'])

        # Get optimizer or approach name
        if 'optimizer' in final_row:
            label_name = final_row['optimizer']
        elif 'approach' in final_row:
            label_name = final_row['approach'].replace('_', ' ').title()
        else:
            label_name = model_key.upper()
    else:
        dense_mse = float('nan')
        grid_size = -1
        label_name = model_key.upper()

    config_info = {
        'grid_size': grid_size,
        'dense_mse': dense_mse,
        'label': label_name
    }

    return model, config_info


def plot_2d_heatmap(dataset_idx, section='section2_1', timestamp=None, device='cpu', save_path=None, show=False):
    """
    Create heatmap visualization for a 2D function fitting task.

    Args:
        dataset_idx: Index of the dataset to visualize (0-3)
        section: Section name ('section2_1' or 'section2_2')
        timestamp: Specific timestamp to load, or None for most recent
        device: Device to run on ('cpu' or 'cuda')
        save_path: Path to save the figure, or None to use default

    Returns:
        matplotlib figure object
    """
    # Find latest timestamp if not provided
    if timestamp is None:
        timestamp = find_latest_timestamp(section)
        print(f"Using most recent timestamp: {timestamp}")

    print(f"\nLoading results and models from {section}_{timestamp}...")

    # Load results and models
    results, meta, models = load_run(section, timestamp, load_models=True)

    # Get true functions
    true_functions, dataset_names = get_2d_functions()

    if dataset_idx >= len(true_functions):
        raise ValueError(f"Dataset index {dataset_idx} out of range (0-{len(true_functions)-1})")

    true_func = true_functions[dataset_idx]
    func_name = dataset_names[dataset_idx]

    print(f"Generating heatmap for Dataset {dataset_idx}: {func_name}")

    # Generate 2D grid for evaluation
    x1 = torch.linspace(0, 1, 50, device=device)
    x2 = torch.linspace(0, 1, 50, device=device)
    X1, X2 = torch.meshgrid(x1, x2, indexing='ij')

    # Flatten for model input
    x_test = torch.stack([X1.flatten(), X2.flatten()], dim=1)

    # Get ground truth
    with torch.no_grad():
        y_true = true_func(x_test).cpu().numpy().reshape(50, 50)

    X1_np = X1.cpu().numpy()
    X2_np = X2.cpu().numpy()

    # Determine which models to load based on section
    if section == 'section2_1':
        # Optimizer comparison: LBFGS, LM, Adam (if available)
        model_keys = ['lbfgs', 'lm', 'adam']
    elif section == 'section2_2':
        # Adaptive density: adaptive_only, adaptive_regular, baseline
        model_keys = ['adaptive_only', 'adaptive_regular', 'baseline']
    else:
        raise ValueError(f"Unknown section: {section}")

    # Load models
    model_data = []
    for key in model_keys:
        model, info = load_kan_model(models, results, key, dataset_idx, device)
        if model is not None:
            with torch.no_grad():
                y_pred = model(x_test).cpu().numpy().reshape(50, 50)
            mse = info['dense_mse']
            model_data.append((info['label'], y_pred, mse))
        else:
            model_data.append((None, None, None))

    # Create figure with gridspec layout
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

    # Main title
    fig.suptitle(f'2D Function Fit: {func_name} ({section})', fontsize=16, fontweight='bold')

    # Row 0: True Function (3D surface + contour in center)
    # 3D Surface
    ax_true_3d = fig.add_subplot(gs[0, 1], projection='3d')
    surf = ax_true_3d.plot_surface(X1_np, X2_np, y_true, cmap=cm.viridis, alpha=0.8)
    ax_true_3d.set_xlabel('x₁')
    ax_true_3d.set_ylabel('x₂')
    ax_true_3d.set_zlabel('y')
    ax_true_3d.set_title(f'True Function: {func_name}', fontsize=12, fontweight='bold')

    # Contour
    ax_true_contour = fig.add_subplot(gs[0, 2])
    contour = ax_true_contour.contourf(X1_np, X2_np, y_true, levels=20, cmap=cm.viridis)
    plt.colorbar(contour, ax=ax_true_contour)
    ax_true_contour.set_xlabel('x₁')
    ax_true_contour.set_ylabel('x₂')
    ax_true_contour.set_title('True Function (Contour)', fontsize=12, fontweight='bold')

    # Rows 1-2: Model predictions
    # Positions for 3 models (each takes 2 columns for 3D and contour)
    positions = [
        (1, 0),  # Model 1 3D
        (1, 1),  # Model 1 Contour
        (1, 2),  # Model 2 3D
        (1, 3),  # Model 2 Contour
        (2, 0),  # Model 3 3D
        (2, 1),  # Model 3 Contour
    ]

    for idx, (label, y_pred, mse) in enumerate(model_data):
        if idx >= 3:  # Only handle up to 3 models
            break

        row_3d = positions[idx * 2][0]
        col_3d = positions[idx * 2][1]
        row_contour = positions[idx * 2 + 1][0]
        col_contour = positions[idx * 2 + 1][1]

        if label is None or y_pred is None:
            # Model not available
            ax_3d = fig.add_subplot(gs[row_3d, col_3d])
            ax_3d.text(0.5, 0.5, 'Model not available', ha='center', va='center')
            ax_3d.axis('off')

            ax_contour = fig.add_subplot(gs[row_contour, col_contour])
            ax_contour.text(0.5, 0.5, 'Model not available', ha='center', va='center')
            ax_contour.axis('off')
            continue

        # 3D Surface
        ax_3d = fig.add_subplot(gs[row_3d, col_3d], projection='3d')
        surf = ax_3d.plot_surface(X1_np, X2_np, y_pred, cmap=cm.plasma, alpha=0.8)
        ax_3d.set_xlabel('x₁')
        ax_3d.set_ylabel('x₂')
        ax_3d.set_zlabel('y')
        ax_3d.set_title(f'{label} Prediction', fontsize=12, fontweight='bold')

        # Contour with MSE
        ax_contour = fig.add_subplot(gs[row_contour, col_contour])
        contour = ax_contour.contourf(X1_np, X2_np, y_pred, levels=20, cmap=cm.plasma)
        plt.colorbar(contour, ax=ax_contour)
        ax_contour.set_xlabel('x₁')
        ax_contour.set_ylabel('x₂')
        ax_contour.set_title(f'{label} (Contour)', fontsize=12, fontweight='bold')

        # Add MSE text box
        mse_text = f'MSE: {mse:.6f}'
        ax_contour.text(0.05, 0.95, mse_text, transform=ax_contour.transAxes,
                       fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Hide unused subplots
    for row in range(3):
        for col in range(4):
            # Check if this subplot should be empty
            if (row == 0 and col in [0, 3]) or (row == 2 and col in [2, 3]):
                ax_empty = fig.add_subplot(gs[row, col])
                ax_empty.axis('off')

    # Save figure
    if save_path is None:
        output_dir = Path(__file__).parent
        save_path = output_dir / f'heatmap_2d_{section}_dataset_{dataset_idx}_{func_name.replace(" ", "_")}_{timestamp}.png'
    else:
        save_path = Path(save_path)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved heatmap to: {save_path}")

    return fig


def plot_all_2d_heatmaps(section='section2_1', timestamp=None, device='cpu', show=False,
                         strategy='latest', epochs=None, verbose=False):
    """
    Create heatmap visualizations for all 2D datasets.

    Args:
        section: Section name ('section2_1' or 'section2_2')
        timestamp: Specific timestamp to load, or None for most recent
        device: Device to run on ('cpu' or 'cuda')
        strategy: Run selection strategy ('latest', 'max_epochs', 'min_epochs', 'exact_epochs')
        epochs: Epoch count for exact_epochs strategy
        verbose: Print run selection details
    """
    if timestamp is None:
        results_base = Path(__file__).parent.parent / 'results'
        timestamp = select_run(section, results_base,
                             strategy=strategy,
                             epochs=epochs,
                             verbose=verbose)
        print(f"Using selected timestamp: {timestamp}")

    true_functions, dataset_names = get_2d_functions()

    print(f"\nGenerating heatmaps for all {len(true_functions)} datasets in {section}...")

    for dataset_idx in range(len(true_functions)):
        print(f"\n{'='*60}")
        print(f"Dataset {dataset_idx}: {dataset_names[dataset_idx]}")
        print('='*60)

        try:
            plot_2d_heatmap(dataset_idx, section, timestamp, device)
        except Exception as e:
            print(f"Error generating heatmap for dataset {dataset_idx}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print("All heatmaps generated successfully!")
    print("="*60)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Plot 2D function fits as heatmaps (3D surface + contour) for Section 2'
    )
    parser.add_argument('--section', type=str, default='section2_1',
                       choices=['section2_1', 'section2_2'],
                       help='Section to visualize (section2_1 or section2_2)')
    parser.add_argument('--dataset', type=int, default=None,
                       help='Dataset index to plot (0-3), or None for all datasets')
    parser.add_argument('--timestamp', type=str, default=None,
                       help='Specific timestamp to load (default: most recent)')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (cpu or cuda)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for the figure (default: auto-generated)')
    parser.add_argument('--show', action='store_true',
                       help='Display the plot in a window (default: only save to file)')
    parser.add_argument('--strategy', type=str, default='latest',
                       choices=['latest', 'max_epochs', 'min_epochs', 'exact_epochs'],
                       help='Run selection strategy (default: latest)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Epoch count for exact_epochs strategy')
    parser.add_argument('--verbose', action='store_true',
                       help='Print run selection details')

    args = parser.parse_args()

    try:
        if args.dataset is None:
            # Plot all datasets
            plot_all_2d_heatmaps(section=args.section, timestamp=args.timestamp, device=args.device,
                               strategy=args.strategy, epochs=args.epochs, verbose=args.verbose)
        else:
            # Plot specific dataset
            plot_2d_heatmap(
                dataset_idx=args.dataset,
                section=args.section,
                timestamp=args.timestamp,
                device=args.device,
                save_path=args.output
            )

        if args.show:
            plt.show()
        else:
            plt.close('all')

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"\nMake sure you have run the {args.section} training script first.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
