"""
Plot 2D function fits as heatmaps (3D surface + contour plots).

This script creates beautiful visualizations for 2D function fitting tasks following
the specifications from reference/cool_spec.md.

For each dataset, it creates:
- 3D surface plot showing the true function
- Contour plot of the true function
- 3D surface and contour plots for each model type (MLP, SIREN, KAN, KAN+Pruning)
- MSE values displayed on contour plots for quantitative comparison

The layout uses a 3x4 grid:
- Row 0: True function (3D surface + contour)
- Row 1: MLP and SIREN predictions (each with 3D + contour)
- Row 2: KAN and KAN+Pruning predictions (each with 3D + contour)
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
from utils import load_run, trad_nn as tnn, data_funcs as dfs
from kan import KAN


def find_latest_timestamp(section='section1_3'):
    """Find the most recent timestamp for section1_3"""
    sec_num = section.split('_')[-1]
    results_dir = Path(__file__).parent.parent / 'results' / f'sec{sec_num}_results'

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    timestamps = set()
    for f in results_dir.glob(f'{section}_*_mlp.pkl'):
        timestamp = f.stem.replace(f'{section}_', '').replace('_mlp', '')
        timestamps.add(timestamp)

    if not timestamps:
        raise FileNotFoundError(f"No results found for {section}")

    return sorted(timestamps)[-1]


def get_2d_functions():
    """Get 2D function definitions for section1_3"""
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


def load_and_reconstruct_mlp(models, results, dataset_idx, device='cpu'):
    """Load and reconstruct MLP model"""
    if 'mlp' not in models or dataset_idx not in models['mlp']:
        return None, None

    state_dict = models['mlp'][dataset_idx]
    mlp_df = results['mlp']
    mlp_dataset = mlp_df[mlp_df['dataset_idx'] == dataset_idx]

    final_epoch_rows = mlp_dataset.loc[
        mlp_dataset.groupby(['depth', 'activation'])['epoch'].idxmax()
    ]
    final_epoch_rows = final_epoch_rows[final_epoch_rows['dense_mse'].notna()]

    if len(final_epoch_rows) == 0:
        return None, None

    best_row = final_epoch_rows.loc[final_epoch_rows['dense_mse'].idxmin()]
    depth = int(best_row['depth'])
    activation = best_row['activation']
    dense_mse = best_row['dense_mse']

    model = tnn.MLP(in_features=2, width=5, depth=depth, activation=activation)

    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"  Error loading MLP: {e}")
        return None, None

    model.to(device)
    model.eval()

    config_info = {
        'depth': depth,
        'activation': activation,
        'dense_mse': dense_mse,
        'label': 'MLP'
    }

    return model, config_info


def load_and_reconstruct_siren(models, results, dataset_idx, device='cpu'):
    """Load and reconstruct SIREN model"""
    if 'siren' not in models or dataset_idx not in models['siren']:
        return None, None

    state_dict = models['siren'][dataset_idx]
    siren_df = results['siren']
    siren_dataset = siren_df[siren_df['dataset_idx'] == dataset_idx]

    final_epoch_rows = siren_dataset.loc[
        siren_dataset.groupby('depth')['epoch'].idxmax()
    ]
    final_epoch_rows = final_epoch_rows[final_epoch_rows['dense_mse'].notna()]

    if len(final_epoch_rows) == 0:
        return None, None

    best_row = final_epoch_rows.loc[final_epoch_rows['dense_mse'].idxmin()]
    depth = int(best_row['depth'])
    dense_mse = best_row['dense_mse']

    model = tnn.SIREN(in_features=2, hidden_features=5, hidden_layers=depth-2, out_features=1)

    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"  Error loading SIREN: {e}")
        return None, None

    model.to(device)
    model.eval()

    config_info = {
        'depth': depth,
        'dense_mse': dense_mse,
        'label': 'SIREN'
    }

    return model, config_info


def load_kan_model(models, results, dataset_idx, device='cpu'):
    """Load KAN model from checkpoint"""
    if 'kan' not in models or dataset_idx not in models['kan']:
        return None, None

    checkpoint_path = models['kan'][dataset_idx]
    model = KAN.loadckpt(checkpoint_path)
    model.to(device)
    model.eval()

    kan_df = results['kan']
    kan_dataset = kan_df[(kan_df['dataset_idx'] == dataset_idx) & (~kan_df['is_pruned'])]

    if len(kan_dataset) > 0:
        final_row = kan_dataset.iloc[-1]
        dense_mse = final_row['dense_mse']
        grid_size = int(final_row['grid_size'])
    else:
        dense_mse = float('nan')
        grid_size = -1

    config_info = {
        'grid_size': grid_size,
        'dense_mse': dense_mse,
        'label': 'KAN'
    }

    return model, config_info


def load_kan_pruned_model(models, results, dataset_idx, device='cpu'):
    """Load pruned KAN model from checkpoint"""
    if 'kan_pruned' not in models or dataset_idx not in models['kan_pruned']:
        return None, None

    checkpoint_path = models['kan_pruned'][dataset_idx]
    model = KAN.loadckpt(checkpoint_path)
    model.to(device)
    model.eval()

    kan_pruning_df = results['kan_pruning']
    kan_dataset = kan_pruning_df[(kan_pruning_df['dataset_idx'] == dataset_idx) & (kan_pruning_df['is_pruned'])]

    if len(kan_dataset) > 0:
        pruned_row = kan_dataset.iloc[-1]
        dense_mse = pruned_row['dense_mse']
        grid_size = int(pruned_row['grid_size'])
    else:
        dense_mse = float('nan')
        grid_size = -1

    config_info = {
        'grid_size': grid_size,
        'dense_mse': dense_mse,
        'label': 'KAN Pruning'
    }

    return model, config_info


def plot_2d_heatmap(dataset_idx, timestamp=None, device='cpu', save_path=None):
    """
    Create heatmap visualization for a 2D function fitting task.

    Args:
        dataset_idx: Index of the dataset to visualize (0-3)
        timestamp: Specific timestamp to load, or None for most recent
        device: Device to run on ('cpu' or 'cuda')
        save_path: Path to save the figure, or None to use default

    Returns:
        matplotlib figure object
    """
    section = 'section1_3'

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

    # Load models
    mlp_model, mlp_info = load_and_reconstruct_mlp(models, results, dataset_idx, device)
    siren_model, siren_info = load_and_reconstruct_siren(models, results, dataset_idx, device)
    kan_model, kan_info = load_kan_model(models, results, dataset_idx, device)
    kan_pruned_model, kan_pruned_info = load_kan_pruned_model(models, results, dataset_idx, device)

    # Get predictions
    model_data = []
    for model, info in [(mlp_model, mlp_info), (siren_model, siren_info),
                        (kan_model, kan_info), (kan_pruned_model, kan_pruned_info)]:
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
    fig.suptitle(f'2D Function Fit: {func_name}', fontsize=16, fontweight='bold')

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
    positions = [
        (1, 0),  # MLP 3D
        (1, 1),  # MLP Contour
        (1, 2),  # SIREN 3D
        (1, 3),  # SIREN Contour
        (2, 0),  # KAN 3D
        (2, 1),  # KAN Contour
        (2, 2),  # KAN Pruning 3D
        (2, 3),  # KAN Pruning Contour
    ]

    for idx, (label, y_pred, mse) in enumerate(model_data):
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

    # Save figure
    if save_path is None:
        output_dir = Path(__file__).parent
        save_path = output_dir / f'heatmap_2d_dataset_{dataset_idx}_{func_name.replace(" ", "_")}_{timestamp}.png'
    else:
        save_path = Path(save_path)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved heatmap to: {save_path}")

    return fig


def plot_all_2d_heatmaps(timestamp=None, device='cpu', output_dir=None):
    """
    Create heatmap visualizations for all 2D datasets in section1_3.

    Args:
        timestamp: Specific timestamp to load, or None for most recent
        device: Device to run on ('cpu' or 'cuda')
        output_dir: Directory to save plots (default: same directory as script)
    """
    section = 'section1_3'

    if timestamp is None:
        timestamp = find_latest_timestamp(section)
        print(f"Using most recent timestamp: {timestamp}")

    true_functions, dataset_names = get_2d_functions()

    print(f"\nGenerating heatmaps for all {len(true_functions)} datasets...")

    # Set up output directory
    if output_dir:
        save_dir = Path(output_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = None

    for dataset_idx in range(len(true_functions)):
        print(f"\n{'='*60}")
        print(f"Dataset {dataset_idx}: {dataset_names[dataset_idx]}")
        print('='*60)

        try:
            # Generate save path if output_dir is specified
            if save_dir:
                func_name = dataset_names[dataset_idx]
                save_path = save_dir / f'heatmap_2d_dataset_{dataset_idx}_{func_name.replace(" ", "_")}_{timestamp}.png'
            else:
                save_path = None

            plot_2d_heatmap(dataset_idx, timestamp, device, save_path=save_path)
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
        description='Plot 2D function fits as heatmaps (3D surface + contour)'
    )
    parser.add_argument('--dataset', type=int, default=None,
                       help='Dataset index to plot (0-3), or None for all datasets')
    parser.add_argument('--timestamp', type=str, default=None,
                       help='Specific timestamp to load (default: most recent)')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (cpu or cuda)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for the figure (default: auto-generated)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory to save plots when plotting all datasets (default: script directory)')
    parser.add_argument('--show', action='store_true',
                       help='Display the plot in a window (default: only save to file)')

    args = parser.parse_args()

    try:
        if args.dataset is None:
            # Plot all datasets
            plot_all_2d_heatmaps(timestamp=args.timestamp, device=args.device, output_dir=args.output_dir)
        else:
            # Plot specific dataset
            plot_2d_heatmap(
                dataset_idx=args.dataset,
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
        print("\nMake sure you have run the section1_3 training script first.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)