"""
Plot 1D cross-sections through 3D and 4D functions.

For dimensions where spatial visualization is still feasible (3D, 4D),
this script creates 1D slices through the learned function to show
how well the KAN captures the true structure.

For a d-dimensional function f(x₁, ..., xd):
- Fix all but one variable at 0.5
- Vary one coordinate across [0,1]
- Compare true function vs KAN prediction

This provides insight into:
- How well KAN approximates along each coordinate direction
- Whether errors are uniform or concentrated in certain directions
- Qualitative assessment of function fit quality

Usage:
    python plot_cross_sections_1d.py --dim 3 --architecture deep
    python plot_cross_sections_1d.py --dim 4 --architecture shallow
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_run, data_funcs as dfs
from kan import KAN


def find_latest_highd_timestamp(dim, architecture):
    """Find the most recent timestamp for a high-D experiment"""
    results_dir = Path(__file__).parent.parent / 'results' / 'sec1_results'

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    pattern = f'section2_1_highd_{dim}d_{architecture}_*_lbfgs.pkl'
    timestamps = set()

    for f in results_dir.glob(pattern):
        parts = f.stem.split('_')
        for i, part in enumerate(parts):
            if part == architecture and i + 1 < len(parts):
                timestamp = parts[i + 1]
                if timestamp not in ['lbfgs', 'lm']:
                    timestamps.add(timestamp)

    if not timestamps:
        raise FileNotFoundError(f"No results found for {dim}D {architecture}")

    return sorted(timestamps)[-1]


def get_true_function(dim):
    """Get the true function for a given dimension"""
    functions = {
        3: dfs.f_poisson_3d_sin,
        4: dfs.f_poisson_4d_sin,
    }

    if dim not in functions:
        raise ValueError(f"1D cross-sections only supported for 3D and 4D (got {dim}D)")

    return functions[dim]


def compute_1d_cross_section(func, dim, coord_idx, num_points=100, device='cpu'):
    """
    Compute 1D cross-section along one coordinate.

    Args:
        func: Function to evaluate (true function or model)
        dim: Dimensionality (3 or 4)
        coord_idx: Which coordinate to vary (0 to dim-1)
        num_points: Number of points along the slice
        device: 'cpu' or 'cuda'

    Returns:
        x_vals: Coordinate values (numpy array)
        y_vals: Function values (numpy array)
    """
    x_vals = np.linspace(0, 1, num_points)
    y_vals = []

    for x in x_vals:
        # Create input: all coordinates at 0.5 except coord_idx
        input_point = torch.ones(1, dim, device=device) * 0.5
        input_point[0, coord_idx] = x

        # Evaluate function
        with torch.no_grad():
            if callable(func):
                # True function
                if isinstance(func, torch.nn.Module):
                    # Model
                    output = func(input_point)
                else:
                    # Lambda function
                    output = func(input_point)
            else:
                raise ValueError(f"Invalid function type: {type(func)}")

            y_vals.append(output.cpu().item())

    return x_vals, np.array(y_vals)


def plot_1d_cross_sections(dim, architecture, optimizer='lbfgs', timestamp=None, save=True, show=False):
    """
    Plot 1D cross-sections for all coordinate directions.

    Args:
        dim: Dimension (3 or 4)
        architecture: 'shallow' or 'deep'
        optimizer: Which optimizer's model to use ('lbfgs' or 'lm')
        timestamp: Specific timestamp, or None for most recent
        save: Whether to save the plot
    """
    if dim not in [3, 4]:
        raise ValueError("1D cross-sections only supported for 3D and 4D")

    # Load results and model
    if timestamp is None:
        timestamp = find_latest_highd_timestamp(dim, architecture)
        print(f"Using most recent timestamp: {timestamp}")

    section = f'section2_1_highd_{dim}d_{architecture}'
    print(f"Loading results from {section}_{timestamp}...")

    results, meta = load_run(section, timestamp)

    # Load model
    models_dir = Path(__file__).parent.parent / 'results' / 'sec1_results'
    model_file = models_dir / f'{section}_{timestamp}_models.pkl'

    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    import pickle
    with open(model_file, 'rb') as f:
        models = pickle.load(f)

    if optimizer not in models or 0 not in models[optimizer]:
        raise ValueError(f"No model found for optimizer={optimizer}, dataset_idx=0")

    # Reconstruct model
    model_state = models[optimizer][0]
    device = 'cpu'

    # Get architecture from metadata
    kan_width = meta.get('kan_width', None)
    if kan_width is None:
        # Try to infer from results
        if optimizer in results:
            arch_str = results[optimizer]['architecture'].iloc[0]
            kan_width = eval(arch_str) if isinstance(arch_str, str) else arch_str

    print(f"Reconstructing KAN with width: {kan_width}")

    # Reconstruct KAN model
    # Note: This assumes the model was saved with state_dict
    # You may need to adjust based on how models are actually saved
    try:
        model = KAN(width=kan_width, grid=3, k=3, seed=1, device=device)
        if isinstance(model_state, dict):
            model.load_state_dict(model_state)
        else:
            model = model_state
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Attempting alternative model loading...")
        # Fallback: model_state might already be the model
        model = model_state
        if hasattr(model, 'eval'):
            model.eval()

    # Get true function
    true_func = get_true_function(dim)

    # Create figure
    coord_names = ['x', 'y', 'z', 'w'][:dim]
    fig, axes = plt.subplots(1, dim, figsize=(5*dim, 4))

    if dim == 1:
        axes = [axes]

    # Plot each coordinate direction
    for coord_idx in range(dim):
        ax = axes[coord_idx]

        # Compute cross-sections
        print(f"Computing cross-section along {coord_names[coord_idx]}...")
        x_true, y_true = compute_1d_cross_section(true_func, dim, coord_idx, device=device)
        x_pred, y_pred = compute_1d_cross_section(model, dim, coord_idx, device=device)

        # Plot
        ax.plot(x_true, y_true, 'k-', linewidth=2.5, label='True', alpha=0.8)
        ax.plot(x_pred, y_pred, 'r--', linewidth=2, label=f'KAN ({optimizer.upper()})', alpha=0.8)

        # Compute error
        mse = np.mean((y_true - y_pred) ** 2)

        ax.set_xlabel(f'{coord_names[coord_idx]}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Function Value', fontsize=12, fontweight='bold')
        ax.set_title(f'Cross-section along {coord_names[coord_idx]}\n(other coords = 0.5)\nMSE: {mse:.2e}',
                    fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='best')

    plt.suptitle(f'{dim}D Poisson PDE - 1D Cross-Sections ({architecture.capitalize()} Architecture)',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save plot
    if save:
        output_dir = Path(__file__).parent
        output_file = output_dir / f'cross_sections_1d_{dim}d_{architecture}_{optimizer}_{timestamp}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {output_file}")

    return fig, axes


def plot_comparison_all_optimizers(dim, architecture, timestamp=None, save=True, show=False):
    """
    Plot 1D cross-sections comparing all optimizers.

    Args:
        dim: Dimension (3 or 4)
        architecture: 'shallow' or 'deep'
        timestamp: Specific timestamp, or None for most recent
        save: Whether to save the plot
    """
    if dim not in [3, 4]:
        raise ValueError("1D cross-sections only supported for 3D and 4D")

    # Load results
    if timestamp is None:
        timestamp = find_latest_highd_timestamp(dim, architecture)

    section = f'section2_1_highd_{dim}d_{architecture}'
    results, meta = load_run(section, timestamp)

    # Get available optimizers
    optimizers = list(results.keys())

    # Load all models
    models_dir = Path(__file__).parent.parent / 'results' / 'sec1_results'
    model_file = models_dir / f'{section}_{timestamp}_models.pkl'

    import pickle
    with open(model_file, 'rb') as f:
        all_models = pickle.load(f)

    # Get true function
    true_func = get_true_function(dim)

    # Create figure: one row per optimizer, one column per coordinate
    coord_names = ['x', 'y', 'z', 'w'][:dim]
    fig, axes = plt.subplots(len(optimizers), dim, figsize=(5*dim, 4*len(optimizers)))

    if len(optimizers) == 1:
        axes = axes.reshape(1, -1)

    device = 'cpu'
    kan_width = meta.get('kan_width', eval(results[optimizers[0]]['architecture'].iloc[0]))

    colors = {'lbfgs': '#2E86AB', 'lm': '#A23B72', 'LBFGS': '#2E86AB', 'LM': '#A23B72'}

    for opt_idx, opt in enumerate(optimizers):
        # Load model for this optimizer
        model_state = all_models[opt][0]
        model = KAN(width=kan_width, grid=3, k=3, seed=1, device=device)

        try:
            if isinstance(model_state, dict):
                model.load_state_dict(model_state)
            else:
                model = model_state
        except:
            model = model_state

        if hasattr(model, 'eval'):
            model.eval()

        for coord_idx in range(dim):
            ax = axes[opt_idx, coord_idx]

            # Compute cross-sections
            x_true, y_true = compute_1d_cross_section(true_func, dim, coord_idx, device=device)
            x_pred, y_pred = compute_1d_cross_section(model, dim, coord_idx, device=device)

            # Plot
            ax.plot(x_true, y_true, 'k-', linewidth=2.5, label='True', alpha=0.7)
            ax.plot(x_pred, y_pred, '--', linewidth=2,
                   color=colors.get(opt, 'C0'),
                   label=f'{opt.upper()}', alpha=0.8)

            mse = np.mean((y_true - y_pred) ** 2)

            if opt_idx == len(optimizers) - 1:
                ax.set_xlabel(f'{coord_names[coord_idx]}', fontsize=11, fontweight='bold')

            if coord_idx == 0:
                ax.set_ylabel(f'{opt.upper()}\nValue', fontsize=11, fontweight='bold')

            title = f'{coord_names[coord_idx]}-slice'
            if opt_idx == 0:
                title += f'\nMSE: {mse:.2e}'
            else:
                ax.set_title(f'MSE: {mse:.2e}', fontsize=9)

            if opt_idx == 0:
                ax.set_title(title, fontsize=11, fontweight='bold')

            ax.grid(True, alpha=0.3)
            if coord_idx == dim - 1:
                ax.legend(fontsize=9, loc='best')

    plt.suptitle(f'{dim}D Poisson - 1D Cross-Sections: Optimizer Comparison ({architecture.capitalize()})',
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    if save:
        output_dir = Path(__file__).parent
        output_file = output_dir / f'cross_sections_1d_comparison_{dim}d_{architecture}_{timestamp}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nComparison plot saved to: {output_file}")

    return fig, axes


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Plot 1D cross-sections for 3D/4D Poisson PDEs'
    )
    parser.add_argument('--dim', type=int, required=True, choices=[3, 4],
                       help='Dimension (3 or 4)')
    parser.add_argument('--architecture', type=str, required=True,
                       choices=['shallow', 'deep'],
                       help='Architecture type')
    parser.add_argument('--optimizer', type=str, default='lbfgs',
                       choices=['lbfgs', 'lm'],
                       help='Optimizer to visualize (default: lbfgs)')
    parser.add_argument('--timestamp', type=str, default=None,
                       help='Specific timestamp (default: most recent)')
    parser.add_argument('--compare', action='store_true',
                       help='Compare all optimizers in one plot')
    parser.add_argument('--show', action='store_true',
                       help='Display the plot in a window (default: only save to file)')


    args = parser.parse_args()

    try:
        if args.compare:
            plot_comparison_all_optimizers(
                dim=args.dim,
                architecture=args.architecture,
                timestamp=args.timestamp,
                save=True
            )
        else:
            plot_1d_cross_sections(
                dim=args.dim,
                architecture=args.architecture,
                optimizer=args.optimizer,
                timestamp=args.timestamp,
                save=True
            )

        if args.show:
            plt.show()
        else:
            plt.close('all')

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"\nMake sure you have run:")
        print(f"  python section2_1_highd.py --dim {args.dim} --architecture {args.architecture}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
