"""
Main script to run comprehensive PDE tests comparing KAN and MLP models.

Usage:
    python run_pde_tests.py --pde 2d_poisson --models kan mlp siren --epochs 100
"""

import sys
from pathlib import Path

# Add pykan to path (parent directory of madoc)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
import pickle
from datetime import datetime
import torch
import numpy as np

from kan import KAN
import pde_data
import models
import trainer
import metrics


def run_experiment(pde_name, model_type, config, device='cpu', save_dir='results'):
    """
    Run a single experiment.

    Args:
        pde_name: Name of PDE problem
        model_type: 'kan', 'mlp', 'siren', or 'pde_mlp'
        config: Configuration dictionary
        device: Device to use
        save_dir: Directory to save results

    Returns:
        Dictionary of results
    """
    print(f"\n{'=' * 60}")
    print(f"Running: {pde_name} with {model_type}")
    print(f"{'=' * 60}")

    # Get PDE problem
    sol_func, source_func, grad_func = pde_data.get_pde_problem(pde_name)

    # Determine dimensionality
    n_dims = config.get('n_dims', 2)

    # Create datasets
    print("Creating datasets...")
    if n_dims == 1:
        dataset = pde_data.create_pde_dataset_1d(
            sol_func,
            ranges=config.get('ranges', [-1, 1]),
            train_num=config.get('train_num', 1000),
            test_num=config.get('test_num', 1000),
            device=device,
            seed=config.get('seed', 0)
        )
    else:
        dataset = pde_data.create_pde_dataset_2d(
            sol_func,
            ranges=config.get('ranges', [-1, 1]),
            train_num=config.get('train_num', 1000),
            test_num=config.get('test_num', 1000),
            device=device,
            seed=config.get('seed', 0)
        )

    # Create dense test set for accurate error measurement
    print("Creating dense test set...")
    dense_test_points = config.get('dense_test_points', 101)
    if n_dims == 1:
        x_dense = metrics.create_dense_test_set(
            config.get('ranges', [-1, 1]),
            dense_test_points,
            device=device
        )
    else:
        ranges_2d = np.array([config.get('ranges', [-1, 1])] * 2)
        x_dense = metrics.create_dense_test_set(
            ranges_2d,
            dense_test_points,
            device=device
        )

    y_dense = sol_func(x_dense)
    dense_dataset = {
        'test_input': x_dense,
        'test_label': y_dense
    }

    # Create metrics tracker
    metrics_tracker = metrics.MetricsTracker(
        None,  # Will set model later
        dense_dataset,
        solution_func=sol_func,
        gradient_func=grad_func,
        source_func=source_func
    )

    # Create model
    print(f"Creating {model_type} model...")
    if model_type == 'kan':
        width = config.get('kan_width', [n_dims, 5, 5, 1])
        grid = config.get('kan_grid', 5)
        k = config.get('kan_k', 3)
        model = KAN(width=width, grid=grid, k=k, seed=config.get('seed', 0), device=device)
        model = model.speed()
    elif model_type == 'kan_progressive':
        width = config.get('kan_width', [n_dims, 5, 5, 1])
        grids = config.get('kan_grids', [5, 10, 20])
        k = config.get('kan_k', 3)
        model = KAN(width=width, grid=grids[0], k=k, seed=config.get('seed', 0), device=device)
        model = model.speed()
    else:
        hidden_features = config.get('hidden_features', 64)
        hidden_layers = config.get('hidden_layers', 3)
        activation = config.get('activation', 'tanh')
        model = models.create_model(
            model_type,
            in_features=n_dims,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            out_features=1,
            activation=activation,
            device=device
        )

    # Update metrics tracker with model
    metrics_tracker.model = model

    # Count parameters
    n_params = models.count_parameters(model)
    print(f"Model has {n_params:,} trainable parameters")

    # Training
    training_mode = config.get('training_mode', 'supervised')
    epochs = config.get('epochs', 100)

    if training_mode == 'supervised':
        print("Training with supervised learning...")
        x_interior = pde_data.create_interior_points_2d(
            config.get('ranges', [-1, 1]),
            config.get('n_interior', 51),
            mode='mesh',
            device=device
        ) if n_dims == 2 else dataset['train_input']

        pde_trainer = trainer.PDETrainer(model, device)
        history = pde_trainer.train_supervised(
            dataset,
            epochs=epochs,
            lr=config.get('lr', 1e-3),
            optimizer_type=config.get('optimizer', 'adam'),
            metrics_tracker=metrics_tracker,
            update_grid_every=config.get('update_grid_every', 5) if model_type == 'kan' else None,
            x_interior=x_interior if model_type == 'kan' else None
        )

    elif training_mode == 'pde_residual':
        print("Training with PDE residual loss...")
        x_interior = pde_data.create_interior_points_2d(
            config.get('ranges', [-1, 1]),
            config.get('n_interior', 51),
            mode=config.get('interior_mode', 'mesh'),
            device=device,
            seed=config.get('seed', 0)
        ) if n_dims == 2 else dataset['train_input']

        x_boundary = pde_data.create_boundary_points_2d(
            config.get('ranges', [-1, 1]),
            config.get('n_boundary', 51),
            device=device
        ) if n_dims == 2 else None

        pde_trainer = trainer.PDETrainer(model, device)
        history = pde_trainer.train_pde_residual(
            x_interior,
            x_boundary,
            sol_func,
            source_func,
            epochs=epochs,
            alpha=config.get('alpha', 0.01),
            lr=config.get('lr', 1.0),
            metrics_tracker=metrics_tracker,
            update_grid_every=config.get('update_grid_every', 5)
        )

    elif training_mode == 'progressive' and model_type in ['kan', 'kan_progressive']:
        print("Training with progressive grid refinement...")
        x_interior = pde_data.create_interior_points_2d(
            config.get('ranges', [-1, 1]),
            config.get('n_interior', 51),
            mode='mesh',
            device=device
        )

        grids = config.get('kan_grids', [5, 10, 20])
        steps_per_grid = config.get('steps_per_grid', 50)

        prog_trainer = trainer.KANProgressiveTrainer(model, grids, device)
        history = prog_trainer.train_progressive(
            dataset,
            x_interior,
            sol_func,
            source_func,
            steps_per_grid=steps_per_grid,
            alpha=config.get('alpha', 0.01),
            metrics_tracker=metrics_tracker
        )
        model = prog_trainer.model  # Get final model
    else:
        raise ValueError(f"Unknown training mode: {training_mode}")

    # Final evaluation on dense test set
    print("\nFinal evaluation on dense test set...")
    metrics_tracker.model = model
    final_metrics = metrics_tracker.compute_all_metrics()

    print("\nFinal Metrics:")
    for key, value in final_metrics.items():
        print(f"  {key}: {value:.6e}")

    # Prepare results
    results = {
        'pde_name': pde_name,
        'model_type': model_type,
        'config': config,
        'n_params': n_params,
        'training_history': history,
        'metrics_history': metrics_tracker.get_history(),
        'final_metrics': final_metrics,
        'timestamp': datetime.now().isoformat()
    }

    # Save results
    save_results(results, pde_name, model_type, save_dir)

    return results


def save_results(results, pde_name, model_type, save_dir):
    """Save results to disk."""
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True, parents=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{pde_name}_{model_type}_{timestamp}"

    # Save as pickle
    with open(save_path / f"{filename}.pkl", 'wb') as f:
        pickle.dump(results, f)

    # Save metrics as JSON (for easy reading)
    json_results = {
        'pde_name': results['pde_name'],
        'model_type': results['model_type'],
        'n_params': results['n_params'],
        'final_metrics': results['final_metrics'],
        'timestamp': results['timestamp']
    }
    with open(save_path / f"{filename}.json", 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"\nResults saved to {save_path / filename}")


def main():
    parser = argparse.ArgumentParser(description='Run PDE tests with KAN and MLP models')
    parser.add_argument('--pde', type=str, default='2d_poisson',
                       choices=list(pde_data.PDE_PROBLEMS.keys()),
                       help='PDE problem to solve')
    parser.add_argument('--models', type=str, nargs='+',
                       default=['kan', 'mlp', 'siren'],
                       choices=['kan', 'kan_progressive', 'mlp', 'pde_mlp', 'siren'],
                       help='Models to test')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--training-mode', type=str, default='supervised',
                       choices=['supervised', 'pde_residual', 'progressive'],
                       help='Training mode')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (auto, cpu, cuda, mps)')
    parser.add_argument('--save-dir', type=str, default='sec2_results',
                       help='Directory to save results')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed')

    args = parser.parse_args()

    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Configuration
    config = {
        'epochs': args.epochs,
        'training_mode': args.training_mode,
        'seed': args.seed,
        'n_dims': 2 if '2d' in args.pde else 1,
        'ranges': [-1, 1],
        'train_num': 1000,
        'test_num': 1000,
        'dense_test_points': 101,
        # KAN specific
        'kan_width': [2, 5, 5, 1] if '2d' in args.pde else [1, 5, 5, 1],
        'kan_grid': 5,
        'kan_grids': [5, 10, 20],
        'kan_k': 3,
        'steps_per_grid': 50,
        'update_grid_every': 5,
        # MLP specific
        'hidden_features': 64,
        'hidden_layers': 4,
        'activation': 'tanh',
        # Training
        'optimizer': 'lbfgs',
        'lr': 1.0,
        'alpha': 0.01,
        'n_interior': 51,
        'n_boundary': 51,
        'interior_mode': 'mesh'
    }

    # Run experiments
    all_results = []
    for model_type in args.models:
        try:
            results = run_experiment(
                args.pde,
                model_type,
                config,
                device=device,
                save_dir=args.save_dir
            )
            all_results.append(results)
        except Exception as e:
            print(f"\nError running {model_type} on {args.pde}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for result in all_results:
        print(f"\n{result['model_type']} ({result['n_params']:,} params):")
        for key, value in result['final_metrics'].items():
            print(f"  {key}: {value:.6e}")


if __name__ == '__main__':
    main()