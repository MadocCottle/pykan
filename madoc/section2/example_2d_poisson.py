"""
Simple example: Solving 2D Poisson equation with KAN and MLP.

This demonstrates the basic usage of the PDE testing infrastructure.
"""

import sys
from pathlib import Path

# Add pykan to path (parent directory of madoc)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from kan import KAN
import pde_data
import models
import trainer
import metrics
import matplotlib.pyplot as plt


def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Get 2D Poisson problem
    print("\n1. Setting up 2D Poisson equation")
    print("   PDE: ∇²u = f")
    print("   Solution: u(x,y) = sin(πx)sin(πy)")
    sol_func, source_func, grad_func = pde_data.get_pde_problem('2d_poisson')

    # Create dataset
    print("\n2. Creating datasets...")
    dataset = pde_data.create_pde_dataset_2d(
        sol_func,
        ranges=[-1, 1],
        train_num=1000,
        test_num=1000,
        device=device,
        seed=0
    )
    print(f"   Training points: {dataset['train_input'].shape[0]}")
    print(f"   Test points: {dataset['test_input'].shape[0]}")

    # Create dense test set for accurate metrics
    print("\n3. Creating dense test set (101x101 = 10,201 points)...")
    import numpy as np
    ranges_2d = np.array([[-1, 1], [-1, 1]])
    x_dense = metrics.create_dense_test_set(ranges_2d, 101, device=device)
    y_dense = sol_func(x_dense)
    dense_dataset = {
        'test_input': x_dense,
        'test_label': y_dense
    }
    print(f"   Dense test points: {x_dense.shape[0]}")

    # Create interior and boundary points for PDE loss
    print("\n4. Creating interior and boundary points...")
    x_interior = pde_data.create_interior_points_2d(
        ranges=[-1, 1],
        n_points=51,
        mode='mesh',
        device=device
    )
    x_boundary = pde_data.create_boundary_points_2d(
        ranges=[-1, 1],
        n_points=51,
        device=device
    )
    print(f"   Interior points: {x_interior.shape[0]}")
    print(f"   Boundary points: {x_boundary.shape[0]}")

    # ============= Train KAN =============
    print("\n" + "=" * 60)
    print("Training KAN Model")
    print("=" * 60)

    kan_model = KAN(width=[2, 5, 5, 1], grid=5, k=3, seed=0, device=device)
    kan_model = kan_model.speed()

    print(f"KAN parameters: {models.count_parameters(kan_model):,}")

    # Create metrics tracker for KAN
    kan_metrics = metrics.MetricsTracker(
        kan_model,
        dense_dataset,
        solution_func=sol_func,
        gradient_func=grad_func,
        source_func=source_func
    )

    # Train KAN using supervised learning
    kan_trainer = trainer.PDETrainer(kan_model, device)
    kan_history = kan_trainer.train_supervised(
        dataset,
        epochs=50,
        lr=1e-3,
        optimizer_type='lbfgs',
        metrics_tracker=kan_metrics,
        update_grid_every=5,
        x_interior=x_interior
    )

    # Evaluate KAN
    print("\nKAN Final Metrics on Dense Test Set:")
    kan_final_metrics = kan_metrics.compute_all_metrics()
    for key, value in kan_final_metrics.items():
        print(f"  {key}: {value:.6e}")

    # ============= Train MLP =============
    print("\n" + "=" * 60)
    print("Training MLP Model")
    print("=" * 60)

    mlp_model = models.create_model(
        'pde_mlp',
        in_features=2,
        hidden_features=64,
        hidden_layers=4,
        out_features=1,
        activation='tanh',
        device=device
    )

    print(f"MLP parameters: {models.count_parameters(mlp_model):,}")

    # Create metrics tracker for MLP
    mlp_metrics = metrics.MetricsTracker(
        mlp_model,
        dense_dataset,
        solution_func=sol_func,
        gradient_func=grad_func,
        source_func=source_func
    )

    # Train MLP
    mlp_trainer = trainer.PDETrainer(mlp_model, device)
    mlp_history = mlp_trainer.train_supervised(
        dataset,
        epochs=50,
        lr=1e-3,
        optimizer_type='lbfgs',
        metrics_tracker=mlp_metrics
    )

    # Evaluate MLP
    print("\nMLP Final Metrics on Dense Test Set:")
    mlp_final_metrics = mlp_metrics.compute_all_metrics()
    for key, value in mlp_final_metrics.items():
        print(f"  {key}: {value:.6e}")

    # ============= Train SIREN =============
    print("\n" + "=" * 60)
    print("Training SIREN Model")
    print("=" * 60)

    siren_model = models.create_model(
        'siren',
        in_features=2,
        hidden_features=64,
        hidden_layers=3,
        out_features=1,
        device=device,
        first_omega_0=30.0,
        hidden_omega_0=30.0
    )

    print(f"SIREN parameters: {models.count_parameters(siren_model):,}")

    # Create metrics tracker for SIREN
    siren_metrics = metrics.MetricsTracker(
        siren_model,
        dense_dataset,
        solution_func=sol_func,
        gradient_func=grad_func,
        source_func=source_func
    )

    # Train SIREN
    siren_trainer = trainer.PDETrainer(siren_model, device)
    siren_history = siren_trainer.train_supervised(
        dataset,
        epochs=50,
        lr=1e-3,
        optimizer_type='lbfgs',
        metrics_tracker=siren_metrics
    )

    # Evaluate SIREN
    print("\nSIREN Final Metrics on Dense Test Set:")
    siren_final_metrics = siren_metrics.compute_all_metrics()
    for key, value in siren_final_metrics.items():
        print(f"  {key}: {value:.6e}")

    # ============= Comparison =============
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    print(f"\n{'Metric':<20} {'KAN':<15} {'MLP':<15} {'SIREN':<15}")
    print("-" * 65)

    for key in kan_final_metrics.keys():
        kan_val = kan_final_metrics[key]
        mlp_val = mlp_final_metrics[key]
        siren_val = siren_final_metrics[key]
        print(f"{key:<20} {kan_val:<15.6e} {mlp_val:<15.6e} {siren_val:<15.6e}")

    # Plot training curves
    print("\n5. Plotting training curves...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Training loss
    axes[0].semilogy(kan_history['train_loss'], label='KAN', marker='o', markevery=5)
    axes[0].semilogy(mlp_history['train_loss'], label='MLP', marker='s', markevery=5)
    axes[0].semilogy(siren_history['train_loss'], label='SIREN', marker='^', markevery=5)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Training Loss')
    axes[0].set_title('Training Loss Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Test loss
    axes[1].semilogy(kan_history['test_loss'], label='KAN', marker='o', markevery=5)
    axes[1].semilogy(mlp_history['test_loss'], label='MLP', marker='s', markevery=5)
    axes[1].semilogy(siren_history['test_loss'], label='SIREN', marker='^', markevery=5)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Test Loss')
    axes[1].set_title('Test Loss Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('2d_poisson_comparison.png', dpi=150, bbox_inches='tight')
    print("   Saved plot to 2d_poisson_comparison.png")

    plt.show()


if __name__ == '__main__':
    main()