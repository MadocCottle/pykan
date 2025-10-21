import sys
from pathlib import Path

# Add pykan to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "models"))
sys.path.insert(0, str(Path(__file__).parent.parent / "adaptive"))

import torch
import numpy as np
import argparse

from adaptive_selective_kan import AdaptiveSelectiveKAN, AdaptiveSelectiveTrainer

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Section 2.2: Adaptive Densification')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs for training (default: 200)')
parser.add_argument('--initial_grid', type=int, default=5, help='Initial grid size (default: 5)')
parser.add_argument('--max_grid', type=int, default=15, help='Maximum grid size (default: 15)')
parser.add_argument('--densify_every', type=int, default=50, help='Densify every N epochs (default: 50)')
args = parser.parse_args()

epochs = args.epochs
initial_grid = args.initial_grid
max_grid = args.max_grid
densify_every = args.densify_every

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print(f"Running with {epochs} epochs, grid: {initial_grid}->{max_grid}, densify_every: {densify_every}")

# Section 2.2: Adaptive Densification
# ============= Create Dataset =============
print("\n" + "="*60)
print("Creating synthetic dataset...")
print("="*60)

X_train = torch.randn(500, 3).to(device)
y_train = (2*X_train[:, 0] + torch.sin(X_train[:, 1]) + X_train[:, 2]**2).reshape(-1, 1)

X_test = torch.randn(200, 3).to(device)
y_test = (2*X_test[:, 0] + torch.sin(X_test[:, 1]) + X_test[:, 2]**2).reshape(-1, 1)

# ============= Train Adaptive KAN =============
print("\n" + "="*60)
print("Training Adaptive Selective KAN...")
print("="*60)

kan = AdaptiveSelectiveKAN(
    input_dim=3,
    hidden_dim=16,
    output_dim=1,
    depth=3,
    initial_grid=initial_grid,
    max_grid=max_grid,
    kan_variant='rbf',
    device=device
)

print(f"\nModel configuration:")
print(f"  Architecture: [3, 16, 16, 16, 1]")
print(f"  Initial grid: {initial_grid}")
print(f"  Max grid: {max_grid}")
print(f"  Densify every: {densify_every} epochs")

trainer = AdaptiveSelectiveTrainer(
    kan,
    densify_every=densify_every,
    densify_k=5,
    densify_delta=2
)

history = trainer.train(
    X_train, y_train,
    epochs=epochs,
    lr=0.01,
    verbose=True
)

# ============= Evaluation =============
print("\n" + "="*60)
print("Evaluation Results")
print("="*60)

kan.eval()
with torch.no_grad():
    y_pred = kan(X_test)
    test_mse = torch.nn.functional.mse_loss(y_pred, y_test).item()

print(f"\nTest MSE: {test_mse:.6f}")

# Grid statistics
stats = kan.get_grid_statistics()
print(f"\nFinal Grid Statistics:")
print(f"  Total nodes: {stats['n_nodes']}")
print(f"  Mean grid size: {stats['mean_grid_size']:.2f}")
print(f"  Grid range: [{stats['min_grid_size']}, {stats['max_grid_size']}]")
print(f"  Total grid points: {stats['total_grid_points']:.0f}")

# Efficiency metrics
uniform_grid_points = stats['n_nodes'] * max_grid
reduction = (uniform_grid_points - stats['total_grid_points']) / uniform_grid_points * 100
print(f"\nEfficiency:")
print(f"  Uniform grid points: {uniform_grid_points:.0f}")
print(f"  Adaptive grid points: {stats['total_grid_points']:.0f}")
print(f"  Reduction: {reduction:.1f}%")

# Cleanup
kan.cleanup()

print("\n" + "="*60)
print("Section 2.2 Complete")
print("="*60)
