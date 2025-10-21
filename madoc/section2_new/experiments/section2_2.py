import sys
from pathlib import Path

# Add pykan to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "models"))

import torch
import numpy as np
import argparse

from adaptive_selective_kan import AdaptiveSelectiveKAN, AdaptiveSelectiveTrainer

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Section 2.2: Adaptive Densification')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs for training (default: 200)')
parser.add_argument('--initial_grid', type=int, default=5, help='Initial grid size (default: 5)')
parser.add_argument('--max_grid', type=int, default=15, help='Maximum grid size (default: 15)')
args = parser.parse_args()

epochs = args.epochs
initial_grid = args.initial_grid
max_grid = args.max_grid

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print(f"Running with {epochs} epochs, initial_grid={initial_grid}, max_grid={max_grid}")

# Section 2.2: Adaptive Densification
# ============= Create Dataset =============
print("\n" + "="*60)
print("Creating synthetic dataset...")
print("="*60)

X_train = torch.randn(500, 3).to(device)
y_train = (2*X_train[:, 0] + torch.sin(X_train[:, 1]) + X_train[:, 2]**2).reshape(-1, 1)

X_test = torch.randn(200, 3).to(device)
y_test = (2*X_test[:, 0] + torch.sin(X_test[:, 1]) + X_test[:, 2]**2).reshape(-1, 1)

# ============= Create Adaptive KAN =============
print("\n" + "="*60)
print("Creating Adaptive Selective KAN...")
print("="*60)

kan = AdaptiveSelectiveKAN(
    input_dim=3,
    hidden_dim=10,
    output_dim=1,
    depth=2,
    initial_grid=initial_grid,
    max_grid=max_grid,
    kan_variant='rbf',
    device=device
)

# Get initial statistics
stats_before = kan.get_grid_statistics()
print(f"\nInitial grid statistics:")
print(f"  Mean grid size: {stats_before['mean_grid_size']:.1f}")
print(f"  Min/Max: [{stats_before['min_grid_size']}, {stats_before['max_grid_size']}]")
print(f"  Total grid points: {stats_before['total_grid_points']}")

# ============= Train with Adaptive Densification =============
print("\n" + "="*60)
print("Training with Adaptive Densification...")
print("="*60)

trainer = AdaptiveSelectiveTrainer(
    model=kan,
    densify_every=50,
    densify_k=2,
    densify_delta=2
)

history = trainer.train(X_train, y_train, epochs=epochs, lr=0.01, verbose=True)

# ============= Evaluation =============
print("\n" + "="*60)
print("Evaluation Results")
print("="*60)

# Get final statistics
stats_after = kan.get_grid_statistics()
print(f"\nFinal grid statistics:")
print(f"  Mean grid size: {stats_after['mean_grid_size']:.1f}")
print(f"  Min/Max: [{stats_after['min_grid_size']}, {stats_after['max_grid_size']}]")
print(f"  Total grid points: {stats_after['total_grid_points']}")

# Calculate savings
uniform_grid_points = 10 * stats_after['max_grid_size']
grid_point_reduction = ((uniform_grid_points - stats_after['total_grid_points']) / uniform_grid_points * 100)

print(f"\nEfficiency:")
print(f"  Uniform densification: {uniform_grid_points} grid points")
print(f"  Adaptive densification: {stats_after['total_grid_points']} grid points")
print(f"  Reduction: {grid_point_reduction:.1f}%")

# Test accuracy
kan.eval()
with torch.no_grad():
    y_pred = kan(X_test)
    test_mse = torch.nn.functional.mse_loss(y_pred, y_test).item()

print(f"\nTest MSE: {test_mse:.6f}")
print(f"Final training loss: {history['loss_history'][-1]:.6f}")
print(f"Densification events: {len(history['densification_epochs'])}")

print("\n" + "="*60)
print("Section 2.2 Complete")
print("="*60)
