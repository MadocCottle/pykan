import sys
from pathlib import Path

# Add pykan to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "models"))

import torch
import numpy as np
import argparse

from heterogeneous_kan import HeterogeneousBasisKAN

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Section 2.3: Heterogeneous Basis Functions')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs for training (default: 200)')
args = parser.parse_args()

epochs = args.epochs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print(f"Running with {epochs} epochs")

# Section 2.3: Heterogeneous Basis Functions
# ============= Create Dataset =============
print("\n" + "="*60)
print("Creating synthetic dataset with mixed signals...")
print("="*60)

# Create mixed signal: Feature 0 is periodic, Feature 1 is smooth
X_train = torch.randn(500, 2).to(device)
y_train = (torch.sin(5 * X_train[:, 0]) + X_train[:, 1]**2).reshape(-1, 1)

X_test = torch.randn(200, 2).to(device)
y_test = (torch.sin(5 * X_test[:, 0]) + X_test[:, 1]**2).reshape(-1, 1)

print("Dataset features:")
print("  Feature 0: Periodic signal (sin)")
print("  Feature 1: Smooth signal (x^2)")

# ============= Baseline: Uniform Basis =============
print("\n" + "="*60)
print("Training Baseline (Uniform RBF)...")
print("="*60)

kan_uniform = HeterogeneousBasisKAN(
    layer_dims=[2, 10, 1],
    basis_config='rbf'
).to(device)

optimizer = torch.optim.Adam(kan_uniform.parameters(), lr=0.01)
for epoch in range(epochs):
    optimizer.zero_grad()
    pred = kan_uniform(X_train)
    loss = torch.nn.functional.mse_loss(pred, y_train)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

kan_uniform.eval()
with torch.no_grad():
    y_pred_uniform = kan_uniform(X_test)
    mse_uniform = torch.nn.functional.mse_loss(y_pred_uniform, y_test).item()

print(f"Baseline (uniform RBF) Test MSE: {mse_uniform:.6f}")

# ============= Heterogeneous Basis =============
print("\n" + "="*60)
print("Training Heterogeneous KAN...")
print("="*60)

# Use Fourier for periodic feature, RBF for smooth feature
kan_hetero = HeterogeneousBasisKAN(
    layer_dims=[2, 10, 1],
    basis_config=[
        {0: 'fourier', 1: 'rbf'},  # Layer 0: different bases per input
        'rbf'                      # Layer 1: uniform RBF
    ]
).to(device)

print("\nBasis assignment:")
print("  Layer 0, Input 0 (periodic): Fourier")
print("  Layer 0, Input 1 (smooth): RBF")
print("  Layer 1: RBF (all edges)")

optimizer = torch.optim.Adam(kan_hetero.parameters(), lr=0.01)
for epoch in range(epochs):
    optimizer.zero_grad()
    pred = kan_hetero(X_train)
    loss = torch.nn.functional.mse_loss(pred, y_train)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

kan_hetero.eval()
with torch.no_grad():
    y_pred_hetero = kan_hetero(X_test)
    mse_hetero = torch.nn.functional.mse_loss(y_pred_hetero, y_test).item()

# ============= Evaluation =============
print("\n" + "="*60)
print("Evaluation Results")
print("="*60)

print(f"\nUniform basis (RBF) MSE: {mse_uniform:.6f}")
print(f"Heterogeneous basis MSE: {mse_hetero:.6f}")
print(f"Improvement: {((mse_uniform - mse_hetero) / mse_uniform * 100):.2f}%")

# Get basis usage statistics
usage = kan_hetero.get_all_basis_usage()
print("\nBasis usage by layer:")
for layer_idx, layer_usage in usage.items():
    print(f"  Layer {layer_idx}: {dict(layer_usage)}")

print("\n" + "="*60)
print("Section 2.3 Complete")
print("="*60)
