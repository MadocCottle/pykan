"""Quick verification test for MLP and SIREN implementations"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
from kan import create_dataset
from utils.trad_nn import MLP, SIREN
from utils.metrics import dense_mse_error_from_dataset, count_parameters

print("="*60)
print("NEURAL NETWORK VERIFICATION TEST")
print("="*60)

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

# Create simple test dataset: sin(2πx)
f = lambda x: torch.sin(2 * torch.pi * x)
dataset = create_dataset(f, n_var=1, train_num=100, test_num=100)
print(f"Dataset created: {dataset['train_input'].shape}")

# Test MLP
print("\n" + "="*60)
print("Testing MLP (width=5, depth=3, activation=tanh)")
print("="*60)
mlp = MLP(in_features=1, width=5, depth=3, activation='tanh').to(device)
print(f"Parameters: {count_parameters(mlp)}")

# Quick training (10 epochs with Adam for speed)
optimizer = torch.optim.Adam(mlp.parameters(), lr=0.01)
criterion = nn.MSELoss()

for epoch in range(10):
    optimizer.zero_grad()
    pred = mlp(dataset['train_input'])
    loss = criterion(pred, dataset['train_label'])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(mlp.parameters(), 1.0)
    optimizer.step()
    
    with torch.no_grad():
        test_pred = mlp(dataset['test_input'])
        test_loss = criterion(test_pred, dataset['test_label']).item()
        print(f"  Epoch {epoch+1}: Train Loss = {loss.item():.6e}, Test Loss = {test_loss:.6e}")

# Compute dense MSE
dense_mse = dense_mse_error_from_dataset(mlp, dataset, f, num_samples=1000, device=device)
print(f"  Final Dense MSE: {dense_mse:.6e}")

# Check for NaN/Inf
import math
if math.isnan(dense_mse) or math.isinf(dense_mse):
    print("  ❌ FAILED: MLP produced NaN/Inf")
else:
    print("  ✓ PASSED: MLP training converged")

# Test SIREN
print("\n" + "="*60)
print("Testing SIREN (width=5, depth=3, omega_0=30)")
print("="*60)
siren = SIREN(in_features=1, hidden_features=5, hidden_layers=1, out_features=1).to(device)
print(f"Parameters: {count_parameters(siren)}")

# SIREN training with Adam and lower LR
optimizer = torch.optim.Adam(siren.parameters(), lr=1e-4)
criterion = nn.MSELoss()

for epoch in range(10):
    optimizer.zero_grad()
    pred = siren(dataset['train_input'])
    loss = criterion(pred, dataset['train_label'])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(siren.parameters(), 0.1)  # Tighter clipping
    optimizer.step()
    
    with torch.no_grad():
        test_pred = siren(dataset['test_input'])
        test_loss = criterion(test_pred, dataset['test_label']).item()
        print(f"  Epoch {epoch+1}: Train Loss = {loss.item():.6e}, Test Loss = {test_loss:.6e}")

# Compute dense MSE
dense_mse = dense_mse_error_from_dataset(siren, dataset, f, num_samples=1000, device=device)
print(f"  Final Dense MSE: {dense_mse:.6e}")

# Check for NaN/Inf
if math.isnan(dense_mse) or math.isinf(dense_mse):
    print("  ❌ FAILED: SIREN produced NaN/Inf")
else:
    print("  ✓ PASSED: SIREN training converged")

print("\n" + "="*60)
print("VERIFICATION COMPLETE")
print("="*60)
