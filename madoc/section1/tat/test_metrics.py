"""Test script for the dense MSE error metrics"""
import sys
from pathlib import Path

# Add pykan to path (parent directory of madoc)
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from kan import KAN, create_dataset
import torch
from utils.metrics import dense_mse_error, dense_mse_error_from_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Test 1: Simple 1D sinusoid
print("\n" + "="*50)
print("Test 1: 1D Sinusoid - sin(2πx)")
print("="*50)

f = lambda x: torch.sin(2 * torch.pi * x)
dataset = create_dataset(f, n_var=1, train_num=100, test_num=100, device=device)

model = KAN(width=[1, 5, 1], grid=5, k=3, seed=1, device=device)
print("\nTraining model...")
results = model.fit(dataset, opt="LBFGS", steps=20, log=1)

print(f"\nTraining MSE (from last epoch): {results['train_loss'][-1]:.6e}")
print(f"Test MSE (from last epoch):     {results['test_loss'][-1]:.6e}")

# Compute dense MSE error
dense_error = dense_mse_error(model, f, n_var=1, num_samples=10000, device=device)
print(f"Dense MSE (10,000 samples):     {dense_error:.6e}")

# Alternative using dataset
dense_error_from_ds = dense_mse_error_from_dataset(model, dataset, f, num_samples=10000, device=device)
print(f"Dense MSE (from dataset):       {dense_error_from_ds:.6e}")


# Test 2: Higher frequency 1D sinusoid
print("\n" + "="*50)
print("Test 2: 1D High Frequency - sin(10πx)")
print("="*50)

f_highfreq = lambda x: torch.sin(10 * torch.pi * x)
dataset_highfreq = create_dataset(f_highfreq, n_var=1, train_num=200, test_num=100, device=device)

model2 = KAN(width=[1, 5, 1], grid=10, k=3, seed=1, device=device)
print("\nTraining model...")
results2 = model2.fit(dataset_highfreq, opt="LBFGS", steps=20, log=1)

print(f"\nTraining MSE (from last epoch): {results2['train_loss'][-1]:.6e}")
print(f"Test MSE (from last epoch):     {results2['test_loss'][-1]:.6e}")

dense_error2 = dense_mse_error_from_dataset(model2, dataset_highfreq, f_highfreq,
                                            num_samples=10000, device=device)
print(f"Dense MSE (10,000 samples):     {dense_error2:.6e}")


# Test 3: 2D function
print("\n" + "="*50)
print("Test 3: 2D Function - sin(2πx₀) * sin(2πx₁)")
print("="*50)

f_2d = lambda x: torch.sin(2*torch.pi*x[:,[0]]) * torch.sin(2*torch.pi*x[:,[1]])
dataset_2d = create_dataset(f_2d, n_var=2, train_num=500, test_num=100, device=device)

model3 = KAN(width=[2, 5, 1], grid=5, k=3, seed=1, device=device)
print("\nTraining model...")
results3 = model3.fit(dataset_2d, opt="LBFGS", steps=20, log=1)

print(f"\nTraining MSE (from last epoch): {results3['train_loss'][-1]:.6e}")
print(f"Test MSE (from last epoch):     {results3['test_loss'][-1]:.6e}")

dense_error3 = dense_mse_error_from_dataset(model3, dataset_2d, f_2d,
                                            num_samples=10000, device=device)
print(f"Dense MSE (10,000 samples):     {dense_error3:.6e}")


# Summary
print("\n" + "="*50)
print("Summary")
print("="*50)
print("\nThe dense MSE error metric provides a more comprehensive evaluation")
print("by sampling the function much more densely (10,000 points) than the")
print("typical train/test sets (100-500 points).")
print("\nThis helps identify:")
print("  - Regions where the model may be overfitting")
print("  - Areas of the input space with poor generalization")
print("  - True model performance across the entire domain")
