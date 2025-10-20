"""Example: Using dense MSE metrics with Section 1 experiments

This demonstrates how to use the dense_mse_error metric alongside
the standard training/test metrics to get a more comprehensive view
of model performance.
"""
from kan import KAN, create_dataset
import torch
from utils import data_funcs as dfs
from utils.metrics import dense_mse_error_from_dataset, evaluate_all_models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

# Example 1: Compare train/test MSE with dense MSE for a single model
print("="*60)
print("Example 1: Single Model Evaluation")
print("="*60)

# Create dataset for sinusoid with frequency 3
f = dfs.sinusoid_1d(3)
dataset = create_dataset(f, n_var=1, train_num=1000, test_num=1000, device=device)

# Train KAN model
model = KAN(width=[1, 5, 1], grid=10, k=3, seed=1, device=device)
print("\nTraining KAN model on sin(6πx)...")
results = model.fit(dataset, opt="LBFGS", steps=20, log=1)

# Compare metrics
train_mse = results['train_loss'][-1]
test_mse = results['test_loss'][-1]
dense_mse = dense_mse_error_from_dataset(model, dataset, f, num_samples=10000, device=device)

print(f"\nResults:")
print(f"  Train MSE (1000 samples):  {train_mse:.6e}")
print(f"  Test MSE (1000 samples):   {test_mse:.6e}")
print(f"  Dense MSE (10000 samples): {dense_mse:.6e}")
print(f"\nDense MSE / Test MSE ratio: {dense_mse/test_mse:.3f}")


# Example 2: Evaluate multiple models across datasets
print("\n" + "="*60)
print("Example 2: Multiple Models and Datasets")
print("="*60)

# Create multiple datasets
datasets = []
true_funcs = []
dataset_names = []

for freq in [1, 2, 5]:
    f = dfs.sinusoid_1d(freq)
    datasets.append(create_dataset(f, n_var=1, train_num=500, test_num=500, device=device))
    true_funcs.append(f)
    dataset_names.append(f"sin({2*freq}πx)")

# Train models with different grid sizes
models = {}
grids = [5, 10, 20]

for grid_size in grids:
    print(f"\nTraining models with grid={grid_size}...")
    models[grid_size] = []

    for i, dataset in enumerate(datasets):
        model = KAN(width=[1, 5, 1], grid=grid_size, k=3, seed=1, device=device)
        model.fit(dataset, opt="LBFGS", steps=10, log=1)
        models[grid_size].append(model)

# Evaluate with dense MSE
print("\n" + "-"*60)
print("Dense MSE Error Results (10,000 samples):")
print("-"*60)
print(f"{'Grid Size':<12} | " + " | ".join([f"{name:^15}" for name in dataset_names]))
print("-"*60)

for grid_size in grids:
    errors = []
    for i, (dataset, true_func) in enumerate(zip(datasets, true_funcs)):
        error = dense_mse_error_from_dataset(
            models[grid_size][i], dataset, true_func,
            num_samples=10000, device=device
        )
        errors.append(error)

    error_str = " | ".join([f"{err:.6e}" for err in errors])
    print(f"Grid={grid_size:<6} | {error_str}")


# Example 3: Tracking how dense MSE changes during training
print("\n" + "="*60)
print("Example 3: Tracking Dense MSE During Training")
print("="*60)

f = dfs.f_piecewise
dataset = create_dataset(f, n_var=1, train_num=500, test_num=500, device=device)

model = KAN(width=[1, 5, 1], grid=5, k=3, seed=1, device=device)

print("\nTraining piecewise function with periodic dense MSE evaluation...")
print(f"{'Epoch':<8} | {'Train MSE':<12} | {'Test MSE':<12} | {'Dense MSE':<12}")
print("-"*55)

epochs_to_track = [1, 5, 10, 15, 20]
for epoch in epochs_to_track:
    steps = 1 if epoch == 1 else (epoch - max([e for e in epochs_to_track if e < epoch]))
    results = model.fit(dataset, opt="LBFGS", steps=steps, log=1)

    train_mse = results['train_loss'][-1]
    test_mse = results['test_loss'][-1]
    dense_mse = dense_mse_error_from_dataset(model, dataset, f, num_samples=10000, device=device)

    print(f"{epoch:<8} | {train_mse:<12.6e} | {test_mse:<12.6e} | {dense_mse:<12.6e}")

print("\n" + "="*60)
print("Summary")
print("="*60)
print("""
The dense_mse_error metric provides valuable insights:

1. It reveals true generalization performance across the entire domain
2. It helps detect overfitting (dense MSE >> test MSE indicates poor
   generalization to unseen regions)
3. It's useful for comparing models trained on different sample sizes
4. For smooth functions, dense MSE often matches or is lower than test MSE
5. For complex/piecewise functions, it may reveal underfitting in certain regions

Use this metric when you need high confidence in model quality beyond
standard train/test splits.
""")
