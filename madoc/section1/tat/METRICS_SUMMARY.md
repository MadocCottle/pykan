# Dense MSE Metrics - Summary

## What Was Added

A new `metrics.py` module in `section1/utils/` that provides dense sampling-based error metrics for better model evaluation.

## Key Features

### 1. `dense_mse_error()`
Evaluates model performance by sampling the true function much more densely (default: 10,000 samples) than typical train/test splits (100-1,000 samples).

### 2. `dense_mse_error_from_dataset()`
Convenience wrapper that automatically infers input ranges from the dataset.

### 3. `evaluate_all_models()`
Batch evaluation of multiple models across multiple datasets.

## Why This Matters

Standard MSE loss computed during training only evaluates the model at training points. Similarly, test MSE only checks a limited number of points. Dense MSE:

- **Better Coverage**: Samples the entire domain more thoroughly
- **Overfitting Detection**: Reveals if model only works at training points
- **True Generalization**: Shows performance across the whole input space
- **Fair Comparison**: Allows comparing models trained on different sample sizes

## Quick Start

```python
from kan import KAN, create_dataset
from utils.metrics import dense_mse_error_from_dataset
import torch

# Define function and create dataset
f = lambda x: torch.sin(2 * torch.pi * x)
dataset = create_dataset(f, n_var=1, train_num=100)

# Train model
model = KAN(width=[1, 5, 1], grid=5, k=3, device='cpu')
results = model.fit(dataset, opt="LBFGS", steps=20)

# Compare metrics
print(f"Train MSE:  {results['train_loss'][-1]:.6e}")
print(f"Test MSE:   {results['test_loss'][-1]:.6e}")

# Dense MSE - samples 10,000 points
dense_error = dense_mse_error_from_dataset(model, dataset, f, num_samples=10000)
print(f"Dense MSE:  {dense_error:.6e}")
```

## Files Created

1. **`utils/metrics.py`** - Core implementation
2. **`test_metrics.py`** - Basic tests demonstrating functionality
3. **`example_using_metrics.py`** - Integration examples with Section 1 code
4. **`utils/README_metrics.md`** - Complete documentation

## Integration

The metrics module is now importable via:

```python
from utils.metrics import dense_mse_error, dense_mse_error_from_dataset, evaluate_all_models
```

Or directly:

```python
from utils import dense_mse_error_from_dataset
```

## Testing

Run the test suite:
```bash
python test_metrics.py
```

Run the examples:
```bash
python example_using_metrics.py
```

## Interpretation

- **Dense MSE ≈ Test MSE**: Good generalization
- **Dense MSE >> Test MSE**: Possible overfitting to sparse samples
- **Dense MSE < Test MSE**: Model generalizes better than test set suggests

## Performance

- 1D functions: Uses uniform linspace sampling
- Multi-dimensional: Uses random sampling
- Default 10,000 samples balances accuracy and speed
- Can increase to 50,000+ for 1D functions for higher precision
