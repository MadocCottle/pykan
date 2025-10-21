# Dense MSE Metrics

## Overview

The `metrics.py` module provides dense sampling-based error metrics that evaluate model performance more thoroughly than standard train/test splits. By sampling the true function much more densely (default: 10,000 samples), these metrics give a better understanding of model generalization across the entire input domain.

## Why Use Dense MSE?

Standard train/test MSE only evaluates the model at a limited number of points (typically 100-1000). Dense MSE provides:

1. **Better coverage** - Samples the entire input domain more thoroughly
2. **Overfitting detection** - Reveals if the model only performs well at training points
3. **True generalization** - Shows how well the model approximates the function everywhere
4. **Comparison fairness** - Allows fair comparison between models trained on different sample sizes

## Functions

### `dense_mse_error(model, true_function, n_var, ranges, num_samples, device)`

Compute MSE by densely sampling from the true function.

**Parameters:**
- `model`: The trained model to evaluate
- `true_function`: Ground truth function `f(x)` that returns torch tensors
- `n_var`: Number of input variables (1 for 1D, 2 for 2D, etc.)
- `ranges`: Input domain ranges, e.g., `[-1, 1]` or `[[-1, 1], [0, 2]]`
- `num_samples`: Number of samples to evaluate (default: 10000)
- `device`: Computation device ('cpu' or 'cuda')

**Returns:**
- Float: Mean squared error over dense samples

**Example:**
```python
from utils.metrics import dense_mse_error

f = lambda x: torch.sin(2 * torch.pi * x)
error = dense_mse_error(model, f, n_var=1, ranges=[-1, 1],
                        num_samples=10000, device='cpu')
print(f"Dense MSE: {error:.6e}")
```

### `dense_mse_error_from_dataset(model, dataset, true_function, num_samples, device)`

Compute dense MSE with ranges automatically inferred from the dataset.

**Parameters:**
- `model`: The trained model
- `dataset`: Dataset dict with 'train_input' key
- `true_function`: Ground truth function
- `num_samples`: Number of samples (default: 10000)
- `device`: Computation device

**Returns:**
- Float: Mean squared error

**Example:**
```python
from kan import create_dataset
from utils.metrics import dense_mse_error_from_dataset

f = lambda x: torch.sin(2 * torch.pi * x)
dataset = create_dataset(f, n_var=1, train_num=100)
# ... train model ...
error = dense_mse_error_from_dataset(model, dataset, f, num_samples=10000)
```

### `evaluate_all_models(models, datasets, true_functions, num_samples, device)`

Evaluate multiple models on multiple datasets.

**Parameters:**
- `models`: Dict or list of models
- `datasets`: List of dataset dicts
- `true_functions`: List of ground truth functions
- `num_samples`: Samples per evaluation (default: 10000)
- `device`: Computation device

**Returns:**
- Dict: Nested dict mapping `{model_idx: {dataset_idx: error}}`

**Example:**
```python
from utils.metrics import evaluate_all_models

results = evaluate_all_models(
    models={0: model1, 1: model2},
    datasets=[dataset1, dataset2],
    true_functions=[f1, f2],
    num_samples=10000
)
# results[0][1] = error of model1 on dataset2
```

## Integration with Existing Code

### Basic Usage in Section 1 Experiments

```python
from kan import KAN, create_dataset
from utils import data_funcs as dfs
from utils.metrics import dense_mse_error_from_dataset

# Create dataset
f = dfs.sinusoid_1d(3)
dataset = create_dataset(f, n_var=1, train_num=1000, test_num=1000)

# Train model
model = KAN(width=[1, 5, 1], grid=10, k=3, device='cpu')
results = model.fit(dataset, opt="LBFGS", steps=20)

# Compare metrics
train_mse = results['train_loss'][-1]
test_mse = results['test_loss'][-1]
dense_mse = dense_mse_error_from_dataset(model, dataset, f, num_samples=10000)

print(f"Train MSE:  {train_mse:.6e}")
print(f"Test MSE:   {test_mse:.6e}")
print(f"Dense MSE:  {dense_mse:.6e}")
```

### Adding to model_tests.py

You can modify the testing functions to return dense MSE alongside train/test losses:

```python
from .metrics import dense_mse_error_from_dataset

def run_kan_grid_tests_with_dense_mse(datasets, grids, epochs, device,
                                       true_functions, prune=False):
    results = {}
    models = {}

    for i, dataset in enumerate(datasets):
        # ... existing training code ...

        # Add dense MSE evaluation
        dense_error = dense_mse_error_from_dataset(
            model, dataset, true_functions[i],
            num_samples=10000, device=device
        )
        results[i]['dense_mse'] = dense_error

    return results, models
```

## Interpretation Guide

### When Dense MSE ≈ Test MSE
- Model generalizes well across the domain
- Training data is representative of the full domain
- Good sign for deployment

### When Dense MSE >> Test MSE
- Possible overfitting to training/test points
- Model struggles in regions with sparse sampling
- Consider: more training data, regularization, or simpler model

### When Dense MSE < Test MSE
- Test set may have been unlucky (bad sampling)
- Model generalizes better than test set suggests
- Can happen with smooth functions and grid-based models

## Performance Notes

- For 1D functions: Uses linspace sampling for uniform coverage
- For multi-dimensional: Uses random sampling to avoid curse of dimensionality
- Default 10,000 samples provides good balance of accuracy and speed
- For 1D, you can use 50,000+ samples for very precise estimates
- For high dimensions (>3), may need fewer samples for speed

## Examples

See:
- `test_metrics.py` - Basic functionality tests
- `example_using_metrics.py` - Integration examples with Section 1 code
