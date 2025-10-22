# Quick Start Guide

## Installation

No additional installation needed beyond pykan dependencies. Just navigate to the section2 directory:

```bash
cd madoc/section2
```

## 30-Second Test

Run the basic infrastructure test:

```bash
python test_basic.py
```

This verifies all modules are working correctly.

## 5-Minute Example

Run a simple comparison of KAN vs MLP vs SIREN on 2D Poisson:

```bash
python example_2d_poisson.py
```

This will:
- Train all three models on 2D Poisson equation
- Evaluate on a dense 101×101 test grid
- Compute comprehensive metrics (MSE, MAE, H1 norm, PDE residual)
- Generate a comparison plot

## Command Line Interface

### Basic Usage

```bash
# Run with default settings (2D Poisson, KAN+MLP+SIREN, 100 epochs)
python run_pde_tests.py

# Specify PDE problem
python run_pde_tests.py --pde 2d_poisson_multiscale

# Choose specific models
python run_pde_tests.py --models kan mlp

# Set training parameters
python run_pde_tests.py --epochs 200 --training-mode pde_residual
```

### Available Options

```
--pde: PDE problem
  Choices: 2d_poisson, 2d_poisson_multiscale, 1d_poisson, 1d_poisson_highfreq,
           1d_heat, burgers, 2d_helmholtz

--models: Models to test (space-separated)
  Choices: kan, kan_progressive, mlp, pde_mlp, siren

--epochs: Number of training epochs (default: 100)

--training-mode: Training approach
  Choices: supervised, pde_residual, progressive

--device: Computing device
  Choices: auto, cpu, cuda, mps

--save-dir: Results directory (default: sec2_results)

--seed: Random seed (default: 0)
```

## Common Use Cases

### Compare Models on Standard Problem

```bash
python run_pde_tests.py --pde 2d_poisson --models kan mlp siren --epochs 100
```

### Physics-Informed Training

```bash
python run_pde_tests.py --pde 2d_poisson --models kan mlp \
    --training-mode pde_residual --epochs 200
```

### Progressive Grid Refinement (KAN Only)

```bash
python run_pde_tests.py --pde 2d_poisson --models kan_progressive \
    --training-mode progressive
```

### Test on Different PDE

```bash
# High-frequency problem
python run_pde_tests.py --pde 2d_poisson_multiscale --models kan siren

# Helmholtz equation
python run_pde_tests.py --pde 2d_helmholtz --models kan mlp siren

# 1D problems
python run_pde_tests.py --pde 1d_poisson_highfreq --models kan mlp
```

## Python API

### Minimal Example

```python
import torch
from kan import KAN
import pde_data, metrics, trainer

# Setup
device = torch.device('cpu')
sol_func, src_func, grad_func = pde_data.get_pde_problem('2d_poisson')

# Create data
dataset = pde_data.create_pde_dataset_2d(sol_func, train_num=1000, device=device)

# Create model
model = KAN(width=[2, 5, 5, 1], grid=5, k=3, device=device).speed()

# Train
pde_trainer = trainer.PDETrainer(model, device)
history = pde_trainer.train_supervised(dataset, epochs=50)

# Evaluate
import numpy as np
x_test = metrics.create_dense_test_set(np.array([[-1,1], [-1,1]]), 101, device)
y_test = sol_func(x_test)
mse = metrics.compute_mse_error(model, x_test, y_test)
print(f"MSE: {mse.item():.6e}")
```

### With Full Metrics

```python
# Create dense test set
dense_dataset = {'test_input': x_test, 'test_label': y_test}

# Create tracker
tracker = metrics.MetricsTracker(
    model, dense_dataset,
    solution_func=sol_func,
    gradient_func=grad_func,
    source_func=src_func
)

# Train with tracking
history = pde_trainer.train_supervised(
    dataset, epochs=50, metrics_tracker=tracker
)

# Get all metrics
final_metrics = tracker.compute_all_metrics()
print(final_metrics)
# {'mse_error': ..., 'h1_norm': ..., 'pde_residual': ...}
```

## Understanding Results

### Saved Files

Results are saved to `sec2_results/` (or your specified directory):

```
sec2_results/
├── 2d_poisson_kan_20241020_143052.pkl    # Full results (Python pickle)
├── 2d_poisson_kan_20241020_143052.json   # Summary (human-readable)
├── 2d_poisson_mlp_20241020_143521.pkl
└── 2d_poisson_mlp_20241020_143521.json
```

### JSON Summary Format

```json
{
  "pde_name": "2d_poisson",
  "model_type": "kan",
  "n_params": 1234,
  "final_metrics": {
    "mse_error": 1.234e-05,
    "mae_error": 2.345e-03,
    "max_error": 5.678e-03,
    "relative_l2_error": 3.456e-03,
    "h1_seminorm": 1.234e-02,
    "h1_norm": 2.345e-02,
    "pde_residual": 4.567e-04
  },
  "timestamp": "2024-10-20T14:30:52.123456"
}
```

### Loading Results

```python
import pickle

# Load full results
with open('sec2_results/2d_poisson_kan_20241020_143052.pkl', 'rb') as f:
    results = pickle.load(f)

# Access data
print(results['final_metrics'])
print(results['training_history']['train_loss'])
print(results['metrics_history']['h1_norm'])
```

## Key Metrics Explained

- **MSE error**: Mean squared error on dense test set (101×101 points)
- **MAE error**: Mean absolute error
- **Max error**: Maximum absolute error (L∞ norm)
- **Relative L2 error**: ||u_pred - u_true|| / ||u_true||
- **H1 semi-norm**: ||∇(u_pred - u_true)||_{L2} (gradient error)
- **H1 norm**: Full Sobolev norm including function and gradient
- **PDE residual**: How well solution satisfies the PDE

## Tips

1. **Start small**: Use `--epochs 50` for quick tests, increase to 200+ for publication-quality results
2. **GPU acceleration**: Use `--device cuda` if available for faster training
3. **Dense test sets**: The infrastructure automatically uses 101×101 test grids for accurate metrics
4. **Progressive refinement**: For KAN models, use `--training-mode progressive` for best results
5. **Compare fairly**: Use the same number of epochs and similar parameter counts

## Troubleshooting

**Import errors**: Make sure you're running from the section2 directory or pykan root

**CUDA errors**: Use `--device cpu` if GPU memory is insufficient

**NaN losses**: Try reducing learning rate or using different optimizer

**Slow training**: Use fewer epochs initially, or reduce dense test set size in code

## Next Steps

- See [README.md](README.md) for detailed documentation
- Check [example_2d_poisson.py](example_2d_poisson.py) for complete example
- Modify [pde_data.py](pde_data.py) to add custom PDEs
- Explore different architectures in [models.py](models.py)
