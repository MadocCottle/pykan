# Section 2: PDE Solving with KAN and Traditional Neural Networks

This directory contains a comprehensive testing infrastructure for comparing Kolmogorov-Arnold Networks (KAN) with traditional neural networks (MLP, SIREN) on various Partial Differential Equations (PDEs).

## Features

- **Multiple PDE Problems**: 2D Poisson, 1D Poisson, Heat equation, Burgers equation, Helmholtz equation
- **Multiple Models**: KAN, MLP, SIREN, PDE-optimized MLP
- **Comprehensive Metrics**:
  - MSE error on dense test sets
  - Mean Absolute Error (MAE)
  - Maximum error (L∞ norm)
  - Relative L2 error
  - H1 semi-norm (gradient norm)
  - H1 norm (Sobolev norm)
  - PDE residual
- **Training Modes**:
  - Supervised learning (fitting to solution values)
  - Physics-informed (PDE residual loss)
  - Progressive grid refinement (for KAN)

## Project Structure

```
section2/
├── README.md                 # This file
├── pde_data.py              # PDE problem definitions and data generation
├── models.py                # Neural network model definitions (MLP, SIREN)
├── metrics.py               # Evaluation metrics (MSE, H1 norm, etc.)
├── trainer.py               # Training utilities
├── run_pde_tests.py         # Main test runner script
└── example_2d_poisson.py    # Simple example script
```

## Quick Start

### Simple Example

Run a simple comparison of KAN, MLP, and SIREN on the 2D Poisson equation:

```bash
python example_2d_poisson.py
```

This will:
1. Set up the 2D Poisson equation: ∇²u = f where u(x,y) = sin(πx)sin(πy)
2. Train all three models (KAN, MLP, SIREN)
3. Evaluate on a dense 101×101 test grid
4. Print comprehensive metrics including H1 norm
5. Plot training curves

### Comprehensive Tests

Run comprehensive tests with the main runner:

```bash
# Test 2D Poisson with all models
python run_pde_tests.py --pde 2d_poisson --models kan mlp siren --epochs 100

# Test with PDE residual loss (physics-informed)
python run_pde_tests.py --pde 2d_poisson --models kan mlp --training-mode pde_residual

# Test with progressive grid refinement for KAN
python run_pde_tests.py --pde 2d_poisson --models kan_progressive --training-mode progressive

# Test different PDE problems
python run_pde_tests.py --pde 2d_poisson_multiscale --models kan mlp siren
python run_pde_tests.py --pde 1d_poisson_highfreq --models kan mlp
python run_pde_tests.py --pde 2d_helmholtz --models kan siren
```

### Available PDEs

- `2d_poisson`: 2D Poisson equation with sin(πx)sin(πy) solution
- `2d_poisson_multiscale`: 2D Poisson with multiple frequencies
- `1d_poisson`: 1D Poisson equation
- `1d_poisson_highfreq`: 1D Poisson with high frequency
- `1d_heat`: 1D heat equation
- `burgers`: Burgers equation
- `2d_helmholtz`: 2D Helmholtz equation

### Available Models

- `kan`: Kolmogorov-Arnold Network
- `kan_progressive`: KAN with progressive grid refinement
- `mlp`: Standard Multi-Layer Perceptron
- `pde_mlp`: MLP optimized for PDEs (Xavier init)
- `siren`: Sinusoidal Representation Network

## Module Documentation

### pde_data.py

Provides PDE problem definitions and data generation utilities.

**Key Functions:**
- `get_pde_problem(name)`: Get solution, source, and gradient functions for a PDE
- `create_pde_dataset_2d(solution_func, ...)`: Create 2D dataset
- `create_interior_points_2d(ranges, n_points, mode)`: Create interior points
- `create_boundary_points_2d(ranges, n_points)`: Create boundary points

**Example:**
```python
import pde_data

# Get 2D Poisson problem
sol_func, source_func, grad_func = pde_data.get_pde_problem('2d_poisson')

# Create dataset
dataset = pde_data.create_pde_dataset_2d(sol_func, ranges=[-1, 1], train_num=1000)

# Create points for PDE loss
x_interior = pde_data.create_interior_points_2d([-1, 1], n_points=51, mode='mesh')
x_boundary = pde_data.create_boundary_points_2d([-1, 1], n_points=51)
```

### metrics.py

Provides comprehensive metrics for PDE solution evaluation.

**Key Functions:**
- `compute_mse_error(model, x_test, y_test)`: MSE on test set
- `compute_h1_seminorm(model, x, true_gradient)`: H1 semi-norm (gradient norm)
- `compute_h1_norm(model, x, y_true, true_gradient)`: H1 norm (Sobolev norm)
- `compute_pde_residual_poisson_2d(model, x, source_func)`: PDE residual
- `create_dense_test_set(ranges, n_points_per_dim)`: Create dense mesh for testing

**MetricsTracker Class:**
```python
import metrics

# Create tracker
tracker = metrics.MetricsTracker(
    model,
    test_dataset,
    solution_func=sol_func,
    gradient_func=grad_func,
    source_func=source_func
)

# Compute all metrics
metrics_dict = tracker.compute_all_metrics()
# Returns: {'mse_error', 'mae_error', 'max_error', 'relative_l2_error',
#           'h1_seminorm', 'h1_norm', 'pde_residual'}

# Log metrics during training
tracker.log_metrics()  # Appends to history

# Get full history
history = tracker.get_history()
```

### models.py

Neural network model definitions.

**Available Models:**
- `MLP`: Standard MLP with configurable activation (tanh, relu, silu, gelu)
- `PDEMLP`: MLP with Xavier initialization, optional batch norm
- `SIREN`: Sinusoidal Representation Networks

**Example:**
```python
import models

# Create MLP
mlp = models.create_model('mlp', in_features=2, hidden_features=64,
                          hidden_layers=4, out_features=1, activation='tanh')

# Create SIREN
siren = models.create_model('siren', in_features=2, hidden_features=64,
                           hidden_layers=3, out_features=1,
                           first_omega_0=30.0, hidden_omega_0=30.0)

# Count parameters
n_params = models.count_parameters(mlp)
```

### trainer.py

Training utilities for PDE solving.

**PDETrainer Class:**

```python
from trainer import PDETrainer

trainer = PDETrainer(model, device='cpu')

# Supervised learning
history = trainer.train_supervised(
    dataset,
    epochs=100,
    lr=1e-3,
    optimizer_type='lbfgs',  # 'adam', 'sgd', or 'lbfgs'
    metrics_tracker=tracker,
    update_grid_every=5  # For KAN models
)

# Physics-informed training (PDE residual)
history = trainer.train_pde_residual(
    x_interior,
    x_boundary,
    solution_func,
    source_func,
    epochs=100,
    alpha=0.01,  # Weight for PDE loss
    lr=1.0,
    metrics_tracker=tracker
)
```

**KANProgressiveTrainer Class:**

```python
from trainer import KANProgressiveTrainer

prog_trainer = KANProgressiveTrainer(model, grids=[5, 10, 20], device='cpu')

history = prog_trainer.train_progressive(
    dataset,
    x_interior,
    solution_func,
    source_func,
    steps_per_grid=50,
    alpha=0.01,
    metrics_tracker=tracker
)

final_model = prog_trainer.model
```

## Metrics Explanation

### MSE Error
Standard Mean Squared Error computed on a dense test set (typically 101×101 points for 2D problems) to get accurate error estimates beyond the training/test sets.

### H1 Semi-Norm
The H1 semi-norm measures the L2 norm of the gradient:
```
||∇u||_{L2}² = ∫ |∇u|² dx
```
This is important for PDEs as it measures how well the model captures derivatives.

### H1 Norm (Sobolev Norm)
The full H1 norm combines function values and gradients:
```
||u||_{H1}² = ||u||_{L2}² + ||∇u||_{L2}²
```
This is the natural norm for many PDE problems.

### PDE Residual
For the Poisson equation ∇²u = f, the residual is:
```
Residual = ||∇²u_pred - f||_{L2}²
```
This measures how well the predicted solution satisfies the PDE.

## Usage Examples

### Example 1: Basic Comparison

```python
import torch
from kan import KAN
import pde_data, models, trainer, metrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Setup problem
sol_func, source_func, grad_func = pde_data.get_pde_problem('2d_poisson')
dataset = pde_data.create_pde_dataset_2d(sol_func, train_num=1000, device=device)

# Create dense test set
import numpy as np
x_dense = metrics.create_dense_test_set(np.array([[-1,1], [-1,1]]), 101, device)
y_dense = sol_func(x_dense)
dense_dataset = {'test_input': x_dense, 'test_label': y_dense}

# Train KAN
kan = KAN(width=[2, 5, 5, 1], grid=5, k=3, device=device).speed()
kan_tracker = metrics.MetricsTracker(kan, dense_dataset, sol_func, grad_func, source_func)
kan_trainer = trainer.PDETrainer(kan, device)
kan_history = kan_trainer.train_supervised(dataset, epochs=50, metrics_tracker=kan_tracker)

# Train MLP
mlp = models.create_model('mlp', 2, 64, 4, 1, device=device)
mlp_tracker = metrics.MetricsTracker(mlp, dense_dataset, sol_func, grad_func, source_func)
mlp_trainer = trainer.PDETrainer(mlp, device)
mlp_history = mlp_trainer.train_supervised(dataset, epochs=50, metrics_tracker=mlp_tracker)

# Compare
print("KAN:", kan_tracker.compute_all_metrics())
print("MLP:", mlp_tracker.compute_all_metrics())
```

### Example 2: Physics-Informed Training

```python
# Create interior and boundary points
x_interior = pde_data.create_interior_points_2d([-1, 1], 51, mode='mesh', device=device)
x_boundary = pde_data.create_boundary_points_2d([-1, 1], 51, device=device)

# Train with PDE residual
kan = KAN(width=[2, 5, 5, 1], grid=5, k=3, device=device).speed()
trainer_obj = trainer.PDETrainer(kan, device)

history = trainer_obj.train_pde_residual(
    x_interior,
    x_boundary,
    sol_func,
    source_func,
    epochs=100,
    alpha=0.01,
    lr=1.0
)

print("PDE Loss:", history['pde_loss'][-1])
print("BC Loss:", history['bc_loss'][-1])
```

### Example 3: Progressive Grid Refinement

```python
from trainer import KANProgressiveTrainer

# Create KAN with initial grid
kan = KAN(width=[2, 5, 5, 1], grid=5, k=3, device=device).speed()

# Progressive training
prog_trainer = KANProgressiveTrainer(kan, grids=[5, 10, 20, 50], device=device)

x_interior = pde_data.create_interior_points_2d([-1, 1], 51, device=device)

history = prog_trainer.train_progressive(
    dataset,
    x_interior,
    sol_func,
    source_func,
    steps_per_grid=50,
    alpha=0.01
)

final_kan = prog_trainer.model
```

## Results Directory

Results are saved to `sec2_results/` (configurable) with:
- `.pkl` files: Full results including training history and metrics
- `.json` files: Summary with final metrics for easy reading

## Adding New PDEs

To add a new PDE problem:

1. Define the solution, source, and gradient functions in [pde_data.py](pde_data.py:1)
2. Add to `PDE_PROBLEMS` dictionary
3. Run tests with `--pde your_pde_name`

Example:
```python
def my_custom_pde():
    sol_fun = lambda x: torch.exp(-x[:, [0]]**2 - x[:, [1]]**2)
    source_fun = lambda x: ...  # Compute -∇²sol
    grad_fun = lambda x: ...     # Compute ∇sol
    return sol_fun, source_fun, grad_fun

PDE_PROBLEMS['my_pde'] = my_custom_pde
```

## References

- KAN: [Kolmogorov-Arnold Networks](https://arxiv.org/abs/2404.19756)
- SIREN: [Implicit Neural Representations with Periodic Activation Functions](https://arxiv.org/abs/2006.09661)
- PINNs: [Physics-Informed Neural Networks](https://www.sciencedirect.com/science/article/pii/S0021999118307125)

## Notes

- The infrastructure prioritizes using pykan modules where possible
- Dense test sets (101×101 points for 2D) provide accurate error estimation
- H1 norms are computed using automatic differentiation
- All metrics are logged during training for comprehensive analysis