# Section 2: Optimizer Selection & Hyperparameter Sensitivity - Implementation Specification

**Version:** 1.0.0
**Purpose:** Systematic comparison of optimizers for KAN-based PDE surrogates
**Dependencies:** Section 1 infrastructure (models, utils, training)

---

## Overview

Section 2 investigates optimizer performance and hyperparameter sensitivity for Kolmogorov-Arnold Networks (KANs) on function approximation and physics-informed neural network (PINN) tasks. It provides a comprehensive framework for comparing different optimization algorithms and identifying the best optimizer for specific task types.

### Research Motivation

Based on [Krishnapriyan et al. (2021)](https://arxiv.org/abs/2205.07430):
- Adam/BFGS/L-BFGS struggle with low-amplitude components
- Levenberg-Marquardt (LM) converges to machine precision rapidly
- Adam requires **26× more parameters** to match LM performance
- Second-order methods may be more suitable for smooth PDE loss landscapes

### Key Research Questions

1. Which optimizer works best for KAN on function approximation?
2. Which optimizer works best for KAN-based PINNs?
3. How do convergence speed and final accuracy trade off?
4. Are second-order methods (L-BFGS, LM) beneficial for small KANs?

---

## Project Structure

```
section2/
├── README.md                          # User-facing documentation
├── IMPLEMENTATION_SPEC.md             # This file - technical specification
├── __init__.py                        # Package metadata
├── run_section_2.py                   # Main experiment runner with timeout management
├── experiments/
│   ├── __init__.py
│   └── optimizer_comparison.py        # Core experiment implementation
├── utils/
│   └── __init__.py                    # Re-exports from section1.utils
├── visualization/
│   ├── __init__.py
│   └── convergence_plots.py           # Plotting and visualization
└── results/                           # Generated results (not version controlled)
    └── section_2_1/
        ├── optimizer_comparison_function_approx.json
        ├── optimizer_comparison_piecewise.json
        ├── optimizer_comparison_poisson_1d.json
        ├── optimizer_comparison_summary.json
        ├── metadata.json
        └── figures/
            ├── function_approx_bar.png
            ├── function_approx_efficiency.png
            ├── piecewise_bar.png
            ├── poisson_1d_bar.png
            └── optimizer_heatmap_all_tasks.png
```

---

## Core Components

### 1. Main Experiment Runner (`run_section_2.py`)

**Purpose:** Command-line interface with timeout enforcement to prevent hanging on slow optimizers.

#### Features

- **Timeout Management:** Each experiment has hard timeout limits
- **Multiple Run Modes:** Quick test, full run, individual experiments
- **Error Recovery:** Graceful handling of optimizer failures
- **Progress Reporting:** Real-time feedback on experiment status

#### Command-Line Interface

```bash
# Quick test (reduced parameters, ~2 minutes)
python run_section_2.py --quick

# Full run with all experiments (~30 minutes)
python run_section_2.py --all

# Specific experiments
python run_section_2.py --function-approx  # ~5 min
python run_section_2.py --pde-1d          # ~10 min
python run_section_2.py --pde-2d          # ~20 min (if implemented)

# Custom parameters
python run_section_2.py --all --n-seeds 3 --n-epochs 1000 --timeout 600
```

#### Key Functions

**`run_with_timeout(func, timeout_seconds, *args, **kwargs)`**
- Executes experiment with hard timeout
- Uses `section1.utils.timeout_enforcer.enforce_timeout`
- Returns result or None on timeout/error
- Prints elapsed time and status

**`run_quick_test(output_dir)`**
- Minimal parameter test (~2 minutes)
- Parameters:
  - Frequencies: [1, 5]
  - Seeds: 2
  - Epochs: 500
  - Optimizers: ['Adam', 'AdamW', 'LBFGS']
  - Timeout: 60s per experiment

**`run_function_approx_full(output_dir)`**
- Full function approximation comparison
- Parameters:
  - Frequencies: [1, 5, 8]
  - Training points: 256
  - Evaluation points: 1000
  - Epochs: 2000
  - Seeds: 5
  - Timeout: 300s (5 minutes)

**`run_pde_1d_full(output_dir)`**
- Full 1D Poisson PDE comparison
- Parameters:
  - Interior points: 100
  - Boundary points: 2
  - Evaluation points: 200
  - Epochs: 2000
  - Seeds: 3
  - Timeout: 600s (10 minutes)

**`run_pde_2d_full(output_dir)`**
- Full 2D Poisson PDE comparison (optional)
- Parameters:
  - Interior points: 400
  - Boundary points: 40
  - Evaluation points: 1000
  - Epochs: 3000
  - Seeds: 3
  - Timeout: 1200s (20 minutes)

**`run_all_experiments(output_dir)`**
- Runs all experiments sequentially
- Total estimated time: 30-40 minutes
- Generates summary report

---

### 2. Optimizer Comparison Experiments (`experiments/optimizer_comparison.py`)

**Purpose:** Core experiment logic for comparing optimizers across different tasks.

#### Class: `OptimizerComparisonExperiment`

**Constructor Parameters:**
- `output_dir` (str): Directory for saving results (default: "results/section_2_1")
- `device` (Optional[str]): Compute device (default: auto-detected)

**Attributes:**
- `output_dir` (Path): Results directory
- `device` (str): Compute device
- `optimizers` (List[str]): List of optimizers to test
  - Default: `['Adam', 'AdamW', 'LBFGS', 'SGD']`
  - Optional: `'LM'` (Levenberg-Marquardt, if stable)

#### Core Methods

##### `run_function_approx_comparison(frequencies, n_train, n_eval, n_epochs, n_seeds)`

**Purpose:** Compare optimizers on 1D sinusoidal function approximation.

**Parameters:**
- `frequencies` (List[float]): Frequencies to test (e.g., [1, 5, 8] Hz)
- `n_train` (int): Number of training points (default: 256)
- `n_eval` (int): Number of evaluation points (default: 1000)
- `n_epochs` (int): Training epochs (default: 2000)
- `n_seeds` (int): Random seeds for statistical significance (default: 5)

**Procedure:**
1. For each optimizer in `self.optimizers`:
   - For each frequency:
     - For each seed:
       - Set random seed
       - Generate training data: `x_train, y_train = generate_1d_sinusoid(n_train, frequency, device)`
       - Generate evaluation data: `x_eval, y_eval = generate_1d_sinusoid(n_eval, frequency, device)`
       - Create KAN model: `KAN(input_dim=1, hidden_dim=32, depth=2, num_knots=8)`
       - Create optimizer with task-specific learning rate:
         - SGD: lr=1e-2
         - Others: lr=1e-3
       - Train using `Trainer(model, optimizer, device).train(x_train, y_train, n_epochs, verbose=False)`
       - Record wall-clock time
       - Evaluate: `compute_all_metrics(model, x_eval, y_eval, task_type='function_approx')`
       - Collect metrics: l2_error, linf_error, relative_l2, wall_time
     - Aggregate metrics over seeds: `aggregate_metrics_over_seeds(seed_results)`
     - Compute mean and std for time
2. Save results to JSON: `optimizer_comparison_function_approx.json`

**Returns:** Dictionary with structure:
```python
{
    'task': 'function_approximation_1d_sinusoid',
    'frequencies': [1, 5, 8],
    'optimizers': {
        'Adam': {
            'frequencies': {
                1: {'l2_error': {'mean': 0.001, 'std': 0.0002}, 'mean_time': 5.2, ...},
                5: {...},
                8: {...}
            }
        },
        'AdamW': {...},
        ...
    }
}
```

##### `run_pde_1d_comparison(n_interior, n_boundary, n_eval, n_epochs, n_seeds)`

**Purpose:** Compare optimizers on 1D Poisson PDE.

**PDE Definition:**
- Equation: `-u''(x) = f(x)` for x ∈ [0,1]
- Boundary conditions: `u(0) = u(1) = 0`
- Forcing function: `f(x) = -π² sin(πx)`
- Analytical solution: `u(x) = sin(πx)`

**Parameters:**
- `n_interior` (int): Interior collocation points (default: 100)
- `n_boundary` (int): Boundary points (default: 2)
- `n_eval` (int): Evaluation grid points (default: 200)
- `n_epochs` (int): Training epochs (default: 2000)
- `n_seeds` (int): Random seeds (default: 3)

**Procedure:**
1. Define forcing function: `forcing_fn = lambda x: -pi^2 * sin(pi*x)`
2. For each optimizer:
   - For each seed:
     - Set random seed
     - Generate PDE problem: `poisson_1d_problem(n_interior, n_boundary, forcing_fn, device)`
       - Returns: `{'x_interior': Tensor, 'x_boundary': Tensor, 'u_boundary': Tensor}`
     - Create KAN model: `KAN(input_dim=1, hidden_dim=32, depth=2, num_knots=8)`
     - Create optimizer with task-specific learning rate:
       - SGD: lr=5e-3
       - Others: lr=5e-4
     - Train PINN: `PINNTrainer(model, optimizer, device).train(...)`
       - Parameters: pde_weight=1.0, bc_weight=10.0, pde_type='1d'
     - Record wall-clock time
     - Evaluate on fine grid: `x_eval = linspace(0, 1, n_eval)`
     - Compute analytical solution: `u_exact = sin(pi*x_eval)`
     - Compute metrics: `compute_all_metrics(model, x_eval, u_exact, task_type='pde_1d')`
   - Aggregate metrics over seeds
3. Save results to JSON: `optimizer_comparison_poisson_1d.json`

**Returns:** Dictionary with structure:
```python
{
    'task': 'poisson_1d',
    'optimizers': {
        'Adam': {
            'l2_error': {'mean': 0.01, 'std': 0.002},
            'mean_time': 12.5,
            'std_time': 1.2,
            ...
        },
        'LBFGS': {...},
        ...
    }
}
```

##### `run_piecewise_comparison(n_train, n_eval, n_epochs, n_seeds)`

**Purpose:** Compare optimizers on piecewise constant function (challenging for gradient-based methods).

**Function Definition:**
- Breakpoints: [0.3, 0.6, 0.8]
- Values: [-0.5, 0.3, 1.0, -0.2]
- Creates discontinuous target function

**Parameters:**
- `n_train` (int): Training points (default: 256)
- `n_eval` (int): Evaluation points (default: 1000)
- `n_epochs` (int): Training epochs (default: 2000)
- `n_seeds` (int): Random seeds (default: 5)

**Procedure:**
1. Define breakpoints and values
2. For each optimizer:
   - For each seed:
     - Generate piecewise data: `generate_1d_piecewise(n_train, breakpoints, values, device)`
     - Create KAN with more knots: `KAN(input_dim=1, hidden_dim=32, depth=2, num_knots=12)`
     - Create optimizer (same learning rates as function_approx)
     - Train using standard `Trainer`
     - Evaluate and collect metrics
   - Aggregate over seeds
3. Save results to JSON: `optimizer_comparison_piecewise.json`

##### `run_all()`

**Purpose:** Run all experiments and generate summary.

**Procedure:**
1. Run `run_function_approx_comparison()`
2. Run `run_piecewise_comparison()`
3. Run `run_pde_1d_comparison()`
4. Generate summary: `_generate_summary(results)`
5. Return combined results

##### `_generate_summary(results)`

**Purpose:** Analyze results and identify best optimizer per task.

**Procedure:**
1. For each task in results:
   - Extract L2 errors for each optimizer
   - If multiple frequencies, average over frequencies
   - Rank optimizers by L2 error (ascending)
   - Identify best optimizer (lowest error)
2. Create summary structure:
   ```python
   {
       'best_optimizers': {
           'function_approx': {'optimizer': 'LBFGS', 'l2_error': 0.0001},
           'piecewise': {'optimizer': 'Adam', 'l2_error': 0.05},
           'poisson_1d': {'optimizer': 'LBFGS', 'l2_error': 0.001}
       },
       'optimizer_rankings': {
           'function_approx': [
               {'optimizer': 'LBFGS', 'l2_error': 0.0001},
               {'optimizer': 'Adam', 'l2_error': 0.0005},
               ...
           ],
           ...
       }
   }
   ```
3. Save to JSON: `optimizer_comparison_summary.json`
4. Print formatted summary to console

#### Helper Functions

**`set_seed(seed)`**
- Sets torch, numpy, and CUDA random seeds for reproducibility

---

### 3. Visualization (`visualization/convergence_plots.py`)

**Purpose:** Generate publication-quality plots for optimizer comparison.

#### Visualization Functions

##### `plot_convergence_curves(training_histories, output_file, title, log_scale)`

**Purpose:** Plot loss vs epoch for multiple optimizers.

**Parameters:**
- `training_histories` (Dict[str, List[float]]): Optimizer name → loss history
- `output_file` (str): Output file path
- `title` (str): Plot title
- `log_scale` (bool): Use log scale for y-axis (default: True)

**Plot Features:**
- Multi-colored lines (one per optimizer)
- X-axis: Epoch
- Y-axis: Loss (log scale)
- Legend with optimizer names
- Grid for readability
- DPI: 300 (publication quality)

##### `plot_efficiency_frontier(optimizer_results, output_file, title, metric)`

**Purpose:** Plot error vs wall-clock time (efficiency frontier).

**Parameters:**
- `optimizer_results` (Dict[str, Dict]): Optimizer name → results dict
- `output_file` (str): Output file path
- `title` (str): Plot title
- `metric` (str): Metric for y-axis (default: 'l2_error')

**Plot Features:**
- Scatter plot: x=time, y=error
- Error bars showing standard deviation
- Log scale on y-axis
- Large markers (s=200) for visibility
- Lower-left corner is optimal (fast + accurate)
- Identifies "best bang for buck" optimizer

##### `plot_optimizer_heatmap(results_by_task, output_file, metric, title)`

**Purpose:** Create heatmap of optimizer performance across tasks.

**Parameters:**
- `results_by_task` (Dict[str, Dict[str, Dict]]): task → optimizer → results
- `output_file` (str): Output file path
- `metric` (str): Metric to visualize (default: 'l2_error')
- `title` (str): Plot title

**Data Processing:**
- Creates matrix: rows=tasks, columns=optimizers
- Normalizes by row (best optimizer per task = 1.0)
- NaN for missing data

**Plot Features:**
- Colormap: RdYlGn_r (reversed, so green=good)
- Value range: [1.0, 3.0]
- Text annotations with normalized values
- Colorbar showing "Relative Error (1.0 = best)"

##### `plot_optimizer_comparison_bar(optimizer_results, output_file, metric, title)`

**Purpose:** Create bar plot comparing optimizers by error.

**Parameters:**
- `optimizer_results` (Dict[str, Dict]): Optimizer → results
- `output_file` (str): Output file path
- `metric` (str): Metric to compare (default: 'l2_error')
- `title` (str): Plot title

**Plot Features:**
- Bars sorted by ascending error (best first)
- Error bars showing standard deviation
- Log scale on y-axis
- Value labels on top of bars (scientific notation)
- Color gradient (viridis)

##### `create_optimizer_report_figures(results_dir, output_dir)`

**Purpose:** Generate all figures from JSON results.

**Procedure:**
1. Load all result JSON files from `results_dir`:
   - `optimizer_comparison_function_approx.json`
   - `optimizer_comparison_piecewise.json`
   - `optimizer_comparison_poisson_1d.json`
2. For each task:
   - Generate bar plot: `{task_name}_bar.png`
   - Generate efficiency plot: `{task_name}_efficiency.png`
3. Generate combined heatmap: `optimizer_heatmap_all_tasks.png`
4. Save all figures to `output_dir`

**Usage:**
```python
create_optimizer_report_figures(
    results_dir=Path("results/section_2_1"),
    output_dir=Path("results/section_2_1/figures")
)
```

---

### 4. Utilities (`utils/__init__.py`)

**Purpose:** Re-export optimizer utilities from Section 1.

**Exports:**
- `create_optimizer(model, optimizer_name, lr, **kwargs)` - Optimizer factory
- `get_optimizer_config(optimizer_name)` - Get default config for optimizer
- `create_scheduler(optimizer, scheduler_name, **kwargs)` - Learning rate scheduler factory
- `OPTIMIZER_METADATA` - Dict with optimizer descriptions and references

**Implementation:**
```python
from section1.utils import (
    create_optimizer,
    get_optimizer_config,
    create_scheduler,
    OPTIMIZER_METADATA
)
```

---

## Optimizer Specifications

### Supported Optimizers

| Optimizer | Type | Parameters | Use Case | Notes |
|-----------|------|------------|----------|-------|
| **Adam** | First-order adaptive | lr=1e-3, betas=(0.9, 0.999) | Baseline for all tasks | Standard choice |
| **AdamW** | First-order adaptive | lr=1e-3, weight_decay=0.01 | Regularized training | Better generalization |
| **L-BFGS** | Quasi-Newton | lr=1.0, max_iter=20, history_size=100 | Smooth PDEs | Full-batch only |
| **SGD** | First-order | lr=1e-2, momentum=0.9, nesterov=True | Classical baseline | Tests if adaptive methods needed |
| **LM** (optional) | Second-order | lr=1e-3, damping=1e-3 | Small networks, MSE loss | Experimental, may be unstable |

### Learning Rate Guidelines

**Function Approximation:**
- Adam/AdamW/LBFGS: `1e-3`
- SGD: `1e-2` (needs higher lr for momentum)
- LM: `1e-3`

**PDE (PINN):**
- Adam/AdamW/LBFGS: `5e-4` (lower for stability)
- SGD: `5e-3`
- LM: `5e-4`

### Optimizer-Specific Considerations

**L-BFGS:**
- Requires closure function (implemented in `Trainer` and `PINNTrainer`)
- Full-batch training (no mini-batching)
- Higher memory usage (stores history)
- May diverge on non-smooth functions

**Levenberg-Marquardt:**
- Only for MSE loss
- Requires Jacobian computation
- Excellent for small-medium networks
- May have numerical instability

---

## Data Flow

### Typical Experiment Flow

```
1. run_section_2.py (CLI)
   ↓
2. run_with_timeout() wrapper
   ↓
3. OptimizerComparisonExperiment.run_*_comparison()
   ↓
4. For each optimizer:
   ├─ For each seed:
   │  ├─ set_seed(seed)
   │  ├─ Generate data (section1.utils)
   │  ├─ Create model (section1.models.KAN)
   │  ├─ Create optimizer (section1.utils.create_optimizer)
   │  ├─ Train (section1.training.Trainer/PINNTrainer)
   │  ├─ Evaluate (section1.utils.compute_all_metrics)
   │  └─ Record metrics + time
   └─ Aggregate over seeds
   ↓
5. Save results to JSON
   ↓
6. create_optimizer_report_figures()
   ↓
7. Generate plots
```

### Result JSON Schema

**Function Approximation:**
```json
{
  "task": "function_approximation_1d_sinusoid",
  "frequencies": [1, 5, 8],
  "optimizers": {
    "Adam": {
      "frequencies": {
        "1": {
          "l2_error": {"mean": 0.001, "std": 0.0001, "all_values": [...]},
          "linf_error": {"mean": 0.002, "std": 0.0002, "all_values": [...]},
          "relative_l2": {"mean": 0.05, "std": 0.01, "all_values": [...]},
          "mean_time": 5.2,
          "std_time": 0.3
        },
        "5": {...},
        "8": {...}
      }
    },
    "AdamW": {...},
    "LBFGS": {...},
    "SGD": {...}
  }
}
```

**PDE:**
```json
{
  "task": "poisson_1d",
  "optimizers": {
    "Adam": {
      "l2_error": {"mean": 0.01, "std": 0.002, "all_values": [...]},
      "linf_error": {"mean": 0.02, "std": 0.004, "all_values": [...]},
      "relative_l2": {"mean": 0.1, "std": 0.02, "all_values": [...]},
      "pde_residual": {"mean": 0.005, "std": 0.001, "all_values": [...]},
      "mean_time": 12.5,
      "std_time": 1.2
    },
    "LBFGS": {...},
    ...
  }
}
```

**Summary:**
```json
{
  "best_optimizers": {
    "function_approx": {
      "optimizer": "LBFGS",
      "l2_error": 0.0001
    },
    "piecewise": {
      "optimizer": "Adam",
      "l2_error": 0.05
    },
    "poisson_1d": {
      "optimizer": "LBFGS",
      "l2_error": 0.001
    }
  },
  "optimizer_rankings": {
    "function_approx": [
      {"optimizer": "LBFGS", "l2_error": 0.0001},
      {"optimizer": "Adam", "l2_error": 0.0005},
      {"optimizer": "AdamW", "l2_error": 0.0006},
      {"optimizer": "SGD", "l2_error": 0.002}
    ],
    ...
  }
}
```

---

## Dependencies

### External Libraries
- `torch` - PyTorch for neural networks and optimization
- `numpy` - Numerical computations
- `matplotlib` - Plotting and visualization
- `scipy` - Scientific computing (not directly used in Section 2)
- `json` - Result serialization
- `pathlib` - File path handling

### Internal Dependencies (from Section 1)

**Models:**
- `section1.models.KAN` - Kolmogorov-Arnold Network
- `section1.models.MLP` - Multi-layer perceptron (not used currently)
- `section1.models.SIREN` - Sinusoidal representation network (not used currently)

**Training:**
- `section1.training.Trainer` - Standard supervised training
- `section1.training.PINNTrainer` - Physics-informed neural network training

**Utilities:**
- `section1.utils.generate_1d_sinusoid` - Generate sinusoidal data
- `section1.utils.generate_1d_piecewise` - Generate piecewise function data
- `section1.utils.poisson_1d_problem` - Generate 1D Poisson PDE problem
- `section1.utils.poisson_2d_problem` - Generate 2D Poisson PDE problem
- `section1.utils.compute_all_metrics` - Evaluate model performance
- `section1.utils.aggregate_metrics_over_seeds` - Statistical aggregation
- `section1.utils.get_device` - Auto-detect compute device
- `section1.utils.create_optimizer` - Optimizer factory
- `section1.utils.OPTIMIZER_METADATA` - Optimizer metadata
- `section1.utils.timeout_enforcer.enforce_timeout` - Timeout context manager
- `section1.utils.timeout_enforcer.TimeoutError` - Timeout exception

---

## Implementation Guidelines

### To Implement Section 2 in a New Project

#### Step 1: Set Up Directory Structure
```bash
mkdir -p section2/{experiments,utils,visualization,results}
touch section2/__init__.py
touch section2/experiments/__init__.py
touch section2/utils/__init__.py
touch section2/visualization/__init__.py
```

#### Step 2: Implement Core Experiment Class

Create `experiments/optimizer_comparison.py`:
1. Import necessary dependencies from Section 1
2. Implement `set_seed()` helper
3. Implement `OptimizerComparisonExperiment` class with:
   - `__init__(output_dir, device)`
   - `run_function_approx_comparison()`
   - `run_pde_1d_comparison()`
   - `run_piecewise_comparison()`
   - `run_all()`
   - `_generate_summary()`

#### Step 3: Implement Experiment Runner

Create `run_section_2.py`:
1. Import experiment class and timeout utilities
2. Implement `run_with_timeout()` wrapper
3. Implement experiment runners:
   - `run_quick_test()`
   - `run_function_approx_full()`
   - `run_pde_1d_full()`
   - `run_pde_2d_full()`
   - `run_all_experiments()`
4. Implement `main()` with argparse CLI

#### Step 4: Implement Visualization

Create `visualization/convergence_plots.py`:
1. Implement individual plot functions:
   - `plot_convergence_curves()`
   - `plot_efficiency_frontier()`
   - `plot_optimizer_heatmap()`
   - `plot_optimizer_comparison_bar()`
2. Implement `create_optimizer_report_figures()`

#### Step 5: Set Up Utilities

Create `utils/__init__.py`:
1. Re-export optimizer utilities from Section 1
2. Document what each function does

#### Step 6: Create Package Init

Create `__init__.py`:
1. Set version and author
2. Add docstring describing section purpose

#### Step 7: Create Documentation

Create `README.md` with:
1. Overview and research questions
2. Quick start guide
3. CLI usage examples
4. Result interpretation guide
5. Troubleshooting section

### Key Design Decisions

**1. Reuse Section 1 Infrastructure**
- Don't reimplement models, trainers, or data generators
- Import from `section1.*` to ensure consistency
- This allows direct comparison with Section 1 baselines

**2. Timeout Enforcement**
- Essential for comparing optimizers with varying convergence speeds
- Prevents experiments from hanging indefinitely
- Use `enforce_timeout()` context manager

**3. Statistical Rigor**
- Always run multiple seeds (3-5)
- Report mean and std for all metrics
- Store all raw values for post-hoc analysis

**4. Modular Design**
- Each task (function_approx, piecewise, pde) is separate method
- Easy to add new tasks or modify existing ones
- Results are independent JSON files

**5. Visualization Separation**
- Plotting is separate from experiments
- Can regenerate plots without re-running experiments
- Supports multiple visualization styles

### Common Pitfalls to Avoid

1. **Forgetting to set random seeds** → Non-reproducible results
2. **Using wrong learning rates** → Poor optimizer performance
3. **Not handling optimizer failures** → Experiments crash
4. **Missing timeout on L-BFGS** → Hangs indefinitely
5. **Forgetting full-batch for L-BFGS** → Incorrect implementation
6. **Not normalizing heatmap data** → Misleading visualizations
7. **Hardcoding paths** → Not portable across systems

### Testing Checklist

- [ ] Quick test completes in ~2 minutes
- [ ] All optimizers complete without errors
- [ ] Results JSONs are well-formed
- [ ] Figures generate successfully
- [ ] Summary correctly identifies best optimizer
- [ ] Timeout enforcement works
- [ ] Handles optimizer creation failures gracefully
- [ ] Multiple seeds produce statistical metrics
- [ ] PDE residuals are computed correctly
- [ ] Efficiency plots show time vs error tradeoff

---

## Advanced Features (Future Extensions)

### Section 2.2: Hyperparameter Sensitivity

**Planned features:**
- Learning rate schedules (StepLR, CosineAnnealing, ReduceLROnPlateau)
- L-BFGS history size tuning
- LM damping parameter sensitivity
- Interaction effects (optimizer × learning rate)

**Implementation approach:**
- New class: `HyperparameterSearchExperiment`
- Grid search or random search
- Visualize sensitivity with contour plots

### Section 2.3: KAN Variant × Optimizer Interaction

**Research questions:**
- Does LM work better with B-splines vs Fourier basis?
- Optimizer recommendations per KAN variant
- Basis function smoothness vs optimizer choice

**Implementation approach:**
- Modify experiment to loop over KAN variants
- Add dimension: task × optimizer × variant
- Create 3D visualization (heatmaps per variant)

### Section 2.4: Scaling Experiments

**Planned features:**
- How do optimizers scale with model size?
- Memory profiling (L-BFGS vs Adam)
- Batch size effects on convergence

**Implementation approach:**
- New class: `ScalingExperiment`
- Vary hidden dimensions, depth, dataset size
- Plot scaling curves (performance vs size)

---

## References

1. **Krishnapriyan, A., et al. (2021).** "Characterizing possible failure modes in physics-informed neural networks." NeurIPS. [arXiv:2205.07430](https://arxiv.org/abs/2205.07430)
   - Motivation for optimizer comparison
   - LM vs Adam parameter efficiency findings

2. **Liu, D. C., & Nocedal, J. (1989).** "On the limited memory BFGS method for large scale optimization." Mathematical Programming.
   - L-BFGS algorithm description

3. **Levenberg, K. (1944).** "A method for the solution of certain non-linear problems in least squares."
   - Levenberg-Marquardt algorithm part 1

4. **Marquardt, D. W. (1963).** "An algorithm for least-squares estimation of nonlinear parameters."
   - Levenberg-Marquardt algorithm part 2

5. **Kingma, D. P., & Ba, J. (2015).** "Adam: A method for stochastic optimization." ICLR.
   - Adam optimizer

6. **Loshchilov, I., & Hutter, F. (2019).** "Decoupled weight decay regularization." ICLR.
   - AdamW optimizer

---

## API Reference

### OptimizerComparisonExperiment

```python
class OptimizerComparisonExperiment:
    """Systematic optimizer comparison across tasks."""

    def __init__(
        self,
        output_dir: str = "results/section_2_1",
        device: Optional[str] = None
    )

    def run_function_approx_comparison(
        self,
        frequencies: List[float] = [1, 5, 8],
        n_train: int = 256,
        n_eval: int = 1000,
        n_epochs: int = 2000,
        n_seeds: int = 5
    ) -> Dict

    def run_pde_1d_comparison(
        self,
        n_interior: int = 100,
        n_boundary: int = 2,
        n_eval: int = 200,
        n_epochs: int = 2000,
        n_seeds: int = 3
    ) -> Dict

    def run_piecewise_comparison(
        self,
        n_train: int = 256,
        n_eval: int = 1000,
        n_epochs: int = 2000,
        n_seeds: int = 5
    ) -> Dict

    def run_all(self) -> Dict
```

### Visualization Functions

```python
def plot_convergence_curves(
    training_histories: Dict[str, List[float]],
    output_file: str,
    title: str = "Training Convergence",
    log_scale: bool = True
) -> None

def plot_efficiency_frontier(
    optimizer_results: Dict[str, Dict],
    output_file: str,
    title: str = "Optimizer Efficiency (Error vs Time)",
    metric: str = 'l2_error'
) -> None

def plot_optimizer_heatmap(
    results_by_task: Dict[str, Dict[str, Dict]],
    output_file: str,
    metric: str = 'l2_error',
    title: str = "Optimizer Performance Heatmap"
) -> None

def plot_optimizer_comparison_bar(
    optimizer_results: Dict[str, Dict],
    output_file: str,
    metric: str = 'l2_error',
    title: str = "Optimizer Comparison"
) -> None

def create_optimizer_report_figures(
    results_dir: Path,
    output_dir: Path
) -> None
```

---

## Changelog

### Version 1.0.0
- Initial implementation
- Function approximation comparison
- 1D PDE comparison
- Piecewise function comparison
- Visualization suite
- Timeout enforcement
- Summary generation

---

## Contact & Support

For questions about Section 2 implementation:
1. Check console output for error messages
2. Verify Section 1 dependencies are available
3. Review generated JSON files for detailed results
4. Consult Section 1 README for baseline information

**Author:** One Two
**Version:** 1.0.0
**Last Updated:** 2024
