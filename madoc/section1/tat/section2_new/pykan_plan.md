# PyKAN Integration Analysis for section2_new

**Date:** 2025-10-21
**Purpose:** Identify all functionality in section2_new that can be replaced with pykan or PyTorch components
**Goal:** Redesign section2_new as a proper extension of the original pykan repo for maximum compatibility

---

## Executive Summary

After comprehensive analysis of both section2_new (~8,000 LOC) and pykan core (~10,000+ LOC), this document identifies what **CAN** and **CANNOT** be replaced with existing pykan functionality. The goal is to make section2_new a minimal, compatible extension that leverages pykan's infrastructure wherever possible.

### Key Findings

- **✅ CAN REPLACE:** ~60% of functionality (data generation, KAN models, training loops, basis functions)
- **❌ CANNOT REPLACE:** ~40% of functionality (ensemble methods, evolution, adaptive densification, meta-learning)
- **⚠️ HYBRID:** Some components should use pykan's KAN classes but add new logic on top

---

## Section 1: Data Generation

### Current Implementation in section2_new
- Custom data generation functions scattered throughout experiments
- Synthetic function generation (sinusoids, piecewise, etc.)
- Dataset creation utilities

### Available in PyKAN
✅ **YES - Full replacement possible**

**File:** `/Users/main/Desktop/pykan/kan/utils.py`

**Functions:**
```python
create_dataset(
    f: Callable,           # Lambda function for target
    n_var: int,            # Number of input variables
    ranges: List[float],   # Input ranges
    train_num: int,        # Training samples
    test_num: int,         # Test samples
    normalize_input: bool,
    normalize_label: bool,
    device: str,
    seed: int
) -> dict

create_dataset_from_data(
    X: array,
    y: array,
    train_ratio: float,
    device: str
) -> dict
```

**Recommendation:**
- **REPLACE ALL** custom data generation with `pykan.kan.utils.create_dataset()`
- Use lambda functions for custom targets
- Benefits:
  - Automatic train/test splitting
  - Built-in normalization
  - Consistent format with pykan examples
  - Reduced code maintenance

**Example Migration:**
```python
# BEFORE (section2_new custom)
def generate_sinusoid(n_samples, frequency):
    x = torch.linspace(0, 1, n_samples)
    y = torch.sin(2 * np.pi * frequency * x)
    return x, y

# AFTER (using pykan)
from kan.utils import create_dataset
import numpy as np

dataset = create_dataset(
    f=lambda x: np.sin(2 * np.pi * frequency * x[0]),
    n_var=1,
    ranges=[-1, 1],
    train_num=256,
    test_num=1000,
    device='cpu'
)
X_train, y_train = dataset['train_input'], dataset['train_label']
X_test, y_test = dataset['test_input'], dataset['test_label']
```

**Files to Update:**
- `section2_new/experiments/exp_1_ensemble_complete.py`
- `section2_new/DEMO.py`
- All test files

---

## Section 2: KAN Models & Basis Functions

### Current Implementation in section2_new

**Custom KAN implementations:**
- Uses `section1.models.kan_variants` which has:
  - `ChebyshevKAN`
  - `FourierKAN`
  - `WaveletKAN`
  - `RBF_KAN`

**Used in:**
- `ensemble/expert_training.py` - Creates experts with different KAN variants
- `evolution/genome.py` - Instantiates KANs from genomes
- `models/adaptive_selective_kan.py` - Wraps RBF_KAN
- `population/population_trainer.py` - Creates population of KANs

### Available in PyKAN

✅ **YES - Partial replacement, but needs careful integration**

**PyKAN Core KAN:**
- **File:** `/Users/main/Desktop/pykan/kan/MultKAN.py` (2808 LOC)
- **Main Class:** `MultKAN` (formerly `KAN`)
- **Features:**
  - B-spline basis functions (configurable order k)
  - Adaptive grid management
  - Symbolic reasoning support
  - Pruning and network modification
  - Checkpoint management

**PyKAN Basis Functions:**
- **File:** `/Users/main/Desktop/pykan/kan/spline.py`
- **B-spline utilities:**
  - `B_batch()` - Evaluate B-spline bases
  - `coef2curve()` - Convert coefficients to curves
  - `curve2coef()` - Fit curves with least squares
  - `extend_grid()` - Grid refinement

**PyKAN Symbolic Functions:**
- **File:** `/Users/main/Desktop/pykan/kan/utils.py`
- **SYMBOLIC_LIB contains:**
  - Trigonometric: `sin`, `cos`, `tan`, `tanh`, `arcsin`, `arccos`, `arctan`, `arctanh`
  - Exponential/Log: `exp`, `log`
  - Powers: `x`, `x^2`, `x^3`, `x^4`, `x^5`, `sqrt`
  - Inverse powers: `1/x`, `1/x^2`, `1/x^3`, `1/x^4`, `1/x^5`
  - Special: `abs`, `sgn`, `gaussian`

### Analysis: Can We Replace?

**⚠️ HYBRID APPROACH NEEDED**

**Problem:**
1. PyKAN's `MultKAN` uses **B-splines only** (no Chebyshev, Fourier, Wavelet, RBF)
2. section2_new explicitly requires **different basis types** for:
   - Heterogeneous KAN (Extension 3) - mixed basis per edge
   - Expert diversity (Extension 1) - different basis per expert
   - Evolution (Extension 5) - basis type as genome parameter

**Solution:**

1. **For simple KANs:** Use `pykan.kan.MultKAN` with B-splines
2. **For basis variety:** Keep custom `section1.models.kan_variants` BUT:
   - Ensure they follow MultKAN API conventions
   - Add compatibility layer to convert between formats
   - Document differences clearly

**Recommendation:**

✅ **KEEP** custom basis implementations (Chebyshev, Fourier, Wavelet, RBF) because:
- PyKAN doesn't provide them
- They're core to section2_new's functionality
- Extension 3 (heterogeneous basis) requires multiple types

✅ **REPLACE** simple B-spline usage with `MultKAN` where:
- Only B-splines needed
- Adaptive grid features beneficial
- Symbolic reasoning useful

⚠️ **CREATE** compatibility wrapper:
```python
# New file: section2_new/models/pykan_wrapper.py

from kan.MultKAN import MultKAN

class PyKAN_Compatible(MultKAN):
    """Wrapper to make pykan's MultKAN compatible with section2_new API."""

    def __init__(self, input_dim, hidden_dim, output_dim, depth, **kwargs):
        # Convert section2_new API to MultKAN API
        width = [input_dim] + [hidden_dim] * (depth - 1) + [output_dim]
        super().__init__(width=width, **kwargs)
```

**Files to Update:**
- Add `section2_new/models/pykan_wrapper.py` (new compatibility layer)
- Update `evolution/genome.py` to optionally use MultKAN for B-splines
- Update `ensemble/expert_training.py` to support `variant='bspline'`

---

## Section 3: Training & Optimization

### Current Implementation in section2_new

**Training loops:**
- Custom training in `ensemble/expert_training.py`
- Custom PINN trainer (not in section2_new, but in section1)
- Manual epoch loops with loss computation

**Code example from expert_training.py:**
```python
for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred = expert(X_train)
    loss = loss_fn(y_pred, y_train)
    loss.backward()
    optimizer.step()
```

### Available in PyKAN

✅ **YES - Can use pykan's training infrastructure**

**PyKAN Training:**
- **File:** `/Users/main/Desktop/pykan/kan/MultKAN.py`
- **Method:** `MultKAN.fit()`

**Features:**
```python
model.fit(
    dataset: dict,           # {'train_input', 'train_label', 'test_input', 'test_label'}
    opt: str = 'LBFGS',     # Optimizer type
    steps: int = 100,        # Training steps
    lamb: float = 0.0,       # L1 regularization
    lamb_l1: float = 1.0,    # Edge sparsity
    lamb_entropy: float = 2.0,
    lamb_coef: float = 0.0,
    lamb_coefdiff: float = 0.0,
    update_grid: bool = True,
    grid_update_num: int = 10,
    loss_fn: Callable = None,
    lr: float = 1.0,
    batch: int = -1,         # -1 = full batch
    ...
)
```

**Benefits:**
- Automatic grid updating
- Multiple regularization options
- LBFGS optimizer built-in
- Batch support
- Loss history tracking

### Recommendation

**⚠️ PARTIAL REPLACEMENT**

**Use pykan's `fit()` when:**
- Training single models
- Want grid adaptation
- Using B-spline KANs
- Need regularization

**Keep custom training when:**
- Training ensembles (need independent loops)
- Evolution (short training, custom stopping)
- Population-based (need synchronization points)
- Adaptive densification (need importance tracking hooks)

**Suggested approach:**
```python
# For single model training
from kan.MultKAN import MultKAN

model = MultKAN(width=[3, 16, 1])
dataset = create_dataset(...)
results = model.fit(dataset, steps=100, opt='LBFGS')

# For ensemble/evolution - keep custom loops but use pykan utilities
from torch.optim import Adam

for expert in ensemble:
    optimizer = Adam(expert.parameters(), lr=0.01)
    # Custom loop with importance tracking, sync, etc.
```

**Files to Update:**
- Add option in `ensemble/expert_training.py` to use `MultKAN.fit()`
- Keep custom loops for advanced features

---

## Section 4: Optimizers

### Current Implementation in section2_new

**Optimizers:**
- Uses PyTorch standard optimizers: `torch.optim.Adam`, `torch.optim.SGD`
- No custom optimizer implementations

### Available in PyKAN

✅ **YES - PyKAN has LBFGS implementation**

**File:** `/Users/main/Desktop/pykan/kan/LBFGS.py`

**Features:**
- Full L-BFGS optimizer
- Line search with Wolfe conditions
- Compatible with PyTorch `torch.optim` API

**Recommendation:**
✅ **REPLACE** any LBFGS usage with `pykan.kan.LBFGS.LBFGS`

**Example:**
```python
# BEFORE
from torch.optim import LBFGS

# AFTER
from kan.LBFGS import LBFGS

optimizer = LBFGS(
    model.parameters(),
    lr=1.0,
    max_iter=20,
    history_size=100
)
```

**Files to Update:**
- `section2_new/evolution/fitness.py` (if using LBFGS)
- `section2_new/population/population_trainer.py` (if using LBFGS)

---

## Section 5: Evaluation & Metrics

### Current Implementation in section2_new

**Metrics:**
- Custom L2 error computation
- Custom relative error
- MSE loss

### Available in PyKAN

✅ **YES - PyKAN has evaluation utilities**

**File:** `/Users/main/Desktop/pykan/kan/MultKAN.py`

**Method:**
```python
model.evaluate(dataset: dict) -> dict
# Returns loss metrics on test set
```

**Recommendation:**
⚠️ **PARTIAL REPLACEMENT**

- Use `model.evaluate()` for basic metrics
- Keep custom metrics for:
  - PDE residuals
  - Uncertainty quantification
  - Multi-objective fitness (accuracy + complexity + speed)

---

## Section 6: Ensemble Methods (Extension 1)

### Current Implementation in section2_new

**Components:**
1. `ensemble/expert_training.py` (452 LOC) - Multi-seed training
2. `ensemble/variable_importance.py` (418 LOC) - Feature importance
3. `ensemble/clustering.py` (447 LOC) - Expert clustering
4. `ensemble/stacking.py` (417 LOC) - Meta-learning

### Available in PyKAN

❌ **NO - Not available in pykan**

**PyKAN does NOT provide:**
- Ensemble training
- Multi-seed management
- Expert clustering
- Meta-learners
- Uncertainty quantification

**Recommendation:**
❌ **KEEP ALL** ensemble code as-is

**Rationale:**
- This is core novelty of section2_new
- No equivalent in pykan
- Well-designed and production-ready
- Only change: Use pykan's KANs as base models

**Files to Keep:**
- ✅ `ensemble/expert_training.py`
- ✅ `ensemble/variable_importance.py`
- ✅ `ensemble/clustering.py`
- ✅ `ensemble/stacking.py`

---

## Section 7: Adaptive Densification (Extension 2)

### Current Implementation in section2_new

**Components:**
1. `adaptive/importance_tracker.py` (387 LOC) - Per-node importance tracking
2. `models/adaptive_selective_kan.py` (487 LOC) - Selective grid densification

**Features:**
- Gradient-based importance
- Activation-based importance
- Weight-based importance
- Selective densification of top-k nodes

### Available in PyKAN

⚠️ **PARTIAL - PyKAN has grid management but not importance-based selection**

**PyKAN Grid Features:**

**File:** `/Users/main/Desktop/pykan/kan/MultKAN.py`

**Methods:**
```python
model.update_grid_from_samples(x)  # Adaptive grid from data quantiles
model.refine()                      # Double grid resolution uniformly
model.get_grid()                    # Get current grid configuration
```

**File:** `/Users/main/Desktop/pykan/kan/KANLayer.py`

**Features:**
- Grid extension: `extend_grid()`
- Adaptive grid updates based on data distribution
- Uniform refinement (all edges equally)

### Analysis: Can We Use PyKAN's Grid Management?

**⚠️ HYBRID APPROACH**

**What PyKAN provides:**
- ✅ Uniform grid refinement
- ✅ Data-adaptive grid placement (quantile-based)
- ✅ Grid initialization and extension

**What section2_new adds (not in pykan):**
- ❌ **Per-node** grid size tracking (PyKAN is uniform)
- ❌ Importance-based **selective** densification
- ❌ Gradient/activation/weight importance computation
- ❌ Top-k node selection

**Recommendation:**

✅ **KEEP** adaptive densification code BUT:
- Use pykan's `KANLayer` as base
- Hook into pykan's grid management utilities
- Extend (don't replace) pykan's grid logic

**Suggested refactor:**
```python
# models/adaptive_selective_kan.py

from kan.KANLayer import KANLayer
from kan.MultKAN import MultKAN

class AdaptiveSelectiveKAN(MultKAN):
    """Extends MultKAN with importance-based selective densification."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add importance tracking on top of base MultKAN
        self.importance_tracker = NodeImportanceTracker(self)

    def densify_important_nodes(self, k=5):
        # Use base MultKAN's grid utilities
        # Apply selectively based on importance
        important_nodes = self.importance_tracker.get_top_k_nodes(k)
        for layer_idx, node_idx in important_nodes:
            # Use pykan's extend_grid() on specific nodes
            self.layers[layer_idx].extend_grid(node_idx)
```

**Files to Update:**
- ✅ `models/adaptive_selective_kan.py` - Inherit from `MultKAN`
- ✅ `adaptive/importance_tracker.py` - Keep as-is (new functionality)

---

## Section 8: Heterogeneous Basis Functions (Extension 3)

### Current Implementation in section2_new

**File:** `models/heterogeneous_kan.py` (590 LOC)

**Features:**
1. `HeterogeneousKANLayer` - Mixed basis per edge/input
2. `HeterogeneousBasisKAN` - Full network with layer-specific configs
3. `AutoBasisSelector` - Heuristic basis selection from signal analysis

**Supported bases:**
- Chebyshev
- Fourier
- RBF
- Wavelet (planned)

### Available in PyKAN

❌ **NO - Not available in pykan**

**PyKAN only provides:**
- B-splines (all edges use same basis)
- Symbolic functions (but not as learnable basis)

**PyKAN does NOT provide:**
- Per-edge basis selection
- Mixed basis in single layer
- Learnable basis assignment (Gumbel-softmax)
- Automatic basis selection

**Recommendation:**

❌ **KEEP ALL** heterogeneous basis code

**Rationale:**
- This is core novelty of section2_new
- No equivalent in pykan
- Requires fundamental changes to pykan's layer structure
- Well-designed implementation

**Files to Keep:**
- ✅ `models/heterogeneous_kan.py` (keep entirely)

**Note:** Could potentially use pykan's `SYMBOLIC_LIB` for basis function implementations, but current design is already good.

---

## Section 9: Population-Based Training (Extension 4)

### Current Implementation in section2_new

**File:** `population/population_trainer.py` (458 LOC)

**Features:**
- Train multiple models in parallel
- Periodic synchronization (average, best, tournament)
- Diversity tracking
- Population-level optimization

### Available in PyKAN

❌ **NO - Not available in pykan**

**PyKAN provides:**
- Single model training only
- No population management
- No synchronization primitives

**Recommendation:**

❌ **KEEP ALL** population-based training code

**Rationale:**
- Completely new functionality
- No overlap with pykan
- Only change: Use pykan KANs as base models

**Files to Keep:**
- ✅ `population/population_trainer.py`

---

## Section 10: Evolutionary Search (Extension 5)

### Current Implementation in section2_new

**Components:**
1. `evolution/genome.py` (345 LOC) - Genome representation
2. `evolution/fitness.py` (298 LOC) - Multi-objective fitness
3. `evolution/operators.py` (274 LOC) - Selection, Pareto frontier
4. `evolution/evolutionary_search.py` (411 LOC) - Evolution loop

**Features:**
- Genome encoding (architecture + hyperparameters)
- Mutation and crossover
- Multi-objective optimization (accuracy, complexity, speed)
- Pareto frontier tracking
- Fitness caching

### Available in PyKAN

❌ **NO - Not available in pykan**

**PyKAN provides:**
- No architecture search
- No evolutionary algorithms
- No multi-objective optimization
- No genome representation

**Recommendation:**

❌ **KEEP ALL** evolutionary code

**Rationale:**
- Completely novel functionality
- No overlap with pykan
- Well-designed implementation
- Only change: Use `MultKAN` in `genome.to_model()` for B-spline variants

**Files to Keep:**
- ✅ `evolution/genome.py`
- ✅ `evolution/fitness.py`
- ✅ `evolution/operators.py`
- ✅ `evolution/evolutionary_search.py`

**Minor update to genome.py:**
```python
# In genome.py

def to_model(self, device='cpu'):
    if self.basis_type == 'bspline':
        # Use pykan's MultKAN
        from kan.MultKAN import MultKAN
        width = [self.layer_sizes[0]] + [self.layer_sizes[1]] * (self.depth - 1) + [self.layer_sizes[-1]]
        return MultKAN(width=width, grid=self.grid_size, device=device)
    elif self.basis_type == 'rbf':
        # Use custom RBF_KAN
        from section1.models.kan_variants import RBF_KAN
        return RBF_KAN(...)
    # ... etc
```

---

## Section 11: Visualization

### Current Implementation in section2_new

**Files:**
- `visualization/__init__.py` (empty placeholder)
- Custom plotting in experiments

### Available in PyKAN

⚠️ **PARTIAL - PyKAN has KAN visualization but not ensemble/evolution viz**

**PyKAN Visualization:**

**File:** `/Users/main/Desktop/pykan/kan/MultKAN.py`

**Methods:**
```python
model.plot(
    beta: int = 3,           # Contrast parameter
    mask: bool = False,
    mode: str = 'supervised',
    scale: float = 0.5,
    tick: bool = False,
    sample: bool = False,
    in_vars: List[str] = None,
    out_vars: List[str] = None,
    title: str = None
)
# Visualizes network architecture with edge importance

model.plot_tree()  # Computation tree visualization
```

**Recommendation:**

✅ **USE** pykan's `plot()` for individual KAN visualization
❌ **KEEP** custom plots for:
- Ensemble diversity
- Evolution progress
- Pareto frontiers
- Importance heatmaps
- Cluster visualizations

**Files to Update:**
- Add imports from `kan.MultKAN` for basic plotting
- Keep custom ensemble/evolution visualizations

---

## Section 12: Utilities & Helper Functions

### Current Implementation in section2_new

**File:** `utils/__init__.py` (minimal)

**Custom utilities scattered throughout:**
- Seed setting
- Device management
- Metric aggregation

### Available in PyKAN

✅ **YES - PyKAN has many utilities**

**File:** `/Users/main/Desktop/pykan/kan/utils.py`

**Functions:**
```python
# Seed setting
torch.manual_seed(seed)
np.random.seed(seed)

# Data utilities
create_dataset()
create_dataset_from_data()

# Symbolic utilities
fit_params()  # Fit a, b, c, d for y = c*f(a*x+b) + d
ex_round()    # Round expressions

# Sparse masks
sparse_mask(in_dim, out_dim, sparsity)

# Input augmentation
augment_input(X, auxiliary_vars)
```

**Recommendation:**

✅ **MIGRATE** to pykan utilities:
- Replace custom `set_seed()` → use torch/numpy directly
- Replace custom dataset creation → use `create_dataset()`
- Use `fit_params()` for symbolic function fitting

**Files to Update:**
- Centralize imports in `utils/__init__.py`:
```python
# section2_new/utils/__init__.py

from kan.utils import (
    create_dataset,
    create_dataset_from_data,
    fit_params,
    sparse_mask,
    augment_input
)

__all__ = [
    'create_dataset',
    'create_dataset_from_data',
    'fit_params',
    'sparse_mask',
    'augment_input'
]
```

---

## Section 13: Advanced PyKAN Features Not Yet Used

### Symbolic Reasoning

**Available in PyKAN:**
- **File:** `/Users/main/Desktop/pykan/kan/MultKAN.py`, `/Users/main/Desktop/pykan/kan/Symbolic_KANLayer.py`

**Features:**
```python
model.suggest_symbolic(0, 0, 0)  # Suggest symbolic function for edge
model.fix_symbolic(layer, i, j, 'sin')  # Lock to symbolic function
model.auto_symbolic()  # Automatically discover symbolic activations
model.symbolic_formula()  # Extract symbolic formula
```

**Potential use in section2_new:**
- ✅ Genome could include symbolic function genes
- ✅ Evolution could optimize symbolic vs numerical activations
- ✅ Interpretability analysis post-training

**Recommendation:**
⚠️ **FUTURE ENHANCEMENT** - Add symbolic evolution:
```python
# evolution/genome.py - future extension

@dataclass
class SymbolicKANGenome(KANGenome):
    symbolic_edges: Dict[Tuple[int,int,int], str] = field(default_factory=dict)
    # (layer, i, j) -> 'sin' | 'exp' | 'x^2' | etc.
```

### Pruning & Network Modification

**Available in PyKAN:**
- **File:** `/Users/main/Desktop/pykan/kan/MultKAN.py`

**Methods:**
```python
model.prune()  # Combined node + edge pruning
model.prune_node(threshold=0.01)
model.prune_edge(threshold=0.01)
model.remove_node(layer, idx)
model.remove_edge(layer, i, j)
model.get_subset(input_ids, output_ids)  # Extract subnetwork
```

**Potential use in section2_new:**
- ✅ Pruning during evolution (reduce complexity)
- ✅ Pruning after ensemble training (compress experts)
- ✅ Mutation operator: prune + regrow

**Recommendation:**
⚠️ **FUTURE ENHANCEMENT** - Add pruning mutations:
```python
# evolution/genome.py

def prune_mutation(self) -> 'KANGenome':
    """Create pruned variant of genome."""
    model = self.to_model()
    model.prune(threshold=0.05)
    # Extract new architecture from pruned model
    return KANGenome.from_model(model)
```

### Interpretability Analysis

**Available in PyKAN:**
- **File:** `/Users/main/Desktop/pykan/kan/hypothesis.py`

**Functions:**
```python
detect_separability(model, X)  # Detect additive/multiplicative separability
test_symmetry(model, X)        # Detect symmetries
model.attribute(X)             # Node/edge importance attribution
model.feature_interaction(X)   # Feature interaction detection
```

**Potential use in section2_new:**
- ✅ Variable importance (Extension 1) could use `model.attribute()`
- ✅ Feature interaction analysis for ensemble
- ✅ Symmetry detection for architecture simplification

**Recommendation:**
✅ **INTEGRATE** into variable importance:
```python
# ensemble/variable_importance.py

def compute_attribution_importance(self, X: torch.Tensor) -> np.ndarray:
    """Compute importance using pykan's attribution."""
    from kan.hypothesis import attribute

    importances = []
    for expert in self.ensemble.experts:
        attr = attribute(expert, X)
        importances.append(attr['input'])

    return np.mean(importances, axis=0)
```

### Feynman Physics Datasets

**Available in PyKAN:**
- **File:** `/Users/main/Desktop/pykan/kan/feynman.py`

**Features:**
- 100+ physics equations from Feynman Lectures
- Pre-built symbolic expressions
- Suggested input ranges

**Example:**
```python
from kan.feynman import get_feynman_dataset

dataset = get_feynman_dataset('I.6.20a')  # Gaussian distribution
# Returns: {'expr': SymPy expr, 'fn': PyTorch lambda, 'ranges': [...]}
```

**Potential use in section2_new:**
- ✅ Benchmark datasets for ensemble/evolution
- ✅ Test heterogeneous basis on physics problems
- ✅ Symbolic reasoning integration

**Recommendation:**
✅ **ADD** Feynman benchmarks:
```python
# experiments/feynman_benchmarks.py (new file)

from kan.feynman import get_feynman_dataset
from section2_new.ensemble.expert_training import KANExpertEnsemble

# Test ensemble on Feynman equations
for eq_name in ['I.6.20a', 'I.9.18', 'I.13.4']:
    dataset = get_feynman_dataset(eq_name)
    ensemble = KANExpertEnsemble(...)
    ensemble.train_experts(dataset['X'], dataset['y'])
```

---

## Summary Tables

### Table 1: Functionality Replacement Status

| Component | Status | Action | Files Affected |
|-----------|--------|--------|----------------|
| **Data Generation** | ✅ Replace | Use `kan.utils.create_dataset()` | All experiments, DEMO.py |
| **KAN Models (B-spline)** | ⚠️ Hybrid | Use `MultKAN` for B-splines, keep custom for other bases | genome.py, expert_training.py |
| **KAN Models (Other bases)** | ❌ Keep | Keep Chebyshev, Fourier, Wavelet, RBF | All model files |
| **Training (simple)** | ✅ Replace | Use `model.fit()` for single models | - |
| **Training (ensemble/evolution)** | ❌ Keep | Custom loops needed | expert_training.py, fitness.py |
| **LBFGS Optimizer** | ✅ Replace | Use `kan.LBFGS.LBFGS` | fitness.py, population_trainer.py |
| **Basic Metrics** | ✅ Replace | Use `model.evaluate()` | - |
| **Advanced Metrics** | ❌ Keep | PDE residuals, uncertainty, multi-objective | All experiments |
| **Ensemble Training** | ❌ Keep | Not in pykan | ensemble/* (all files) |
| **Variable Importance** | ⚠️ Hybrid | Keep code, optionally use `model.attribute()` | variable_importance.py |
| **Expert Clustering** | ❌ Keep | Not in pykan | clustering.py |
| **Meta-Learning** | ❌ Keep | Not in pykan | stacking.py |
| **Adaptive Densification** | ⚠️ Hybrid | Extend `MultKAN`'s grid management | adaptive_selective_kan.py, importance_tracker.py |
| **Heterogeneous Basis** | ❌ Keep | Not in pykan | heterogeneous_kan.py |
| **Population Training** | ❌ Keep | Not in pykan | population_trainer.py |
| **Evolution** | ❌ Keep | Not in pykan | evolution/* (all files) |
| **Visualization (KAN)** | ✅ Replace | Use `model.plot()` | - |
| **Visualization (Ensemble/Evolution)** | ❌ Keep | Custom plots needed | visualization/* |
| **Utilities** | ✅ Replace | Use `kan.utils` | utils/__init__.py |

### Table 2: Code Volume Impact

| Category | Current LOC | Can Replace | Must Keep | % Reusable |
|----------|-------------|-------------|-----------|------------|
| Data Generation | ~200 | 200 | 0 | 100% |
| KAN Models | ~1000 | 200 | 800 | 20% |
| Training | ~500 | 100 | 400 | 20% |
| Ensemble (Ext 1) | 1734 | 0 | 1734 | 0% |
| Adaptive (Ext 2) | 874 | 200 | 674 | 23% |
| Heterogeneous (Ext 3) | 590 | 0 | 590 | 0% |
| Population (Ext 4) | 458 | 0 | 458 | 0% |
| Evolution (Ext 5) | 1328 | 0 | 1328 | 0% |
| Visualization | ~200 | 50 | 150 | 25% |
| Utilities | ~100 | 100 | 0 | 100% |
| **TOTAL** | **~7000** | **~850** | **~6150** | **12%** |

**Key Insight:** Only ~12% of section2_new code can be directly replaced with pykan. The remaining 88% is novel functionality that extends pykan.

---

## Integration Roadmap

### Phase 1: Foundation (Immediate)

**Goal:** Replace all data generation and utilities

**Tasks:**
1. ✅ Update all experiments to use `kan.utils.create_dataset()`
2. ✅ Update `utils/__init__.py` to re-export pykan utilities
3. ✅ Remove custom data generation functions
4. ✅ Update DEMO.py with new data generation

**Files:**
- `section2_new/utils/__init__.py`
- `section2_new/experiments/exp_1_ensemble_complete.py`
- `section2_new/DEMO.py`
- All test files

**Estimated Impact:** -200 LOC, improved consistency

---

### Phase 2: Model Integration (High Priority)

**Goal:** Support MultKAN as base model option

**Tasks:**
1. ✅ Create `models/pykan_wrapper.py` compatibility layer
2. ✅ Update `evolution/genome.py` to support `basis_type='bspline'`
3. ✅ Update `ensemble/expert_training.py` to support `kan_variant='bspline'`
4. ⚠️ Refactor `models/adaptive_selective_kan.py` to inherit from `MultKAN`
5. ✅ Add LBFGS import from `kan.LBFGS`

**Files:**
- `section2_new/models/pykan_wrapper.py` (new)
- `section2_new/evolution/genome.py`
- `section2_new/ensemble/expert_training.py`
- `section2_new/models/adaptive_selective_kan.py`
- `section2_new/population/population_trainer.py`

**Estimated Impact:** -100 LOC, +1 new file, improved compatibility

---

### Phase 3: Advanced Features (Medium Priority)

**Goal:** Integrate pykan's interpretability and symbolic reasoning

**Tasks:**
1. ⚠️ Add `kan.hypothesis.attribute()` to variable importance
2. ⚠️ Add optional symbolic function genes to evolution
3. ✅ Add pruning mutations to evolution
4. ✅ Add Feynman physics benchmarks

**Files:**
- `section2_new/ensemble/variable_importance.py`
- `section2_new/evolution/genome.py`
- `section2_new/experiments/feynman_benchmarks.py` (new)

**Estimated Impact:** +300 LOC new features, enhanced interpretability

---

### Phase 4: Optimization (Low Priority)

**Goal:** Performance improvements using pykan infrastructure

**Tasks:**
1. ⚠️ Use `model.fit()` for simple single-model training
2. ⚠️ Leverage pykan's grid management in adaptive densification
3. ⚠️ Use pykan's checkpoint system for evolution state saving
4. ⚠️ Integrate pykan's device management

**Files:**
- Various training loops
- `section2_new/models/adaptive_selective_kan.py`
- `section2_new/evolution/evolutionary_search.py`

**Estimated Impact:** Better performance, reduced code complexity

---

### Phase 5: Documentation (Ongoing)

**Goal:** Clear documentation of pykan vs custom components

**Tasks:**
1. ✅ Document which features come from pykan
2. ✅ Document which features are section2_new extensions
3. ✅ Add compatibility notes
4. ✅ Update README with pykan dependency info
5. ✅ Add migration guide for users

**Files:**
- `section2_new/README.md`
- `section2_new/PYKAN_INTEGRATION.md` (new)
- All docstrings

**Estimated Impact:** Better user experience, clearer maintenance

---

## Detailed Replacement Examples

### Example 1: Data Generation

**BEFORE:**
```python
# section2_new/experiments/exp_1_ensemble_complete.py

import torch
import numpy as np

def generate_synthetic_data(n_samples=1000):
    """Generate synthetic data for testing."""
    X = torch.randn(n_samples, 3)
    y = torch.sin(X[:, 0]) + torch.cos(X[:, 1]) + X[:, 2]**2
    y = y.unsqueeze(1)

    # Split train/test
    n_train = int(0.8 * n_samples)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    return X_train, y_train, X_test, y_test
```

**AFTER:**
```python
# section2_new/experiments/exp_1_ensemble_complete.py

from kan.utils import create_dataset
import numpy as np

def generate_synthetic_data(n_samples=1000):
    """Generate synthetic data using pykan utilities."""
    # Define target function
    def target_fn(x):
        return np.sin(x[0]) + np.cos(x[1]) + x[2]**2

    # Create dataset using pykan
    dataset = create_dataset(
        f=target_fn,
        n_var=3,
        ranges=[-3, 3],
        train_num=int(0.8 * n_samples),
        test_num=int(0.2 * n_samples),
        device='cpu'
    )

    return (
        dataset['train_input'],
        dataset['train_label'],
        dataset['test_input'],
        dataset['test_label']
    )
```

**Benefits:**
- ✅ Consistent data format across project
- ✅ Automatic train/test splitting
- ✅ Normalization options available
- ✅ Less custom code to maintain

---

### Example 2: KAN Model Creation

**BEFORE:**
```python
# section2_new/evolution/genome.py

from section1.models.kan_variants import RBF_KAN

class KANGenome:
    def to_model(self, device='cpu'):
        if self.basis_type == 'rbf':
            return RBF_KAN(
                input_dim=self.layer_sizes[0],
                hidden_dim=self.layer_sizes[1],
                output_dim=self.layer_sizes[-1],
                depth=self.depth,
                n_centers=self.grid_size
            ).to(device)
```

**AFTER:**
```python
# section2_new/evolution/genome.py

from section1.models.kan_variants import RBF_KAN
from kan.MultKAN import MultKAN

class KANGenome:
    def to_model(self, device='cpu'):
        if self.basis_type == 'bspline':
            # Use pykan's MultKAN for B-splines
            width = self.layer_sizes
            return MultKAN(
                width=width,
                grid=self.grid_size,
                k=3,  # Cubic B-splines
                device=device
            )
        elif self.basis_type == 'rbf':
            # Use custom RBF_KAN
            return RBF_KAN(
                input_dim=self.layer_sizes[0],
                hidden_dim=self.layer_sizes[1],
                output_dim=self.layer_sizes[-1],
                depth=self.depth,
                n_centers=self.grid_size
            ).to(device)
        # ... other basis types
```

**Benefits:**
- ✅ Access to pykan's grid adaptation
- ✅ Access to symbolic reasoning features
- ✅ Better maintenance (pykan updates automatically)
- ✅ Still supports custom bases

---

### Example 3: Variable Importance

**BEFORE:**
```python
# section2_new/ensemble/variable_importance.py

class VariableImportanceAnalyzer:
    def compute_weight_importance(self):
        """Compute importance from first-layer weights."""
        importances = []
        for expert in self.ensemble.experts:
            # Custom weight extraction
            first_layer = expert.layers[0]
            weights = first_layer.weight.detach().cpu().numpy()
            importance = np.abs(weights).sum(axis=0)
            importances.append(importance)
        return np.mean(importances, axis=0)
```

**AFTER:**
```python
# section2_new/ensemble/variable_importance.py

class VariableImportanceAnalyzer:
    def compute_weight_importance(self):
        """Compute importance from first-layer weights."""
        importances = []
        for expert in self.ensemble.experts:
            # Use pykan's attribution if available
            if hasattr(expert, 'attribute'):
                attr = expert.attribute(self.X_val)
                importance = attr['input_attribution']
            else:
                # Fallback to custom method
                first_layer = expert.layers[0]
                weights = first_layer.weight.detach().cpu().numpy()
                importance = np.abs(weights).sum(axis=0)
            importances.append(importance)
        return np.mean(importances, axis=0)

    def compute_pykan_importance(self, X):
        """Compute importance using pykan's attribution (new method)."""
        from kan.hypothesis import attribute

        importances = []
        for expert in self.ensemble.experts:
            if hasattr(expert, 'attribute'):  # Is MultKAN
                attr = expert.attribute(X)
                importances.append(attr['node_importance'][0])  # First layer

        if importances:
            return np.mean(importances, axis=0)
        else:
            # Fallback if no MultKAN experts
            return self.compute_weight_importance()
```

**Benefits:**
- ✅ Leverages pykan's tested attribution
- ✅ Backward compatible with custom KANs
- ✅ More interpretability options

---

### Example 4: Adaptive Densification

**BEFORE:**
```python
# section2_new/models/adaptive_selective_kan.py

class AdaptiveSelectiveKAN(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # Create custom RBF_KAN
        self.model = RBF_KAN(...)
        self.node_grid_sizes = {}

    def densify_node(self, layer_idx, node_idx):
        # Custom grid densification logic
        current_grid = self.node_grid_sizes[(layer_idx, node_idx)]
        new_grid = min(current_grid + 2, self.max_grid)
        # Manually update RBF centers
        # ... custom implementation
```

**AFTER:**
```python
# section2_new/models/adaptive_selective_kan.py

from kan.MultKAN import MultKAN

class AdaptiveSelectiveKAN(MultKAN):
    """Extends MultKAN with selective densification."""

    def __init__(self, width, initial_grid=5, max_grid=20, **kwargs):
        super().__init__(width=width, grid=initial_grid, **kwargs)
        self.max_grid = max_grid
        self.node_grid_sizes = {}
        # Initialize tracking
        for layer_idx in range(len(self.layers)):
            for node_idx in range(self.width[layer_idx+1]):
                self.node_grid_sizes[(layer_idx, node_idx)] = initial_grid

    def densify_node(self, layer_idx, node_idx):
        """Selectively densify specific node."""
        current_grid = self.node_grid_sizes[(layer_idx, node_idx)]
        if current_grid >= self.max_grid:
            return False

        # Use pykan's grid refinement infrastructure
        layer = self.layers[layer_idx]

        # Selectively refine only this node's incoming edges
        # (This requires accessing pykan's grid extension utilities)
        from kan.spline import extend_grid

        # Update specific node
        # ... use pykan's extend_grid for proper B-spline extension

        self.node_grid_sizes[(layer_idx, node_idx)] += 2
        return True
```

**Benefits:**
- ✅ Inherits all MultKAN features (pruning, symbolic, etc.)
- ✅ Uses pykan's tested grid utilities
- ✅ Compatible with pykan's visualization
- ✅ Can use `model.fit()` for training

---

## Dependencies & Imports Structure

### Recommended Import Organization

```python
# section2_new/utils/__init__.py

"""
Utilities for section2_new.

This module re-exports utilities from pykan and adds custom extensions.
"""

# ============================================================
# FROM PYKAN (use these instead of custom implementations)
# ============================================================

from kan.utils import (
    create_dataset,              # Dataset generation
    create_dataset_from_data,    # Convert numpy/torch to dataset format
    fit_params,                   # Fit symbolic function parameters
    sparse_mask,                  # Generate sparse connection masks
    augment_input,                # Input augmentation
    ex_round,                     # Expression rounding
)

from kan.LBFGS import LBFGS      # L-BFGS optimizer

from kan.hypothesis import (
    attribute,                    # Node/edge attribution
    detect_separability,          # Separability detection
    test_symmetry,                # Symmetry testing
    feature_interaction,          # Feature interaction analysis
)

# ============================================================
# CUSTOM SECTION2_NEW UTILITIES (not in pykan)
# ============================================================

from .ensemble_utils import (    # If needed
    aggregate_predictions,
    compute_ensemble_diversity,
)

from .evolution_utils import (   # If needed
    pareto_dominance,
    crowding_distance,
)

# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    # PyKAN utilities
    'create_dataset',
    'create_dataset_from_data',
    'fit_params',
    'sparse_mask',
    'augment_input',
    'ex_round',
    'LBFGS',
    'attribute',
    'detect_separability',
    'test_symmetry',
    'feature_interaction',

    # Custom utilities
    'aggregate_predictions',
    'compute_ensemble_diversity',
    'pareto_dominance',
    'crowding_distance',
]
```

### Recommended Model Imports

```python
# section2_new/models/__init__.py

"""
Model architectures for section2_new.

Combines pykan's MultKAN with custom extensions and variants.
"""

# ============================================================
# FROM PYKAN (primary KAN implementation)
# ============================================================

from kan.MultKAN import MultKAN  # B-spline KAN (use as default)
from kan.MLP import MLP           # Baseline MLP for comparison

# ============================================================
# CUSTOM VARIANTS (not in pykan - needed for heterogeneous basis)
# ============================================================

from section1.models.kan_variants import (
    ChebyshevKAN,    # Chebyshev polynomial basis
    FourierKAN,      # Fourier basis
    WaveletKAN,      # Wavelet basis
    RBF_KAN,         # Radial basis functions
)

# ============================================================
# SECTION2_NEW EXTENSIONS (novel architectures)
# ============================================================

from .adaptive_selective_kan import (
    AdaptiveSelectiveKAN,         # Extension 2: Selective densification
    AdaptiveSelectiveTrainer,
)

from .heterogeneous_kan import (
    HeterogeneousKANLayer,        # Extension 3: Mixed basis
    HeterogeneousBasisKAN,
    AutoBasisSelector,
)

from .pykan_wrapper import (      # Compatibility layer (new)
    PyKANCompatible,
)

# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    # PyKAN models
    'MultKAN',
    'MLP',

    # Custom variants
    'ChebyshevKAN',
    'FourierKAN',
    'WaveletKAN',
    'RBF_KAN',

    # Section2_new extensions
    'AdaptiveSelectiveKAN',
    'AdaptiveSelectiveTrainer',
    'HeterogeneousKANLayer',
    'HeterogeneousBasisKAN',
    'AutoBasisSelector',
    'PyKANCompatible',
]
```

---

## Testing Strategy

### Integration Tests

**File:** `section2_new/tests/test_pykan_integration.py` (new)

```python
"""
Test compatibility between section2_new and pykan.

Ensures that:
1. Data generated by pykan works with section2_new models
2. MultKAN can be used in ensembles, evolution, population training
3. Section2_new extensions work with pykan utilities
"""

import pytest
import torch
from kan.utils import create_dataset
from kan.MultKAN import MultKAN

from section2_new.ensemble.expert_training import KANExpertEnsemble
from section2_new.evolution.genome import KANGenome


def test_pykan_data_with_ensemble():
    """Test that pykan datasets work with ensemble training."""
    # Generate data using pykan
    dataset = create_dataset(
        f=lambda x: x[0]**2 + x[1]**2,
        n_var=2,
        ranges=[-1, 1],
        train_num=100,
        test_num=50,
    )

    # Create ensemble
    ensemble = KANExpertEnsemble(
        input_dim=2,
        hidden_dim=5,
        output_dim=1,
        n_experts=3,
        kan_variant='rbf'
    )

    # Train with pykan data
    results = ensemble.train_experts(
        dataset['train_input'],
        dataset['train_label'],
        epochs=50
    )

    assert results is not None
    assert len(ensemble.experts) == 3


def test_multkan_in_evolution():
    """Test that MultKAN can be used in evolutionary search."""
    genome = KANGenome(
        layer_sizes=[2, 5, 1],
        basis_type='bspline',  # Use MultKAN
        grid_size=5,
        learning_rate=0.01
    )

    # Convert to model
    model = genome.to_model()

    # Should be MultKAN
    assert isinstance(model, MultKAN)

    # Should be trainable
    X = torch.randn(10, 2)
    y = model(X)
    assert y.shape == (10, 1)


def test_pykan_visualization():
    """Test that pykan's plot() works with section2_new models."""
    from section2_new.models.adaptive_selective_kan import AdaptiveSelectiveKAN

    # Create adaptive KAN
    kan = AdaptiveSelectiveKAN(
        input_dim=2,
        hidden_dim=5,
        output_dim=1,
        initial_grid=5
    )

    # If inherits from MultKAN, should have plot()
    if isinstance(kan, MultKAN):
        # Should not raise error
        try:
            kan.plot(beta=3)
        except Exception as e:
            pytest.fail(f"plot() failed: {e}")


def test_pykan_attribute_with_ensemble():
    """Test that pykan's attribution works with ensemble."""
    from kan.hypothesis import attribute
    from section2_new.ensemble.variable_importance import VariableImportanceAnalyzer

    # Create ensemble with MultKAN
    ensemble = KANExpertEnsemble(
        input_dim=2,
        hidden_dim=5,
        output_dim=1,
        n_experts=2,
        kan_variant='bspline'  # MultKAN
    )

    # Generate data
    X = torch.randn(50, 2)
    y = X[:, 0]**2 + X[:, 1]**2
    y = y.unsqueeze(1)

    # Train
    ensemble.train_experts(X, y, epochs=10)

    # Test attribution
    analyzer = VariableImportanceAnalyzer(ensemble)
    importance = analyzer.compute_pykan_importance(X[:10])

    assert importance is not None
    assert importance.shape[0] == 2  # 2 input features
```

---

## Migration Checklist

### Immediate Actions (Phase 1)

- [ ] Create `section2_new/models/pykan_wrapper.py`
- [ ] Update `section2_new/utils/__init__.py` to re-export pykan utilities
- [ ] Replace all custom data generation with `kan.utils.create_dataset()`
- [ ] Update `section2_new/DEMO.py` to use pykan data generation
- [ ] Update `section2_new/experiments/exp_1_ensemble_complete.py` data generation
- [ ] Add LBFGS import: `from kan.LBFGS import LBFGS`

### High Priority (Phase 2)

- [ ] Update `evolution/genome.py` to support `basis_type='bspline'` → MultKAN
- [ ] Update `ensemble/expert_training.py` to support `kan_variant='bspline'`
- [ ] Refactor `models/adaptive_selective_kan.py` to inherit from MultKAN
- [ ] Add compatibility tests in `tests/test_pykan_integration.py`
- [ ] Update all docstrings to reference pykan where appropriate

### Medium Priority (Phase 3)

- [ ] Integrate `kan.hypothesis.attribute()` into variable importance
- [ ] Add Feynman physics benchmarks in `experiments/feynman_benchmarks.py`
- [ ] Add pruning mutations to evolution
- [ ] Consider symbolic genome genes for evolution

### Low Priority (Phase 4)

- [ ] Optimize training loops to use `model.fit()` where possible
- [ ] Leverage pykan's checkpoint system for evolution
- [ ] Performance benchmarking: pykan vs custom implementations
- [ ] Memory profiling and optimization

### Documentation (Phase 5)

- [ ] Update `README.md` with pykan dependency information
- [ ] Create `PYKAN_INTEGRATION.md` guide
- [ ] Document which features require pykan vs work standalone
- [ ] Add migration examples to documentation
- [ ] Update installation instructions

---

## Conclusion

### Summary of Integration Strategy

1. **USE PyKAN for:** Data generation, B-spline KANs, LBFGS optimizer, basic utilities
2. **KEEP Custom for:** Ensemble methods, evolution, adaptive densification, heterogeneous basis, population training
3. **HYBRID Approach for:** Variable importance (add pykan attribution), adaptive KAN (extend MultKAN), genome (support bsplines)

### Benefits of Integration

✅ **Reduced maintenance burden** - Leverage pykan's updates and bug fixes
✅ **Better compatibility** - Ensure section2_new works with pykan ecosystem
✅ **Access to advanced features** - Symbolic reasoning, pruning, interpretability
✅ **Consistency** - Same data formats and APIs across projects
✅ **Future-proofing** - Benefit from pykan improvements automatically

### Remaining Custom Code Justification

The 88% of code that remains custom (ensembles, evolution, adaptive, heterogeneous, population) represents **genuine novel contributions** that extend pykan in meaningful ways. This is appropriate and valuable - section2_new should be an extension, not a replacement.

### Next Steps

1. Implement Phase 1 (data generation + utilities) - **Estimated 2-4 hours**
2. Implement Phase 2 (model compatibility) - **Estimated 4-8 hours**
3. Add tests and documentation - **Estimated 4-6 hours**
4. Consider Phase 3+ enhancements as time permits

**Total estimated integration effort: 10-18 hours**

---

## Appendix: Quick Reference

### PyKAN Components Available for Reuse

| Component | File | Key Classes/Functions |
|-----------|------|----------------------|
| **Data** | `kan/utils.py` | `create_dataset()`, `create_dataset_from_data()` |
| **KAN Model** | `kan/MultKAN.py` | `MultKAN` (B-spline KAN) |
| **KAN Layer** | `kan/KANLayer.py` | `KANLayer` |
| **Symbolic** | `kan/Symbolic_KANLayer.py` | `Symbolic_KANLayer` |
| **Splines** | `kan/spline.py` | `B_batch()`, `coef2curve()`, `curve2coef()` |
| **Optimizer** | `kan/LBFGS.py` | `LBFGS` |
| **Interpretability** | `kan/hypothesis.py` | `attribute()`, `detect_separability()`, `test_symmetry()` |
| **Symbolic Lib** | `kan/utils.py` | `SYMBOLIC_LIB` (sin, cos, exp, etc.) |
| **Symbolic Compiler** | `kan/compiler.py` | `expr2kan()` |
| **Physics Datasets** | `kan/feynman.py` | `get_feynman_dataset()` |
| **MLP Baseline** | `kan/MLP.py` | `MLP` |

### Section2_new Novel Components (Not in PyKAN)

| Component | Files | Purpose |
|-----------|-------|---------|
| **Ensemble** | `ensemble/*` (4 files) | Multi-seed training, uncertainty, clustering, stacking |
| **Adaptive** | `adaptive/*`, `models/adaptive_selective_kan.py` | Importance-based selective densification |
| **Heterogeneous** | `models/heterogeneous_kan.py` | Mixed basis per edge |
| **Population** | `population/population_trainer.py` | Population-based training with sync |
| **Evolution** | `evolution/*` (4 files) | Evolutionary architecture search |
| **Other Bases** | `section1/models/kan_variants.py` | Chebyshev, Fourier, Wavelet, RBF |

---

**End of Analysis**