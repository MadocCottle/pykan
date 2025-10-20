# Section 2 New: Evolutionary KAN Implementation

This section implements six major extensions to the KAN architecture for geophysical inverse problems.

**Built on PyKAN:** This implementation leverages pykan's core utilities and supports pykan's MultKAN (B-spline basis) alongside custom KAN variants.

## PyKAN Integration

Section2_new is designed as a compatible extension of the original pykan repository (Liu et al., 2024):

- **Data Generation**: Uses `kan.utils.create_dataset()` for consistent data handling
- **B-spline KANs**: Supports pykan's MultKAN with grid adaptation and symbolic reasoning
- **Optimizers**: Uses pykan's LBFGS optimizer for second-order optimization
- **Custom Variants**: Maintains custom Chebyshev, Fourier, Wavelet, and RBF implementations

**Reference:**
```
Liu, Ziming, et al. "KAN: Kolmogorov-Arnold Networks."
arXiv preprint arXiv:2404.19756 (2024).
https://arxiv.org/abs/2404.19756

Liu, Ziming, et al. "KAN 2.0: Kolmogorov-Arnold Networks Meet Science."
arXiv preprint arXiv:2408.10205 (2024).
https://arxiv.org/abs/2408.10205
```

## Extensions

1. **Hierarchical Ensemble of KAN Experts** - Multi-seed expert training with clustering and stacking
2. **Adaptive Densification Based on Node Importance** - Per-node selective grid refinement
3. **Heterogeneous Basis Functions** - Mixed-basis KAN layers with learnable/heuristic selection
4. **Population-Based Training** - Multi-seed coordination with gradient sharing
5. **Evolutionary Architecture Search** - Genome-based evolution of KAN architectures
6. **Geophysical Application** - Physics-informed constraints and uncertainty quantification

## Directory Structure

- `models/` - New KAN architectures (heterogeneous, ensemble, adaptive)
- `ensemble/` - Expert training, clustering, stacking
- `adaptive/` - Importance tracking and selective densification
- `population/` - Population-based training with synchronization
- `evolution/` - Evolutionary search with genome representation
- `geophysics/` - Physics constraints and uncertainty quantification
- `experiments/` - Experimental scripts for each extension
- `utils/` - Parallel training, serialization, visualization utilities
- `visualization/` - Ensemble, evolution, and uncertainty plots
- `tests/` - Unit tests for all components

## Quick Start

### Phase 1: Ensemble and Heterogeneous Basis

```python
# Train ensemble of KAN experts with pykan's MultKAN (B-spline)
from section2_new.ensemble.expert_training import KANExpertEnsemble
from kan.utils import create_dataset

# Generate data using pykan
dataset = create_dataset(
    f=lambda x: x[0]**2 + x[1],
    n_var=2,
    ranges=[-1, 1],
    train_num=100,
    test_num=50
)

# Create ensemble with B-spline KAN (pykan's MultKAN)
ensemble = KANExpertEnsemble(
    input_dim=2,
    hidden_dim=8,
    output_dim=1,
    depth=3,
    n_experts=10,
    kan_variant='bspline'  # Uses pykan's MultKAN
)

# Or use custom RBF KAN
ensemble_rbf = KANExpertEnsemble(
    input_dim=2,
    hidden_dim=8,
    output_dim=1,
    depth=3,
    n_experts=10,
    kan_variant='rbf'  # Custom RBF implementation
)

results = ensemble.train_experts(
    dataset['train_input'],
    dataset['train_label'],
    epochs=500
)
y_pred, uncertainty = ensemble.predict_with_uncertainty(dataset['test_input'])
```

### Phase 2: Adaptive and Population-Based Training

```python
# Adaptive densification
from section2_new.models.adaptive_selective_kan import AdaptiveSelectiveKAN

kan = AdaptiveSelectiveKAN(architecture=[2, 5, 1], initial_grid=3, max_grid=10)
# Training loop with periodic densification included
```

### Phase 3: Evolution and Geophysics

```python
# Evolutionary architecture search
from section2_new.evolution.evolutionary_search import EvolutionaryKANSearch

evolver = EvolutionaryKANSearch(
    population_size=30,
    n_generations=50,
    objectives=['accuracy', 'complexity', 'speed']
)
best_genomes = evolver.evolve(X_train, y_train, X_val, y_val)
```

## Implementation Status

See [plan.md](plan.md) for detailed implementation roadmap and viability assessment.

## Requirements

All essential dependencies are already installed. Optional dependencies:

```bash
pip install scikit-learn  # For clustering (optional, can use custom)
pip install deap  # For evolutionary algorithms (optional)
pip install tensorboard  # For experiment tracking (optional)
```

## Testing

```bash
# Run all tests
pytest section2_new/tests/

# Test pykan integration
python section2_new/tests/test_pykan_integration.py

# Run specific test
pytest section2_new/tests/test_ensemble.py
```

## Available KAN Variants

section2_new supports multiple basis functions:

| Variant | Source | Description |
|---------|--------|-------------|
| `bspline` | pykan (MultKAN) | B-spline basis with grid adaptation, symbolic reasoning, pruning |
| `rbf` | Custom | Radial basis functions |
| `chebyshev` | Custom | Chebyshev polynomial basis |
| `fourier` | Custom | Fourier basis for periodic functions |
| `wavelet` | Custom | Wavelet basis (multi-resolution) |

**Note:** `bspline` variant requires pykan to be available and provides access to advanced features like `model.plot()`, `model.prune()`, and `model.auto_symbolic()`.

## PyKAN Compatibility

For details on pykan integration, see [pykan_plan.md](pykan_plan.md).