# Section 2.3: Merge_KAN Implementation

## Overview

This section implements a novel **Merge_KAN** approach inspired by mixture-of-experts, where multiple specialized KAN "expert" models are trained independently, then merged into a single unified architecture. The key idea is to leverage KANs' lack of catastrophic forgetting to combine diverse learned representations.

## Implementation Strategy

### Phase 1: Expert Generation
Train a pool of diverse expert KANs with varying configurations:
- **Depth variations**: 2-layer and 3-layer networks
- **Spline order variations**: k=2 (quadratic) and k=3 (cubic) B-splines
- **Multiple seeds**: 5 different random initializations per configuration
- **Total**: ~15 experts per dataset (2 depths × 1 k-variation + 1 depth × 1 k × 5 seeds)

### Phase 2: Dependency Detection
Use PyKAN's built-in attribution system to identify which input variables each expert depends on:
```python
model.attribute()  # Compute attribution scores
input_scores = model.node_scores[0]  # First layer scores
active_inputs = (input_scores > 1e-2).nonzero()  # Default threshold from KAN paper
```

Each expert learns a **unique functional dependency pattern**, e.g.:
- Expert 1: `{x_0, x_1}`
- Expert 2: `{x_0, x_2}`
- Expert 3: `{x_1, x_2}`

### Phase 3: Expert Selection
**Strategy: Best performer per unique dependency set**

1. Group experts by their detected dependencies
2. Within each group, select the expert with lowest `dense_mse`
3. This ensures:
   - **Diversity**: Different dependency patterns represented
   - **Quality**: Best-performing expert for each pattern

### Phase 4: Merging Architecture

**Option A (Implemented)**: Full input space with masked connections

```
Expert_1: [2, 5, 1] with deps {x_0, x_1}
Expert_2: [2, 5, 1] with deps {x_0, x_1}
Expert_3: [2, 3, 1] with deps {x_1}

↓ Merge ↓

Merged_KAN: [2, 13, 3, 1]
            ↑   ↑   ↑  ↑
            │   │   │  └─ Final output
            │   │   └──── Aggregation layer (3 neurons)
            │   └──────── Combined hidden (5+5+3=13 neurons)
            └──────────── Full input space (all variables)
```

**Merge implementation using PyKAN primitives:**
- `expand_width()` - Add neurons to layers
- `expand_depth()` - Add aggregation layer
- Direct weight transfer for matching grid/k parameters
- Masked connections preserve expert specializations

### Phase 5: Training Merged KAN

**Hybrid Grid Refinement Strategy:**

Uses a fixed schedule with early stopping:

```python
grids = [3, 5, 10, 20]
steps_per_grid = 200
```

**Early stopping rules:**
- Stop if test loss increases by >5% from best
- Requires 2 consecutive increases to prevent false stops
- Based on interpolation threshold theory from KAN paper

**Interpolation threshold consideration:**
```
For merged KAN [8, 10, 3, 1]:
  Total params ≈ (8×10 + 10×3 + 3×1) × G = 113G

With 1000 training samples:
  Optimal G ≈ 1000/113 ≈ 9 grid intervals

Strategy: Stop refinement before heavy overfitting sets in
```

## File Structure

```
section2/
├── section2_3.py              # Main experiment script
├── utils/
│   ├── merge_kan.py           # Merge_KAN implementation
│   └── __init__.py            # Export merge functions
└── section2_3_README.md       # This file
```

## Key Functions

### `generate_expert_pool(dataset, device, true_function, dataset_name, n_seeds=5)`
Trains multiple expert KANs with varied configurations.

**Returns:** List of expert dicts with:
- `model`: Trained KAN
- `dense_mse`: Performance metric
- `dependencies`: Detected input dependencies
- `config`: Training configuration

### `select_best_experts(experts)`
Groups experts by dependency pattern and selects best per group.

**Returns:** List of selected experts (best of each dependency pattern)

### `merge_kans(expert_models, input_dim, device, grid=None, k=None)`
Merges multiple KANs into wider architecture.

**Strategy:**
- Concatenates expert hidden layers horizontally
- Adds aggregation layer on top
- Transfers weights when grid/k parameters match
- Masks inactive connections (Option A)

**Returns:** Merged KAN model

### `train_merged_kan_with_refinement(merged_model, dataset, ...)`
Trains merged KAN with grid refinement and early stopping.

**Returns:** Dict with training results and metrics

### `run_merge_kan_experiment(dataset, ...)`
Complete experiment pipeline: generate → select → merge → train.

## Usage

### Basic Usage
```bash
cd madoc/section2
python section2_3.py
```

### Test Mode (Faster)
```bash
python section2_3.py --test-mode
```
- Reduces expert pool from 15 to 6 experts
- Useful for debugging and quick validation

### Custom Number of Seeds
```bash
python section2_3.py --n-seeds 10
```

## Output

Results saved to `results/section2_3/`:

- **summary.csv**: High-level results per dataset
  - `n_experts_trained`, `n_experts_selected`
  - `merged_dense_mse`, `merged_num_params`

- **experts.csv**: All trained experts
  - Configuration (depth, k, seed)
  - Performance (dense_mse, train_time)
  - Dependencies detected
  - Whether selected for merging

- **selected_experts.csv**: Experts chosen for merging
  - One per unique dependency pattern
  - Best performer in each group

- **grid_history.csv**: Grid refinement progression
  - Train/test loss at each grid size
  - Shows when early stopping triggered

- **models.pt**: Saved merged KAN models

## Evaluation Metrics

Compare Merge_KAN results with Section 1.3 baselines:

1. **Dense MSE**: Fine-grained error on 10k sample grid
2. **Parameter efficiency**: Performance vs model size
3. **Generalization**: Test loss trajectory during refinement
4. **Diversity**: Number of unique dependency patterns discovered

## Theoretical Motivation

### Why Merge_KAN?

1. **Lack of Catastrophic Forgetting**: KANs retain learned representations better than traditional NNs, making them suitable for ensemble-like approaches

2. **Functional Decomposition**: Real-world functions often decompose into simpler subfunctions with different variable dependencies

3. **Curse of Dimensionality**: In high-dimensional spaces, different regions may be better approximated by specialized sub-models

4. **Exploration vs Exploitation**: Multiple experts explore different optima; merging exploits their collective knowledge

## Potential Extensions (Section 2.3.1, 2.3.2)

### Merge Variants
- **Normal merge** (implemented): Direct weight transfer
- **Snap merge**: Apply symbolic regression to snap functions before merging
- **Freeze merge**: Freeze expert weights, only train aggregation layer

### Multi-stage Merging
Instead of single merge 15 → 1, try staged:
```
15 experts → 8 merged → 3 merged → 1 final
```

### Adaptive Merging
Use validation performance to decide which experts to merge at each stage.

## Known Limitations

1. **Grid/k mismatch**: If experts trained with different grid sizes or spline orders, weight transfer is approximate
   - Current: Falls back to random initialization with masked connections
   - Better: Implement spline interpolation/resampling

2. **Input dependency detection**: Currently assumes experts use first N inputs sequentially
   - TODO: Track actual input mapping from dependency detection
   - Need to handle arbitrary input permutations

3. **Computational cost**: Training 15 experts per dataset is expensive
   - Mitigate: Use `--test-mode` for development
   - Parallelize expert training (future work)

4. **Merge architecture heuristics**: Aggregation layer size is heuristic (max(3, n_experts))
   - Could be optimized based on dataset complexity

## Comparison with Section 2.4 (KAN_Probe)

Merge_KAN is a simplified version of the genetic KAN_Probe approach:
- **Merge_KAN**: Single-stage selection and merging
- **KAN_Probe**: Multi-generation evolution with crossover/mutation

Section 2.3 serves as a foundation for the more complex Section 2.4.

## References

- Original plan: `section2/plan.md` lines 32-73
- Attribution system: PyKAN `MultKAN.py:1913` (`model.attribute()`)
- Pruning thresholds: KAN paper defaults (node: 1e-2, edge: 3e-2)
- Interpolation threshold: KAN paper Section on grid extension
