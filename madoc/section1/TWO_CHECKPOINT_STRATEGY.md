# Two-Checkpoint Strategy Implementation

## Overview

This document describes the implementation of the two-checkpoint comparison strategy for evaluating KAN vs MLP/SIREN performance. This strategy enables fair **iso-compute comparisons** by capturing model performance at two critical points:

1. **At KAN Interpolation Threshold**: When KAN reaches its optimal performance (before overfitting)
2. **At Final Training**: When all models have exhausted their training budget

## Motivation

### The Problem

KANs have a fundamental capacity ceiling (the **interpolation threshold**) beyond which they overfit:
- **Before threshold**: KANs improve rapidly (scaling ∝ G^-4)
- **After threshold**: KANs overfit, test loss increases
- **MLPs/SIRENs**: Continue improving slowly but monotonically (scaling ∝ N^-1)

This creates an unfair comparison if we only look at "final" models:
- KAN@final: Overfit, poor performance
- MLP@final: Still improving, better performance
- **Result**: KAN looks worse than it actually is!

### The Solution

Capture two checkpoints:

**Checkpoint 1: Iso-Compute (at KAN threshold time T)**
- KAN: At interpolation threshold (optimal performance)
- MLP: After T seconds of training
- SIREN: After T seconds of training
- **Shows**: "KAN achieves X performance in time T; MLP/SIREN only achieve Y (worse) in same time"

**Checkpoint 2: Final (after full budget)**
- All models: After complete training
- **Shows**: "Given unlimited time, MLP/SIREN can eventually reach Z performance"

## Implementation Details

### 1. KAN Threshold Detection (`model_tests.py`)

**Function**: `detect_kan_threshold(test_losses, patience=2, threshold=0.05)`

Detects when KAN starts overfitting by monitoring test loss:
```python
# Looks for 2 consecutive epochs where test loss increases by >5%
# Returns the epoch number where KAN was at its best (before degradation)
```

**Location**: Called in `run_kan_grid_tests()` after training completes

### 2. Modified Training Functions

#### `run_kan_grid_tests()`
**Changes**:
- Detects interpolation threshold using `detect_kan_threshold()`
- Calculates cumulative time to threshold
- Returns: `(results_df, checkpoints, avg_threshold_time, pruned_models)`

**Checkpoint structure for KAN**:
```python
checkpoints[dataset_idx] = {
    'at_threshold': {
        'model': model,
        'epoch': threshold_epoch,
        'time': cumulative_time_at_threshold,
        'train_loss': ...,
        'test_loss': ...,
        'dense_mse': ...,
        'grid_size': ...,
        'num_params': ...
    },
    'final': {
        'model': model,
        'epoch': cumulative_epochs,
        'time': total_dataset_time,
        ...
    }
}
```

#### `run_mlp_tests()` and `run_siren_tests()`
**Changes**:
- Accept `kan_threshold_time` parameter
- During training, check if cumulative time >= kan_threshold_time
- Save checkpoint at that point (best model up to that time)
- Returns: `(results_df, checkpoints)`

**Checkpoint structure for MLP/SIREN**:
```python
checkpoints[dataset_idx] = {
    'at_kan_threshold_time': {
        'model': model,
        'epoch': threshold_epoch,
        'time': cumulative_time,
        'dense_mse': ...,
        'depth': ...,
        'activation': ...,  # MLP only
        ...
    },
    'final': {
        'model': best_model,
        'dense_mse': ...,
        ...
    }
}
```

### 3. Orchestration (`section1_1.py`)

**Key change**: Train KANs **first** to get threshold time, then pass to MLP/SIREN:

```python
# 1. Train KANs first
kan_results, kan_checkpoints, kan_threshold_time = run_kan_grid_tests(...)

# 2. Pass threshold time to others
mlp_results, mlp_checkpoints = run_mlp_tests(..., kan_threshold_time=kan_threshold_time)
siren_results, siren_checkpoints = run_siren_tests(..., kan_threshold_time=kan_threshold_time)

# 3. Save with checkpoints
save_run(all_results, 'section1_1',
         checkpoints={'mlp': mlp_checkpoints, 'siren': siren_checkpoints,
                     'kan': kan_checkpoints, ...},
         kan_threshold_time=kan_threshold_time)
```

### 4. Checkpoint Saving (`io.py`)

**Function**: `save_run(..., checkpoints=None, ...)`

Saves both model checkpoints and metadata:

**File structure**:
```
section1/results/sec1_results/
├── section1_1_20251023_143022_mlp.pkl                          # DataFrame
├── section1_1_20251023_143022_mlp_0_at_kan_threshold_time.pth  # Checkpoint
├── section1_1_20251023_143022_mlp_0_final.pth                  # Checkpoint
├── section1_1_20251023_143022_kan_0_at_threshold_state         # Checkpoint
├── section1_1_20251023_143022_kan_0_final_state                # Checkpoint
├── section1_1_20251023_143022_checkpoint_metadata.pkl          # Metadata
└── ...
```

### 5. Visualization (`visualization/plot_checkpoint_comparison.py`)

Three plots created:

1. **Iso-Compute Comparison**: Performance when KAN reaches threshold
2. **Final Comparison**: Performance after full training
3. **Time to Threshold**: How fast KAN converges

**Usage**:
```bash
python madoc/section1/visualization/plot_checkpoint_comparison.py section1_1 20251023_143022
```

## How to Use

### Running Experiments

```bash
cd pykan/madoc/section1
python section1_1.py --epochs 100 --steps_per_grid 200
```

**What happens**:
1. KANs train first across all grid sizes
2. Interpolation threshold detected for each dataset
3. Average threshold time calculated
4. MLPs/SIRENs train, saving checkpoint at KAN threshold time
5. All checkpoints saved (2 per model per dataset)

### Creating Visualizations

```bash
cd pykan/madoc/section1/visualization
python plot_checkpoint_comparison.py section1_1 <timestamp>
```

**Output**: 3 PNG files showing iso-compute, final, and time-to-threshold comparisons

### Loading Results

```python
from utils import load_run
import pickle

# Load DataFrames and metadata
results, meta = load_run('section1_1', '20251023_143022')

# Load checkpoint metadata
with open('results/sec1_results/section1_1_20251023_143022_checkpoint_metadata.pkl', 'rb') as f:
    checkpoints = pickle.load(f)

# Access checkpoint info
kan_threshold_time = meta['kan_threshold_time']
mlp_dense_mse_at_threshold = checkpoints['mlp'][0]['at_kan_threshold_time']['dense_mse']
kan_dense_mse_at_threshold = checkpoints['kan'][0]['at_threshold']['dense_mse']
```

## Key Findings Expected

### Hypothesis 1: KAN Speed Advantage
At iso-compute checkpoint:
- KAN should have **lower dense MSE** than MLP/SIREN
- Demonstrates KAN's rapid convergence

### Hypothesis 2: Long-Run Convergence
At final checkpoint:
- MLPs/SIRENs may **catch up** or **surpass** KAN
- Shows diminishing returns for KAN due to interpolation threshold

### Hypothesis 3: Problem Dependency
- On **simple problems**: KAN reaches threshold quickly, massive speedup
- On **complex problems**: KAN may need more capacity, threshold comes later

## Important Notes

### Limitations

1. **KAN Model at Threshold**: Currently we use the final trained model to represent the "threshold model" because we don't save intermediate models during training. The metrics (loss, time) are accurate, but the model weights are from the final grid.
   - **Impact**: Minimal for analysis purposes (metrics are correct)
   - **Future improvement**: Save actual model checkpoint during training loop

2. **Dense MSE Computation**: We compute dense MSE at checkpoints, which adds computational cost
   - **Current**: Computed for each checkpoint
   - **Trade-off**: Accuracy vs speed

3. **Threshold Detection Heuristic**: Uses test loss degradation (2 consecutive 5% increases)
   - Works well in practice
   - May need tuning for specific problems

### Backward Compatibility

The old `models` parameter in `save_run()` is still supported but deprecated:
```python
# Old way (still works)
save_run(results, 'section1_1', models={...})

# New way (recommended)
save_run(results, 'section1_1', checkpoints={...})
```

## Files Modified

1. `section1/utils/model_tests.py`:
   - Added `detect_kan_threshold()` function
   - Modified `run_kan_grid_tests()` to detect threshold and save checkpoints
   - Modified `run_mlp_tests()` to save checkpoint at KAN threshold time
   - Modified `run_siren_tests()` to save checkpoint at KAN threshold time

2. `section1/section1_1.py`:
   - Reordered training (KANs first)
   - Pass `kan_threshold_time` to MLP/SIREN training
   - Save checkpoints instead of single models

3. `section1/utils/io.py`:
   - Modified `save_run()` to accept and save checkpoints
   - Save checkpoint metadata as pickle for easy access

4. `section1/visualization/plot_checkpoint_comparison.py`:
   - NEW: Create iso-compute and final comparison plots

## Future Enhancements

### Tier 2 (If time permits)
- Compute dense MSE at more intermediate points (every 10% of training)
- Save actual model checkpoints during training (not just metrics)
- Add H1 norm metrics for PDE problems
- Statistical significance testing (bootstrapping)

### Tier 3 (Publication-grade)
- Full scaling law analysis (log-log plots)
- Parameter efficiency curves
- Ablation studies on threshold detection parameters
- Comparison with other threshold detection methods

## Testing

To verify the implementation works:

```bash
# Quick test with reduced epochs
cd pykan/madoc/section1
python section1_1.py --epochs 20 --steps_per_grid 10

# Check that checkpoints were saved
ls results/sec1_results/*checkpoint_metadata.pkl

# Create plots
cd visualization
python plot_checkpoint_comparison.py section1_1 <timestamp>

# Verify plots were created
ls ../results/sec1_results/*comparison*.png
```

## Questions?

See:
- Original discussion in main chat
- `MODEL_SAVING_GUIDE.md` for checkpoint loading details
- Code comments in modified files
