# Checkpoint Visualization Guide for Section 2

This guide explains how to use the checkpoint comparison visualizations for Section 2 experiments.

## Overview

The two-checkpoint strategy enables **iso-compute comparisons** by capturing model performance at:
1. **At Threshold**: When the reference approach reaches its interpolation threshold
2. **Final**: After full training budget is exhausted

This allows us to answer two key questions:
- **Speed**: Which approach reaches target performance fastest?
- **Asymptotic Performance**: Given unlimited time, which approach is best?

---

## Section 2.1: Optimizer Comparison

Compares three optimizers on KAN training: **Adam**, **LBFGS**, and **LM** (Levenberg-Marquardt)

### Available Plots

#### 1. Iso-Compute Comparison
```bash
python plot_checkpoint_comparison.py section2_1 --timestamp <TIMESTAMP>
# Output: section2_1_<TIMESTAMP>_iso_compute.png
```

**Shows**: Performance when LBFGS reaches its interpolation threshold
- Answers: "At the point when LBFGS reaches optimal, how do Adam/LM compare?"
- Fair comparison: all optimizers evaluated at same time point
- Lower is better (log scale)

**Expected findings**:
- LBFGS typically fastest to converge
- LM may have similar or better performance at threshold
- Adam may lag behind at early stage

#### 2. Final Performance Comparison
```bash
python plot_checkpoint_comparison.py section2_1 --timestamp <TIMESTAMP>
# Output: section2_1_<TIMESTAMP>_final.png
```

**Shows**: Performance after full training budget
- Answers: "Given unlimited time, which optimizer achieves best results?"
- Long-run convergence behavior
- Lower is better (log scale)

**Expected findings**:
- All optimizers may converge to similar final values
- Some optimizers may continue improving while others plateau

#### 3. Time-to-Threshold Analysis
```bash
python plot_checkpoint_comparison.py section2_1 --timestamp <TIMESTAMP>
# Output: section2_1_<TIMESTAMP>_time_to_threshold.png
```

**Shows**: Time (seconds) to reach interpolation threshold
- Answers: "Which optimizer converges fastest?"
- Direct speed comparison
- Lower is better

**Expected findings**:
- LBFGS typically fastest (second-order optimization)
- LM competitive due to adaptive damping
- Adam slower but more stable

### Usage Examples

**Basic usage** (uses latest run):
```bash
cd /Users/main/Desktop/my_pykan/pykan/madoc/section2/visualization
python plot_checkpoint_comparison.py section2_1
```

**Specific timestamp**:
```bash
python plot_checkpoint_comparison.py section2_1 --timestamp 20251024_143022
```

**Show plots interactively**:
```bash
python plot_checkpoint_comparison.py section2_1 --show
```

**Custom output directory**:
```bash
python plot_checkpoint_comparison.py section2_1 --output-dir /path/to/output
```

---

## Section 2.2: Adaptive Density Comparison

Compares three approaches: **Adaptive Only**, **Adaptive+Regular**, and **Baseline**

### Available Plots

#### 1. Iso-Compute Comparison
```bash
python plot_checkpoint_comparison.py section2_2 --timestamp <TIMESTAMP>
# Output: section2_2_<TIMESTAMP>_iso_compute.png
```

**Shows**: Performance when baseline reaches its interpolation threshold
- Answers: "At the point when baseline reaches optimal, how do adaptive approaches compare?"
- Tests if adaptive density provides speed advantage
- Lower is better (log scale)

**Expected findings**:
- Adaptive approaches may reach similar performance faster
- Benefit depends on problem structure (importance distribution)
- Some problems may not benefit from adaptive density

#### 2. Final Performance Comparison
```bash
python plot_checkpoint_comparison.py section2_2 --timestamp <TIMESTAMP>
# Output: section2_2_<TIMESTAMP>_final.png
```

**Shows**: Performance after full training budget
- Answers: "Given unlimited time, does adaptive density help?"
- Long-run comparison of approaches
- Lower is better (log scale)

**Expected findings**:
- All approaches likely converge to similar final values
- Adaptive+Regular may have slight edge from selective densification
- Main benefit is speed, not asymptotic performance

### Usage Examples

**Basic usage** (uses latest run):
```bash
cd /Users/main/Desktop/my_pykan/pykan/madoc/section2/visualization
python plot_checkpoint_comparison.py section2_2
```

**Specific timestamp**:
```bash
python plot_checkpoint_comparison.py section2_2 --timestamp 20251024_150000
```

---

## Understanding the Results

### Interpreting Iso-Compute Comparisons

**Lower bars = Better performance at same time**

Example interpretation:
```
At threshold (50s):
  Adam:   1.2e-3  ← Slower, worse performance
  LBFGS:  3.4e-4  ← Reference point (reached threshold here)
  LM:     2.8e-4  ← Faster, better performance
```

**Conclusion**: LM reaches better accuracy in less time than LBFGS

### Interpreting Final Comparisons

**Lower bars = Better final performance**

Example interpretation:
```
After 200s:
  Adam:   2.1e-4  ← Eventually caught up
  LBFGS:  2.0e-4  ← Slightly better
  LM:     1.9e-4  ← Best final result
```

**Conclusion**: All converge to similar values; LM has slight edge

### Interpreting Time-to-Threshold

**Lower bars = Faster convergence**

Example interpretation:
```
Time to threshold:
  Adam:   80s   ← Slowest
  LBFGS:  50s   ← Baseline
  LM:     45s   ← Fastest (10% speedup)
```

**Conclusion**: LM provides 10% speedup over LBFGS

---

## Command-Line Options

```bash
python plot_checkpoint_comparison.py <section> [OPTIONS]

Required Arguments:
  section              Section name: section2_1 or section2_2

Optional Arguments:
  --timestamp STR      Specific timestamp to load (default: latest)
  --strategy STR       Run selection strategy: latest, max_epochs, min_epochs (default: latest)
  --show              Show plots interactively instead of saving
  --output-dir PATH    Custom output directory (default: results directory)
  -h, --help          Show help message
```

---

## File Structure

After running experiments and creating visualizations:

```
section2/results/sec1_results/
├── section2_1_20251024_143022_e10_adam.pkl
├── section2_1_20251024_143022_e10_lbfgs.pkl
├── section2_1_20251024_143022_e10_lm.pkl
├── section2_1_20251024_143022_e10_checkpoint_metadata.pkl
├── section2_1_20251024_143022_e10_adam_0_at_threshold_*
├── section2_1_20251024_143022_e10_adam_0_final_*
├── ... (more checkpoints)
├── section2_1_20251024_143022_iso_compute.png          ← Generated plot
├── section2_1_20251024_143022_final.png                ← Generated plot
└── section2_1_20251024_143022_time_to_threshold.png    ← Generated plot

section2/results/sec2_results/
├── section2_2_20251024_150000_e10_adaptive_only.pkl
├── section2_2_20251024_150000_e10_adaptive_regular.pkl
├── section2_2_20251024_150000_e10_baseline.pkl
├── section2_2_20251024_150000_e10_checkpoint_metadata.pkl
├── ... (checkpoints)
├── section2_2_20251024_150000_iso_compute.png          ← Generated plot
└── section2_2_20251024_150000_final.png                ← Generated plot
```

---

## Checkpoint Metadata Structure

The `checkpoint_metadata.pkl` file contains:

```python
{
    'optimizer_name': {  # e.g., 'adam', 'lbfgs', 'lm'
        dataset_idx: {
            'at_threshold': {
                'epoch': int,
                'time': float,
                'train_loss': float,
                'test_loss': float,
                'dense_mse': float,
                'grid_size': int,
                'num_params': int,
                'optimizer': str
            },
            'final': {
                'epoch': int,
                'time': float,
                'train_loss': float,
                'test_loss': float,
                'dense_mse': float,
                'grid_size': int,
                'num_params': int,
                'optimizer': str
            }
        }
    }
}
```

**Note**: The actual model weights are stored separately in checkpoint files (e.g., `*_at_threshold_*`, `*_final_*`)

---

## Troubleshooting

### No checkpoint metadata found

**Error**: `No checkpoint metadata found for section2_1 <timestamp>`

**Solution**: Ensure you've run the experiment with the updated code that saves checkpoints:
```bash
cd /Users/main/Desktop/my_pykan/pykan/madoc/section2
python section2_1.py --epochs 10
```

### Plots look wrong / missing bars

**Cause**: Some checkpoints may be missing or have invalid data

**Check**:
1. Verify checkpoint_metadata.pkl exists
2. Check that all optimizers/approaches completed training
3. Look for NaN or inf values in dense_mse

**Debug**:
```python
import pickle
with open('checkpoint_metadata.pkl', 'rb') as f:
    data = pickle.load(f)
    print(data.keys())  # Should show all optimizers/approaches
    print(data['adam'][0])  # Check structure
```

### Different number of datasets

**Cause**: Dataset names mismatch or incomplete results

**Solution**: Ensure all approaches trained on same datasets

---

## Integration with Other Visualizations

The checkpoint comparison plots complement other Section 2 visualizations:

| Plot Script | Purpose | Checkpoint Use |
|------------|---------|----------------|
| `plot_checkpoint_comparison.py` | Iso-compute & final comparison | ✅ Uses checkpoints |
| `plot_optimizer_comparison.py` | Dense MSE over epochs | Uses DataFrames |
| `plot_best_loss.py` | Loss evolution | Uses DataFrames |
| `plot_function_fit.py` | Learned functions | Can use checkpoint models |
| `plot_heatmap_2d.py` | 2D visualizations | Can use checkpoint models |

---

## Example Workflow

### Complete Analysis Pipeline

```bash
# 1. Run experiment
cd /Users/main/Desktop/my_pykan/pykan/madoc/section2
python section2_1.py --epochs 10

# 2. Create checkpoint comparison plots
cd visualization
python plot_checkpoint_comparison.py section2_1

# 3. Create other visualizations
python plot_optimizer_comparison.py section2_1
python plot_best_loss.py --section section2_1 --dataset 0
python plot_function_fit.py --section section2_1

# 4. View all results
ls ../results/sec1_results/*png
```

### Batch Processing Multiple Runs

```bash
# Find all timestamps
python -c "from utils.result_finder import find_all_runs; print(find_all_runs('section2_1'))"

# Process each
for ts in 20251024_143022 20251024_150000; do
    python plot_checkpoint_comparison.py section2_1 --timestamp $ts
done
```

---

## Advanced Usage

### Custom Plotting

You can import the plotting functions in your own scripts:

```python
from plot_checkpoint_comparison import (
    plot_iso_compute_comparison_optimizers,
    plot_final_comparison_optimizers,
    plot_time_to_threshold
)
import pickle

# Load checkpoint metadata
with open('checkpoint_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

# Create custom plots
dataset_names = ['sin', 'poly', 'highfreq', 'spec']
fig = plot_iso_compute_comparison_optimizers(metadata, dataset_names)
fig.savefig('custom_iso_compute.png', dpi=300)
```

### Analyzing Specific Datasets

```python
import pickle

with open('checkpoint_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

# Compare optimizers on dataset 0
for opt in ['adam', 'lbfgs', 'lm']:
    threshold_mse = metadata[opt][0]['at_threshold']['dense_mse']
    final_mse = metadata[opt][0]['final']['dense_mse']
    time_to_threshold = metadata[opt][0]['at_threshold']['time']

    print(f"{opt.upper()}:")
    print(f"  Threshold MSE: {threshold_mse:.6e}")
    print(f"  Final MSE: {final_mse:.6e}")
    print(f"  Time to threshold: {time_to_threshold:.2f}s")
    print(f"  Improvement: {(threshold_mse/final_mse - 1)*100:.1f}%")
```

---

## See Also

- [TWO_CHECKPOINT_STRATEGY.md](../TWO_CHECKPOINT_STRATEGY.md) - Implementation details
- [METADATA_OPTIMIZATION.md](../METADATA_OPTIMIZATION.md) - Metadata storage approach
- [README.md](README.md) - General visualization guide
- [loading_guide.md](loading_guide.md) - Data loading patterns

---

## Questions?

For issues or questions about checkpoint visualizations:
1. Check that experiments ran successfully with new checkpoint code
2. Verify checkpoint_metadata.pkl file exists and is not corrupted
3. Review Phase 1 and Phase 2 implementation summaries in main documentation
