# PyKAN Section 2 Implementation Documentation

This documentation covers the comprehensive implementation effort to bring Section 2 up to Section 1's standards for runability, consistency, and data visualization pipelines.

## Quick Navigation

### Implementation Guides
- **[Phase Implementation Summary](PHASE_IMPLEMENTATION_SUMMARY.md)** - Complete record of all changes made across four phases
  - Phase 1: Core Infrastructure (Section 2.1)
  - Phase 2: Extension to All Experiments (Section 2.2, 2.3)
  - Phase 3: Visualization System
  - Phase 4: Documentation

### Technical Documentation
- **[Two-Checkpoint Strategy](TWO_CHECKPOINT_STRATEGY.md)** - Comprehensive guide to checkpoint implementation
  - Motivation and problem statement
  - Implementation details
  - Usage examples
  - Expected findings

- **[Metadata Optimization](METADATA_OPTIMIZATION.md)** - Storage philosophy and patterns
  - Essential vs derivable metadata
  - Self-contained DataFrames
  - File organization
  - Best practices

### Visualization Guides
- **[Checkpoint Visualization Guide](../section2/visualization/CHECKPOINT_VISUALIZATION_GUIDE.md)** - Using visualization tools
  - Section 2.1: Optimizer comparison plots
  - Section 2.2: Adaptive density comparison plots
  - Interpretation guidelines
  - Troubleshooting

## Overview

### The Challenge

Section 2 of the PyKAN project needed to be brought up to Section 1's level in terms of:
1. **Runability** - Consistent execution patterns and reliable checkpointing
2. **Consistency** - Uniform data generation and storage standards
3. **Data Pipeline** - Comprehensive visualization and analysis capabilities

### The Solution

A four-phase implementation strategy:

```
Phase 1: Core Infrastructure
├── Threshold detection for Section 2.1
├── Two-checkpoint strategy
├── Metadata in DataFrame.attrs
└── LBFGS as reference optimizer

Phase 2: Extension
├── Apply to Section 2.2 (adaptive density)
├── Baseline as reference approach
└── Verify Section 2.3

Phase 3: Visualization
├── Checkpoint comparison plots
├── Iso-compute analysis
└── Time-to-threshold analysis

Phase 4: Documentation
├── Implementation guides
├── Technical documentation
└── Usage examples
```

## Key Concepts

### Two-Checkpoint Strategy

Each experiment saves two checkpoints per model/dataset combination:

1. **`at_threshold`** - State when interpolation threshold is detected
   - Best generalization point before overfitting
   - Used for iso-compute comparisons

2. **`final`** - State at end of training
   - Final converged performance
   - Used for long-run comparisons

**Why?** Enables fair temporal comparisons between approaches with different convergence speeds.

### Interpolation Threshold Detection

Automatic detection of when model begins overfitting:

```python
def detect_kan_threshold(test_losses, patience=2, threshold=0.05):
    """Detect when test loss increases by 5% for 2 consecutive epochs"""
    # Returns: best epoch before overfitting
```

**Why?** KANs can overfit on training data while test performance degrades. The threshold marks the sweet spot.

### Iso-Compute Comparisons

Compare different approaches at the same computational budget (time):

```
LBFGS reaches threshold at t=45s
Adam at t=45s: Compare performance here
LM at t=45s: Compare performance here
```

**Why?** Fair comparison - some optimizers converge faster but may not be better given equal time.

### Reference Approach Pattern

Each experiment trains a "reference" approach first:
- **Section 2.1**: LBFGS optimizer (most stable)
- **Section 2.2**: Baseline refinement (standard approach)

**Why?** Establishes the threshold time baseline for iso-compute comparisons.

### Metadata Storage Philosophy

Store only **essential** metadata; derive everything else from DataFrames:

**Essential** (stored):
- Training configuration: epochs, device
- Reference times: lbfgs_threshold_time, baseline_threshold_time
- Checkpoint metadata: epochs, losses, times

**Derivable** (computed on demand):
- Grid sizes: `df['grid_size'].unique()`
- Depths: `df['depth'].unique()`
- Dataset names: `df['dataset_name'].unique()`
- Frequencies, activations, etc.

**Why?** Reduces redundancy, prevents sync issues, keeps files lean.

## Implementation Highlights

### Phase 1: Core Infrastructure

**Modified Files**:
- `section2/utils/optimizer_tests.py` - Added threshold detection
- `section2/section2_1.py` - Reordered to train LBFGS first
- `section2/utils/io.py` - Enhanced checkpoint saving

**Key Addition**: `detect_kan_threshold()` function
```python
# Detects when test loss increases by 5% for 2 consecutive epochs
threshold_epoch = detect_kan_threshold(test_losses_per_epoch)
```

**Result**: Section 2.1 now has full checkpoint infrastructure matching Section 1.

### Phase 2: Extension to All Experiments

**Modified Files**:
- `section2/utils/optimizer_tests.py` - Updated adaptive density and baseline tests
- `section2/section2_2.py` - Reordered to train baseline first
- `section2/utils/io.py` - Added baseline_threshold_time metadata

**Result**: All Section 2 experiments use consistent checkpoint infrastructure.

### Phase 3: Visualization System

**Created Files**:
- `section2/visualization/plot_checkpoint_comparison.py` (430 lines)
- `section2/visualization/CHECKPOINT_VISUALIZATION_GUIDE.md` (400+ lines)

**Updated Files**:
- `section2/visualization/README.md` - Added checkpoint comparison section

**Key Features**:
- Iso-compute comparison plots
- Final performance comparison plots
- Time-to-threshold analysis
- Full CLI with argparse
- Support for both Section 2.1 and 2.2

**Usage**:
```bash
# Generate all Section 2.1 checkpoint plots
python plot_checkpoint_comparison.py --section section2_1 --all

# Generate all Section 2.2 checkpoint plots
python plot_checkpoint_comparison.py --section section2_2 --all
```

**Result**: Comprehensive visualization system for checkpoint-based analysis.

### Phase 4: Documentation

**Created Files**:
- `documentation/TWO_CHECKPOINT_STRATEGY.md` (300+ lines)
- `documentation/METADATA_OPTIMIZATION.md` (250+ lines)
- `documentation/PHASE_IMPLEMENTATION_SUMMARY.md` (250+ lines)
- `documentation/INDEX.md` (this file)

**Result**: Complete documentation of implementation and design philosophy.

## File Organization

```
pykan/madoc/
├── documentation/                    # Implementation documentation
│   ├── INDEX.md                     # Main entry point (you are here)
│   ├── TWO_CHECKPOINT_STRATEGY.md   # Checkpoint implementation guide
│   ├── METADATA_OPTIMIZATION.md     # Storage philosophy
│   └── PHASE_IMPLEMENTATION_SUMMARY.md  # Detailed change log
│
├── section1/                        # Section 1 experiments
│   ├── section1_1.py               # Function approximation
│   ├── utils/                      # Section 1 utilities
│   └── visualization/              # Section 1 plots
│
├── section2/                        # Section 2 experiments
│   ├── section2_1.py               # Optimizer comparison
│   ├── section2_2.py               # Adaptive density
│   ├── section2_3.py               # Regularization
│   │
│   ├── utils/                      # Section 2 utilities
│   │   ├── optimizer_tests.py      # Training functions (modified)
│   │   └── io.py                   # I/O functions (modified)
│   │
│   └── visualization/              # Section 2 plots
│       ├── plot_checkpoint_comparison.py  # Checkpoint plots (new)
│       ├── CHECKPOINT_VISUALIZATION_GUIDE.md  # Guide (new)
│       └── README.md               # Visualization overview (updated)
│
└── results/                        # Saved results
    ├── section2_1_results.pkl
    ├── section2_1_checkpoints.pkl
    ├── section2_1_checkpoint_metadata.pkl
    ├── section2_2_results.pkl
    ├── section2_2_checkpoints.pkl
    └── section2_2_checkpoint_metadata.pkl
```

## Usage Examples

### Running Experiments

**Section 2.1: Optimizer Comparison**
```bash
cd pykan/madoc/section2
python section2_1.py --epochs 100

# Generates:
# - section2_1_results.pkl (DataFrames with attrs)
# - section2_1_checkpoints.pkl (model states)
# - section2_1_checkpoint_metadata.pkl (fast access)
```

**Section 2.2: Adaptive Density**
```bash
cd pykan/madoc/section2
python section2_2.py --epochs 100

# Generates:
# - section2_2_results.pkl
# - section2_2_checkpoints.pkl
# - section2_2_checkpoint_metadata.pkl
```

### Loading Results

```python
import pickle
import pandas as pd

# Load results
with open('section2_1_results.pkl', 'rb') as f:
    results = pickle.load(f)

# Access DataFrames
adam_df = results['adam']
lbfgs_df = results['lbfgs']
lm_df = results['lm']

# Access metadata from DataFrame.attrs
epochs = adam_df.attrs['epochs']
device = adam_df.attrs['device']
lbfgs_threshold_time = adam_df.attrs['lbfgs_threshold_time']

# Derive information from DataFrame
grid_sizes = adam_df['grid_size'].unique()
dataset_names = adam_df['dataset_name'].unique()
num_datasets = adam_df['dataset_idx'].nunique()
```

### Loading Checkpoints

```python
import pickle

# Load checkpoint metadata (fast)
with open('section2_1_checkpoint_metadata.pkl', 'rb') as f:
    checkpoint_meta = pickle.load(f)

# Access metadata
adam_meta = checkpoint_meta['adam']
threshold_info = adam_meta['0_5']['at_threshold']
final_info = adam_meta['0_5']['final']

print(f"Threshold epoch: {threshold_info['epoch']}")
print(f"Threshold dense MSE: {threshold_info['dense_mse']}")
print(f"Final epoch: {final_info['epoch']}")
print(f"Final dense MSE: {final_info['dense_mse']}")

# Load actual checkpoints (slower, includes model states)
with open('section2_1_checkpoints.pkl', 'rb') as f:
    checkpoints = pickle.load(f)

# Access model state
adam_checkpoint = checkpoints['adam']['0_5']['at_threshold']
model_state = adam_checkpoint['model_state']
```

### Generating Visualizations

**Section 2.1: Optimizer Comparison**
```bash
cd pykan/madoc/section2/visualization

# All plots
python plot_checkpoint_comparison.py --section section2_1 --all

# Specific plots
python plot_checkpoint_comparison.py --section section2_1 --iso-compute
python plot_checkpoint_comparison.py --section section2_1 --final
python plot_checkpoint_comparison.py --section section2_1 --time-to-threshold
```

**Section 2.2: Adaptive Density Comparison**
```bash
cd pykan/madoc/section2/visualization

# All plots
python plot_checkpoint_comparison.py --section section2_2 --all

# Specific plots
python plot_checkpoint_comparison.py --section section2_2 --iso-compute
python plot_checkpoint_comparison.py --section section2_2 --final
```

## Interpreting Results

### Iso-Compute Comparison

Shows performance at the reference approach's interpolation threshold time.

**Section 2.1 (Optimizers)**:
- Compare Adam, LBFGS, LM at `lbfgs_threshold_time`
- Lower dense MSE = better generalization
- Answers: "Which optimizer performs best given equal time?"

**Section 2.2 (Approaches)**:
- Compare adaptive_only, adaptive_regular, baseline at `baseline_threshold_time`
- Lower dense MSE = better generalization
- Answers: "Does adaptive density help within baseline's convergence time?"

### Final Comparison

Shows performance at end of training (all models fully converged).

**Section 2.1**:
- Compare final checkpoint performance across optimizers
- Answers: "Which optimizer achieves best long-run performance?"

**Section 2.2**:
- Compare final checkpoint performance across approaches
- Answers: "Does adaptive density achieve better final performance?"

### Time-to-Threshold (Section 2.1 only)

Shows how long each optimizer takes to reach its interpolation threshold.

**Interpretation**:
- Lower time = faster convergence
- Answers: "Which optimizer converges fastest?"

### Key Insights

If an approach is:
- **Better at iso-compute** → More sample-efficient, faster convergence
- **Better at final** → Better long-run optimization
- **Better at both** → Clear winner across all metrics

If an approach is:
- **Better at iso-compute, worse at final** → Fast early, but suboptimal convergence
- **Worse at iso-compute, better at final** → Slow start, but superior final solution

## Common Workflows

### 1. Run Complete Section 2.1 Pipeline

```bash
# Run experiments
cd pykan/madoc/section2
python section2_1.py --epochs 100

# Generate visualizations
cd visualization
python plot_checkpoint_comparison.py --section section2_1 --all

# Results in:
# - optimizer_iso_compute_comparison.png
# - optimizer_final_comparison.png
# - optimizer_time_to_threshold.png
```

### 2. Run Complete Section 2.2 Pipeline

```bash
# Run experiments
cd pykan/madoc/section2
python section2_2.py --epochs 100

# Generate visualizations
cd visualization
python plot_checkpoint_comparison.py --section section2_2 --all

# Results in:
# - approach_iso_compute_comparison.png
# - approach_final_comparison.png
```

### 3. Analyze Specific Dataset

```python
import pickle
import pandas as pd

# Load results
with open('section2_1_results.pkl', 'rb') as f:
    results = pickle.load(f)

# Filter for specific dataset
dataset_name = 'poisson_2d_sin'
adam_df = results['adam']
dataset_results = adam_df[adam_df['dataset_name'] == dataset_name]

# Analyze grid progression
for _, row in dataset_results.iterrows():
    grid = row['grid_size']
    dense_mse = row['dense_mse']
    test_loss = row['test_loss']
    print(f"Grid {grid}: Dense MSE = {dense_mse:.6f}, Test Loss = {test_loss:.6f}")
```

### 4. Compare Threshold vs Final Performance

```python
import pickle

# Load checkpoint metadata
with open('section2_1_checkpoint_metadata.pkl', 'rb') as f:
    checkpoint_meta = pickle.load(f)

# Compare for Adam optimizer on dataset 0, grid 10
adam_meta = checkpoint_meta['adam']['0_10']

threshold_mse = adam_meta['at_threshold']['dense_mse']
final_mse = adam_meta['final']['dense_mse']

print(f"At threshold: {threshold_mse:.6f}")
print(f"At final: {final_mse:.6f}")

if final_mse < threshold_mse:
    print("Continued training improved performance")
else:
    print(f"Overfitting detected: {((final_mse/threshold_mse - 1) * 100):.1f}% worse")
```

## Troubleshooting

### Missing Checkpoint Files

**Error**: `FileNotFoundError: section2_1_checkpoints.pkl not found`

**Solution**: Run the experiment first:
```bash
cd pykan/madoc/section2
python section2_1.py --epochs 100
```

### Missing Metadata

**Error**: `KeyError: 'lbfgs_threshold_time'`

**Solution**: The results file may be from an old version. Re-run the experiment with updated code.

### Visualization Issues

**Error**: Plots show unexpected patterns or missing data

**Solution**:
1. Check that checkpoint metadata exists
2. Verify DataFrame has expected columns
3. Ensure threshold times are reasonable (> 0)
4. Check console output for warnings during visualization

## Performance Considerations

### Checkpoint File Sizes

Typical sizes for Section 2 experiments:
- `section2_1_results.pkl`: ~500 KB (DataFrames only)
- `section2_1_checkpoints.pkl`: ~50-100 MB (includes model states)
- `section2_1_checkpoint_metadata.pkl`: ~50 KB (metadata only)

**Tip**: Load checkpoint_metadata first for fast access to metrics without loading full model states.

### Memory Usage

Loading all checkpoints can be memory-intensive. For analysis tasks, prefer loading:
1. Results DataFrames (minimal memory)
2. Checkpoint metadata (small memory footprint)
3. Specific checkpoints only when needed

Example:
```python
# Fast: Load only what you need
with open('section2_1_checkpoint_metadata.pkl', 'rb') as f:
    meta = pickle.load(f)
# Analyze metrics from metadata...

# Only load full checkpoints if you need model states
if need_model_states:
    with open('section2_1_checkpoints.pkl', 'rb') as f:
        checkpoints = pickle.load(f)
```

## Contributing

### Adding New Experiments

When adding new Section 2 experiments, follow the established patterns:

1. **Use checkpoint infrastructure** from `optimizer_tests.py`:
   ```python
   results, checkpoints, threshold_time = run_your_test(...)
   ```

2. **Train reference approach first** to establish threshold time

3. **Save with checkpoints**:
   ```python
   save_run(all_results, 'section2_X',
            checkpoints={...},
            epochs=epochs, device=str(device),
            reference_threshold_time=ref_threshold_time)
   ```

4. **Update visualization tools** if needed

5. **Document in this INDEX** under new section

### Code Style

Follow existing patterns:
- Use `detect_kan_threshold()` for threshold detection
- Store essential metadata in DataFrame.attrs
- Separate checkpoint data from checkpoint metadata
- Include dense MSE computation for fair comparisons
- Use consistent naming: `{approach}_threshold_time`

## References

### Related Documentation
- [Section 1 README](../section1/README.md) - Original implementation
- [Visualization README](../section2/visualization/README.md) - All visualization tools
- [Utils Documentation](../section2/utils/README.md) - Utility functions

### External Resources
- PyKAN Paper: [Link to paper when available]
- KAN Architecture: [Link to architecture docs]
- Optimizer Comparison Studies: [Relevant papers]

## Version History

### v1.0 (2025-10-24)
- Phase 1: Core infrastructure for Section 2.1
- Phase 2: Extension to Section 2.2 and 2.3
- Phase 3: Visualization system
- Phase 4: Documentation

### Future Versions
- Statistical significance testing
- Interactive visualizations
- Checkpoint-based ensembles
- Automated report generation

---

**Document Status**: Complete (Phases 1-4)
**Last Updated**: 2025-10-24
**Maintained By**: PyKAN Team

For questions or issues, please refer to the [Phase Implementation Summary](PHASE_IMPLEMENTATION_SUMMARY.md) or individual technical documents.
