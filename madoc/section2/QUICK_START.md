# Section 2: Quick Start Guide

## What is Section 2?

Section 2 implements advanced KAN techniques (ensembles, adaptive grids, pruning) in the concise style of section1, with full I/O integration for analysis.

## Run Your First Experiment

```bash
cd /Users/main/Desktop/my_pykan/pykan/madoc

# Run ensemble experiment (3 experts, 20 epochs - fast test)
python section2/section2_1.py --epochs 20 --n_experts 3

# Expected output:
# ============================================================
# Section 2.1: KAN Expert Ensemble
# ============================================================
# Expert 1/3 (seed=0)
#   Test MSE: 0.069670
# Expert 2/3 (seed=1)
#   Test MSE: 0.069670
# Expert 3/3 (seed=2)
#   Test MSE: 0.069670
# Ensemble (averaging) MSE: 0.069670
# Saved to section2/results/sec1_results/section2_1_TIMESTAMP.*
```

## Load and Analyze Results

```python
from section2.analysis import load_run

# Load latest run
results, meta, models_dir = load_run('section2_1')

# Explore results
print(f"Ensemble MSE: {results['ensemble_mse']}")
print(f"Expert losses: {results['expert_losses']}")
print(f"Mean uncertainty: {results['mean_uncertainty']}")

# Check metadata
print(f"Epochs: {meta['epochs']}")
print(f"N experts: {meta['n_experts']}")
print(f"Device: {meta['device']}")
```

## All Experiments

### section2_1: Ensemble Training
Train multiple KAN experts, quantify uncertainty

```bash
python section2/section2_1.py --epochs 100 --n_experts 10
```

**Results**: `expert_losses`, `ensemble_mse`, `mean_uncertainty`

### section2_2: Adaptive Grid Refinement
Progressive grid densification (3 → 5 → 10)

```bash
python section2/section2_2.py --epochs 100 --initial_grid 3 --max_grid 10
```

**Results**: `grid_3`, `grid_5`, `grid_10` (each with MSE)

### section2_3: Grid Comparison
Compare different grid sizes [3, 5, 10, 20]

```bash
python section2/section2_3.py --epochs 100
```

**Results**: `grid_3`, `grid_5`, `grid_10`, `grid_20` (MSE + n_params)

### section2_4: Uncertainty Analysis
Ensemble with noisy data, calibration metrics

```bash
python section2/section2_4.py --epochs 100 --n_experts 5
```

**Results**: `ensemble_mse`, `uncertainty_error_correlation`

### section2_5: Pruning
Train, prune, compare parameter reduction

```bash
python section2/section2_5.py --epochs 100
```

**Results**: `base` (MSE, n_params), `pruned` (MSE, n_params)

## File Organization

**After running section2_1.py**, you'll have:

```
section2/results/sec1_results/
├── section2_1_TIMESTAMP.json      # Metadata + results (human-readable)
├── section2_1_TIMESTAMP.pkl       # Results (for Python)
└── section2_1_TIMESTAMP_models/   # Model checkpoints
    ├── expert_0_cache_data
    ├── expert_0_config.yml
    ├── expert_0_state
    ├── expert_1_cache_data
    └── ...
```

## Common Patterns

### Save Results (in experiment scripts)
```python
from section2.utils import save_run

save_run(
    results={'mse': 0.05, 'loss_history': [...]},
    section='section2_1',
    models=trained_models,  # List or dict of KAN models
    epochs=100,
    n_experts=10
)
```

### Load Results (for analysis)
```python
from section2.analysis import load_run, list_runs

# Load latest
results, meta, models_dir = load_run('section2_1')

# Load specific timestamp
results, meta, models_dir = load_run('section2_1', timestamp='20251022_151408')

# List all runs
timestamps = list_runs('section2_1')
for ts in timestamps:
    print(f"Available run: {ts}")
```

## Comparison with section1

Same I/O patterns:

| Section 1 | Section 2 |
|-----------|-----------|
| `section1/utils/io.py` | `section2/utils/io.py` |
| `section1/analysis/io.py` | `section2/analysis/io.py` |
| `section1/results/sec{N}_results/` | `section2/results/sec{N}_results/` |
| `section1_1.py`, `section1_2.py` | `section2_1.py`, `section2_2.py` |

Both use `save_run()` and `load_run()` with identical signatures.

## Tips

1. **Quick testing**: Use `--epochs 20 --n_experts 3` for fast validation
2. **Full experiments**: Use `--epochs 100 --n_experts 10` for research
3. **Check results**: `ls section2/results/sec1_results/` to see all runs
4. **Load in analysis**: All results are pickle + JSON for easy loading

## Next Steps

After running experiments, create analysis scripts:

```python
# section2/analysis/compare_all.py (future work)
from section2.analysis import load_run

sections = ['section2_1', 'section2_2', 'section2_3', 'section2_4', 'section2_5']
for sec in sections:
    try:
        results, meta, _ = load_run(sec)
        print(f"{sec}: MSE = {results.get('ensemble_mse', 'N/A')}")
    except FileNotFoundError:
        print(f"{sec}: No runs found")
```

## Troubleshooting

**Import errors**: Make sure you're in `/madoc` directory
```bash
cd /Users/main/Desktop/my_pykan/pykan/madoc
python section2/section2_1.py
```

**FileNotFoundError**: No runs saved yet, run an experiment first
```bash
python section2/section2_1.py --epochs 20 --n_experts 3
```

**Device errors**: Scripts auto-detect CPU/CUDA, use `--device cpu` if needed (currently automatic)

## Summary

Section 2 provides:
- ✓ 5 experiments (ensemble, grids, pruning)
- ✓ Section1-style I/O (save_run, load_run)
- ✓ Automatic timestamping
- ✓ Model checkpoint saving
- ✓ Analysis-ready results

**Total code**: ~620 lines (vs 3000+ in section2_new)
**Style**: Concise, functional, section1-compatible
