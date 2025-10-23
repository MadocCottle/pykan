# Section 1 Table Generation Methodology

## Overview

This document explains the evaluation methodology used for generating comparison tables in Section 1 experiments. The methodology ensures scientifically rigorous, reproducible, and fair comparisons between MLP, SIREN, and KAN models.

## Key Principles

### 1. Two-Checkpoint Evaluation Strategy

All experiments use a **two-checkpoint strategy** that captures model performance at two critical stages:

1. **Iso-Compute Checkpoint** (`at_kan_threshold_time` / `at_threshold`):
   - Captured when KAN reaches its interpolation threshold
   - Represents a **fair time-matched comparison** across all models
   - Answers: "Given equal training time, which model performs best?"

2. **Final Checkpoint** (`final`):
   - Captured after completing the full training budget
   - Represents **best achievable performance** for each model
   - Answers: "Given unlimited time, which model achieves best accuracy?"

### 2. Dense MSE Evaluation

All accuracy metrics use **dense_mse** (Dense Mean Squared Error):

- **What it is**: MSE computed on 10,000 densely sampled points from the true function
- **Why not test_mse**: The `test_mse` used during training is computed on a sparse test set (typically 1,000 points) that may not represent the full function behavior
- **Why it matters**: Dense evaluation provides a more rigorous assessment of how well the model approximates the true function across the entire domain

### 3. Fair Comparisons

#### Iso-Compute Comparison (Tables 1a, 2a, 3a)
- All models evaluated at the **same wall-clock time**
- Time is set by when KAN first reaches interpolation threshold
- MLPs and SIRENs are checkpointed at this exact time
- **Purpose**: Answers "which model learns faster given equal compute resources?"

#### Final Performance Comparison (Tables 1b, 2b, 3b)
- All models evaluated after **full training budget is exhausted**
- Each model trained to convergence with its optimal hyperparameters
- **Purpose**: Answers "which model achieves best accuracy eventually?"

## Implementation Details

### Data Storage

During training ([section1/utils/model_tests.py](../utils/model_tests.py)), the pipeline saves:

1. **DataFrame** for each model type:
   - Per-epoch training history: `train_loss`, `test_loss`, `time_per_epoch`
   - Final metrics: `dense_mse`, `num_params`
   - Configuration: `depth/grid_size`, `activation`, `dataset_name`

2. **Checkpoint Metadata** (pickle file):
   ```python
   {
       'mlp': {
           dataset_idx: {
               'at_kan_threshold_time': {model, epoch, time, dense_mse, train_loss, test_loss, ...},
               'final': {model, epoch, time, dense_mse, train_loss, test_loss, ...}
           }
       },
       'siren': {...},
       'kan': {
           dataset_idx: {
               'at_threshold': {...},  # Note: different key name for KAN
               'final': {...}
           }
       },
       'kan_pruning': {...}
   }
   ```

### Table Generation

Table scripts ([tables/*.py](./)) use:

1. `load_checkpoint_metadata(section_name)` - Loads checkpoint pkl file
2. `compare_models_from_checkpoints(checkpoint_metadata, dataset_names, checkpoint_type)`:
   - `checkpoint_type='iso_compute'` → uses `at_kan_threshold_time` / `at_threshold`
   - `checkpoint_type='final'` → uses `final` checkpoints
3. All comparisons report `dense_mse` from checkpoints, **not** `test_mse` from DataFrames

### Key Functions ([tables/utils.py](./utils.py))

- `load_checkpoint_metadata()`: Loads two-checkpoint data
- `compare_models_from_checkpoints()`: Creates fair comparison tables
- `get_dataset_names()`: Returns dataset lists for each section
- `format_scientific()`: Formats numbers in scientific notation

## Table Organization

### Primary Comparison Tables

| Table | Section | Comparison Type | Purpose |
|-------|---------|----------------|---------|
| 1a | Function Approximation (1.1) | Iso-compute | Time-matched comparison |
| 1b | Function Approximation (1.1) | Final | Best achievable accuracy |
| 2a | 1D PDEs (1.2) | Iso-compute | Time-matched comparison |
| 2b | 1D PDEs (1.2) | Final | Best achievable accuracy |
| 3a | 2D PDEs (1.3) | Iso-compute | Time-matched comparison |
| 3b | 2D PDEs (1.3) | Final | Best achievable accuracy |

### Secondary Analysis Tables

| Table | Purpose |
|-------|---------|
| 4 | Parameter efficiency (params vs accuracy trade-offs) |
| 5 | Training efficiency (convergence speed, time metrics) |
| 6 | KAN grid size ablation study |
| 7 | MLP/SIREN depth ablation study |

## Why This Methodology?

### Problem with Previous Approach

The original table scripts had several issues:

1. **Wrong Metric**: Used `test_mse` (sparse test set) instead of `dense_mse` (10k samples)
   - **Impact**: Inaccurate performance assessment
   - **Fix**: Now uses `dense_mse` from checkpoints

2. **Unfair Comparison**: Compared "best overall" configurations regardless of training time
   - **Impact**: KAN might be compared at 1000 epochs vs MLP at 100 epochs
   - **Fix**: Iso-compute comparison at matched timestamps

3. **Missing Context**: Only showed final results, not learning dynamics
   - **Impact**: Couldn't tell if model won due to faster convergence or better asymptotic performance
   - **Fix**: Separate iso-compute and final tables

4. **Data Source**: Pulled from DataFrame rows, not checkpoints
   - **Impact**: Had to recompute or approximate metrics
   - **Fix**: Uses pre-computed checkpoint metadata with exact dense_mse values

### Benefits of New Approach

1. **Scientific Rigor**:
   - Dense evaluation on 10,000 samples
   - Fair time-matched comparisons
   - Reproducible methodology

2. **Clear Narrative**:
   - Iso-compute: Shows learning efficiency
   - Final: Shows best achievable accuracy
   - Improvement: Shows how much additional training helps

3. **Thesis-Ready**:
   - Publication-quality comparisons
   - Well-documented methodology
   - Defensible experimental design

4. **Reproducibility**:
   - Checkpoint metadata preserved
   - Exact evaluation protocol documented
   - All metrics derived from same source

## Example: Interpreting Results

### Scenario: KAN vs MLP Comparison

**Iso-Compute Table (1a):**
```
Dataset    | MLP Dense MSE | KAN Dense MSE
sin_freq1  | 1.23e-03     | 8.45e-04
```
**Interpretation**: At same training time, KAN achieves 1.46x lower error than MLP.

**Final Table (1b):**
```
Dataset    | MLP Dense MSE | KAN Dense MSE
sin_freq1  | 5.67e-04     | 2.34e-04
```
**Interpretation**: With unlimited training, KAN achieves 2.42x lower error than MLP.

**Conclusion**:
- KAN is both **faster** (better at iso-compute) and **more accurate** (better at final)
- Additional training helps both models, but KAN benefits more

## Validation

### Checklist for Table Correctness

- [ ] Loads checkpoint metadata (not DataFrame)
- [ ] Uses `dense_mse` (not `test_mse`)
- [ ] Iso-compute tables use matching timestamps
- [ ] Final tables use `final` checkpoints
- [ ] All datasets have checkpoint data
- [ ] Parameter counts match checkpoints
- [ ] Architecture strings extracted from checkpoints

### Common Pitfalls to Avoid

1. ❌ Using `df['test_mse'].min()` → ✅ Use `checkpoint['dense_mse']`
2. ❌ Comparing across arbitrary epochs → ✅ Use iso-compute checkpoints
3. ❌ Mixing DataFrame and checkpoint data → ✅ Use checkpoints exclusively for tables
4. ❌ Reporting "best across all time" → ✅ Separate iso-compute vs final

## Running the Tables

### Generate All Tables

```bash
cd pykan/madoc/section1/tables
python table1_function_approximation.py
python table2_pde_1d_comparison.py
python table3_pde_2d_comparison.py
```

### Prerequisites

1. Run training scripts to generate checkpoint data:
   ```bash
   cd pykan/madoc/section1
   python section1_1.py --epochs 100
   python section1_2.py --epochs 100
   python section1_3.py --epochs 100
   ```

2. Verify checkpoint metadata files exist:
   ```bash
   ls results/sec1_results/*_checkpoint_metadata.pkl
   ```

### Output Files

Each table script generates:
- `tableXa_*_iso_compute.tex` - LaTeX table for iso-compute comparison
- `tableXa_*_iso_compute.csv` - CSV for iso-compute comparison
- `tableXb_*_final.tex` - LaTeX table for final performance
- `tableXb_*_final.csv` - CSV for final performance
- `tableX_summary_statistics.csv` - Summary statistics
- `tableX_improvement_analysis.csv` - Improvement ratios

## References

### Code Files
- Training pipeline: [section1/utils/model_tests.py](../utils/model_tests.py)
- I/O utilities: [section1/utils/io.py](../utils/io.py)
- Evaluation metrics: [section1/utils/metrics.py](../utils/metrics.py)
- Table utilities: [section1/tables/utils.py](./utils.py)

### Related Documentation
- Main README: [section1/README.md](../README.md)
- Table README: [section1/tables/README.md](./README.md)
- KAN Paper Comparison: [section1/tables/KAN_PAPER_COMPARISON.md](./KAN_PAPER_COMPARISON.md)

## Contact

For questions about this methodology:
1. Check that you've read this entire document
2. Verify checkpoint metadata files exist
3. Review the code in [tables/utils.py](./utils.py)
4. Check training output for checkpoint confirmation messages

---

**Last Updated**: 2025-01-24
**Version**: 2.0 (Checkpoint-based methodology)
