# Section 1 IO System Migration Guide

**Date:** 2025-10-22

## Summary

The Section 1 data saving and loading system has been redesigned to be more concise and aligned with the codebase style (like `section1_1.py`).

### Key Changes

- **Old system**: 611 lines (97 save + 514 load)
- **New system**: 60 lines save + 110 lines load = **170 lines total**
- **Reduction**: 72% fewer lines of code

## Old System (Archived)

Located in: [section1/archive/old_io_system/](section1/archive/old_io_system/)

### Problems
1. Over-engineered with complex path resolution and auto-discovery
2. SECTION_CONFIG dictionary duplicated training script information
3. Many abstraction layers made simple operations complex
4. Inconsistent with concise section1_X.py style

### Files Archived
- `save.py` - Original `section1/utils/io.py` (97 lines)
- `load.py` - Original `holding/analysis/data_io.py` (514 lines)
- `ARCHIVED.md` - Documentation of why it was archived

## New System

### Saving Data (Training Scripts)

**Location:** [section1/utils/io.py](section1/utils/io.py)

**New API:**
```python
from utils import save_run

save_run(all_results, 'section1_1',
         models={'kan': kan_models, 'kan_pruned': kan_pruned_models},
         epochs=epochs, device=str(device), grids=grids.tolist(),
         depths=depths, activations=activations, frequencies=freq)
```

**Files Saved:**
- `sec{N}_results/section1_{N}_{timestamp}.pkl` - Full results
- `sec{N}_results/section1_{N}_{timestamp}.json` - JSON + metadata
- `sec{N}_results/section1_{N}_{timestamp}_kan_{idx}` - KAN model checkpoints
- `sec{N}_results/section1_{N}_{timestamp}_pruned_{idx}` - Pruned KAN checkpoints

### Loading Data (Analysis Scripts)

**Location:** [holding/analysis/io.py](holding/analysis/io.py)

**New API:**
```python
from analysis import io

# Load latest run
results, metadata, models_dir = io.load_run('section1_1')

# Load specific timestamp
results, metadata, models_dir = io.load_run('section1_1', timestamp='20251022_143000')

# Check if section has 2D data
is_2d = io.is_2d('section1_3')  # True

# Get model path
model_path = io.get_model_path(models_dir, 'section1_1', timestamp, 'kan', dataset_idx=0)
```

**Available sections:**
```python
io.SECTIONS = ['section1_1', 'section1_2', 'section1_3']
io.SECTION_DIRS = {
    'section1_1': 'sec1_results',
    'section1_2': 'sec2_results',
    'section1_3': 'sec3_results'
}
io.SECTION_IS_2D = {
    'section1_1': False,
    'section1_2': False,
    'section1_3': True
}
```

## Migration Steps (Already Completed)

### 1. Training Scripts ✓
- [x] `section1/section1_1.py` - Updated to use `save_run()`
- [x] `section1/section1_2.py` - Updated to use `save_run()`
- [x] `section1/section1_3.py` - Updated to use `save_run()`
- [x] `section1/utils/__init__.py` - Export `save_run` instead of `save_results`

### 2. Analysis Scripts ✓
- [x] `holding/analysis/comparative_metrics.py` - Use new `io` module
- [x] `holding/analysis/function_fitting.py` - Use new `io` module
- [x] `holding/analysis/heatmap_2d_fits.py` - Use new `io` module
- [x] `holding/analysis/run_analysis.py` - Use new `io` module
- [x] `holding/analysis/analyze_all_section1.py` - Use new `io` module

### 3. Archive ✓
- [x] Created `section1/archive/old_io_system/`
- [x] Moved old files to archive with documentation

## Data Format (Unchanged)

The underlying data structure remains identical:

```python
results = {
    'mlp': {
        dataset_idx: {
            depth: {
                activation: {
                    'train': [...],
                    'test': [...],
                    'dense_mse': [...],
                    'total_time': float,
                    'time_per_epoch': float
                }
            }
        }
    },
    'siren': {
        dataset_idx: {
            depth: {
                'train': [...],
                'test': [...],
                ...
            }
        }
    },
    'kan': {
        dataset_idx: {
            grid_size: {
                'train': [...],
                'test': [...],
                ...
            }
        }
    },
    'kan_pruning': { ... }
}
```

## Backward Compatibility

✅ **Old `.pkl` files are still readable** by the new system
- File format unchanged (still uses pickle)
- Data structure unchanged
- Only the save/load interface simplified

## Benefits

1. **Concise**: Matches section1_X.py style (~60 lines each)
2. **Simple**: Direct glob/pickle operations, no complex abstractions
3. **Clear**: Obvious what each function does
4. **Maintainable**: Easy to modify and extend
5. **Analysis-driven**: `load_run()` returns exactly what analysis needs

## Example: Full Workflow

### Training
```bash
cd section1
python section1_1.py --epochs 100
# Saves to: sec1_results/section1_1_20251022_143000.*
```

### Analysis
```bash
cd holding/analysis
python run_analysis.py ../../section1/sec1_results/section1_1_20251022_143000.pkl
# Or simply:
# python analyze_all_section1.py  # Analyzes latest from all sections
```

## Questions?

See the source code:
- Saving: [section1/utils/io.py](section1/utils/io.py)
- Loading: [holding/analysis/io.py](holding/analysis/io.py)
- Old system: [section1/archive/old_io_system/](section1/archive/old_io_system/)
