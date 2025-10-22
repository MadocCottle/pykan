# Old IO System - Archived

**Date Archived:** 2025-10-22

## Why Archived

The old IO system consisted of:
- `save.py` (originally `section1/utils/io.py`) - 97 lines
- `load.py` (originally `holding/analysis/data_io.py`) - 514 lines
- **Total: 611 lines**

### Problems with Old System

1. **Over-engineered loading**: 514 lines of path resolution, auto-discovery, and helper functions
2. **Complex configuration**: SECTION_CONFIG dictionary with metadata that duplicated training script info
3. **Inconsistent with codebase style**: section1_X.py scripts are ~60 lines, but IO was 10x longer
4. **Hard to maintain**: Many abstraction layers made simple tasks complex

### What Was Saved

- `save.py`: Original saving logic with NaN cleaning, JSON/pickle dual save, metadata extraction
- `load.py`: Original loading with auto-discovery, SECTION_CONFIG, path inference, etc.

## New System

Replaced with:
- `section1/utils/io.py` - ~30 lines (concise save function)
- `holding/analysis/io.py` - ~50 lines (simple load function)
- **Total: ~80 lines (87% reduction)**

### Benefits

1. **Matches codebase style**: Concise like section1_1.py
2. **Simple to understand**: Direct glob/pickle operations
3. **Analysis-driven**: Load function returns exactly what analysis needs
4. **Easy to maintain**: No complex abstractions

## Migration Notes

Old `.pkl` files remain readable by new system. The data structure is unchanged:
```python
results[model_type][dataset_idx][config][subconfig] = {'train': [...], 'test': [...], ...}
```

Only the save/load interface simplified.
