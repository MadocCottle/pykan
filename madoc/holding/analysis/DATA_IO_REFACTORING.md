# Analysis Module Refactoring Summary

## Overview

The analysis module has been refactored to use a centralized IO system that makes loading experimental results much easier and more robust.

## Changes Made

### 1. New `data_io.py` Module

Created a centralized I/O module (`data_io.py`) that handles:
- **Auto-discovery** of latest results for each section
- **Auto-discovery** of metadata files
- **Auto-discovery** of KAN model directories
- **Smart path resolution** - accepts both section IDs and explicit paths
- **Format-agnostic loading** - handles both .pkl and .json seamlessly
- **Comprehensive error handling** with helpful error messages

### 2. Refactored Analysis Classes

Updated all analyzer classes to use the new IO module:
- `MetricsAnalyzer` ([comparative_metrics.py](comparative_metrics.py))
- `FunctionFittingVisualizer` ([function_fitting.py](function_fitting.py))
- `Heatmap2DAnalyzer` ([heatmap_2d_fits.py](heatmap_2d_fits.py))

**Key improvements:**
- Removed duplicate `_load_results()` and `_load_metadata()` methods
- Added support for section ID shortcuts
- Auto-discovery of model directories when not explicitly provided
- Maintained backward compatibility with explicit paths

### 3. Updated Package Exports

Updated `__init__.py` to export the `data_io` module for easy access.

## Usage Examples

### Before Refactoring

```python
# Had to specify full paths
analyzer = MetricsAnalyzer(
    '/path/to/section1/sec1_results/section1_1_results_20251021_215324.pkl'
)

visualizer = FunctionFittingVisualizer(
    '/path/to/section1/sec1_results/section1_1_results_20251021_215324.pkl',
    models_dir='/path/to/section1/sec1_results/kan_models_20251021_215324'
)
```

### After Refactoring

```python
# Simple section ID - auto-discovers everything!
analyzer = MetricsAnalyzer('section1_1')

visualizer = FunctionFittingVisualizer('section1_1')
# Models directory is auto-discovered!
```

### Using data_io Directly

```python
from analysis import data_io

# Check what's available
data_io.print_available_results()

# Load latest results for a section
results, metadata = data_io.load_results('section1_1')

# Get complete section info
section_data = data_io.load_section_results('section1_1')
print(section_data['timestamp'])
print(section_data['models_dir'])

# Find models directory for any results file
models_dir = data_io.find_models_dir('/path/to/results.pkl')
```

## Benefits

1. **Less Code** - No need to manually construct paths or find timestamps
2. **Auto-Discovery** - Models, metadata, and latest results found automatically
3. **Easier to Use** - Just pass section ID like 'section1_1'
4. **Centralized Logic** - All IO code in one tested module
5. **Better Errors** - Helpful error messages when files not found
6. **Backward Compatible** - Explicit paths still work
7. **Type Safe** - Full type hints throughout

## File Structure

```
analysis/
├── data_io.py                 # NEW: Centralized I/O module
├── test_io.py                 # NEW: Comprehensive test suite for data_io
├── USAGE_EXAMPLE.py           # NEW: Usage examples
├── DATA_IO_REFACTORING.md     # This file
├── comparative_metrics.py     # UPDATED: Uses data_io
├── function_fitting.py        # UPDATED: Uses data_io
├── heatmap_2d_fits.py         # UPDATED: Uses data_io
├── run_analysis.py            # Works with updated analyzers
├── analyze_all_section1.py    # Works with updated analyzers
└── __init__.py                # UPDATED: Exports data_io
```

## Section ID Mapping

The following section IDs are recognized:

| Section ID | Results Directory | Description |
|-----------|------------------|-------------|
| `section1_1` | `sec1_results/` | Function approximation experiments |
| `section1_2` | `sec2_results/` | 1D Poisson PDE experiments |
| `section1_3` | `sec3_results/` | 2D Poisson PDE experiments |

## Testing

All functionality has been tested:

```bash
# Test the IO module
cd /path/to/section1/analysis
python test_io.py

# Test refactored analyzers
cd /path/to/section1
python -c "from analysis import MetricsAnalyzer; MetricsAnalyzer('section1_1')"

# Run usage examples
python analysis/USAGE_EXAMPLE.py
```

## API Reference

### Main Functions

#### `load_results(path_or_section, timestamp=None)`
Load results and metadata from path or section ID.

**Args:**
- `path_or_section`: Either a file path or section ID ('section1_1', etc.)
- `timestamp`: Optional specific timestamp (otherwise loads latest)

**Returns:** `(results_dict, metadata_dict)`

**Example:**
```python
results, metadata = data_io.load_results('section1_1')
# or
results, metadata = data_io.load_results('/path/to/results.pkl')
```

#### `load_section_results(section_id, timestamp=None, load_models_info=True)`
Load complete section data including model paths.

**Returns:** Dictionary with:
- `results`: Results data
- `metadata`: Metadata (or None)
- `results_file`: Path to results file
- `metadata_file`: Path to metadata file (or None)
- `models_dir`: Path to models directory (or None)
- `pruned_models_dir`: Path to pruned models (or None)
- `timestamp`: Timestamp string

#### `find_latest_results(section_id, base_dir=None, timestamp=None)`
Find latest (or specific) results for a section.

#### `find_models_dir(results_path, pruned=False)`
Find KAN models directory corresponding to results file.

#### `print_available_results(base_dir=None)`
Print summary of available results across all sections.

## Migration Guide

### For Existing Scripts

If you have existing scripts using the old API, they will continue to work! The refactoring maintains backward compatibility:

```python
# This still works:
analyzer = MetricsAnalyzer('/full/path/to/results.pkl')

# But you can now simplify to:
analyzer = MetricsAnalyzer('section1_1')
```

### For New Scripts

Use the simpler section ID syntax:

```python
from analysis import MetricsAnalyzer, FunctionFittingVisualizer

# Load and analyze section1_1
analyzer = MetricsAnalyzer('section1_1')
analyzer.generate_all_visualizations('output_dir/')

# Function fitting with auto-discovered models
viz = FunctionFittingVisualizer('section1_1')  # models_dir found automatically
viz.generate_all_function_fits('fitting_output/')
```

## Troubleshooting

### "ResultsNotFoundError"
Make sure you've run the experiments for that section. Check what's available:
```python
from analysis import data_io
data_io.print_available_results()
```

### "InvalidSectionError"
Valid section IDs are: `section1_1`, `section1_2`, `section1_3`. Check spelling!

### Models directory not found
Some sections may not have saved KAN models. Check with:
```python
info = data_io.find_latest_results('section1_1')
print(info['models_dir'])  # Will be None if not found
```

## Future Enhancements

Possible future improvements:
- Support for loading specific experiment runs by name/tag
- Caching of loaded results to avoid re-reading files
- Export functions to save results in standardized format
- Integration with experiment tracking systems

---

**Author:** PyKAN Analysis Tools Team
**Version:** 1.1.0
**Date:** 2025-10-21
