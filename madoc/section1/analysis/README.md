# Section 1 Analysis Tools

This directory contains comprehensive analysis and visualization tools for Section 1 experiment results.

## Overview

The analysis suite provides three main types of visualizations:

1. **Comparative Metrics** - Compare models across epochs, training times, and performance metrics
2. **Function Fitting** - Visualize how well NNs approximate the underlying functions
3. **2D Heatmaps** - Detailed spatial analysis for 2D equations (Section 1.3)

## Quick Start

### Run Complete Analysis

The easiest way to analyze your results is using the main runner script:

```bash
cd /Users/main/Desktop/my_pykan/pykan/madoc/section1/analysis

# With saved KAN models (recommended for best visualizations)
python run_analysis.py ../sec1_results/section1_1_results_YYYYMMDD_HHMMSS.pkl \
       --models-dir ../sec1_results/kan_models_YYYYMMDD_HHMMSS

# Without models (limited function fitting visualizations)
python run_analysis.py ../sec1_results/section1_1_results_YYYYMMDD_HHMMSS.pkl
```

This will:
- Auto-detect whether results are 1D or 2D
- Generate all appropriate visualizations
- Create a comprehensive summary report
- Organize outputs into subdirectories

### Output Structure

```
analysis_output_<section>_<timestamp>/
├── 01_comparative_metrics/
│   ├── dataset_0_comparison_table.csv
│   ├── dataset_0_learning_curves_test.png
│   ├── dataset_0_training_times.png
│   └── all_datasets_heatmap_test.png
├── 02_function_fitting/
│   └── function_fit_dataset_<N>_<name>.png
├── 03_heatmap_analysis/          # Only for 2D (Section 1.3)
│   ├── heatmap_<N>_<function>_<model>.png
│   ├── cross_section_<N>_<function>.png
│   └── error_quantile_<N>_<function>_<model>.png
└── ANALYSIS_SUMMARY.md
```

## Individual Analysis Scripts

Each script can also be run independently for specific analyses.

### 1. Comparative Metrics Analysis

**Script:** `comparative_metrics.py`

Generates tables and graphs comparing all models across metrics.

```bash
python comparative_metrics.py ../sec1_results/section1_1_results_YYYYMMDD_HHMMSS.pkl \
       --output-dir metrics_output
```

**Outputs:**
- Comparison tables (CSV) with final MSE, training times, etc.
- Learning curves for train/test/dense MSE
- Training time bar charts
- Performance heatmaps across all datasets

**Use cases:**
- Determine which model type performs best
- Analyze training dynamics and convergence
- Compare computational efficiency
- Identify overfitting

### 2. Function Fitting Visualization

**Script:** `function_fitting.py`

Creates visualizations comparing NN predictions with true functions.

```bash
# For 1D functions (Section 1.1, 1.2)
python function_fitting.py ../sec1_results/section1_1_results_YYYYMMDD_HHMMSS.pkl \
       --models-dir ../sec1_results/kan_models_YYYYMMDD_HHMMSS \
       --output-dir function_fits

# For 2D functions (Section 1.3)
python function_fitting.py ../sec1_results/section1_3_results_YYYYMMDD_HHMMSS.pkl \
       --models-dir ../sec1_results/kan_models_YYYYMMDD_HHMMSS \
       --output-dir function_fits_2d \
       --2d
```

**Outputs:**

For 1D:
- Line plots showing true function vs NN predictions
- All model types compared side-by-side
- MSE displayed for each model

For 2D:
- Surface plots for true function and predictions
- Contour plots for easier comparison
- All model types in one figure

**Use cases:**
- Visually assess approximation quality
- Identify regions where models struggle
- Compare model behaviors on different function types

### 3. 2D Heatmap Analysis

**Script:** `heatmap_2d_fits.py`

Detailed spatial analysis for 2D functions (Section 1.3 only).

```bash
python heatmap_2d_fits.py ../sec1_results/section1_3_results_YYYYMMDD_HHMMSS.pkl \
       --models-dir ../sec1_results/kan_models_YYYYMMDD_HHMMSS \
       --output-dir heatmap_output
```

**Outputs:**
- **Comparison heatmaps**: True function, prediction, absolute/signed/relative error
- **Cross-sections**: 1D slices at x₁=0.25, 0.5, 0.75 and x₂=0.25, 0.5, 0.75
- **Error quantile maps**: Categorize errors into quantiles to identify problem regions
- **Error statistics**: Quantitative breakdown by spatial region

**Use cases:**
- Identify problematic regions in 2D domain
- Understand spatial error distribution
- Compare model behaviors across dimensions
- Find patterns in approximation errors

## Understanding the Visualizations

### Learning Curves

![Learning curve example](https://via.placeholder.com/600x400?text=Learning+Curves)

- **Y-axis**: MSE (log scale)
- **X-axis**: Training epoch
- Different lines for each model configuration
- Look for: convergence, overfitting (train/test gap), stability

**Interpretation:**
- Decreasing curves = model is learning
- Plateau = convergence (or stuck in local minimum)
- Train MSE << Test MSE = overfitting
- Noisy curves = unstable training

### Training Time Comparisons

- **Total Time**: Full training duration
- **Time per Epoch**: Average epoch time

**Key insights:**
- KAN models typically slower than MLPs
- Deeper networks take longer
- Helps balance accuracy vs speed

### Performance Heatmaps

Color-coded grid showing final MSE for each model on each dataset.

- **Rows**: Model types (MLP, SIREN, KAN, KAN Pruning)
- **Columns**: Datasets
- **Color**: Lower (green) = better, Higher (red) = worse

**Use to:**
- Quickly identify best model for each dataset
- Find patterns (e.g., which models excel at periodic functions)
- Spot datasets where all models struggle

### Function Fitting Plots (1D)

- **Black dashed line**: True function
- **Colored solid lines**: NN predictions
- **MSE value**: Quantitative error measure

**Good fit indicators:**
- Prediction line overlaps true function
- Low MSE value
- Smooth predictions (not jittery)

### Function Fitting Plots (2D)

- **Top row**: True function (surface + contour)
- **Other rows**: Model predictions
- **Color differences**: Indicate approximation errors

**Good fit indicators:**
- Similar color patterns to true function
- Smooth surfaces (no artifacts)
- Low MSE values

### Heatmap Error Analysis

- **Absolute error**: |true - predicted|
- **Signed error**: true - predicted (shows over/under-estimation)
- **Relative error**: % error (normalized by true value)
- **Error quantiles**: Spatial regions categorized by error magnitude

**Use to:**
- Find where models make largest errors
- Identify systematic biases (signed error)
- Understand relative vs absolute performance

## Requirements

The analysis scripts require the following Python packages:

```bash
pip install numpy matplotlib seaborn pandas torch
```

Plus the PyKAN library (already in your environment).

## Tips and Best Practices

### 1. Always Save Models

When running experiments, make sure to save KAN models:

```python
save_results(all_results, 'section1_1', output_dir='sec1_results',
             kan_models=kan_models, pruned_models=kan_pruned_models)
```

This enables the function fitting visualizations.

### 2. Interpret Metrics Together

Don't rely on a single metric:
- **Train MSE**: Can be misleading (overfitting)
- **Test MSE**: Better but still sampled
- **Dense MSE**: Most reliable for true approximation quality
- **Visual inspection**: Catches issues metrics might miss

### 3. Check Learning Curves First

Before diving into detailed analysis:
1. Check if models converged (learning curves plateau)
2. If not converged, consider re-running with more epochs
3. Look for overfitting (train/test gap)

### 4. Use Comparison Tables for Exact Values

Visual plots are great for patterns, but use CSV tables for:
- Exact numerical comparisons
- Statistical analysis
- Generating your own custom visualizations

### 5. 2D Analysis is Computationally Intensive

Heatmap generation for 2D functions can be slow:
- Uses high resolution grids (100x100 or 50x50 points)
- Evaluates multiple models
- Consider reducing resolution for quick checks

## Customization

### Modify Grid Resolution

In `heatmap_2d_fits.py`, adjust the resolution parameter:

```python
# Lower resolution = faster but less detailed
analyzer.create_comparison_heatmaps(dataset_idx, func, name, resolution=50)

# Higher resolution = slower but more detailed
analyzer.create_comparison_heatmaps(dataset_idx, func, name, resolution=200)
```

### Change Plot Styles

Modify matplotlib/seaborn settings at the top of scripts:

```python
import seaborn as sns
sns.set_style("darkgrid")  # or "whitegrid", "dark", "white", "ticks"
plt.rcParams['figure.figsize'] = (16, 10)  # Larger figures
plt.rcParams['font.size'] = 12  # Bigger fonts
```

### Add New Functions

To analyze additional functions, update the function maps in `function_fitting.py` and `heatmap_2d_fits.py`:

```python
self.function_map_1d = {
    # ... existing functions ...
    9: ('My Custom Func', my_custom_function),
}
```

### Filter Models

To analyze only specific models, modify the model type lists:

```python
# In any script, change:
model_types = ['mlp', 'siren', 'kan', 'kan_pruning']

# To (for example, only KAN models):
model_types = ['kan', 'kan_pruning']
```

## Troubleshooting

### "No predictions available"

**Cause**: Models directory not provided or models not found

**Solution**:
- Ensure you pass `--models-dir` argument
- Verify the path points to the correct directory
- Check that model files exist (e.g., `kan_dataset_0_state`)

### "Cannot detect section type"

**Cause**: Results filename doesn't match expected pattern

**Solution**:
- Ensure filename contains `section1_1`, `section1_2`, or `section1_3`
- Or manually specify section type in code

### Empty or strange plots

**Cause**: Data format issues or missing metrics

**Solution**:
- Verify results file is not corrupted
- Check that experiments ran to completion
- Ensure all required metrics were saved (train, test, dense_mse, timing)

### Import errors

**Cause**: Missing dependencies or path issues

**Solution**:
```bash
# Ensure pykan path is correct
export PYTHONPATH="/Users/main/Desktop/my_pykan:$PYTHONPATH"

# Install missing packages
pip install numpy matplotlib seaborn pandas torch
```

### Memory issues with 2D heatmaps

**Cause**: High resolution grids + multiple models

**Solution**:
- Reduce resolution: `resolution=50` instead of `100`
- Process one function at a time
- Close figures after saving: `plt.close()`

## Advanced Usage

### Batch Analysis

Analyze multiple result files:

```bash
#!/bin/bash
for results in sec1_results/*.pkl; do
    timestamp=$(basename "$results" | grep -o '[0-9]\{8\}_[0-9]\{6\}')
    models_dir="sec1_results/kan_models_${timestamp}"
    python run_analysis.py "$results" --models-dir "$models_dir"
done
```

### Custom Analysis

You can import the analyzer classes in your own scripts:

```python
from comparative_metrics import MetricsAnalyzer

# Load results
analyzer = MetricsAnalyzer('path/to/results.pkl')

# Get comparison table for dataset 0
df = analyzer.create_comparison_table(dataset_idx=0)

# Custom processing
best_model = df.loc[df['Final Test MSE'].idxmin()]
print(f"Best model: {best_model['Model']} {best_model['Configuration']}")

# Generate specific plot
fig = analyzer.plot_learning_curves(dataset_idx=0, metric='test')
# ... customize figure ...
plt.savefig('custom_plot.png')
```

### Integration with Experiments

Add analysis to your experiment pipeline:

```python
# At the end of section1_1.py
from analysis.run_analysis import run_full_analysis

# After save_results()
results_file = f'sec1_results/section1_1_results_{timestamp}.pkl'
models_dir = f'sec1_results/kan_models_{timestamp}'

print("\nRunning analysis...")
run_full_analysis(results_file, models_dir, f'analysis_{timestamp}')
```

## File Descriptions

| File | Purpose | Input | Output |
|------|---------|-------|--------|
| `run_analysis.py` | Main runner - orchestrates all analyses | Results file | Complete analysis suite |
| `comparative_metrics.py` | Model comparison metrics and tables | Results file | Tables, charts, heatmaps |
| `function_fitting.py` | Function approximation quality | Results + models | 1D/2D fit visualizations |
| `heatmap_2d_fits.py` | Detailed 2D spatial analysis | Results + models | Error maps, cross-sections |
| `README.md` | This documentation | - | - |

## Contributing

To add new visualization types:

1. Create a new script following the existing patterns
2. Add a class with appropriate methods
3. Include a `main()` function for standalone usage
4. Import and integrate into `run_analysis.py`
5. Update this README

## Support

For issues or questions:
1. Check the Troubleshooting section above
2. Review experiment logs for errors during result generation
3. Ensure all dependencies are installed
4. Check that results and model files are not corrupted

## Citation

If you use these analysis tools in your research, please cite:

```bibtex
@misc{pykan_section1_analysis,
  title={Section 1 Analysis Tools for PyKAN Experiments},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/pykan}
}
```

---

**Last Updated:** 2025-10-21
**Version:** 1.0
**Maintainer:** Your Name
