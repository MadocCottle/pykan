# Plotting and Visualization Guide

This guide explains how to generate all visualizations and tables for the KAN experiments.

## Quick Start

### Generate Everything
```bash
cd /Users/main/Desktop/my_pykan/pykan/madoc
./plot.sh
```

This will generate all plots and tables for both Section 1 and Section 2.

## plot.sh Script

The `plot.sh` script automates the generation of all visualizations and tables.

### Usage

```bash
./plot.sh [OPTIONS]
```

### Options

| Option | Description |
|--------|-------------|
| `--section1-only` | Generate only Section 1 plots and tables |
| `--section2-only` | Generate only Section 2 plots and tables |
| `--plots-only` | Generate only plots (skip tables) |
| `--tables-only` | Generate only tables (skip plots) |
| `--continue-on-error` | Continue even if some scripts fail |
| `--help, -h` | Show help message |

### Examples

**Generate only Section 1 visualizations:**
```bash
./plot.sh --section1-only --plots-only
```

**Generate only tables:**
```bash
./plot.sh --tables-only
```

**Generate Section 2 plots, continue on errors:**
```bash
./plot.sh --section2-only --plots-only --continue-on-error
```

## What Gets Generated

### Section 1 Visualizations

**Section 1.1 - Function Approximation:**
- Best loss curves comparing all model types
- Function fit plots showing learned vs. true functions
- Checkpoint comparison plots (at threshold vs. final)

**Section 1.2 - 1D Poisson PDE:**
- Best loss curves for PDE solutions
- Function fit plots for 1D solutions

**Section 1.3 - 2D Poisson PDE:**
- Best loss curves for 2D PDEs
- Function fit plots for 2D solutions
- 2D heatmap visualizations (3D surface + contour plots)

**Output location:** `section1/visualization/`

### Section 1 Tables

All tables are generated using `section1/tables/generate_all_tables.py`:

- Table 0: Executive Summary
- Table 1: Function Approximation Comparison
- Table 2: 1D Poisson PDE Comparison
- Table 3: 2D Poisson PDE Comparison
- Table 4: Parameter Efficiency Analysis
- Table 5: Training Efficiency Summary
- Table 6: KAN Grid Size Ablation
- Table 7: Depth Ablation Study

**Output location:** `section1/tables/`

### Section 2 Visualizations

**Section 2.1 - Optimizer Comparison:**
- Best dense MSE curves comparing optimizers
- Optimizer comparison plots (Adam, LBFGS, LM)
- Training loss comparison across optimizers
- Function fit plots
- 2D heatmap visualizations
- 1D cross-section plots

**Section 2.2 - Adaptive Grid:**
- Best dense MSE curves for adaptive grid methods

**High-Dimensional Experiments** (if results exist):
- Dimension comparison heatmaps
- Architecture depth comparison
- Scaling law plots (deep and shallow architectures)

**Output location:** `section2/visualization/`

## Individual Plot Scripts

All plotting scripts support a `--show` flag to display plots in a window. By default, plots are only saved to files.

### Section 1 Plotting Scripts

```bash
# Best loss curves
python section1/visualization/plot_best_loss.py --section section1_1 [--show]

# Function fits
python section1/visualization/plot_function_fit.py --section section1_1 [--show]

# Checkpoint comparison
python section1/visualization/plot_checkpoint_comparison.py --section section1_1

# 2D heatmaps (for section1_3)
python section1/visualization/plot_heatmap_2d.py [--dataset 0] [--show]
```

### Section 2 Plotting Scripts

```bash
# Best loss curves
python section2/visualization/plot_best_loss.py --section section2_1 --metric dense_mse [--show]

# Optimizer comparison
python section2/visualization/plot_optimizer_comparison.py --section section2_1 --plot-type both [--show]

# Function fits
python section2/visualization/plot_function_fit.py --section section2_1 [--show]

# 2D heatmaps
python section2/visualization/plot_heatmap_2d.py --section section2_1 [--show]

# 1D cross-sections
python section2/visualization/plot_cross_sections_1d.py --section section2_1 [--show]

# Dimension comparison (requires high-D results)
python section2/visualization/plot_dimension_comparison.py --section section2_1 --plot-type both [--show]

# Scaling laws (requires high-D results)
python section2/visualization/plot_scaling_laws.py --dim all --architecture deep [--show]
```

## Prerequisites

Before running the plotting scripts, ensure you have:

1. **Trained models and results** from running the experiment scripts:
   - Section 1: `section1_1.py`, `section1_2.py`, `section1_3.py`
   - Section 2: `section2_1.py`, `section2_2.py`
   - High-D (optional): `section2_1_highd.py`, `section2_2_highd.py`

2. **Required Python packages** installed (should already be installed if you ran the experiments)

## Troubleshooting

### No results found error
```
Error: Results directory not found
```
**Solution:** Run the corresponding experiment script first to generate results.

### Script fails with missing module
```
ModuleNotFoundError: No module named 'X'
```
**Solution:** Ensure you're in the virtual environment:
```bash
source .venv/bin/activate  # or activate.bat on Windows
```

### High-D plots are skipped
```
⚠ Skipping high-D plots (no results found)
```
**Solution:** This is normal if you haven't run the high-dimensional experiments. To generate high-D plots, run:
```bash
python section2/section2_1_highd.py --dim 3 --architecture shallow
python section2/section2_1_highd.py --dim 3 --architecture deep
# Repeat for dims 4, 10, 100
```

## Tips

1. **Use `--continue-on-error`** when generating many plots - this prevents the entire process from stopping if one script fails.

2. **Generate plots for specific sections** to save time during development:
   ```bash
   ./plot.sh --section1-only --plots-only
   ```

3. **Check individual scripts** if a particular plot fails - they provide more detailed error messages.

4. **All plots are saved by default** - no popup windows will appear unless you add `--show`.

5. **Table generation is fast** compared to plotting, so you can run tables separately:
   ```bash
   ./plot.sh --tables-only
   ```

## Next Steps

After generating plots and tables:

1. Check the output directories for generated files
2. Review the plots for quality and correctness
3. Use the tables in your paper/report
4. Add `--show` to individual scripts if you want to view specific plots interactively
