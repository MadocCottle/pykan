# Section 2 Visualization Tools

This directory contains visualization scripts for analyzing Section 2 results, which focus on comparing different optimizers (Adam vs. Levenberg-Marquardt) for training KAN models on 2D Poisson PDE problems.

## Available Visualizations

### 1. Optimizer Comparison (`plot_optimizer_comparison.py`)

Compares the performance of Adam and LM optimizers by plotting dense MSE evolution over training epochs.

**Usage:**
```bash
# Plot optimizer comparison for all datasets (uses most recent run)
python plot_optimizer_comparison.py

# Specify a particular timestamp
python plot_optimizer_comparison.py --timestamp 20231022_143022

# Plot only a specific dataset
python plot_optimizer_comparison.py --dataset 0

# Generate training loss comparison
python plot_optimizer_comparison.py --plot-type training

# Generate both optimizer and training loss plots
python plot_optimizer_comparison.py --plot-type both
```

**Outputs:**
- `optimizer_comparison_all_datasets_{timestamp}.png` - Dense MSE comparison for all datasets
- `optimizer_comparison_dataset_{idx}_{timestamp}.png` - Dense MSE comparison for specific dataset
- `training_loss_comparison_{timestamp}.png` - Training/test loss comparison

### 2. Function Fit Visualization (`plot_function_fit.py`)

Visualizes how well the trained KAN models (using different optimizers) fit the ground truth 2D functions.

**Usage:**
```bash
# Plot function fits for all datasets (uses most recent run)
python plot_function_fit.py

# Save individual plots for each dataset
python plot_function_fit.py --save-individual

# Generate 2D heatmap for a specific dataset
python plot_function_fit.py --heatmap --dataset 0

# Use GPU for inference
python plot_function_fit.py --device cuda
```

**Outputs:**
- `function_fits_all_datasets_{timestamp}.png` - Combined plot showing all dataset fits
- `function_fit_dataset_{idx}_{name}_{timestamp}.png` - Individual plots (if --save-individual)
- `heatmap_2d_dataset_{idx}_{name}_{timestamp}.png` - 2D heatmap visualization (if --heatmap)

## Dataset Information

Section 2.1 uses four 2D Poisson PDE test functions:
- Dataset 0: `poisson_2d_sin` - Sinusoidal source term
- Dataset 1: `poisson_2d_poly` - Polynomial source term
- Dataset 2: `poisson_2d_highfreq` - High-frequency source term
- Dataset 3: `poisson_2d_spec` - Spectral source term

All functions are defined on the unit square [0,1] × [0,1].

## Optimizer Information

- **Adam**: Adaptive moment estimation, a popular first-order gradient-based optimizer
- **LM (Levenberg-Marquardt)**: Second-order optimizer that interpolates between gradient descent and Gauss-Newton method

## Common Options

All scripts support these common arguments:
- `--section`: Section to load (default: `section2_1`)
- `--timestamp`: Specific timestamp to load (default: most recent)
- `--device`: Device to use for computation (`cpu` or `cuda`, default: `cpu`)

## Examples

### Quick Start
```bash
# Run the training script first
cd /Users/main/Desktop/my_pykan/pykan/madoc/section2
python section2_1.py --epochs 20

# Then generate visualizations
cd visualization
python plot_optimizer_comparison.py
python plot_function_fit.py --save-individual
python plot_function_fit.py --heatmap --dataset 0
```

### Analyzing a Specific Run
```bash
# Find available timestamps
ls ../results/sec1_results/

# Use a specific timestamp
python plot_optimizer_comparison.py --timestamp 20231022_143022 --dataset 2
python plot_function_fit.py --timestamp 20231022_143022 --heatmap --dataset 2
```

### Generating Publication-Quality Figures
```bash
# Generate all plots for a run
python plot_optimizer_comparison.py --plot-type both
python plot_function_fit.py --save-individual

# Generate heatmaps for all datasets
for i in 0 1 2 3; do
    python plot_function_fit.py --heatmap --dataset $i
done
```

## Understanding the Plots

### Dense MSE Plot
- **Y-axis (log scale)**: Dense MSE computed on 10,000 test points
- **X-axis**: Training epoch
- **Lower is better**: Models with lower curves are more accurate
- Shows how quickly each optimizer converges to a good solution

### Function Fit Plot
- **Solid line**: Ground truth function
- **Dashed/colored lines**: Model predictions from different optimizers
- For 2D functions, shows a 1D slice at y=0.5
- Visual inspection of approximation quality

### 2D Heatmap
- **Left panel**: Ground truth function over the full 2D domain
- **Middle/Right panels**: Model predictions from Adam and LM optimizers
- **Color scale**: Function values
- Allows visual comparison of spatial accuracy across the entire domain

## Notes

- All plots are saved as high-resolution PNG files (300 DPI)
- The scripts automatically use the most recent training run if no timestamp is specified
- Models must be saved during training for the function fit visualizations to work
- 2D heatmaps may take longer to generate due to the dense grid evaluation

## Troubleshooting

**"No results found" error:**
- Make sure you've run the training script (`section2_1.py`) first
- Check that the results directory exists: `../results/sec1_results/`

**"No models found" error:**
- The training script must save models (this is automatic in the current implementation)
- Check that model checkpoint directories exist in the results folder

**Memory issues with heatmaps:**
- Reduce the resolution in the script (default is 100×100)
- Use CPU instead of GPU if GPU memory is limited
