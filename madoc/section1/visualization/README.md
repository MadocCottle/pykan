# Visualization Tools for Section 1 Experiments

This folder contains tools and guides for visualizing the results from Section 1 model training experiments.

## Files

### Documentation

- **[loading_guide.md](loading_guide.md)**: Comprehensive guide for loading and extracting features from experiment result DataFrames. Essential reading for creating new visualizations.

### Plotting Scripts

- **[plot_best_dense_mse.py](plot_best_dense_mse.py)**: Plots the dense MSE evolution over training epochs for the best performing model from each class (MLP, SIREN, KAN, KAN+Pruning).

- **[plot_function_fit.py](plot_function_fit.py)**: Plots the learned functions vs ground truth for the best model from each class across all datasets. Shows actual function fits to visualize what each model learned.

## Quick Start

### Plot Best Dense MSE Evolution

```bash
# Plot for dataset 0 (most recent run)
python plot_best_dense_mse.py --dataset 0

# Plot for specific dataset and timestamp
python plot_best_dense_mse.py --dataset 5 --timestamp 20251022_201159

# Plot for different section
python plot_best_dense_mse.py --section section1_2 --dataset 0
```

**Options:**
- `--section`: Section to load (default: `section1_1`)
- `--timestamp`: Specific timestamp to load (default: most recent)
- `--dataset`: Dataset index to analyze (default: 0)

**Output:**
- PNG file: `best_dense_mse_dataset_{dataset_idx}_{timestamp}.png`
- Shows log-scale dense MSE vs epoch for the best model from each class

### Plot Function Fits

```bash
# Plot all datasets (most recent run)
python plot_function_fit.py

# Plot for specific timestamp
python plot_function_fit.py --timestamp 20251022_201159

# Plot for different section
python plot_function_fit.py --section section1_2

# Save individual plots for each dataset
python plot_function_fit.py --save-individual

# Use GPU for faster inference
python plot_function_fit.py --device cuda
```

**Options:**
- `--section`: Section to load (default: `section1_1`)
- `--timestamp`: Specific timestamp to load (default: most recent)
- `--device`: Device to use for model inference (default: `cpu`)
- `--save-individual`: Save separate plots for each dataset in addition to combined plot

**Output:**
- PNG file: `function_fits_all_datasets_{timestamp}.png` - Combined grid of all datasets
- PNG files (if `--save-individual`): `function_fit_dataset_{idx}_{name}_{timestamp}.png` - Individual plots
- Shows learned functions vs ground truth for MLP, SIREN, KAN, and KAN+Pruning

## Creating New Visualizations

1. **Read [loading_guide.md](loading_guide.md)** for detailed instructions on:
   - Loading results DataFrames
   - Extracting specific metrics
   - Common filtering and grouping patterns
   - Handling NaN values

2. **Use the template below** for new visualization scripts:

```python
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_run

def plot_my_visualization(section='section1_1', timestamp=None):
    # Find latest timestamp if not provided
    if timestamp is None:
        # ... (see plot_best_dense_mse.py for example)
        pass

    # Load results
    results, meta = load_run(section, timestamp)

    # Extract what you need
    mlp_df = results['mlp']
    # ... your extraction logic ...

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    # ... your plotting logic ...

    # Save
    output_file = Path(__file__).parent / f'my_plot_{timestamp}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    plot_my_visualization()
```

## Available DataFrames

All experiment results are stored as pandas DataFrames with the following structures:

### MLP Results
- Columns: `dataset_idx`, `depth`, `activation`, `epoch`, `train_loss`, `test_loss`, `dense_mse`, `total_time`, `time_per_epoch`, `num_params`

### SIREN Results
- Columns: `dataset_idx`, `depth`, `epoch`, `train_loss`, `test_loss`, `dense_mse`, `total_time`, `time_per_epoch`, `num_params`

### KAN & KAN+Pruning Results
- Columns: `dataset_idx`, `grid_size`, `epoch`, `train_loss`, `test_loss`, `dense_mse`, `total_time`, `time_per_epoch`, `num_params`, `is_pruned`

See [loading_guide.md](loading_guide.md) for detailed column descriptions and usage examples.

## Example Visualizations to Create

Here are some ideas for additional visualizations:

1. **Parameter Efficiency**: Dense MSE vs number of parameters (scatter plot)
2. **Training Time Comparison**: Time to reach certain MSE threshold
3. **Depth Analysis**: How performance varies with network depth for each model type
4. **Grid Refinement**: How KAN performance improves with grid size
5. **Pruning Impact**: Before/after pruning comparison
6. **Activation Function Comparison**: MLP performance across different activations
7. **Dataset Difficulty**: Compare all models across all datasets
8. **Convergence Speed**: Epochs to reach 90% of final performance
9. **Residual Plots**: Plot prediction errors (y_pred - y_true) to see where models fail
10. **2D Heatmaps**: For 2D problems, show function fits as heatmaps instead of slices

## Dataset Reference (Section 1.1)

- Dataset 0-4: Sinusoids with frequencies 1-5
- Dataset 5: Piecewise function
- Dataset 6: Sawtooth function
- Dataset 7: Polynomial function
- Dataset 8: Poisson PDE 1D high frequency

Check the source code or metadata for Section 1.2 and 1.3 dataset definitions.

## Output

All generated plots are saved to this folder with descriptive filenames including:
- What was plotted
- Dataset index (if applicable)
- Timestamp of the source data

Example: `best_dense_mse_dataset_0_20251022_201159.png`
