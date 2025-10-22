# Data Loading and Extraction Guide for Section 1 Experiments

This guide provides comprehensive instructions for loading and extracting features from the experiment results DataFrames for visualization and analysis.

## Quick Start

```python
import sys
from pathlib import Path

# Add section1 to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import load_run
import pandas as pd
import numpy as np

# Load results by section and timestamp
results, meta = load_run('section1_1', '20251022_194219')

# Access individual DataFrames
mlp_df = results['mlp']
siren_df = results['siren']
kan_df = results['kan']
kan_pruning_df = results['kan_pruning']
```

## Available DataFrames

### 1. MLP Results (`mlp_df`)

**Columns:**
- `dataset_idx` (int): Dataset index (0-8 for section1_1)
- `dataset_name` (str): Descriptive name of dataset (e.g., 'sin_freq1', 'piecewise', 'polynomial')
- `depth` (int): Network depth (2, 3, 4, 5, 6)
- `activation` (str): Activation function ('tanh', 'relu', 'silu')
- `epoch` (int): Training epoch (0 to epochs-1)
- `train_loss` (float): Training loss at this epoch
- `test_loss` (float): Test loss at this epoch
- `dense_mse` (float): Dense MSE computed on 10k samples from broader domain
- `total_time` (float): Total training time for this configuration (seconds)
- `time_per_epoch` (float): Average time per epoch (seconds)
- `num_params` (int): Number of model parameters

**Shape:** `(num_datasets × num_depths × num_activations × num_epochs, 11)`

### 2. SIREN Results (`siren_df`)

**Columns:**
- `dataset_idx` (int): Dataset index
- `dataset_name` (str): Descriptive name of dataset
- `depth` (int): Network depth (2, 3, 4, 5, 6)
- `epoch` (int): Training epoch
- `train_loss` (float): Training loss
- `test_loss` (float): Test loss
- `dense_mse` (float): Dense MSE on broader domain
- `total_time` (float): Total training time (seconds)
- `time_per_epoch` (float): Average time per epoch (seconds)
- `num_params` (int): Number of parameters

**Shape:** `(num_datasets × num_depths × num_epochs, 10)`

### 3. KAN Results (`kan_df`)

**Columns:**
- `dataset_idx` (int): Dataset index
- `dataset_name` (str): Descriptive name of dataset
- `grid_size` (int): Grid size (3, 5, 10, 20, 50, 100)
- `epoch` (int): Global epoch (across all grid refinements)
- `train_loss` (float): Training loss
- `test_loss` (float): Test loss
- `dense_mse` (float): Dense MSE on broader domain
- `total_time` (float): Training time for this grid (seconds)
- `time_per_epoch` (float): Time per epoch for this grid (seconds)
- `num_params` (int): Number of parameters
- `is_pruned` (bool): Always False for regular KAN

**Shape:** `(num_datasets × num_grids × epochs_per_grid, 11)`

### 4. KAN Pruning Results (`kan_pruning_df`)

Same columns as KAN results, but includes additional rows where `is_pruned=True` representing the pruned model results.

## Common Extraction Patterns

### Extract Final Epoch Results

```python
# Get final epoch for each configuration
def get_final_epoch(df, group_cols):
    """Get the last epoch for each configuration"""
    return df.loc[df.groupby(group_cols)['epoch'].idxmax()]

# MLP: final epoch for each (dataset, depth, activation)
mlp_final = get_final_epoch(mlp_df, ['dataset_idx', 'depth', 'activation'])

# SIREN: final epoch for each (dataset, depth)
siren_final = get_final_epoch(siren_df, ['dataset_idx', 'depth'])

# KAN: final epoch for each (dataset, grid_size) excluding pruned
kan_final = get_final_epoch(
    kan_df[~kan_df['is_pruned']],
    ['dataset_idx', 'grid_size']
)
```

### Extract Dense MSE for Plotting

```python
# Plot Dense MSE vs Depth for MLPs
for dataset_idx in mlp_df['dataset_idx'].unique():
    for activation in ['tanh', 'relu', 'silu']:
        subset = mlp_final[
            (mlp_final['dataset_idx'] == dataset_idx) &
            (mlp_final['activation'] == activation)
        ]
        x = subset['depth']
        y = subset['dense_mse']
        # Plot x, y...

# Plot Dense MSE vs Grid Size for KANs
for dataset_idx in kan_df['dataset_idx'].unique():
    subset = kan_final[kan_final['dataset_idx'] == dataset_idx]
    x = subset['grid_size']
    y = subset['dense_mse']
    # Plot x, y...
```

### Extract Training Curves (Loss vs Epoch)

```python
# Training curve for specific MLP configuration
dataset_idx = 0
depth = 3
activation = 'tanh'

mlp_curve = mlp_df[
    (mlp_df['dataset_idx'] == dataset_idx) &
    (mlp_df['depth'] == depth) &
    (mlp_df['activation'] == activation)
].sort_values('epoch')

epochs = mlp_curve['epoch']
train_losses = mlp_curve['train_loss']
test_losses = mlp_curve['test_loss']
dense_mses = mlp_curve['dense_mse']

# Plot training curves...
```

### Compare Models Across Same Dataset

```python
# Compare all models on dataset 0 at their best performance
dataset_idx = 0

# Get best MLP (lowest dense_mse at final epoch)
mlp_best = mlp_final[mlp_final['dataset_idx'] == dataset_idx].loc[
    mlp_final[mlp_final['dataset_idx'] == dataset_idx]['dense_mse'].idxmin()
]

# Get best SIREN
siren_best = siren_final[siren_final['dataset_idx'] == dataset_idx].loc[
    siren_final[siren_final['dataset_idx'] == dataset_idx]['dense_mse'].idxmin()
]

# Get best KAN
kan_best = kan_final[kan_final['dataset_idx'] == dataset_idx].loc[
    kan_final[kan_final['dataset_idx'] == dataset_idx]['dense_mse'].idxmin()
]

# Compare
comparison = pd.DataFrame({
    'Model': ['MLP', 'SIREN', 'KAN'],
    'Dense MSE': [mlp_best['dense_mse'], siren_best['dense_mse'], kan_best['dense_mse']],
    'Num Params': [mlp_best['num_params'], siren_best['num_params'], kan_best['num_params']],
    'Time': [mlp_best['total_time'], siren_best['total_time'], kan_best['total_time']]
})
```

### Extract Dense MSE vs Parameters

```python
# Dense MSE vs number of parameters for all models
def extract_mse_vs_params(df_final, model_name):
    """Extract (num_params, dense_mse) pairs"""
    return pd.DataFrame({
        'model': model_name,
        'num_params': df_final['num_params'],
        'dense_mse': df_final['dense_mse'],
        'dataset_idx': df_final['dataset_idx']
    })

mlp_params = extract_mse_vs_params(mlp_final, 'MLP')
siren_params = extract_mse_vs_params(siren_final, 'SIREN')
kan_params = extract_mse_vs_params(kan_final, 'KAN')

# Combine all
all_params = pd.concat([mlp_params, siren_params, kan_params])

# Plot for each dataset
for dataset_idx in all_params['dataset_idx'].unique():
    subset = all_params[all_params['dataset_idx'] == dataset_idx]
    for model in ['MLP', 'SIREN', 'KAN']:
        model_data = subset[subset['model'] == model]
        # Plot model_data['num_params'] vs model_data['dense_mse']
```

### Extract Epoch-by-Epoch Dense MSE Evolution

```python
# Track how dense MSE improves during training
dataset_idx = 0

# MLP with depth=3, activation='tanh'
mlp_evolution = mlp_df[
    (mlp_df['dataset_idx'] == dataset_idx) &
    (mlp_df['depth'] == 3) &
    (mlp_df['activation'] == 'tanh')
].sort_values('epoch')

# SIREN with depth=3
siren_evolution = siren_df[
    (siren_df['dataset_idx'] == dataset_idx) &
    (siren_df['depth'] == 3)
].sort_values('epoch')

# KAN across grid refinements
kan_evolution = kan_df[
    (kan_df['dataset_idx'] == dataset_idx) &
    (~kan_df['is_pruned'])
].sort_values('epoch')

# Plot dense_mse vs epoch for comparison
```

### Extract Best Configuration Per Dataset

```python
def find_best_config(df_final, dataset_idx):
    """Find configuration with lowest dense_mse for a dataset"""
    subset = df_final[df_final['dataset_idx'] == dataset_idx]
    if len(subset) == 0:
        return None
    best_idx = subset['dense_mse'].idxmin()
    return subset.loc[best_idx]

# Find best MLP configuration for each dataset
for dataset_idx in mlp_final['dataset_idx'].unique():
    best = find_best_config(mlp_final, dataset_idx)
    print(f"Dataset {dataset_idx}: depth={best['depth']}, "
          f"activation={best['activation']}, "
          f"dense_mse={best['dense_mse']:.6f}")
```

### Compare Pruned vs Unpruned KAN

```python
# Get pruned results
kan_pruned_only = kan_pruning_df[kan_pruning_df['is_pruned']]

# Compare for each dataset
for dataset_idx in kan_pruned_only['dataset_idx'].unique():
    # Get final unpruned result
    unpruned = kan_final[kan_final['dataset_idx'] == dataset_idx].iloc[-1]

    # Get pruned result
    pruned = kan_pruned_only[kan_pruned_only['dataset_idx'] == dataset_idx].iloc[0]

    print(f"Dataset {dataset_idx}:")
    print(f"  Unpruned: {unpruned['num_params']} params, "
          f"dense_mse={unpruned['dense_mse']:.6f}")
    print(f"  Pruned:   {pruned['num_params']} params, "
          f"dense_mse={pruned['dense_mse']:.6f}")
    print(f"  Param reduction: {100*(1-pruned['num_params']/unpruned['num_params']):.1f}%")
```

## Handling NaN Values

Some training runs may produce NaN values (model divergence). Always filter these out:

```python
# Remove rows with NaN in key metrics
mlp_clean = mlp_df[mlp_df['dense_mse'].notna() &
                    mlp_df['train_loss'].notna()]

# Or use dropna
siren_clean = siren_df.dropna(subset=['dense_mse', 'train_loss'])
```

## Metadata Access

Metadata is now stored in DataFrame attributes for efficiency and simplicity. Each DataFrame carries its own metadata.

### Essential Metadata (Stored in df.attrs)

```python
results, meta = load_run('section1_1', 'TIMESTAMP')

# Access from backward-compatible meta dict
print(f"Epochs: {meta['epochs']}")
print(f"Device: {meta['device']}")
print(f"Section: {meta['section']}")
print(f"Timestamp: {meta['timestamp']}")

# Or access directly from any DataFrame
mlp_df = results['mlp']
print(f"Epochs: {mlp_df.attrs['epochs']}")
print(f"Device: {mlp_df.attrs['device']}")
print(f"Model type: {mlp_df.attrs['model_type']}")
```

### Derivable Metadata (Computed from DataFrames)

Instead of storing redundant information, derive it from the data:

```python
mlp_df = results['mlp']
siren_df = results['siren']
kan_df = results['kan']

# Hyperparameter spaces explored
depths = sorted(mlp_df['depth'].unique())
activations = sorted(mlp_df['activation'].unique())
grids = sorted(kan_df['grid_size'].unique())
num_datasets = mlp_df['dataset_idx'].nunique()
dataset_names = sorted(mlp_df['dataset_name'].unique())

print(f"Depths tested: {depths}")
print(f"Activations tested: {activations}")
print(f"Grids tested: {grids}")
print(f"Number of datasets: {num_datasets}")
print(f"Dataset names: {dataset_names}")
```

**Why this approach?**
- ✅ No redundant storage (depths, activations, grids already in data)
- ✅ Always accurate (derived directly from actual data)
- ✅ Smaller files (no JSON, minimal metadata duplication)
- ✅ Self-contained DataFrames (metadata travels with data)

## Finding Available Timestamps

```python
from pathlib import Path

section = 'section1_1'
sec_num = section.split('_')[-1]
results_dir = Path(__file__).parent.parent / 'results' / f'sec{sec_num}_results'

# Find all timestamps for this section
timestamps = set()
for f in results_dir.glob(f'{section}_*_mlp.pkl'):
    # Extract timestamp from filename: section1_1_TIMESTAMP_mlp.pkl
    timestamp = f.stem.replace(f'{section}_', '').replace('_mlp', '')
    timestamps.add(timestamp)

print(f"Available timestamps: {sorted(timestamps)}")
```

## Complete Example: Plotting Dense MSE Comparison

```python
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_run

# Load results
results, meta = load_run('section1_1', 'TIMESTAMP')

# Get final epoch results
mlp_final = results['mlp'].loc[
    results['mlp'].groupby(['dataset_idx', 'depth', 'activation'])['epoch'].idxmax()
]
siren_final = results['siren'].loc[
    results['siren'].groupby(['dataset_idx', 'depth'])['epoch'].idxmax()
]
kan_final = results['kan'][~results['kan']['is_pruned']].loc[
    results['kan'][~results['kan']['is_pruned']].groupby(['dataset_idx', 'grid_size'])['epoch'].idxmax()
]

# Plot for dataset 0
dataset_idx = 0

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# MLP: Dense MSE vs Depth
for activation in ['tanh', 'relu', 'silu']:
    subset = mlp_final[
        (mlp_final['dataset_idx'] == dataset_idx) &
        (mlp_final['activation'] == activation)
    ].sort_values('depth')
    axes[0].plot(subset['depth'], subset['dense_mse'],
                 marker='o', label=activation)
axes[0].set_xlabel('Depth')
axes[0].set_ylabel('Dense MSE')
axes[0].set_title('MLP')
axes[0].legend()
axes[0].set_yscale('log')

# SIREN: Dense MSE vs Depth
subset = siren_final[siren_final['dataset_idx'] == dataset_idx].sort_values('depth')
axes[1].plot(subset['depth'], subset['dense_mse'], marker='o', color='C3')
axes[1].set_xlabel('Depth')
axes[1].set_ylabel('Dense MSE')
axes[1].set_title('SIREN')
axes[1].set_yscale('log')

# KAN: Dense MSE vs Grid Size
subset = kan_final[kan_final['dataset_idx'] == dataset_idx].sort_values('grid_size')
axes[2].plot(subset['grid_size'], subset['dense_mse'], marker='o', color='C4')
axes[2].set_xlabel('Grid Size')
axes[2].set_ylabel('Dense MSE')
axes[2].set_title('KAN')
axes[2].set_yscale('log')

plt.tight_layout()
plt.savefig(f'dense_mse_comparison_dataset_{dataset_idx}.png', dpi=300)
plt.show()
```

## Tips for Claude Code

1. **Always filter out NaN values** before plotting or analysis
2. **Use `groupby` with `idxmax`** to get final epoch results efficiently
3. **Check unique values** of categorical columns before iterating: `df['column'].unique()`
4. **Use boolean indexing** for filtering: `df[(condition1) & (condition2)]`
5. **Sort before plotting** to ensure correct line connections: `df.sort_values('column')`
6. **Log scale** often works better for dense_mse: `plt.yscale('log')`
7. **The `is_pruned` column** in KAN DataFrames distinguishes regular vs pruned results
8. **Epochs are zero-indexed**: first epoch is 0, last is `epochs-1`
9. **For KAN**, epochs are global across grid refinements - use `grid_size` to separate them
10. **Parameter counts** (`num_params`) are useful for efficiency comparisons

## Filter by Dataset Name

You can filter results by dataset name instead of index for more readable code:

```python
# Filter by specific dataset name
mlp_piecewise = mlp_df[mlp_df['dataset_name'] == 'piecewise']
mlp_sin_freq3 = mlp_df[mlp_df['dataset_name'] == 'sin_freq3']

# Filter multiple sinusoid frequencies
sin_datasets = mlp_df[mlp_df['dataset_name'].str.startswith('sin_freq')]

# Get best result per dataset (by name)
final_epoch = mlp_df.loc[mlp_df.groupby(['dataset_name', 'depth', 'activation'])['epoch'].idxmax()]
best_per_dataset = final_epoch.loc[final_epoch.groupby('dataset_name')['dense_mse'].idxmin()]
print(best_per_dataset[['dataset_name', 'dense_mse', 'depth', 'activation']])
```

## Dataset Names Reference

### Section 1.1 (Function Approximation)
- `sin_freq1`, `sin_freq2`, `sin_freq3`, `sin_freq4`, `sin_freq5`: Sinusoids with frequencies 1-5
- `piecewise`: Piecewise function
- `sawtooth`: Sawtooth wave
- `polynomial`: Polynomial function
- `poisson_1d_highfreq`: Poisson PDE 1D high frequency

### Section 1.2 (1D Poisson PDE)
- `poisson_1d_sin`: 1D Poisson with sin forcing
- `poisson_1d_poly`: 1D Poisson with polynomial forcing
- `poisson_1d_highfreq`: 1D Poisson with high frequency forcing

### Section 1.3 (2D Poisson PDE)
- `poisson_2d_sin`: 2D Poisson with sin forcing
- `poisson_2d_poly`: 2D Poisson with polynomial forcing
- `poisson_2d_highfreq`: 2D Poisson with high frequency forcing
- `poisson_2d_spec`: 2D Poisson with special forcing
