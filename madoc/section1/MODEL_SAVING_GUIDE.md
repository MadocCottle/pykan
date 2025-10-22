# Model Saving and Loading Guide

This guide explains how to save and load trained models (MLPs, SIRENs, and KANs) for function fitting visualization and further analysis.

## What's New

### Models Now Saved
All trained models are now automatically saved:
- **MLP models**: Best model per dataset (lowest final dense_mse) saved as PyTorch state_dict `.pth` files
- **SIREN models**: Best model per dataset (lowest final dense_mse) saved as PyTorch state_dict `.pth` files
- **KAN models**: All final models saved using KAN's checkpoint system
- **Pruned KAN models**: All pruned models saved using KAN's checkpoint system

### JSON Backwards Compatibility Removed
- All metadata is now stored in DataFrame `.attrs` (preserved in pickle files)
- JSON metadata files are no longer created or loaded
- Cleaner, more efficient storage system

## Automatic Saving

When you run any section script, models are automatically saved:

```bash
python madoc/section1/section1_1.py --epochs 20
```

This will save to `madoc/section1/results/sec1_results/`:
- `section1_1_{timestamp}_mlp.pkl` - MLP results DataFrame
- `section1_1_{timestamp}_mlp_{idx}.pth` - Best MLP model for dataset idx
- `section1_1_{timestamp}_siren.pkl` - SIREN results DataFrame
- `section1_1_{timestamp}_siren_{idx}.pth` - Best SIREN model for dataset idx
- `section1_1_{timestamp}_kan.pkl` - KAN results DataFrame
- `section1_1_{timestamp}_kan_{idx}/` - KAN checkpoint for dataset idx
- `section1_1_{timestamp}_kan_pruning.pkl` - KAN pruning results DataFrame
- `section1_1_{timestamp}_pruned_{idx}/` - Pruned KAN checkpoint for dataset idx

## Loading Results (Metrics Only)

To load just the DataFrames (for plotting metrics):

```python
from utils import load_run

# Load results
results, meta = load_run('section1_1', '20251022_194219')

# Access DataFrames
mlp_df = results['mlp']
siren_df = results['siren']
kan_df = results['kan']
kan_pruning_df = results['kan_pruning']

# Access metadata
print(f"Epochs: {meta['epochs']}")
print(f"Device: {meta['device']}")
```

## Loading Models (for Function Fitting)

To load saved models for visualization:

```python
from utils import load_run
import torch
from utils import trad_nn as tnn

# Load results AND models
results, meta, models = load_run('section1_1', '20251022_194219', load_models=True)

# models is a dict with keys: 'mlp', 'siren', 'kan', 'kan_pruned'
# Each contains {dataset_idx: model_data}

# Get the MLP state_dict for dataset 2 (freq 3 sinusoid)
dataset_idx = 2
mlp_state_dict = models['mlp'][dataset_idx]

# Reconstruct the model architecture
# You need to know the architecture used during training
# For Section 1.1: width=5, depth varies, activation varies
# Check the DataFrame to find the best config:
best_mlp = results['mlp'][results['mlp']['dataset_idx'] == dataset_idx]
final_epoch = best_mlp['epoch'].max()
best_config = best_mlp[best_mlp['epoch'] == final_epoch].iloc[0]

depth = int(best_config['depth'])
activation = best_config['activation']
n_var = 1  # Section 1.1 uses 1D inputs

# Recreate the model with same architecture
model = tnn.MLP(in_features=n_var, width=5, depth=depth, activation=activation)

# Load the saved weights
model.load_state_dict(mlp_state_dict)
model.eval()

# Now you can use it for predictions!
import numpy as np
x = torch.linspace(-1, 1, 1000).reshape(-1, 1)
y_pred = model(x).detach().numpy()

# Compare to ground truth
from utils import data_funcs as dfs
true_func = dfs.sinusoid_1d(3)  # freq 3
y_true = true_func(x).numpy()

# Plot
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(x.numpy(), y_pred, label='MLP Prediction', linewidth=2)
plt.plot(x.numpy(), y_true, label='Ground Truth', linewidth=2, linestyle='--')
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'MLP (depth={depth}, {activation}) vs Ground Truth')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('mlp_function_fit.png', dpi=300)
plt.show()
```

## Loading SIREN Models

Similar process for SIREN:

```python
# Get SIREN state_dict
siren_state_dict = models['siren'][dataset_idx]

# Find best config
best_siren = results['siren'][results['siren']['dataset_idx'] == dataset_idx]
final_epoch = best_siren['epoch'].max()
best_config = best_siren[best_siren['epoch'] == final_epoch].iloc[0]

depth = int(best_config['depth'])

# Recreate SIREN with same architecture
model = tnn.SIREN(in_features=1, hidden_features=5, hidden_layers=depth-2, out_features=1)

# Load weights
model.load_state_dict(siren_state_dict)
model.eval()

# Generate predictions...
```

## Loading KAN Models

KAN models are loaded differently (checkpoint paths):

```python
from kan import KAN

# Get KAN checkpoint path
kan_checkpoint_path = models['kan'][dataset_idx]

# Load using KAN's method
model = KAN.loadckpt(kan_checkpoint_path)
model.eval()

# Generate predictions
x = torch.linspace(-1, 1, 1000).reshape(-1, 1)
y_pred = model(x).detach().numpy()
```

## Complete Example: Visualizing All Models

```python
import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_run, data_funcs as dfs, trad_nn as tnn
from kan import KAN

# Load everything
results, meta, models = load_run('section1_1', '20251022_194219', load_models=True)

# Choose dataset
dataset_idx = 2  # freq 3 sinusoid
true_func = dfs.sinusoid_1d(3)

# Create x values for plotting
x = torch.linspace(-1, 1, 1000).reshape(-1, 1)
y_true = true_func(x).detach().numpy()

# === Load and predict with MLP ===
mlp_state_dict = models['mlp'][dataset_idx]
best_mlp = results['mlp'][results['mlp']['dataset_idx'] == dataset_idx]
best_mlp_row = best_mlp.loc[best_mlp.groupby(['depth', 'activation'])['dense_mse'].idxmin().iloc[0]]
mlp_model = tnn.MLP(in_features=1, width=5,
                     depth=int(best_mlp_row['depth']),
                     activation=best_mlp_row['activation'])
mlp_model.load_state_dict(mlp_state_dict)
mlp_model.eval()
y_mlp = mlp_model(x).detach().numpy()

# === Load and predict with SIREN ===
siren_state_dict = models['siren'][dataset_idx]
best_siren = results['siren'][results['siren']['dataset_idx'] == dataset_idx]
best_siren_row = best_siren.loc[best_siren.groupby('depth')['dense_mse'].idxmin()]
siren_model = tnn.SIREN(in_features=1, hidden_features=5,
                        hidden_layers=int(best_siren_row['depth'])-2,
                        out_features=1)
siren_model.load_state_dict(siren_state_dict)
siren_model.eval()
y_siren = siren_model(x).detach().numpy()

# === Load and predict with KAN ===
kan_checkpoint = models['kan'][dataset_idx]
kan_model = KAN.loadckpt(kan_checkpoint)
kan_model.eval()
y_kan = kan_model(x).detach().numpy()

# === Plot all together ===
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(x.numpy(), y_true, 'k--', label='Ground Truth', linewidth=3, alpha=0.7)
ax.plot(x.numpy(), y_mlp, label=f'MLP (depth={int(best_mlp_row["depth"])})', linewidth=2)
ax.plot(x.numpy(), y_siren, label=f'SIREN (depth={int(best_siren_row["depth"])})', linewidth=2)
ax.plot(x.numpy(), y_kan, label='KAN', linewidth=2)

ax.set_xlabel('x', fontsize=14)
ax.set_ylabel('y', fontsize=14)
ax.set_title('Model Comparison: Function Fitting', fontsize=16)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('all_models_comparison.png', dpi=300)
plt.show()
```

## Model Selection Strategy

The system saves the **best model per dataset** based on final dense MSE:
- For MLPs: Best across all (depth, activation) combinations
- For SIRENs: Best across all depths
- For KANs: The final trained model (after all grid refinements)

To find which model was saved:

```python
# Find best MLP configuration for dataset 2
dataset_idx = 2
mlp_final = results['mlp'].loc[
    results['mlp'].groupby(['dataset_idx', 'depth', 'activation'])['epoch'].idxmax()
]
mlp_dataset = mlp_final[mlp_final['dataset_idx'] == dataset_idx]
best_mlp = mlp_dataset.loc[mlp_dataset['dense_mse'].idxmin()]

print(f"Best MLP for dataset {dataset_idx}:")
print(f"  Depth: {best_mlp['depth']}")
print(f"  Activation: {best_mlp['activation']}")
print(f"  Dense MSE: {best_mlp['dense_mse']:.6e}")
```

## Storage Size

Typical storage per run:
- **DataFrames** (pickles): ~100KB - 1MB total
- **MLP models**: ~10-100KB per model
- **SIREN models**: ~10-100KB per model
- **KAN models**: ~1-10MB per model (larger due to spline coefficients)

For Section 1.1 with 9 datasets:
- Total: ~50-100MB per complete run

## Notes

1. **Architecture Required**: To load MLP/SIREN models, you must recreate the exact architecture. Check the DataFrame to find the configuration.

2. **Device Handling**: Models are loaded to CPU by default. Move to GPU if needed:
   ```python
   model = model.to('cuda')
   ```

3. **KAN Checkpoints**: KAN uses a directory-based checkpoint system. The path stored in `models['kan']` points to the checkpoint directory.

4. **Best Models Only**: Only the best-performing model per dataset is saved to minimize storage. All training metrics are still available in DataFrames.

5. **Metadata Access**: All metadata is in DataFrame `.attrs`:
   ```python
   print(results['mlp'].attrs['epochs'])
   print(results['mlp'].attrs['device'])
   ```
