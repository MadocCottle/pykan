# Metadata Storage Optimization for Section 2

## Summary

Section 2 follows Section 1's **hybrid storage approach** that eliminates redundant metadata while maintaining backward compatibility. Metadata is stored as DataFrame attributes instead of separate JSON files, with checkpoint metadata stored separately for efficient access.

## Storage Philosophy

### Essential vs. Derivable Metadata

**Essential Metadata** (must be stored):
- Cannot be computed from the data itself
- Required for reproducibility
- Small in size
- Context that would be lost otherwise

**Derivable Metadata** (should not be stored):
- Can be computed from DataFrame columns
- Redundant with actual data
- May become stale if data changes
- Increases file size unnecessarily

## Implementation for Section 2

### 1. DataFrame Attribute Storage

**Before (Separate JSON)**:
```python
# File 1: section2_1_TIMESTAMP_adam.pkl (DataFrame)
# File 2: section2_1_TIMESTAMP_meta.json (Metadata)
{
    "epochs": 10,
    "device": "cpu",
    "grids": [3, 5, 10, 20, 50, 100],
    "num_datasets": 4,
    "dataset_names": ["poisson_2d_sin", "poisson_2d_poly", ...],
    "lbfgs_threshold_time": 45.2
}
```

**After (DataFrame Attributes)**:
```python
# Single file: section2_1_TIMESTAMP_adam.pkl (DataFrame with attrs)
df.attrs = {
    'section': 'section2_1',
    'timestamp': '20251024_143022',
    'epochs': 10,
    'device': 'cpu',
    'lbfgs_threshold_time': 45.2,      # Essential
    'baseline_threshold_time': None,   # Not applicable
    'model_type': 'adam'
}

# Derivable from DataFrame:
# - grids: df['grid_size'].unique() → [3, 5, 10, 20, 50, 100]
# - num_datasets: df['dataset_idx'].nunique() → 4
# - dataset_names: df['dataset_name'].unique() → ['poisson_2d_sin', ...]
```

### 2. Minimal Essential Metadata

**Section 2.1 (Optimizer Comparison)**:
```python
{
    'section': 'section2_1',
    'timestamp': '20251024_143022',
    'epochs': 10,
    'device': 'cpu',
    'lbfgs_threshold_time': 45.2,  # Reference optimizer threshold
    'model_type': 'adam'  # or 'lbfgs', 'lm'
}
```

**Section 2.2 (Adaptive Density)**:
```python
{
    'section': 'section2_2',
    'timestamp': '20251024_150000',
    'epochs': 10,
    'device': 'cpu',
    'baseline_threshold_time': 52.1,  # Reference approach threshold
    'model_type': 'adaptive_only'  # or 'adaptive_regular', 'baseline'
}
```

**Section 2.3 (Merge_KAN)**:
```python
{
    'section': 'section2_3',
    'timestamp': '20251024_153000',
    'device': 'cpu',
    'n_seeds': 5,
    'n_experts': 20,
    'n_selected': 8
}
```

### 3. Checkpoint Metadata Storage

**Separate File**: `*_checkpoint_metadata.pkl`

**Structure**:
```python
{
    'optimizer_name': {  # e.g., 'adam', 'lbfgs', 'lm'
        dataset_idx: {
            'at_threshold': {
                'epoch': 23,
                'time': 45.2,
                'train_loss': 1.2e-3,
                'test_loss': 1.5e-3,
                'dense_mse': 3.4e-4,
                'grid_size': 10,
                'num_params': 1234,
                'optimizer': 'adam',
                # Note: 'model' NOT included (saved separately)
            },
            'final': {
                'epoch': 60,
                'time': 200.5,
                'train_loss': 5.2e-4,
                'test_loss': 6.1e-4,
                'dense_mse': 2.8e-4,
                'grid_size': 100,
                'num_params': 12345,
                'optimizer': 'adam'
            }
        }
    }
}
```

**Rationale**:
- **Separate file**: Don't mix with DataFrames (different access patterns)
- **Small size**: Metrics only, no model weights (~10KB vs ~10MB for models)
- **Fast access**: Can analyze performance without loading 10MB+ checkpoint files
- **No model weights**: Models saved separately in checkpoint files

## File Organization

### Complete File Structure

```
section2/results/sec1_results/
├── section2_1_20251024_143022_e10_adam.pkl           # DataFrame + attrs
├── section2_1_20251024_143022_e10_adam.parquet       # DataFrame (optional, no attrs)
├── section2_1_20251024_143022_e10_lbfgs.pkl          # DataFrame + attrs
├── section2_1_20251024_143022_e10_lm.pkl             # DataFrame + attrs
│
├── section2_1_20251024_143022_e10_checkpoint_metadata.pkl  # All metrics
│
├── section2_1_20251024_143022_e10_adam_0_at_threshold      # KAN checkpoint
├── section2_1_20251024_143022_e10_adam_0_at_threshold_config.yml
├── section2_1_20251024_143022_e10_adam_0_at_threshold_state
├── section2_1_20251024_143022_e10_adam_0_final
├── section2_1_20251024_143022_e10_adam_0_final_config.yml
├── section2_1_20251024_143022_e10_adam_0_final_state
│
└── ... (more checkpoints for all optimizers/datasets)
```

### File Size Comparison

**Per Dataset (4 datasets)**:

| Component | Old Approach | New Approach | Savings |
|-----------|--------------|--------------|---------|
| DataFrames (3 optimizers) | 3 × 50KB = 150KB | 3 × 50KB = 150KB | 0KB |
| Metadata JSON | 1 × 2KB | 0KB | 2KB |
| DataFrame.attrs | 0KB | 3 × 0.5KB = 1.5KB | -1.5KB |
| Checkpoint metadata | N/A | 1 × 10KB | -10KB |
| Checkpoint files | 0KB | 24 × 2MB = 48MB | -48MB |
| **Total** | **152KB** | **48.16MB** | -48MB |

**Note**: Checkpoints are new feature (not in old approach), so comparison isn't direct. The key improvement is:
- **Eliminated**: Separate metadata JSON (2KB saved, 1 fewer file)
- **Self-contained**: DataFrames include their own metadata
- **Efficient checkpoint access**: Small metadata file enables fast analysis

## Benefits

### ✅ Reduced File Count

**Before** (per run):
- 3 DataFrames (pkl)
- 3 DataFrames (parquet, optional)
- 1 Metadata JSON
- **Total**: 4-7 files

**After** (per run):
- 3 DataFrames (pkl)
- 3 DataFrames (parquet, optional)
- 1 Checkpoint metadata
- 24 Checkpoint files (optional feature)
- **Total**: 4-31 files (but no separate JSON, checkpoints add value)

**Net benefit**: Eliminated metadata JSON, DataFrames are self-contained

### ✅ Self-Contained DataFrames

```python
# Can share a single pickle file with full context
df = pd.read_pickle('section2_1_20251024_143022_e10_adam.pkl')

print(df.attrs['epochs'])                  # 10
print(df.attrs['lbfgs_threshold_time'])    # 45.2
print(df.attrs['section'])                  # 'section2_1'

# Derivable metadata
print(df['grid_size'].unique())            # [3, 5, 10, 20, 50, 100]
print(df['dataset_name'].unique())         # ['poisson_2d_sin', ...]
print(df['optimizer'].unique())            # ['adam']
```

**No risk of**:
- Losing metadata file (travels with DataFrame)
- Metadata-data mismatch (always synchronized)
- Wrong JSON file loaded (metadata is embedded)

### ✅ Eliminated Redundancy

**Before**:
```python
# In JSON
{"grids": [3, 5, 10, 20, 50, 100]}

# In DataFrame (redundant)
df['grid_size'] = [3, 3, 3, ..., 10, 10, 10, ..., 100, 100, 100, ...]
#                   ↑ Stored 720 times (6 grids × 10 epochs × 4 datasets × 3 optimizers)
```

**After**:
```python
# Only in DataFrame
df['grid_size'].unique()  # Derive: [3, 5, 10, 20, 50, 100]
```

**Redundancy eliminated**: ~2KB per run

### ✅ Always Accurate

**Problem with separate metadata**:
```python
# metadata.json says:
{"grids": [3, 5, 10, 20, 50, 100]}

# But DataFrame actually has:
df['grid_size'].unique()  # [3, 5, 10, 20]  ← Training stopped early!
```

**Solution with derivable metadata**:
```python
# Always correct (derived from actual data)
grids = df['grid_size'].unique()  # [3, 5, 10, 20]  ← True state
```

## Usage Patterns

### Saving Results

**Old way** (deprecated but still works):
```python
save_run(results, 'section2_1',
         models={'adam': adam_models, ...},  # Deprecated
         epochs=10, device='cpu')
```

**New way** (recommended):
```python
save_run(results, 'section2_1',
         checkpoints={'adam': adam_checkpoints, ...},  # Two-checkpoint strategy
         epochs=10, device='cpu',
         lbfgs_threshold_time=lbfgs_threshold_time)  # Essential metadata
```

**What NOT to pass**:
```python
# BAD - These are derivable!
save_run(results, 'section2_1',
         grids=[3, 5, 10, 20, 50, 100],           # Redundant
         num_datasets=4,                           # Redundant
         dataset_names=['poisson_2d_sin', ...],   # Redundant
         epochs=10, device='cpu')
```

### Loading and Accessing Metadata

**Essential metadata** (from DataFrame.attrs):
```python
from utils import load_run

results, meta = load_run('section2_1', '20251024_143022')
adam_df = results['adam']

# Method 1: From consolidated meta dict (backward compatible)
print(meta['epochs'])                   # 10
print(meta['device'])                   # 'cpu'
print(meta['lbfgs_threshold_time'])    # 45.2

# Method 2: From DataFrame.attrs (recommended)
print(adam_df.attrs['epochs'])          # 10
print(adam_df.attrs['device'])          # 'cpu'
print(adam_df.attrs['model_type'])      # 'adam'
```

**Derivable metadata** (from DataFrame):
```python
# Hyperparameters explored
grids = sorted(adam_df['grid_size'].unique())              # [3, 5, 10, 20, 50, 100]
optimizers = sorted(adam_df['optimizer'].unique())         # ['adam']

# Dataset information
num_datasets = adam_df['dataset_idx'].nunique()            # 4
dataset_names = sorted(adam_df['dataset_name'].unique())   # ['poisson_2d_sin', ...]

# Training information
total_epochs = adam_df.groupby('dataset_idx')['epoch'].max().mean()  # 60
final_grid = adam_df.groupby('dataset_idx')['grid_size'].last().mode()[0]  # 100
```

**Checkpoint metadata** (from separate file):
```python
import pickle

# Load checkpoint metadata
with open('checkpoint_metadata.pkl', 'rb') as f:
    checkpoints = pickle.load(f)

# Access without loading full models
adam_threshold_mse = checkpoints['adam'][0]['at_threshold']['dense_mse']
adam_threshold_time = checkpoints['adam'][0]['at_threshold']['time']
adam_final_mse = checkpoints['adam'][0]['final']['dense_mse']

print(f"Adam reached {adam_threshold_mse:.6e} in {adam_threshold_time:.2f}s")
```

## Comparison: Section 1 vs Section 2

### Similarities

| Aspect | Both Sections |
|--------|---------------|
| **Storage** | DataFrame.attrs for essential metadata |
| **No JSON** | Eliminated separate metadata files |
| **Derivable** | Compute grids, depths, etc. from data |
| **Checkpoints** | Separate checkpoint_metadata.pkl |
| **Backward compatible** | Old loading code still works |

### Differences

| Aspect | Section 1 | Section 2 |
|--------|-----------|-----------|
| **Threshold field** | `kan_threshold_time` | `lbfgs_threshold_time` or `baseline_threshold_time` |
| **Model types** | mlp, siren, kan, kan_pruning | adam, lbfgs, lm (2.1) / adaptive_only, adaptive_regular, baseline (2.2) |
| **Checkpoint names** | `at_kan_threshold_time` (MLP/SIREN), `at_threshold` (KAN) | `at_threshold` (all), `final` (all) |
| **Comparison** | Model architectures | Optimizers/approaches |

## Design Decisions

### Why DataFrame.attrs Instead of JSON?

**Pros**:
- ✅ Self-contained files (metadata travels with data)
- ✅ No separate file to manage
- ✅ Preserved in pickle format (built-in)
- ✅ Always synchronized with data
- ✅ Simpler code (no JSON I/O)

**Cons**:
- ❌ Not preserved in parquet (older pandas versions)
- ❌ Slightly more work to access (need to read pickle)

**Decision**: Pros outweigh cons. Use pickle for loading, parquet optional.

### Why Separate Checkpoint Metadata File?

**Alternative considered**: Store checkpoint metadata in DataFrame.attrs

**Pros of separate file**:
- ✅ Different access pattern (analyze checkpoints without DataFrames)
- ✅ Smaller file for quick analysis (~10KB vs ~50KB DataFrame)
- ✅ Can include all optimizers in one file
- ✅ Natural structure for nested dicts

**Cons**:
- ❌ One more file to manage
- ❌ Could become unsynchronized

**Decision**: Separate file better for checkpoint analysis workflows.

### Why Not Store Model Weights in Metadata?

**Models are large**: 2-10MB per checkpoint
**Metadata is small**: 1-2KB per checkpoint

**Storage breakdown**:
```
Checkpoint files: 48MB (24 checkpoints × 2MB each)
Checkpoint metadata: 10KB (24 checkpoints × ~400 bytes each)
```

**Analysis workflow**:
```python
# Fast: Load metadata only (10KB)
with open('checkpoint_metadata.pkl', 'rb') as f:
    meta = pickle.load(f)
    print(meta['adam'][0]['at_threshold']['dense_mse'])  # Instant

# Slow: Load full model (2MB)
from kan import KAN
model = KAN.loadckpt('section2_1_..._adam_0_at_threshold')  # Seconds
```

**Decision**: Store metrics separately for fast analysis.

## Best Practices

### When Saving Results

**DO** pass essential metadata:
```python
save_run(results, section,
         checkpoints=checkpoints,
         epochs=10,              # ✅ Can't derive
         device='cpu',           # ✅ Can't derive
         lbfgs_threshold_time=45.2)  # ✅ Can't derive
```

**DON'T** pass derivable metadata:
```python
save_run(results, section,
         grids=[3, 5, 10, 20, 50, 100],  # ❌ Derivable
         num_datasets=4,                  # ❌ Derivable
         dataset_names=[...])             # ❌ Derivable
```

### When Loading Results

**DO** derive metadata from data:
```python
results, meta = load_run('section2_1', timestamp)
adam_df = results['adam']

# Derive from data (always accurate)
grids = adam_df['grid_size'].unique()
datasets = adam_df['dataset_name'].unique()
```

**DON'T** rely on stored derivable metadata:
```python
# May not exist in new format
grids = meta.get('grids', [])  # ❌ Fragile
```

### When Analyzing Checkpoints

**DO** use checkpoint metadata for quick analysis:
```python
with open('checkpoint_metadata.pkl', 'rb') as f:
    meta = pickle.load(f)

# Fast comparison without loading models
for opt in ['adam', 'lbfgs', 'lm']:
    mse = meta[opt][0]['at_threshold']['dense_mse']
    print(f"{opt}: {mse:.6e}")
```

**DON'T** load all models for simple comparisons:
```python
# Slow and unnecessary for simple comparisons
for opt in ['adam', 'lbfgs', 'lm']:
    model = KAN.loadckpt(f'..._{ opt}_0_at_threshold')  # ❌ Slow
    # Just to get a number...
```

## Migration from Old Format

### Backward Compatibility

The loading code handles both old and new formats transparently:

```python
# Loads old format (with JSON) OR new format (with attrs)
results, meta = load_run('section2_1', timestamp)

# Works with both:
print(meta['epochs'])  # From JSON (old) or attrs (new)
```

### No Migration Required

- Old results continue to work
- New results use DataFrame.attrs
- Visualization scripts work with both
- No need to convert existing data

### If You Want to Migrate

Not necessary, but if desired:

```python
# 1. Load old format
results, meta = load_run('section2_1', old_timestamp)

# 2. Add attrs to DataFrames
for model_type, df in results.items():
    df.attrs.update({
        'section': 'section2_1',
        'timestamp': old_timestamp,
        'epochs': meta['epochs'],
        'device': meta['device'],
        'model_type': model_type
    })

# 3. Re-save with new format
import pickle
for model_type, df in results.items():
    df.to_pickle(f'section2_1_{old_timestamp}_e{meta["epochs"]}_{model_type}.pkl')

# 4. Delete old JSON (optional)
os.remove(f'section2_1_{old_timestamp}_meta.json')
```

## Troubleshooting

### Metadata not found in DataFrame.attrs

**Symptom**: `df.attrs` is empty or missing keys

**Possible causes**:
1. Loaded from parquet (doesn't preserve attrs in old pandas)
2. Very old run (before attrs implementation)
3. DataFrame was copied/transformed (attrs not preserved)

**Solutions**:
1. Load from pickle instead: `pd.read_pickle()` not `pd.read_parquet()`
2. Use `load_run()` which handles both formats
3. Check pandas version: `pandas.__version__` (>=1.3 recommended)

### Derivable metadata different from stored

**Symptom**: `meta['grids']` doesn't match `df['grid_size'].unique()`

**This is expected**:
- Old format may have stored intended grids
- Actual training may have stopped early
- Derivable version is ground truth

**Solution**: Always trust derivable metadata from actual data

### Checkpoint metadata file too large

**Symptom**: checkpoint_metadata.pkl is >1MB

**Possible causes**:
1. Accidentally included model weights
2. Many datasets/optimizers
3. Extra data in checkpoint dicts

**Solutions**:
1. Verify model weights excluded: `k != 'model'` in save_run()
2. Normal for many experiments (24 checkpoints × 0.5KB = 12KB is typical)
3. Check what's in checkpoint dict: `print(list(checkpoint_data.keys()))`

## Summary

### Key Principles

1. **Essential Only**: Store only what cannot be derived
2. **Self-Contained**: Metadata travels with data
3. **Separation of Concerns**: Checkpoint metadata separate for fast access
4. **Backward Compatible**: Old code still works
5. **Always Accurate**: Derive from actual data prevents staleness

### Essential Metadata Checklist

**✅ Store**:
- epochs (can't derive)
- device (can't derive)
- threshold times (can't derive)
- section, timestamp, model_type (can't derive)

**❌ Don't Store**:
- grids (derive from df['grid_size'].unique())
- num_datasets (derive from df['dataset_idx'].nunique())
- dataset_names (derive from df['dataset_name'].unique())
- Parameter counts (in DataFrame already)

### File Organization

```
Essential:
  ✅ section2_1_<TS>_e10_adam.pkl           (DataFrame + attrs)
  ✅ section2_1_<TS>_e10_checkpoint_metadata.pkl  (Metrics only)

Optional:
  ⚪ section2_1_<TS>_e10_adam.parquet       (Portability)
  ⚪ section2_1_<TS>_e10_adam_0_at_threshold  (Full models for analysis)

Eliminated:
  ❌ section2_1_<TS>_meta.json             (Redundant)
```

---

**Document Version**: 1.0
**Last Updated**: 2025-10-24
**Related**: [TWO_CHECKPOINT_STRATEGY.md](TWO_CHECKPOINT_STRATEGY.md), [PHASE_IMPLEMENTATION_SUMMARY.md](PHASE_IMPLEMENTATION_SUMMARY.md)
