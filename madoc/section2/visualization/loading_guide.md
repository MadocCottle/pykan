# Model Loading Guide for Section 2

## Overview

Section 2 saves KAN models trained with different optimizers (Adam and LM). The models can be loaded for visualization and further analysis.

## File Structure

When you run `section2_1.py`, models are saved in the `section2/results/sec1_results/` directory with the following pattern:

```
section2_1_{timestamp}_{optimizer}_{dataset_idx}_config.yml
section2_1_{timestamp}_{optimizer}_{dataset_idx}_state
section2_1_{timestamp}_{optimizer}_{dataset_idx}_cache_data
```

For example:
```
section2_1_20251022_213346_adam_0_config.yml
section2_1_20251022_213346_adam_0_state
section2_1_20251022_213346_adam_0_cache_data
section2_1_20251022_213346_lm_0_config.yml
section2_1_20251022_213346_lm_0_state
section2_1_20251022_213346_lm_0_cache_data
```

## Loading Models

### Using the Utility Function

```python
from utils import load_run

# Load results and models
results, meta, models = load_run('section2_1', '20251022_213346', load_models=True)

# Models dict structure:
# models = {
#     'adam': {0: '/path/to/adam_0', 1: '/path/to/adam_1', ...},
#     'lm': {0: '/path/to/lm_0', 1: '/path/to/lm_1', ...}
# }
```

### Loading Individual KAN Models

```python
from kan import KAN

# Get model path from load_run
model_path = models['adam'][0]  # Path to Adam optimizer, dataset 0

# Load the KAN model
model = KAN.loadckpt(model_path)
model.eval()

# Use the model
import torch
x = torch.tensor([[0.5, 0.5]])
y = model(x)
```

## Known Issues

### YAML Loading Error with NumPy Scalars

If you encounter a YAML constructor error when loading models:

```
yaml.constructor.ConstructorError: could not determine a constructor for the tag
'tag:yaml.org,2002:python/object/apply:numpy.core.multiarray.scalar'
```

**Workaround:**

1. **Option 1: Use `yaml.unsafe_load` (requires modifying KAN source)**

   This is not recommended as it's a security risk. Instead, proceed to option 2.

2. **Option 2: Load models without YAML**

   You can manually load the state and bypass the config:

   ```python
   import torch
   from kan import KAN

   # Create a new KAN with the same architecture
   model = KAN(width=[2, 5, 1], grid=100, k=3, device='cpu')

   # Load the state directly
   model.load_state_dict(torch.load(model_path + '_state', map_location='cpu'))
   model.eval()
   ```

3. **Option 3: Re-save models without numpy scalars**

   Modify the training script to ensure all hyperparameters are Python primitives (int, float, str) rather than numpy types.

## Visualization Without Model Loading

If model loading is problematic, you can still use the optimizer comparison plots which only require the results DataFrames:

```bash
# This doesn't require model loading
python plot_optimizer_comparison.py --plot-type both

# Generate training loss comparison
python plot_optimizer_comparison.py --plot-type training
```

## Comparing Performance

### Using Results DataFrames

```python
from utils import load_run
import pandas as pd

# Load results
results, meta = load_run('section2_1', '20251022_213346')

# Compare final performance
adam_df = results['adam']
lm_df = results['lm']

# Get final epoch for each dataset
adam_final = adam_df.groupby('dataset_idx').last()
lm_final = lm_df.groupby('dataset_idx').last()

# Compare dense MSE
comparison = pd.DataFrame({
    'dataset': adam_final['dataset_name'],
    'adam_mse': adam_final['dense_mse'],
    'lm_mse': lm_final['dense_mse'],
    'adam_params': adam_final['num_params'],
    'lm_params': lm_final['num_params']
})

print(comparison)
```

## Best Practices

1. **Save checkpoints incrementally** during long training runs
2. **Use Python primitives** for hyperparameters to avoid YAML issues
3. **Test model loading** immediately after training completes
4. **Keep results DataFrames** even if model checkpoints fail to load
5. **Document model architecture** in your results for later reconstruction

## Additional Resources

- KAN documentation: See `/Users/main/Desktop/my_pykan/pykan/kan/` for source code
- Results visualization: See `plot_optimizer_comparison.py` for examples
- Section 1 reference: See `../section1/visualization/loading_guide.md` for similar patterns
