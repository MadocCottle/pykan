# High-Dimensional Experiment Visualization Guide

This guide explains how to visualize results from higher-dimensional (3D, 4D, 10D, 100D) Poisson PDE experiments in Section 2.

## Overview

The visualization strategy is based on research into the original KAN paper and high-quality ML publications (NeurIPS, ICML, ICLR). The approach separates visualizations into:

1. **Dimension-agnostic plots** - Work for all dimensions (3D-100D)
2. **Spatial visualizations** - Only for low dimensions (3D, 4D)

## Dimension-Agnostic Visualizations (All Dimensions)

### 1. Scaling Laws ⭐ HIGHEST PRIORITY

**File:** `plot_scaling_laws.py`

**What it shows:** Log-log plot of Dense MSE vs # Parameters

**Why it's important:**
- Primary visualization used in KAN paper for high-D results
- Shows whether KANs beat curse of dimensionality
- Validates theoretical scaling exponents (α=4 vs α=4/d)
- Compares deep vs shallow architectures

**Usage:**
```bash
# Single dimension
python plot_scaling_laws.py --dim 3 --architecture deep
python plot_scaling_laws.py --dim 100 --architecture shallow

# Compare all dimensions
python plot_scaling_laws.py --dim all --architecture deep
```

**What to look for:**
- Steeper slopes = better scaling (closer to α=4 is better)
- Deep architectures should outperform shallow
- Lines should follow KAN theory (α=4) rather than classical (α=4/d)

### 2. Dimension Comparison Heatmap

**File:** `plot_dimension_comparison.py`

**What it shows:** Heatmap of final Dense MSE across all experiments

**Layout:**
- Rows: Dimensions (3D, 4D, 10D, 100D)
- Columns: (Architecture × Optimizer) combinations
- Cell colors: Performance (darker green = better)
- Cell values: log₁₀(Dense MSE) with actual MSE in parentheses

**Usage:**
```bash
# Create heatmap
python plot_dimension_comparison.py --plot-type heatmap

# Compare shallow vs deep side-by-side
python plot_dimension_comparison.py --plot-type architecture

# Generate both
python plot_dimension_comparison.py --plot-type both
```

**What to look for:**
- Which (dimension, architecture, optimizer) combination performs best?
- Does deep consistently beat shallow?
- How does error scale with dimension?

### 3. Optimizer Comparison (Loss Curves)

**File:** `plot_optimizer_comparison.py` (existing, works for high-D)

**What it shows:** Dense MSE over training epochs

**Usage:**
```bash
python plot_optimizer_comparison.py --section section2_1_highd_3d_deep
python plot_optimizer_comparison.py --section section2_1_highd_100d_deep --plot-type both
```

**What to look for:**
- Convergence speed (which optimizer converges faster?)
- Final performance (which achieves lower MSE?)
- Stability (smooth vs oscillating curves)

## Spatial Visualizations (3D & 4D Only)

### 4. 1D Cross-Sections

**File:** `plot_cross_sections_1d.py`

**What it shows:** Slices through the function along each coordinate

**How it works:**
- For each coordinate direction (x, y, z, [w])
- Fix all other coordinates at 0.5
- Vary one coordinate across [0, 1]
- Compare true function vs KAN prediction

**Usage:**
```bash
# Single optimizer
python plot_cross_sections_1d.py --dim 3 --architecture deep --optimizer lbfgs

# Compare all optimizers
python plot_cross_sections_1d.py --dim 4 --architecture shallow --compare
```

**What to look for:**
- Visual match between true and predicted curves
- Coordinate-specific errors (some directions harder than others?)
- MSE values per direction

### 5. 2D Heatmap Cross-Sections (3D Only)

**File:** `plot_heatmap_2d.py` (existing, can be adapted)

**What it shows:** 2D slice through 3D function (fix z=0.5)

**Layout:**
- Row 1: True function (3D surface + contour)
- Row 2: Shallow LBFGS, Shallow LM
- Row 3: Deep LBFGS, Deep LM

**Note:** This requires model loading and evaluation on 2D grid. May need custom implementation.

## Visualization Priority by Dimension

### For 3D & 4D:
✅ **Must use:**
1. Scaling laws (dimension-agnostic)
2. Dimension comparison heatmap
3. 1D cross-sections (spatial)

✅ **Nice to have:**
4. Optimizer comparison curves
5. 2D heatmap slices (3D only)

### For 10D & 100D:
✅ **Must use (only dimension-agnostic):**
1. Scaling laws
2. Dimension comparison heatmap
3. Optimizer comparison curves

❌ **Skip (not meaningful):**
- Any spatial visualizations
- Cross-sections
- Heatmaps

## Recommended Workflow

### Step 1: Run Experiments
```bash
# Run all experiments first
for dim in 3 4 10 100; do
    for arch in shallow deep; do
        python section2_1_highd.py --dim $dim --architecture $arch --epochs 10
    done
done
```

### Step 2: Generate Overview Visualizations
```bash
# Get the big picture
python plot_dimension_comparison.py --plot-type both
python plot_scaling_laws.py --dim all --architecture deep
python plot_scaling_laws.py --dim all --architecture shallow
```

### Step 3: Detailed Analysis by Dimension
```bash
# For 3D and 4D: add spatial visualizations
python plot_cross_sections_1d.py --dim 3 --architecture deep --compare
python plot_cross_sections_1d.py --dim 4 --architecture deep --compare

# For all dimensions: individual scaling laws
for dim in 3 4 10 100; do
    python plot_scaling_laws.py --dim $dim --architecture deep
done
```

### Step 4: Optimizer-Specific Analysis
```bash
# Compare optimizer convergence
for dim in 3 4 10 100; do
    for arch in shallow deep; do
        python plot_optimizer_comparison.py \\
            --section section2_1_highd_${dim}d_${arch}
    done
done
```

## Expected Insights

### From Scaling Laws:
1. **Depth matters more in high-D:** Deep architectures should show steeper slopes (better scaling exponents)
2. **Beat curse of dimensionality:** Lines closer to α=4 than α=4/d
3. **Optimizer comparison:** LBFGS vs LM performance across dimensions

### From Dimension Comparison:
1. **Error scaling:** How does MSE increase from 3D → 100D?
2. **Architecture efficiency:** Is deep worth the extra parameters?
3. **Optimizer robustness:** Which optimizer handles high-D better?

### From Cross-Sections (3D/4D):
1. **Directional errors:** Uniform errors or some directions harder?
2. **Qualitative fit:** Visual assessment of approximation quality
3. **Architecture comparison:** Deep vs shallow spatial accuracy

## Color Schemes

**Dimensions:** Sequential colormap (viridis, plasma)
**Architectures:**
- Shallow: Blue (#2E86AB)
- Deep: Orange/Purple (#A23B72)

**Optimizers:**
- LBFGS: Blue/Teal (#2E86AB)
- LM: Red/Purple (#A23B72)

**Performance heatmaps:** Red-Yellow-Green (RdYlGn_r reversed, so darker green = better)

## File Locations

All visualizations are saved to: `section2/visualization/`

**Naming convention:**
- Scaling laws: `scaling_laws_{dim}d_{architecture}_{timestamp}.png`
- Comparisons: `scaling_laws_comparison_{architecture}.png`
- Heatmap: `dimension_comparison_heatmap.png`
- Cross-sections: `cross_sections_1d_{dim}d_{architecture}_{optimizer}_{timestamp}.png`

## Troubleshooting

### "No results found"
Make sure you've run the training scripts first:
```bash
python section2_1_highd.py --dim 3 --architecture shallow --epochs 10
```

### "Model file not found"
Check that models were saved during training. Models are saved to:
```
section2/results/sec1_results/section2_1_highd_{dim}d_{architecture}_{timestamp}_models.pkl
```

### Timestamp issues
Use `--timestamp` argument to specify exact timestamp:
```bash
python plot_scaling_laws.py --dim 3 --architecture deep --timestamp 20241023_140119
```

## References

- **KAN Paper visualizations:**
  - Figure `model_scaling.pdf` - Scaling laws
  - Figure `model_scale_exp100d.pdf` - 100D specific
  - Figure `special_pf.pdf` - Pareto frontiers

- **This implementation:** Based on analysis of KAN paper (kan.tex lines 468-471) and standard ML publication practices

## Future Enhancements

Possible additions (not yet implemented):

1. **Pareto Frontiers** - Parameters vs Accuracy trade-off
2. **Marginal Distributions** - Average over dimensions
3. **Animation** - Morphing across dimensions
4. **Interactive Plots** - Plotly/Bokeh for exploration
5. **2D Cross-sections for 4D** - Fix two coords, show 2D slice
