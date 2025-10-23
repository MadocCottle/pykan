# Advanced Evaluation Metrics for Section 1

## Overview

This document describes the comprehensive suite of evaluation metrics available for assessing model performance on function approximation and PDE solving tasks. These metrics provide multiple perspectives on model quality beyond simple pointwise error.

**Last Updated**: January 2025
**Version**: 2.0 (H¹ and L∞ metrics added)

---

## Quick Reference Table

| Metric | Symbol | What It Measures | When to Use | Cost |
|--------|--------|------------------|-------------|------|
| **Dense MSE (L²)** | ||u - u_true||²_L² | Average squared error | Always (primary metric) | Medium |
| **L∞ Norm** | ||u - u_true||_∞ | Worst-case pointwise error | Safety-critical, robustness | Low |
| **H¹ Seminorm** | ||\∇u - \∇u_true||_L² | Gradient/derivative error | PDEs, physics-informed | High |
| **Train/Test MSE** | - | Sparse training set error | Training monitoring only | Very Low |
| **Parameter Count** | - | Model size | Efficiency analysis | Negligible |

---

## Detailed Metric Descriptions

### 1. Dense MSE (L² Norm) - PRIMARY METRIC

**Implementation**: `dense_mse_error_from_dataset()` in [utils/metrics.py](../utils/metrics.py)

**Formula**:
```
L² error = (1/N) Σ |u(x_i) - u_true(x_i)|²
```
where N = 10,000 dense samples across the entire domain.

**What it measures**: Average squared pointwise error, much more rigorous than sparse test sets.

**Current usage**:
- ✅ Computed at all checkpoints (iso-compute and final)
- ✅ Reported in all comparison tables (Tables 1-3)
- ✅ Used in visualizations

**When to use**: **Always** - this is your primary metric.

**Interpretation**:
- Lower is better
- Sensitive to outliers (squared error)
- Averages over entire domain
- Compare across models at same checkpoint type

**Code example**:
```python
from utils.metrics import dense_mse_error_from_dataset

dense_mse = dense_mse_error_from_dataset(
    model, dataset, true_function,
    num_samples=10000, device='cpu'
)
print(f"Dense MSE: {dense_mse:.6e}")
```

---

### 2. L∞ Norm (Maximum Error) - WORST-CASE PERFORMANCE

**Implementation**: `linf_error_from_dataset()` in [utils/metrics.py](../utils/metrics.py)

**Formula**:
```
L∞ error = max_i |u(x_i) - u_true(x_i)|
```
Maximum absolute error across all sample points.

**What it measures**: The single worst prediction your model makes.

**Current usage**:
- ✅ **NEW**: Available as of January 2025
- ⚠️ **Not yet integrated** into checkpoint metadata
- Can be computed manually post-training

**When to use**:
- When worst-case guarantees matter
- Safety-critical applications
- Detecting localized failures (e.g., near boundaries)
- Complementing L² to see full error distribution

**Interpretation**:
- Lower is better
- Shows maximum deviation anywhere in domain
- Can be much larger than L² if errors are concentrated
- Reveals failure modes L² might hide

**Code example**:
```python
from utils.metrics import linf_error_from_dataset

linf = linf_error_from_dataset(
    model, dataset, true_function,
    num_samples=10000, device='cpu'
)
print(f"L∞ Error (max): {linf:.6e}")
```

**Comparison with L²**:
```python
metrics = compute_all_metrics(model, dataset, true_function, device='cpu')
print(f"L² (average): {metrics['dense_mse']:.6e}")
print(f"L∞ (max):     {metrics['linf_error']:.6e}")
print(f"Ratio (L∞/L²): {metrics['linf_error'] / metrics['dense_mse']:.2f}")
```
If ratio >> 1, errors are concentrated in specific regions.

---

### 3. H¹ Seminorm - GRADIENT QUALITY (FOR PDEs)

**Implementation**: `h1_seminorm_error_from_dataset()` in [utils/metrics.py](../utils/metrics.py)

**Formula**:
```
H¹ seminorm = sqrt( (1/N) Σ_i Σ_d (∂u/∂x_d - ∂u_true/∂x_d)² )
```
where d ranges over spatial dimensions, computed via central finite differences.

**What it measures**: Error in the **gradients** (derivatives) of the solution, not just pointwise values.

**Why it matters for PDEs**:
- PDEs fundamentally involve derivatives: -∇²u = f (Poisson), ∂u/∂t = ... (diffusion)
- A model can have low L² but terrible gradient estimation
- Physics-informed neural networks care deeply about derivative accuracy
- Critical for problems where derivatives enter physics laws

**Current usage**:
- ✅ **NEW**: Available as of January 2025
- ⚠️ **Not yet integrated** into checkpoint metadata
- **RECOMMENDED** for 2D PDE problems (section1_3)
- Optional for 1D PDEs (section1_2)
- Not needed for function approximation (section1_1)

**When to use**:
- **Always** for 2D Poisson PDE problems (section1_3)
- Optionally for 1D PDEs where derivatives matter
- Skip for pure function approximation tasks

**Computational cost**: ~3x more expensive than dense_mse (requires 2*n_var extra forward passes per sample point)

**Interpretation**:
- Lower is better
- Units: Same as gradient of solution (e.g., if u is in meters, H¹ is in meters/meter = dimensionless)
- Can be larger or smaller than L² depending on problem smoothness
- High H¹ with low L² suggests model interpolates well but has poor derivative estimates

**Code example**:
```python
from utils.metrics import h1_seminorm_error_from_dataset

# For 2D PDE (section1_3)
h1_error = h1_seminorm_error_from_dataset(
    model, dataset, true_function,
    num_samples=10000, device='cpu', eps=1e-4
)
print(f"H¹ Seminorm Error: {h1_error:.6e}")
```

**Compute all metrics at once**:
```python
from utils.model_tests import compute_all_metrics

# For 2D PDEs: compute H¹
metrics = compute_all_metrics(
    model, dataset, true_function,
    device='cpu', compute_h1=True
)
print(f"L² error:  {metrics['dense_mse']:.6e}")
print(f"L∞ error:  {metrics['linf_error']:.6e}")
print(f"H¹ error:  {metrics['h1_seminorm']:.6e}")
```

**Implementation details**:
- Uses central finite differences with step size ε = 1e-4
- For each dimension d: ∂u/∂x_d ≈ (u(x + ε*e_d) - u(x - ε*e_d)) / (2ε)
- Samples interior points (away from boundaries) to avoid finite difference issues
- Accumulates squared gradient errors across all dimensions

**Typical H¹/L² ratios**:
- Smooth functions (low frequency): H¹/L² ~ 1-10
- Rough functions (high frequency): H¹/L² ~ 10-1000
- If H¹/L² >> expected, model has poor derivative estimates

---

### 4. Train/Test MSE (Sparse) - TRAINING MONITORING ONLY

**Implementation**: Standard PyTorch MSELoss during training

**What it measures**: MSE on sparse training/test sets (typically 1,000 samples)

**Current usage**:
- ✅ Used during training for convergence monitoring
- ✅ Used for KAN interpolation threshold detection
- ✅ Saved in DataFrames
- ❌ **NOT used** in comparison tables (deprecated for this purpose)

**When to use**:
- Training progress monitoring
- Early stopping decisions
- **DO NOT use** for final model comparisons (use dense_mse instead)

**Why dense_mse is better**:
- 10x more samples (10,000 vs 1,000)
- Covers domain more thoroughly
- More representative of true generalization
- Fair comparison across different training set sizes

---

### 5. Parameter Count - MODEL SIZE

**Implementation**: `count_parameters()` in [utils/metrics.py](../utils/metrics.py)

**What it measures**: Total number of trainable parameters

**Current usage**:
- ✅ Computed for all models
- ✅ Saved in checkpoint metadata
- ✅ Reported in all comparison tables

**When to use**: Always (for efficiency analysis)

**Typical ranges**:
- MLPs (depth 2-6, width 5): 100-500 params
- SIRENs: Similar to MLPs
- KANs (grid 3-100): 200-5000+ params
- KANs after pruning: 50-80% reduction

---

## Integration Status

### ✅ Fully Integrated (Ready to Use)

**Dense MSE (L²)**:
- [x] Implemented in metrics.py
- [x] Integrated into model_tests.py checkpoints
- [x] Used in all comparison tables
- [x] Visualized in plots

**Parameter Count**:
- [x] Implemented
- [x] Integrated everywhere
- [x] Reported in tables

---

### ⚠️ Available But Not Integrated (Manual Use)

**L∞ Norm**:
- [x] Implemented in metrics.py (`linf_error_from_dataset`)
- [ ] **TODO**: Add to checkpoint metadata
- [ ] **TODO**: Add to comparison tables (supplementary)
- [ ] **TODO**: Visualize worst-case error locations

**H¹ Seminorm**:
- [x] Implemented in metrics.py (`h1_seminorm_error_from_dataset`)
- [ ] **TODO**: Add to 2D PDE checkpoints (section1_3)
- [ ] **TODO**: Add to Table 3 (2D PDE comparison)
- [ ] **TODO**: Create gradient error heatmaps

**Helper Function**:
- [x] `compute_all_metrics()` available in model_tests.py
- [ ] **TODO**: Use in checkpoint computation
- [ ] **TODO**: Update section1_3.py to compute H¹

---

## Usage Guide

### For Function Approximation (Section 1.1)

**Recommended metrics**:
- ✅ Dense MSE (L²) - primary
- ✅ L∞ - optional, shows worst-case
- ❌ H¹ - not needed (no PDEs)

```python
# Current (automatic)
dense_mse = computed_automatically_in_checkpoints

# Manual L∞ (if desired)
from utils.metrics import linf_error_from_dataset
linf = linf_error_from_dataset(model, dataset, true_func, device='cpu')
```

---

### For 1D PDEs (Section 1.2)

**Recommended metrics**:
- ✅ Dense MSE (L²) - primary
- ✅ L∞ - optional
- ⚠️ H¹ - optional (less critical for 1D)

```python
# Current (automatic)
dense_mse = computed_automatically_in_checkpoints

# Manual H¹ (if interested in derivatives)
from utils.metrics import h1_seminorm_error_from_dataset
h1_error = h1_seminorm_error_from_dataset(model, dataset, true_func, device='cpu')
```

---

### For 2D PDEs (Section 1.3) - **RECOMMENDED ENHANCEMENT**

**Recommended metrics**:
- ✅ Dense MSE (L²) - primary
- ✅ L∞ - shows worst-case
- ✅ **H¹ - HIGHLY RECOMMENDED** for gradient quality

```python
from utils.model_tests import compute_all_metrics

# Compute all metrics (including H¹)
metrics = compute_all_metrics(
    model, dataset, true_function,
    device='cpu', compute_h1=True
)

print(f"L² (pointwise): {metrics['dense_mse']:.6e}")
print(f"L∞ (worst-case): {metrics['linf_error']:.6e}")
print(f"H¹ (gradient):   {metrics['h1_seminorm']:.6e}")
```

---

## Integration Roadmap

### Phase 1: Manual Computation (CURRENT STATUS)
- [x] Metrics implemented and tested
- [x] Helper function available
- Users can compute manually post-training

### Phase 2: Checkpoint Integration (RECOMMENDED NEXT STEP)
- [ ] Modify `section1_3.py` to compute H¹ + L∞ at checkpoints
- [ ] Update checkpoint metadata structure
- [ ] Save new metrics alongside dense_mse

### Phase 3: Table Enhancement
- [ ] Add H¹ column to Table 3 (2D PDEs)
- [ ] Add L∞ as supplementary metric
- [ ] Create multi-metric comparison table

### Phase 4: Visualization
- [ ] Plot gradient error heatmaps (if H¹ available)
- [ ] Mark worst-case error locations (if L∞ available)
- [ ] Create metric comparison plots

---

## Thesis Justification

### What You Can Defend (Current)

**With Dense MSE only**:
> "We use dense L² error computed on 10,000 samples as our primary evaluation metric, providing rigorous assessment across the entire domain rather than sparse test sets."

### What You Can Defend (With H¹ Added)

**For 2D PDEs**:
> "For PDE problems, we report both L² norm (pointwise accuracy) and H¹ seminorm (gradient accuracy) to comprehensively assess solution quality. Since PDEs fundamentally involve derivatives (-∇²u = f for Poisson), evaluating gradient error is essential. This multi-metric approach follows standard practice in numerical PDE literature."

### What You Can Defend (With L∞ Added)

**For all problems**:
> "We report both average-case (L² norm) and worst-case (L∞ norm) performance to characterize the full error distribution. The L∞ norm reveals localized failures that L² averages might hide, critical for understanding model robustness."

---

## References

### Code Implementation
- [utils/metrics.py](../utils/metrics.py) - All metric implementations
- [utils/model_tests.py](../utils/model_tests.py) - Helper functions and checkpoint computation
- [tables/METHODOLOGY.md](../tables/METHODOLOGY.md) - Checkpoint-based evaluation strategy

### Literature
- **H¹ Sobolev Seminorm**: Standard in numerical PDE analysis (see: Brenner & Scott, "Mathematical Theory of Finite Element Methods")
- **Physics-Informed Neural Networks**: Raissi et al., 2019 - emphasizes derivative accuracy
- **L∞ Error**: Standard in approximation theory and numerical analysis

---

## Questions?

For questions about:
- **Metric implementation**: See code in [utils/metrics.py](../utils/metrics.py:198)
- **When to use which metric**: See usage guide above
- **Integration into training**: See [utils/model_tests.py](../utils/model_tests.py:45) `compute_all_metrics()`
- **Checkpoint methodology**: See [tables/METHODOLOGY.md](../tables/METHODOLOGY.md)
