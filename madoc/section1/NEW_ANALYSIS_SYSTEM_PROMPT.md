# Section 1: New Analysis System Design
**Inspired by Liu et al. (2024) KAN Paper Methodology**

## Executive Summary

This document outlines a revised analysis system for comparing **MLP, SIREN, KAN, and KAN+Pruning** architectures across function approximation and PDE solving tasks. The approach combines:
- **Best practices from the original KAN paper** (Liu et al., 2024) for fair architectural comparisons
- **Your existing heatmap analysis** from Section 1.3 (2D spatial error analysis)
- **Pareto-optimal model selection** instead of exhaustive depth/activation sweeps

---

## Current Issues with Existing Analysis

### What's Wrong:
1. **Exhaustive sweeps are wasteful**: Testing MLPs at depths {2,3,4,5,6} × activations {tanh, relu, silu} = 15 configurations per dataset is excessive and uninformative
2. **No fair comparison protocol**: Different architectures get different parameter budgets without systematic control
3. **Missing key metrics**: No parameter counts, no scaling law analysis, no Pareto frontiers
4. **Visualization gaps**: Learning curves exist but lack parameter-performance plots that reveal architecture efficiency

### What Works Well (Keep These):
✅ **Section 1.3 Heatmap Analysis** - Spatial error visualization with cross-sections and quantile maps
✅ **Function fitting overlays** - Side-by-side visual comparisons of predictions vs ground truth
✅ **Four-way comparisons** - MLP, SIREN, KAN, KAN+Pruning consistently compared

---

## New Analysis Framework (Inspired by KAN Paper)

### Philosophy from Liu et al. (2024)

The original KAN paper doesn't test every possible MLP configuration. Instead:

> **"Widths are set to 5 and depths are swept in {2,3,4,5,6}"** for baseline comparisons

> **Focus on Pareto frontiers**: Models that are "optimal in the sense of no other fit being both simpler and more accurate"

> **Primary metric**: Test RMSE vs. parameter count on log-log plots to reveal scaling laws

---

## Proposed Comparison Protocol

### 1. Model Selection Strategy

#### **For MLPs:**
- **Fixed width**: Choose one representative width (e.g., 64 or 128 neurons)
- **Fixed activation**: Use the best-performing activation from pilot studies (likely `silu` based on your data)
- **Depth sweep**: {2, 3, 4, 5} depths only
- **Rationale**: "We're not optimizing MLPs—we're establishing a baseline that's fair and replicable"

#### **For SIREN:**
- **Fixed depth**: Use 3 or 4 layers (SIREN paper suggests shallow networks work well)
- **No activation sweep**: SIREN uses sine activation by design
- **Rationale**: "SIREN has one canonical configuration; test it faithfully"

#### **For KAN:**
- **Grid sweep**: {3, 5, 10, 20, 50, 100} (matching Liu et al.)
- **Fixed depth**: 2 or 3 layers
- **Rationale**: "KAN performance scales with grid refinement, not depth"

#### **For KAN+Pruning:**
- Same grid sweep as KAN, plus final pruned model
- **Track pruning ratio**: Report what % of parameters survived

---

### 2. Metrics to Report (Matching KAN Paper)

For each model variant, compute:

| Metric | Purpose | Format |
|--------|---------|--------|
| **Test RMSE** | Primary accuracy measure | Scientific notation (e.g., 2.3×10⁻⁵) |
| **Parameter count** | Measure model complexity | Integer (e.g., 1,247 params) |
| **Training time** | Computational cost | Seconds (s) |
| **Train RMSE** | Detect overfitting | Scientific notation |
| **Epochs** | Training efficiency | Integer |

**Drop these metrics:**
- ❌ Dense MSE (inconsistent, hard to interpret)
- ❌ Per-activation MSE for MLPs (too granular)
- ❌ Time per epoch (redundant with total time ÷ epochs)

---

### 3. Visualization Suite

#### **A. Pareto Frontier Plot** (New, inspired by KAN paper)
**File**: `pareto_frontier_<dataset>.png`

**Description**: Log-log plot of test RMSE vs. parameter count

**What to show**:
- Each model type as different color/marker
- Connect points within same architecture family
- Highlight Pareto-optimal models (no model beats them on both axes)
- Include scaling law fit lines (α exponent) if possible

**Why**: "Reveals which architectures are parameter-efficient at achieving target accuracy"

```
Example layout:
   Test RMSE
      ↑
10⁻²  |    MLP-2  MLP-3
      |          ○
10⁻⁴  |    SIREN-3    MLP-4
      |        ●         ○
10⁻⁶  |              KAN-20
      |                  ◆
10⁻⁸  |                      KAN-50
      |                         ◆
      +---------------------------→ Parameters
       10²  10³  10⁴  10⁵  10⁶
```

---

#### **B. Best-Model Comparison Table** (New)
**File**: `best_models_comparison_<dataset>.csv`

**Columns**:
```csv
Architecture,Configuration,Test RMSE,Parameters,Train Time (s),Pareto Optimal
MLP,depth_4_silu,2.3e-05,8192,0.45,No
SIREN,depth_3,1.1e-06,4096,0.38,Yes
KAN,grid_20,8.7e-07,12473,2.1,Yes
KAN_Pruning,grid_50_pruned,3.2e-08,3421,2.8,Yes
```

**Selection criteria**:
1. Pareto-optimal models only (or top 3 per architecture if none qualify)
2. Sort by test RMSE ascending

---

#### **C. Scaling Law Plot** (New, advanced)
**File**: `scaling_laws_<dataset>.png`

**Description**: Test RMSE vs. parameter count with fitted power-law curves

**Math**: Fit `RMSE = A × N^(-α)` where N = parameter count

**Show**:
- Empirical data points
- Fitted curves with α values in legend
- Compare α across architectures

**Interpretation**: "Larger α means faster accuracy gains from adding parameters"

---

#### **D. Training Dynamics** (Keep, but simplify)
**File**: `training_curves_<dataset>.png`

**Show**: Only test RMSE over epochs for best model per architecture

**Dropzone**:
- ❌ Don't plot train RMSE (focus on generalization)
- ❌ Don't plot all configurations (visual clutter)

---

#### **E. Function Fitting Overlays** (Keep as-is)
**File**: `function_fit_<dataset>.png`

**Current approach is good**:
- Side-by-side subplots for each architecture
- Ground truth vs. prediction
- MSE annotated on each subplot

**Enhancement**: Add parameter count to subplot titles
- Example: "KAN (grid=20, 12.5k params)"

---

#### **F. Heatmap Analysis for 2D** (Keep from Section 1.3)
**Files**: `heatmap_*`, `cross_section_*`, `error_quantile_*`

**This is excellent work—don't change it!**

Components to retain:
- ✅ Spatial error maps (signed, absolute, relative)
- ✅ Cross-sections at fixed x₁, x₂
- ✅ Error quantile breakdowns (90th, 95th, 99th percentile regions)
- ✅ Four-way comparisons (MLP, SIREN, KAN, KAN+Pruning)

---

### 4. Report Structure

#### **Per-Dataset Analysis Summary**

For each function/PDE:

```markdown
## Dataset N: <Function Name>

### Best Models (Pareto Frontier)
| Architecture | Config | Test RMSE | Params | Time (s) | Notes |
|--------------|--------|-----------|--------|----------|-------|
| SIREN | depth_3 | 1.1e-06 | 4.1k | 0.38 | Fastest |
| KAN | grid_20 | 8.7e-07 | 12.5k | 2.1 | Most accurate |
| KAN+Pruning | grid_50→pruned | 3.2e-08 | 3.4k | 2.8 | Best efficiency |

### Key Findings
- **Winner by accuracy**: KAN+Pruning (3.2×10⁻⁸ RMSE)
- **Winner by efficiency**: SIREN (4.1k params for 10⁻⁶ RMSE)
- **Scaling laws**: KAN α=0.42, MLP α=0.28 (KAN scales faster)

### Visualizations
- [Pareto frontier](pareto_frontier_0.png)
- [Function fitting](function_fit_0.png)
- [Heatmap analysis](heatmap_0_2D_Sin.png) ← 2D only

### Spatial Error Analysis (2D only)
- **High-error regions**: Corners (>99th percentile)
- **Cross-section accuracy**: KAN outperforms at boundaries
```

---

## Implementation Checklist

### Phase 1: Data Collection Changes
- [ ] Run experiments with **reduced MLP sweep**: depth={2,3,4,5}, activation=silu only
- [ ] Collect **parameter counts** for all models (currently missing!)
- [ ] Ensure grid sweep for KAN: {3,5,10,20,50,100}

### Phase 2: Metrics Module Updates
- [ ] Add `count_parameters(model)` function to utils
- [ ] Modify results dict to include: `{'params': int, 'test_rmse': float, ...}`
- [ ] Remove dense_mse from comparison tables

### Phase 3: Visualization Development
- [ ] Create `pareto_frontier.py` module
  - Input: results dict with params + test_rmse
  - Output: log-log scatter + Pareto curve highlighting
- [ ] Create `scaling_laws.py` module
  - Fit power-law: `scipy.optimize.curve_fit`
  - Plot empirical + fitted curves
- [ ] Update `comparative_metrics.py`:
  - Generate best_models_comparison table
  - Simplify training curves (best model only)
- [ ] Enhance `function_fitting.py`:
  - Add param counts to subplot titles
- [ ] Keep `heatmap_2d_fits.py` **unchanged** (it's already great!)

### Phase 4: Reporting
- [ ] Update report templates with new sections
- [ ] Add Pareto optimality criteria to summary tables
- [ ] Include scaling law exponents in findings

---

## Example Output Structure

```
section1_1_analysis/
├── 01_pareto_analysis/
│   ├── pareto_frontier_0.png
│   ├── pareto_frontier_1.png
│   ├── scaling_laws_0.png
│   └── best_models_comparison_0.csv
├── 02_function_fitting/
│   ├── function_fit_0.png  [enhanced with param counts]
│   └── ...
├── 03_training_dynamics/
│   └── training_curves_0.png  [best models only]
└── ANALYSIS_SUMMARY.md
```

For 2D sections (1.3):
```
section1_3_analysis/
├── 01_pareto_analysis/
├── 02_function_fitting/
├── 03_training_dynamics/
├── 04_heatmap_analysis/  ← Keep existing amazing work!
│   ├── heatmap_0_2D_Sin_MLP.png
│   ├── heatmap_0_2D_Sin_SIREN.png
│   ├── heatmap_0_2D_Sin_KAN.png
│   ├── heatmap_0_2D_Sin_KAN_Pruning.png
│   ├── cross_section_0_2D_Sin.png
│   └── error_quantile_0_2D_Sin_KAN.png
└── ANALYSIS_SUMMARY.md
```

---

## Key Principles (Borrowed from KAN Paper)

1. **Fairness over exhaustiveness**:
   - "We don't need to test every activation—just establish a consistent baseline"

2. **Parameter-aware comparisons**:
   - "A smaller model with same accuracy is strictly better"

3. **Pareto optimality**:
   - "Only show models that can't be beaten on both simplicity AND accuracy"

4. **Scaling laws reveal architecture quality**:
   - "If KAN has α=0.5 and MLP has α=0.3, KAN is fundamentally more parameter-efficient"

5. **Visualize what matters**:
   - "Readers care about: Does it work? How complex is it? How long to train?"

---

## References

**Liu, Z., Wang, Y., Vaidya, S., Ruehle, F., Halverson, J., Soljačić, M., Hou, T. Y., & Tegmark, M. (2024).**
*KAN: Kolmogorov-Arnold Networks.*
arXiv:2404.19756.
https://arxiv.org/abs/2404.19756

**Key methodology insights:**
- Fixed-width MLP baselines (width=5 or 100)
- Depth sweeps {2,3,4,5,6} with consistent training
- Test RMSE vs. parameter count on log-log scales
- Pareto frontier identification
- Identical optimization (LBFGS, 3 seeds per config)
- Grid sweeps for KAN: {3,5,10,20,50,100,200,500,1000}

---

## Questions for Refinement

Before implementing, clarify:

1. **MLP width**: Should we use 64, 128, or match KAN input dimension?
2. **Optimization**: Continue with Adam or switch to LBFGS (as KAN paper uses)?
3. **Seeds**: Currently single-run; adopt 3-seed averaging for robustness?
4. **Grid range**: KAN paper tests up to grid=1000; practical limit for your tasks?
5. **Pruning threshold**: What sparsity target for KAN+Pruning (50%? 70%? 90%)?

---

## Timeline Estimate

| Phase | Tasks | Effort |
|-------|-------|--------|
| Planning | Finalize metrics, update experiment configs | 2-3 hours |
| Data collection | Re-run experiments with reduced MLP sweep | 4-6 hours (compute time) |
| Code development | pareto_frontier.py, scaling_laws.py, updates | 8-10 hours |
| Testing | Verify plots, validate Pareto logic | 2-3 hours |
| Documentation | Update templates, write interpretation guide | 2-3 hours |

**Total**: ~20-25 hours human time + compute time

---

## Final Recommendation

**Start with Section 1.1** (function approximation) as pilot:
1. Implement Pareto frontier + scaling law plots
2. Compare with existing exhaustive analysis
3. If insights are clearer → roll out to 1.2 and 1.3
4. Keep Section 1.3 heatmap analysis unchanged (it's publication-ready!)

**Success criteria**:
- ✅ Can identify "best" model per architecture in <5 seconds of looking at Pareto plot
- ✅ Scaling law exponents quantify which architecture is more parameter-efficient
- ✅ Reduction in total experiment compute time (fewer MLP configs)
- ✅ Cleaner, more interpretable reports for thesis/papers

---

*This document provides the blueprint—implementation can proceed incrementally.*
