# References.md Update Log

**Date:** 2025-10-24
**Related to:** Section 1 Training Fixes and Improvements

---

## Summary

Updated `references.md` to document the implementation details and paper methodology used in Section 1 training fixes. All changes connect our code implementations to the original published papers.

---

## Changes Made

### 1. SIREN Entry Enhancement (lines 347-363)

**Added Section 1 Usage Details:**
- Implementation location: `section1/utils/trad_nn.py`
- Training configuration: Adam optimizer with LR schedule (1e-4 → 1e-5)
- Initialization: Following paper specs (sin(ω₀·x) with ω₀=30)
- Stability: Gradient clipping with max_norm=0.1
- Role: Baseline comparison in all Section 1 experiments

**Why:** Documents the SIREN fixes that resolved the noisy prediction issue (Problem #1 from FIXES_SUMMARY.md)

---

### 2. KAN Entry Enhancement (lines 40-52)

**Added Section 1 Implementation Details:**
- Experiment scripts: `section1_2.py` (1D), `section1_3.py` (2D)
- Grid extension: G={3,5,10,20,50,100}
- **Pruning workflow (Section 3.2.1 of KAN paper):**
  - Stage 1: Sparsification with λ=1e-2 for 200 steps
  - Stage 2: Pruning with node_th=1e-2, edge_th=3e-2
  - Stage 3: Retraining for 200 steps
- Implementation: `section1/utils/model_tests.py:686-805`

**Why:** Documents the paper-aligned pruning workflow that fixed KAN+Pruning performance collapse (Problem #2 from FIXES_SUMMARY.md)

---

### 3. Section 1 Bibliography Entry Update (lines 503-512)

**Enhanced "Papers by Repository Section":**

**Before:**
```markdown
### Section 1 (Core KAN Implementation)
- Liu et al. (2024a) - KAN: Kolmogorov-Arnold Networks
- Kingma & Ba (2015) - Adam optimizer
```

**After:**
```markdown
### Section 1 (Function Approximation & Basic PDE Solving)
- Liu et al. (2024a) - KAN: Kolmogorov-Arnold Networks
  - Grid extension methodology
  - Pruning and simplification (Section 3.2.1)
  - Sparsification regularization (λ=1e-2)
- Sitzmann et al. (2020) - SIREN (baseline comparison)
  - Periodic activation functions: sin(ω₀·x) with ω₀=30
  - Specialized initialization: uniform(-1/n, 1/n) for first layer
  - Adam optimizer training with learning rate schedule
- Kingma & Ba (2015) - Adam optimizer (used for SIREN training)
```

**Why:** Provides clear traceability from code to papers, showing which papers informed which implementations

---

## Connection to Implementation Fixes

| Fix | Paper Reference | Section in references.md |
|-----|----------------|-------------------------|
| SIREN initialization | Sitzmann et al. (2020) | Lines 329-367, now enhanced at 347-363 |
| SIREN Adam optimizer | Kingma & Ba (2015) | Lines 82-104, now linked at 512 |
| KAN pruning workflow | Liu et al. (2024) Section 3.2.1 | Lines 24-53, now enhanced at 40-52 |

---

## Verification

To verify these references are correctly implemented, see:

1. **SIREN Implementation:**
   - Code: `section1/utils/trad_nn.py:35-62`
   - Training: `section1/utils/model_tests.py:149-263`
   - Paper: Sitzmann et al. (2020), specifically initialization section

2. **KAN Pruning Workflow:**
   - Code: `section1/utils/model_tests.py:686-805`
   - Paper: Liu et al. (2024), Section 3.2.1 "KAN Simplification"

3. **Adam Optimizer for SIREN:**
   - Code: `section1/utils/model_tests.py:166-180`
   - Paper: Kingma & Ba (2015)

---

## Future Updates Needed

If additional Section 1 experiments are added that reference other papers, update:
1. The individual paper's "Usage in Repo" section
2. The "Section 1" entry in "Papers by Repository Section"
3. Add to "Complete Bibliography" if it's a new paper

---

## Related Documentation

- **Implementation Details:** `/Users/main/Desktop/my_pykan/pykan/madoc/section1/FIXES_SUMMARY.md`
- **Before/After Comparison:** `/Users/main/Desktop/my_pykan/pykan/madoc/section1/BEFORE_AFTER.md`
- **Quick Reference:** `/Users/main/Desktop/my_pykan/pykan/madoc/section1/QUICK_REFERENCE.md`

---

**Maintained by:** Repository maintainers
**Last Updated:** October 24, 2025
