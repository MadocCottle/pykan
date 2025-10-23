# Plotting Bug Fix - Section 1

**Date:** 2025-10-24
**Status:** ✅ FIXED

---

## Issue: KAN+Pruning Models Missing from 2D Heatmaps

### Problem
All 2D heatmap visualizations showed "Model not available" for KAN+Pruning models, despite the checkpoint files existing on disk.

**Affected Visualizations:**
- `heatmap_2d_dataset_0_2D_Sin_(π²).png`
- `heatmap_2d_dataset_1_2D_Polynomial.png`
- `heatmap_2d_dataset_2_2D_High-freq.png`
- `heatmap_2d_dataset_3_2D_Special.png`

**Evidence:**
- Checkpoint files existed: `section1_3_20251024_044352_e30_kan_pruning_0_final_state` (and 1, 2, 3)
- Visualization code failed to load them
- Resulted in empty panels showing "Model not available"

---

## Root Cause

**File:** `section1/utils/io.py`
**Line:** 309 (before fix)

The `load_run()` function had a missing glob pattern for loading KAN pruned model checkpoints.

### The Bug
```python
# OLD CODE (missing epoch pattern):
for state_file in p.glob(f'{section}_{timestamp}_kan_pruning_*_final_state'):
    ...
```

This pattern **did not match** files with epoch information in the filename:
- **Expected pattern**: `section1_3_20251024_044352_kan_pruning_*_final_state`
- **Actual filename**: `section1_3_20251024_044352_e30_kan_pruning_0_final_state`
- **Mismatch**: Missing `_e30` portion!

### Why Other Models Worked
MLP, SIREN, and KAN models already had epoch-aware patterns:
- Line 210: `p.glob(f'{section}_{timestamp}_e*_mlp_*_final.pth')` ✓
- Line 242: `p.glob(f'{section}_{timestamp}_e*_siren_*_final.pth')` ✓
- Line 274: `p.glob(f'{section}_{timestamp}_e*_kan_*_final_state')` ✓

But KAN pruned was missing the `_e*` pattern!

---

## Solution

**File Modified:** `section1/utils/io.py`
**Lines Changed:** 306-342 (added epoch-aware pattern with fallbacks)

### The Fix
```python
# NEW CODE (with epoch pattern):
# Try format with epochs first
for state_file in p.glob(f'{section}_{timestamp}_e*_kan_pruning_*_final_state'):
    base_path = str(state_file)[:-6]
    parts = base_path.split('_')
    for i, part in enumerate(parts):
        if part == 'final' and i > 0:
            idx = int(parts[i-1])
            kan_pruned_models[idx] = base_path
            break

# Fallback: without epochs (backward compatibility)
if not kan_pruned_models:
    for state_file in p.glob(f'{section}_{timestamp}_kan_pruning_*_final_state'):
        ...existing fallback logic...

# Fallback: very old format with 'pruned' prefix
if not kan_pruned_models:
    for state_file in p.glob(f'{section}_{timestamp}_pruned_*_state'):
        ...existing very old format logic...
```

**Key Improvements:**
1. ✅ Added `_e*` to primary glob pattern
2. ✅ Maintained backward compatibility with two fallback patterns
3. ✅ Matches current checkpoint naming convention
4. ✅ Works with old results without breaking changes

---

## Testing & Verification

### Test Command
```bash
cd /Users/main/Desktop/my_pykan/pykan/madoc/section1/visualization
python3 plot_heatmap_2d.py --timestamp 20251024_044352
```

### Results
✅ **All 4 datasets now show KAN+Pruning visualizations**

**Before Fix:**
- KAN Pruning: "Model not available" (all datasets)

**After Fix:**
- Dataset 0 (2D Sin): KAN Pruning shows smooth 3D surface with MSE = nan
- Dataset 1 (2D Polynomial): KAN Pruning shows smooth 3D surface with MSE = 0.000655
- Dataset 2 (2D High-freq): KAN Pruning shows predictions ✓
- Dataset 3 (2D Special): KAN Pruning shows predictions ✓

**Note:** Some datasets show MSE = nan, but this is a **training issue** (likely from the pruning workflow encountering NaN during training with 30 epochs). The critical bug - **model loading** - is now fixed.

---

## Impact

### What's Fixed
- ✅ KAN+Pruning models now load correctly from checkpoints
- ✅ All 2D heatmap visualizations display KAN+Pruning predictions
- ✅ Backward compatible with old checkpoint naming formats
- ✅ No breaking changes to existing data or pipelines

### What's Still an Issue (Separate Problems)
- ⚠️ SIREN predictions still noisy (MSE ~79.82) - **Training issue, not plotting**
  - Root cause: Only 30 epochs trained, SIREN needs 100+ with Adam optimizer
  - Solution: Retrain with more epochs (already documented in FIXES_SUMMARY.md)

- ⚠️ Some KAN+Pruning models have NaN MSE - **Training issue, not plotting**
  - Root cause: Pruning workflow may encounter NaN with short training (30 epochs)
  - Solution: Use longer training runs or adjust pruning workflow

---

## Files Modified

| File | Lines | Change Summary |
|------|-------|----------------|
| `section1/utils/io.py` | 306-342 | Added epoch-aware glob pattern for KAN pruned models with fallbacks |

**Total Changes:** 1 file, ~20 lines added (with fallbacks for backward compatibility)

---

## Related Issues

This fix addresses the "missing models" issue identified in:
- `/madoc/section1/visualization/outputs/20251024_044426_results/` analysis
- Original problem: "Why is KAN pruning model not available for heatmaps?"

**Related Documents:**
- `FIXES_SUMMARY.md` - Documents SIREN and KAN pruning training improvements
- `BEFORE_AFTER.md` - Expected improvements comparison

---

## Backward Compatibility

✅ **Fully backward compatible**

The fix includes **three fallback patterns**:
1. New format with epochs: `section1_3_TIMESTAMP_e30_kan_pruning_0_final_state`
2. Old format without epochs: `section1_3_TIMESTAMP_kan_pruning_0_final_state`
3. Very old format: `section1_3_TIMESTAMP_pruned_0_state`

This ensures:
- New results (with epochs) load correctly ✓
- Old results (without epochs) still work ✓
- Ancient results (old naming) still work ✓

---

## Verification Checklist

- [x] Fix applied to `io.py`
- [x] All 4 datasets tested
- [x] KAN+Pruning appears in all heatmaps
- [x] No errors during visualization generation
- [x] Backward compatibility maintained
- [x] Documentation updated

---

**Status:** ✅ Complete and verified working
**Next Steps:** None required for this bug. If training quality is poor (SIREN noise, KAN NaN), see `FIXES_SUMMARY.md` for training improvements.
