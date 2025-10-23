## Section 2.1 - Optomiser comparison

Testing different optomisers. Since KANs tend to use fewer parameters than other NNs, applying different optomisers, especially those shown to improve performance on models with a small number of parameters when applied to PDE problems.

This paper claims the the LM optomisers delivers greatly superior performance on physics problems for models with a small number of paramters. This section aims to test that claim on the poisson equation in 2D, using the same tests and metrics as section1/section1_3.py.

This is informed by the paper - "Optimizing the optimizer for data driven deep neural networks and physics informed neural networks" - available at https://arxiv.org/abs/2205.07430

The four optomisers implemented in this paper are ADAM, LGBFS, LGBFS-L, and LM

## Section 2.2 - Adaptive density

Using the same heurisitc as is used in the pykan packages model.prune() function, instead of eliminating underperforming nodes, simply increase the density of the most important/sensitive nodes.

Do two tests. One with this as an alternative to the normal grid densification practice, and another also incorporating the regular grid densification, but with additional densification on imporant nodes.

Follow the instructions in heuristic.md to use the build in features of pykan, rather than making your own system

# 2.2.2 - Adaptive density (reduction)

In the interests of remaining below the interpolation threshold, reduce the density of nodes that would have been pruned down to a minimum. There are limitations here so far as.... hmmm.... ok.

Yeah ok so maybe once we approach the interpolation threshold.... we find some way to prune, reduce density etc on... some nodes? Like try to do what we can before we hit that threshold.

# Idea lol:
Would using a more dense MSE loss function actually prevent overfitting past the interpolation threshold?

More fine tuned grid size selection/changing, rather than picking from a list?

Maybe a one step look ahead to determine like the.... marge of each additional point haha.

## Section 2.3 - Merge_KAN

Similar to mixture of experts but very distinct. Train a variety of different KANs using different depths, basis functions (and maybe optomisers). Train each of these configurations from a variety of starting seeds. Then prune them until smaller KANs each reach the distinct sets of functional dependences mentioned in section 4 of the KANs paper in.

To merge kan, create a new KAN with one more layers. Then use the pykan api to make the output layer of each expert the top hidden layer of the merge KAN. That is if we had a 8 variable input {x_1....x_8} and say we found 3 different pruned KANs, with different dependencies. 

For example, if we have pruned cans with the following dependencies:
KAN_1 {x_1, x_2, x_6}
KAN_2 {x_3, x_4}
KAN_3 {x_2, x_5, x_7, x_8}

Eg if we have KANs of shape:

KAN_1: [3, 5, 1]
KAN_2: [2, 3, 1]
KAN_3: [4, 2, 1]

Then the merged KAN should be of shape [9, 10, 3, 1]

Then continue normal training for the Merge_KAN

Start with a fairly large number of KANs, then reduce the number at each of several stages, eventually combining into a single or handful of surviving KANS.

Eg you could start with 20 kans, then have 8 after the first merge, then 3 after the second.


## Section 2.3 - general notes
Part of the idea behind merge KAN is inspired by the lack of catesrophic forgetting KANS have. I vibesways sus that this mean that different experts will "remember".... different sections/ properties? much more in this l8r I guess haha

I also think this and 2.4 could really benifit from being tested in a very high dimensional space? Like both cause curse of dimensionality and the weird complexity of these kinda spaces maybe that curse of forgetting shit will mega buff KAN probe or just KANS or even MERGE KAN just cause like there would be *so* many relationships and you would want some kind of way to *remember* them.


Is it possible to fold a KAN.... it can't be made into a single basis function?

Maybe try snapping all functions when merging?

Like KAN merge normal and KAN merge snap?

ALso maybe kan merge freeze for when the component KANS are put in, training on *them* is frozen.... results in a deeper models as there would need to be layers above but yeah should make sense right?

<!-- ## Section 2.3.1 -->


## Section 2.4 - KAN_Probe

Genetic evolution of KANs. Similar to Merge_KAN, but greatly extended. 

KAN_probe is best explained by allegory. Imagine a black hole is discovered to a physical reality with a much greater number of spatial dimensions. Each of these probes must explore this space. Probes enter this reality at different points, then figure out which direction to move (optomiser) - from each starting point, they should split into a variety of probes first differentiated simply by their optomisers. Each probe should continue for a specified number of epochs. After this, probes should report back to their mother probe and report their progress (Dense MSE, H1_norm, semi_norm etc). Based on which have progressed most, probes should prune their models, then recombined traits - such as super experts, as per merge_KAN, optomisers, as per section 2.1, and other traits like depth, learning rate etc. - Genes/traits from better performing probes should be given a higher "survival" probability

This should comprise a two part genome of sub-experts and general traits.



## General Section 2 rules

All datasets and metrics should be identical to those used in section 1.3. This is so that the utility of each section can be fairly benchmarked against the basic KAN architecture.

---

# Higher-Dimensional Extension Plan (3D, 4D, 10D, 100D)

## Overview
Extend Section 2.1 (Optimizer Comparison) and Section 2.2 (Adaptive Density) to higher dimensions: 3D, 4D, 10D, and 100D Poisson PDE problems.

## Rationale from KAN Paper Analysis

### Key Findings from Original KAN Paper (kan.tex)

1. **100D Example** (line 468):
   ```
   f(x₁,...,x₁₀₀) = exp(1/100 * Σ sin²(πxᵢ/2))
   ```
   - Original paper used [100,1,1] architecture (shallow, additive structure)

2. **4D Example** (line 469):
   ```
   f(x₁,x₂,x₃,x₄) = exp(1/2*(sin(π(x₁²+x₂²)) + sin(π(x₃²+x₄²))))
   ```
   - Original paper used [4,4,2,1] architecture (deeper, better performance)

3. **Critical Finding** (line 471):
   > "the 2-Layer KAN [4,9,1] behaves **much worse** than the 3-Layer KAN (shape [4,2,2,1]). This highlights the greater expressive power of deeper KANs"

4. **Training Protocol** (line 471):
   - LBFGS optimizer for 1800 steps
   - Grid extension: G={3,5,10,20,50,100,200,500,1000}
   - Demonstrates LBFGS is feasible even for 100D

### Design Decisions

**Function Variants:** Use ONLY sinusoidal variant per dimension
- Rationale: Matches KAN paper's approach for scalability testing
- Cost: 1 variant × 4 dims = 4 test cases (manageable)
- Alternative (rejected): 4 variants × 4 dims = 16 test cases (too expensive)

**Optimizers for High-D:** LBFGS and LM only (no Adam)
- Rationale: Focus on advanced optimizers that perform well on small-parameter models
- KAN paper demonstrates LBFGS scales to 100D
- Reduces computational cost by 33%

**Architectures:** Test both shallow and deeper variants per dimension
- Validates KAN paper finding that depth significantly impacts performance
- Shallow: Baseline for comparison
- Deep: Expected to outperform based on KAN paper evidence

## Implementation Details

### 1. New Test Functions

Add to `pykan/madoc/section2/utils/data_funcs.py`:

```python
# ============= Higher-Dimensional Poisson PDE (forcing functions) =============

# 3D Poisson PDE (sinusoidal forcing)
# -∇²u = f in (0,1)³, u=0 on boundary
# Analytical solution: u(x,y,z) = sin(πx)sin(πy)sin(πz)
f_poisson_3d_sin = lambda x: 3 * (torch.pi**2) * torch.sin(torch.pi*x[:,[0]]) * torch.sin(torch.pi*x[:,[1]]) * torch.sin(torch.pi*x[:,[2]])

# 4D Poisson PDE (sinusoidal forcing)
# -∇²u = f in (0,1)⁴, u=0 on boundary
# Analytical solution: u(x₁,x₂,x₃,x₄) = sin(πx₁)sin(πx₂)sin(πx₃)sin(πx₄)
f_poisson_4d_sin = lambda x: 4 * (torch.pi**2) * torch.sin(torch.pi*x[:,[0]]) * torch.sin(torch.pi*x[:,[1]]) * torch.sin(torch.pi*x[:,[2]]) * torch.sin(torch.pi*x[:,[3]])

# 10D Poisson PDE (sinusoidal forcing)
def f_poisson_10d_sin(x):
    result = torch.ones_like(x[:,[0]])
    for i in range(10):
        result = result * torch.sin(torch.pi*x[:,[i]])
    return 10 * (torch.pi**2) * result

# 100D Poisson PDE (sinusoidal forcing)
def f_poisson_100d_sin(x):
    result = torch.ones_like(x[:,[0]])
    for i in range(100):
        result = result * torch.sin(torch.pi*x[:,[i]])
    return 100 * (torch.pi**2) * result
```

### 2. KAN Architectures by Dimension

Test multiple depths to validate that deeper KANs outperform shallow ones:

| Dimension | Shallow Architecture | Deep Architecture | Rationale |
|-----------|---------------------|-------------------|-----------|
| 3D | [3, 5, 1] | [3, 3, 2, 1] | Extends 2D pattern [2,5,1] |
| 4D | [4, 5, 1] | [4, 4, 2, 1] | Matches KAN paper's 4D architecture |
| 10D | [10, 5, 1] | [10, 10, 5, 1] | Moderate depth with reasonable width |
| 100D | [100, 1, 1] | [100, 10, 1] | Baseline vs reasonable hidden layer |

**Note:** [100,1,1] expected to underperform based on depth findings, but included as KAN paper baseline.

### 3. Grid Sizes by Dimension

Reduce grid sizes for higher dimensions to manage computational cost:

| Dimension | Grid Sizes | # Grids | Rationale |
|-----------|-----------|---------|-----------|
| 2D (existing) | [3, 5, 10, 20, 50, 100] | 6 | Current baseline |
| 3D | [3, 5, 10, 20, 50, 100] | 6 | Same as 2D |
| 4D | [3, 5, 10, 20, 50, 100] | 6 | Same as 2D |
| 10D | [3, 5, 10, 20, 50] | 5 | Remove largest grid (100) |
| 100D | [3, 5, 10, 20] | 4 | Minimal grids for computational feasibility |

### 4. Optimizer Configuration

**2D (existing):** Adam, LBFGS, LM (3 optimizers)
**3D-100D (new):** LBFGS, LM only (2 optimizers)

**Rationale:**
- Focus on advanced optimizers shown to work well on small-parameter models
- Computational savings: 33% reduction per dimension
- KAN paper validates LBFGS works on 100D

### 5. Training Parameters

| Parameter | 3D | 4D | 10D | 100D | Notes |
|-----------|----|----|-----|------|-------|
| Train samples | 1000 | 1000 | 1000 | 1000 | Same as 2D |
| Test samples | 1000 | 1000 | 1000 | 1000 | Same as 2D |
| Epochs per grid | 10 | 10 | 10 | 10 | Default, configurable |
| Dense MSE samples | 10000 | 10000 | 5000 | 1000 | Reduced for 10D/100D |

**Dense MSE sampling:** Reduced for very high dimensions to avoid curse of dimensionality in evaluation phase.

### 6. New Scripts

Create two dimension-parameterized scripts:

#### `section2_1_highd.py` - High-D Optimizer Comparison

**Command-line arguments:**
```python
--dim: int, required, choices=[3, 4, 10, 100]
--architecture: str, required, choices=['shallow', 'deep']
--epochs: int, default=10
```

**Features:**
- Single sinusoidal Poisson PDE per dimension
- LBFGS and LM optimizers only
- Automatic grid size adjustment based on dimension
- Outputs same metrics as section2_1.py for fair comparison

#### `section2_2_highd.py` - High-D Adaptive Density

**Command-line arguments:**
```python
--dim: int, required, choices=[3, 4, 10, 100]
--architecture: str, required, choices=['shallow', 'deep']
--epochs: int, default=10
```

**Features:**
- Tests: adaptive-only, adaptive+regular, baseline
- Same dimension/architecture selection as section2_1_highd.py
- Adam optimizer for consistency with 2D version

### 7. Experimental Matrix

**Section 2.1 (Optimizer Comparison):**
- 2 architectures × 2 optimizers × N grids × 10 epochs

| Dimension | Configs per Dimension | Total Training Runs |
|-----------|----------------------|---------------------|
| 3D | 2 × 2 × 6 = 24 | 24 × 10 = 240 epochs |
| 4D | 2 × 2 × 6 = 24 | 24 × 10 = 240 epochs |
| 10D | 2 × 2 × 5 = 20 | 20 × 10 = 200 epochs |
| 100D | 2 × 2 × 4 = 16 | 16 × 10 = 160 epochs |
| **Total** | **84 configs** | **840 training runs** |

**Section 2.2 (Adaptive Density):**
- 2 architectures × 3 approaches × N grids × 10 epochs

| Dimension | Configs per Dimension | Total Training Runs |
|-----------|----------------------|---------------------|
| 3D | 2 × 3 × 6 = 36 | 36 × 10 = 360 epochs |
| 4D | 2 × 3 × 6 = 36 | 36 × 10 = 360 epochs |
| 10D | 2 × 3 × 5 = 30 | 30 × 10 = 300 epochs |
| 100D | 2 × 3 × 4 = 24 | 24 × 10 = 240 epochs |
| **Total** | **126 configs** | **1260 training runs** |

### 8. Expected Outcomes

1. **Depth Validation:** Deeper architectures outperform shallow ones across all dimensions
2. **Optimizer Performance:** Compare LBFGS vs LM on high-D problems
3. **Scalability Analysis:** Understand how KAN performance degrades (or doesn't) from 2D → 100D
4. **Adaptive Density Benefits:** Determine if adaptive densification provides advantages in high-D

### 9. File Structure

```
section2/
├── section2_1.py              # Existing: 2D optimizer comparison
├── section2_2.py              # Existing: 2D adaptive density
├── section2_1_highd.py        # NEW: High-D optimizer comparison
├── section2_2_highd.py        # NEW: High-D adaptive density
├── utils/
│   ├── data_funcs.py          # MODIFIED: Add 3D/4D/10D/100D functions
│   ├── optimizer_tests.py     # May need minor updates
│   └── ...
└── plan.md                    # This file
```

### 10. Implementation Checklist

- [ ] Add high-D Poisson functions to `data_funcs.py`
- [ ] Create `section2_1_highd.py` with dimension/architecture arguments
- [ ] Create `section2_2_highd.py` with dimension/architecture arguments
- [ ] Test 3D shallow (quick validation)
- [ ] Test 3D deep (validate depth benefit)
- [ ] Run full experiments: 3D, 4D, 10D, 100D
- [ ] Generate comparison plots and tables
- [ ] Document findings and validate against KAN paper claims

### 11. Usage Examples

```bash
# 3D optimizer comparison with shallow architecture
python section2_1_highd.py --dim 3 --architecture shallow --epochs 10

# 4D optimizer comparison with deep architecture
python section2_1_highd.py --dim 4 --architecture deep --epochs 10

# 100D optimizer comparison with both architectures
python section2_1_highd.py --dim 100 --architecture shallow --epochs 10
python section2_1_highd.py --dim 100 --architecture deep --epochs 10

# 10D adaptive density with deep architecture
python section2_2_highd.py --dim 10 --architecture deep --epochs 10
```

### 12. Computational Cost Estimate

Assuming single 2D experiment takes time T:

| Experiment | Estimated Time | Notes |
|------------|---------------|-------|
| 3D shallow | ~1.2T | Similar complexity to 2D |
| 3D deep | ~1.5T | Additional layer |
| 4D shallow | ~1.3T | One more dimension |
| 4D deep | ~1.6T | Deeper + higher-D |
| 10D shallow | ~2T | Much higher dimensionality |
| 10D deep | ~3T | Higher-D + deeper |
| 100D shallow | ~4T | Very high-D, but minimal grids |
| 100D deep | ~6T | Very high-D + deeper |

**Total estimated time: ~20T per section**
**With LBFGS+LM only (vs Adam+LBFGS+LM): 33% time savings**

## References

- Original KAN Paper: `/Users/main/Desktop/my_pykan/KAN_paper/kan.tex` (lines 468-471)
- Section 1 implementations (for reference patterns)
- Current Section 2: `section2_1.py`, `section2_2.py`
- LM Optimizer Paper: https://arxiv.org/abs/2205.07430