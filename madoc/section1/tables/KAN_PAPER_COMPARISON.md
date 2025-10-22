# Comparison with KAN Paper Tables

This document maps the tables in the original KAN paper to the equivalent tables implemented in this repository.

## KAN Paper Tables (Original)

### Table 1: Special Functions
**Purpose:** Comparison of KAN vs MLP on scipy special functions (Bessel, elliptic integrals, Legendre functions, spherical harmonics)

**Columns:**
- Function name
- scipy API
- Minimal KAN shape
- Minimal KAN test RMSE
- Best KAN shape
- Best KAN test RMSE
- MLP test RMSE

**Our Equivalent:** `table1_function_approximation.py`
- **Adaptation:** Uses our function set (sinusoids, piecewise, polynomial, PDEs)
- **Extension:** Includes SIREN and KAN with pruning
- **Metrics:** Test MSE, architecture, parameter count

---

### Table 2: Feynman Dataset
**Purpose:** Physics equations with pruned KAN shapes vs MLP/unpruned KAN

**Columns:**
- Feynman equation ID
- Original formula
- Dimensionless formula
- Variables
- Human-constructed KAN shape
- Pruned KAN shape (smallest with RMSE < 10^-2)
- Pruned KAN shape (lowest loss)
- Various loss comparisons

**Our Equivalent:** Not directly applicable
- **Reason:** Different dataset (we use function approximation and PDEs)
- **Similar Analysis:** `table4_param_efficiency.py` shows parameter reduction like pruning analysis
- **Alternative:** Could implement symbolic formula discovery in future

---

### Table 3: Signature Classification Comparison
**Purpose:** KAN vs MLP parameter efficiency on knot theory problem

**Columns:**
- Method
- Architecture
- Parameter count
- Accuracy

**Our Equivalent:** `table4_param_efficiency.py`
- **Focus:** Parameter efficiency across all our experiments
- **Metrics:** Parameter count, accuracy, reduction ratio
- **Extension:** Covers multiple problem types, not just classification

---

### Table 4: Symbolic Formulas for Signature
**Purpose:** Auto-discovered formulas by KANs vs human formula

**Columns:**
- Formula ID
- Discovered formula
- Discovered by (human/KAN architecture)
- Test accuracy
- R² correlations

**Our Equivalent:** Not implemented (yet)
- **Reason:** Requires symbolic formula extraction
- **Future Work:** Could implement using KAN's `symbolic_formula()` method
- **Note:** This is a key KAN feature we could add

---

### Table 5: Anderson Localization Symbolic Formulas
**Purpose:** Mobility edge formulas for GAAM and MAAM systems

**Columns:**
- System
- Origin (theory/KAN)
- Formula
- Accuracy

**Our Equivalent:** Not applicable
- **Reason:** Different physics problem domain
- **Our Focus:** Function approximation and PDE solving

---

### Table 6: KAN Functionalities
**Purpose:** API reference for KAN methods

**Columns:**
- Functionality
- Description

**Our Equivalent:** Not a results table, but documented in:
- Main README files
- `.pykan_cheatsheet.md`
- `.pykan_usage_guide.md`

---

## Our Additional Tables (Not in KAN Paper)

### Table 0: Executive Summary
**Purpose:** High-level overview of all experimental results

**Why Added:**
- Quick reference for decision makers
- Cross-section comparison
- Key findings extraction

---

### Table 2 & 3: PDE-Specific Comparisons
**Purpose:** Detailed analysis of 1D and 2D PDE solving

**Why Added:**
- PDEs are major use case not emphasized in KAN paper
- Different problem characteristics than function fitting
- Shows KAN performance on physics problems

---

### Table 5: Training Efficiency Summary
**Purpose:** Computational cost and convergence analysis

**Why Added:**
- Practical concern not detailed in KAN paper
- Important for real-world deployment
- Shows time/accuracy trade-offs

---

### Table 6: Grid Size Ablation
**Purpose:** Systematic study of KAN grid parameter

**Why Added:**
- Grid size is key KAN hyperparameter
- Not extensively studied in original paper
- Helps users choose appropriate grid

---

### Table 7: Depth Ablation
**Purpose:** MLP/SIREN depth analysis

**Why Added:**
- Fair comparison requires optimal baselines
- Shows when depth helps vs hurts
- Activation function comparison

---

## Summary Comparison

| Aspect | KAN Paper | Our Implementation |
|--------|-----------|-------------------|
| **Function Approximation** | Special functions (15 functions) | Custom functions (9 functions) |
| **Physics Problems** | Feynman equations, Anderson localization | Poisson PDEs (1D & 2D) |
| **Symbolic Discovery** | Yes (Tables 2, 4, 5) | Not yet (future work) |
| **Parameter Efficiency** | Yes (Table 3) | Yes (Table 4) |
| **Ablation Studies** | Limited | Extensive (Tables 6, 7) |
| **Training Efficiency** | Not emphasized | Detailed (Table 5) |
| **Baseline Models** | MLP only | MLP, SIREN |
| **Pruning Analysis** | Integrated | Separate KAN variant |

## Key Differences

### 1. Problem Domain
- **KAN Paper:** Physics (Feynman, Anderson localization), special functions, knot theory
- **Our Repo:** Function approximation, PDE solving

### 2. Symbolic Formulas
- **KAN Paper:** Major focus, multiple tables
- **Our Repo:** Not yet implemented (opportunity for extension)

### 3. Baseline Comparisons
- **KAN Paper:** Primarily vs MLP
- **Our Repo:** MLP + SIREN (another spectral method)

### 4. Ablation Depth
- **KAN Paper:** Grid size mentioned but not systematically studied
- **Our Repo:** Systematic grid and depth ablations

### 5. Computational Cost
- **KAN Paper:** Not emphasized
- **Our Repo:** Dedicated table for training efficiency

## Opportunities for Future Tables

Based on KAN paper but not yet implemented:

1. **Symbolic Formula Extraction Table**
   - Use `model.symbolic_formula()`
   - Compare auto-discovered vs true formulas
   - Measure symbolic accuracy

2. **Pruning Effectiveness Table**
   - Before/after pruning comparison
   - Sparsity vs accuracy trade-off
   - Visualization of pruned architectures

3. **Interpolation vs Extrapolation**
   - Test on in-distribution vs out-of-distribution
   - Compare model robustness

4. **Compositional Structure Discovery**
   - Identify discovered hierarchies
   - Compare to known compositional structure

5. **Multi-Task Learning**
   - Train on multiple related tasks
   - Transfer learning analysis

## Usage Recommendations

### For Reproducing KAN Paper Results
- Focus on special functions (can add to Section 1.1)
- Implement symbolic formula extraction
- Add Feynman equations dataset

### For New Research
- Use our comprehensive ablation tables
- Leverage PDE solving benchmarks
- Build on parameter efficiency analysis

### For Applications
- Start with Table 0 (Executive Summary)
- Use Table 4 (Parameter Efficiency) for justification
- Reference Table 5 (Training Efficiency) for deployment planning

## Contributing

To add KAN paper-style symbolic formula tables:

1. Implement symbolic extraction in experiments
2. Create `table8_symbolic_formulas.py`
3. Use `model.symbolic_formula()` API
4. Compare with true analytical formulas
5. Measure symbolic accuracy metrics

See `pykan` documentation for symbolic formula methods.
