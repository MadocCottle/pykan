# Section 1 Tables

This directory contains scripts to generate comparison tables from Section 1 experimental results.

## Overview

The table generation scripts analyze checkpoint metadata from Section 1 experiments to create publication-ready comparison tables. All tables use **dense MSE** (computed on 10,000 samples) for rigorous evaluation.

## Available Tables

### Table 1: Function Approximation (Section 1.1)

Compares MLP, SIREN, and KAN on 9 function approximation tasks:
- Sinusoids (frequencies 1-5)
- Piecewise, Sawtooth, Polynomial
- High-frequency 1D Poisson solution

**Generates:**
- `table1a_function_approx_iso.*` - Iso-compute comparison (time-matched at KAN threshold)
- `table1b_function_approx_final.*` - Final performance comparison

**Run:**
```bash
python table1_function_approximation.py
```

**Requires:** Section 1.1 training completed

### Table 2: 1D Poisson PDEs (Section 1.2)

Compares MLP, SIREN, and KAN on 3 1D Poisson PDE tasks:
- Sinusoidal source term
- Polynomial source term
- High-frequency source term

**Generates:**
- `table2_pde_1d_final.*` - Final performance comparison only

**Run:**
```bash
python table2_pde_1d_comparison.py
```

**Note:** Sections 1.2 and 1.3 train models independently (not using iso-compute methodology), so only final performance is compared.

**Requires:** Section 1.2 training completed

### Table 3: 2D Poisson PDEs (Section 1.3)

Compares MLP, SIREN, and KAN on 4 2D Poisson PDE tasks:
- Sinusoidal source term
- Polynomial source term
- High-frequency source term
- Special source term

**Generates:**
- `table3_pde_2d_final.*` - Final performance comparison only

**Run:**
```bash
python table3_pde_2d_comparison.py
```

**Requires:** Section 1.3 training completed

## Quick Start

### Prerequisites

1. **Install dependencies:**
   ```bash
   pip install pandas numpy tabulate
   ```

2. **Run training scripts:**
   ```bash
   cd pykan/madoc/section1
   python section1_1.py --epochs 500  # For full iso-compute analysis
   python section1_2.py --epochs 100
   python section1_3.py --epochs 100
   ```

3. **Verify checkpoint metadata exists:**
   ```bash
   ls results/sec1_results/*_checkpoint_metadata.pkl
   ```

### Generate Tables

**Individual tables:**
```bash
cd pykan/madoc/section1/tables
python table1_function_approximation.py
python table2_pde_1d_comparison.py
python table3_pde_2d_comparison.py
```

**All tables at once:**
```bash
python generate_all_tables.py
```

**Skip specific tables:**
```bash
python generate_all_tables.py --skip 2 3  # Only Table 1
```

## Output Files

Each table script generates:

1. **LaTeX files** (`.tex`) - Ready for `\input{}` in papers
2. **CSV files** (`.csv`) - For further analysis or plotting

### Example Output Structure

```
tables/
├── table1a_function_approx_iso.tex
├── table1a_function_approx_iso.csv
├── table1b_function_approx_final.tex
├── table1b_function_approx_final.csv
├── table2_pde_1d_final.tex
├── table2_pde_1d_final.csv
├── table3_pde_2d_final.tex
└── table3_pde_2d_final.csv
```

## Methodology

### Dense MSE Evaluation

All tables report **dense MSE** - Mean Squared Error computed on 10,000 densely sampled points from the true function. This provides more rigorous evaluation than sparse test sets.

### Two-Checkpoint Strategy (Section 1.1 Only)

Section 1.1 uses a two-checkpoint strategy:

1. **Iso-compute checkpoint:** Captured when KAN reaches interpolation threshold. Provides fair time-matched comparison across all models.

2. **Final checkpoint:** Captured after full training budget. Shows best achievable performance.

### Why No Iso-Compute for Sections 1.2/1.3?

Sections 1.2 and 1.3 do NOT train KAN first (they train all models in parallel), so there is no shared "KAN threshold time" to use for iso-compute comparison. These sections only provide final performance comparisons.

## File Structure

```
tables/
├── README.md                          # This file
├── QUICK_START.md                     # Quick reference guide
├── METHODOLOGY.md                     # Detailed methodology
├── utils.py                           # Shared utility functions
├── table1_function_approximation.py   # Table 1 generator
├── table2_pde_1d_comparison.py        # Table 2 generator
├── table3_pde_2d_comparison.py        # Table 3 generator
├── generate_all_tables.py             # Master script
└── [output files]                     # Generated .tex and .csv files
```

## Utility Functions ([utils.py](utils.py))

- `load_checkpoint_metadata()` - Load checkpoint data from pickle files
- `get_dataset_names()` - Get dataset names for each section
- `compare_models_from_checkpoints()` - Create comparison DataFrames
- `print_table()` - Pretty-print tables to console
- `create_latex_table()` - Convert DataFrame to LaTeX
- `save_table()` - Save table to file
- `format_scientific()` - Format numbers in scientific notation

## Data Source

Tables load checkpoint metadata from:
```
../results/sec1_results/section1_{1,2,3}_*_checkpoint_metadata.pkl
```

These files are generated automatically by running `section1_*.py` training scripts with the checkpoint-based evaluation methodology.

## Troubleshooting

### "No checkpoint metadata found"

**Problem:** Training hasn't been run or checkpoint metadata wasn't saved.

**Solution:** Run the corresponding section training script:
```bash
cd pykan/madoc/section1
python section1_1.py --epochs 500
```

### "ModuleNotFoundError: No module named 'tabulate'"

**Problem:** Missing dependency.

**Solution:**
```bash
pip install tabulate
```

### N/A values in tables

**Problem:** Some models weren't trained or checkpoints are missing.

**Solution:**
- Check that all models (MLP, SIREN, KAN) were trained
- For Section 1.1 iso-compute tables, ensure sufficient epochs (>= 100) for KAN to reach threshold

### Empty or incorrect results

**Problem:** Old checkpoint metadata or wrong section.

**Solution:**
- Check the timestamp of the checkpoint metadata file being loaded
- Re-run training if needed
- Verify you're using the latest results (sorted by timestamp)

## Using Tables in Papers

LaTeX tables can be directly included:

```latex
\input{tables/table1a_function_approx_iso.tex}
```

Or copy the table content from the `.tex` files for customization.

CSV files can be used for additional analysis, plotting, or importing into other tools.

## Comparison to KAN Paper

| KAN Paper Table | Our Equivalent | Notes |
|-----------------|----------------|-------|
| Table 1: Special Functions | Table 1 | Adapted to our 9 function approximation tasks |
| Table 2: Feynman Dataset | Not applicable | Different dataset |
| Table 3: PDE Solutions | Tables 2 & 3 | Split into 1D and 2D PDEs |

## Support

For issues:
1. Check that checkpoint metadata files exist in `../results/sec1_results/`
2. Verify all dependencies are installed (`pip install pandas numpy tabulate`)
3. Ensure training scripts completed successfully
4. Check console output for specific error messages
5. Review [METHODOLOGY.md](METHODOLOGY.md) for detailed methodology
6. See [QUICK_START.md](QUICK_START.md) for common usage patterns
