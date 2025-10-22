# Section 1 Tables

This directory contains scripts to generate comprehensive comparison tables based on the experimental results from Section 1 (function approximation and PDE solving).

## Overview

The table generation scripts are inspired by the tables in the KAN paper but adapted to the specific experiments conducted in this repository. Each script analyzes results from the pickle files stored in `../results/sec1_results/`.

## Available Tables

### Table 1: Function Approximation Comparison (`table1_function_approximation.py`)
Compares MLP, SIREN, KAN, and KAN with pruning on various 1D function approximation tasks:
- Sinusoids (frequencies 1-5)
- Piecewise functions
- Sawtooth functions
- Polynomial functions
- High-frequency Poisson solutions

**Output:**
- Console table with detailed comparisons
- LaTeX table for paper inclusion
- CSV export
- Summary statistics

**Similar to:** KAN paper Table 1 (Special Functions)

### Table 2: 1D Poisson PDE Comparison (`table2_pde_1d_comparison.py`)
Analyzes performance on 1D Poisson PDE solutions with different source terms:
- Sinusoidal source
- Polynomial source
- High-frequency source

**Output:**
- Comparison of test MSE across models
- Parameter efficiency analysis
- LaTeX and CSV exports

### Table 3: 2D Poisson PDE Comparison (`table3_pde_2d_comparison.py`)
Evaluates models on 2D Poisson PDE problems:
- 2D sinusoidal source
- 2D polynomial source
- 2D high-frequency source
- 2D special source

**Output:**
- Performance comparison with parameter counts
- Improvement ratios (KAN vs baselines)
- LaTeX and CSV exports

### Table 4: Parameter Efficiency Analysis (`table4_param_efficiency.py`)
Demonstrates KAN's parameter efficiency compared to MLPs and SIRENs across all sections.

**Key metrics:**
- Parameter count vs accuracy trade-offs
- Parameter reduction ratios (MLP/KAN)
- Best-case efficiency examples

**Similar to:** KAN paper Table 3 (Signature classification comparison)

### Table 5: Training Efficiency and Convergence (`table5_convergence_summary.py`)
Analyzes computational efficiency and training dynamics:
- Time per epoch
- Convergence speed
- Final test MSE
- Relative speed comparisons

### Table 6: KAN Grid Size Ablation Study (`table6_grid_ablation.py`)
Explores the effect of grid resolution on KAN performance:
- Performance across grid sizes [3, 5, 10, 20, 50, 100]
- Best grid size per dataset
- Parameter count vs accuracy trade-offs
- Comparison of regular vs pruned KANs

### Table 7: Depth Ablation Study (`table7_depth_ablation.py`)
Analyzes how network depth affects MLP and SIREN performance:
- Depths [2, 3, 4, 5, 6]
- Activation function comparison (tanh, relu, silu)
- Optimal depth identification per dataset

## Usage

### Running Individual Tables

```bash
# Navigate to the tables directory
cd pykan/madoc/section1/tables

# Run individual table scripts
python table1_function_approximation.py
python table2_pde_1d_comparison.py
python table3_pde_2d_comparison.py
python table4_param_efficiency.py
python table5_convergence_summary.py
python table6_grid_ablation.py
python table7_depth_ablation.py
```

### Running All Tables

```bash
python generate_all_tables.py
```

## Requirements

The table generation scripts require:
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `tabulate` - Console table formatting

Install dependencies:
```bash
pip install pandas numpy tabulate
```

## Output Files

Each table script generates:

1. **Console Output**: Formatted tables printed to terminal using `tabulate`
2. **LaTeX Files**: `.tex` files ready for inclusion in papers
3. **CSV Files**: `.csv` files for further analysis or plotting
4. **Summary Statistics**: Additional analysis and insights

## File Structure

```
tables/
├── README.md                          # This file
├── utils.py                           # Shared utility functions
├── table1_function_approximation.py   # Section 1.1 comparison
├── table2_pde_1d_comparison.py        # Section 1.2 comparison
├── table3_pde_2d_comparison.py        # Section 1.3 comparison
├── table4_param_efficiency.py         # Parameter efficiency analysis
├── table5_convergence_summary.py      # Training efficiency analysis
├── table6_grid_ablation.py            # Grid size ablation
├── table7_depth_ablation.py           # Depth ablation
├── generate_all_tables.py             # Run all tables
└── [output files]                     # Generated .tex and .csv files
```

## Utility Functions

The `utils.py` module provides shared functionality:

- `load_latest_results()`: Load most recent results for a section
- `get_best_result_per_dataset()`: Extract best performance per dataset
- `format_architecture()`: Format model architecture strings
- `format_scientific()`: Scientific notation formatting
- `create_latex_table()`: Convert DataFrame to LaTeX
- `print_table()`: Pretty-print tables to console
- `compare_models()`: Cross-model comparison

## Data Source

All tables load data from pickle files in:
```
../results/sec1_results/section1_{1,2,3}_*_{mlp,siren,kan,kan_pruning}.pkl
```

These files are generated by running the section scripts:
- `section1_1.py` - Function approximation experiments
- `section1_2.py` - 1D PDE experiments
- `section1_3.py` - 2D PDE experiments

## Customization

### Adding New Tables

1. Create a new script `tableN_description.py`
2. Import utilities from `utils.py`
3. Load results using `load_latest_results()`
4. Process data and create tables
5. Save outputs in LaTeX and CSV formats

### Modifying Table Formats

Edit the `column_format` parameter in `create_latex_table()` calls to adjust LaTeX table appearance.

### Changing Metrics

Most functions accept a `metric` parameter (default: `'test_mse'`). You can analyze other metrics like:
- `train_mse`
- `num_params`
- `time_per_epoch`
- `dense_mse_*` (at various grid densities)

## Example Output

### Console Table Example
```
================================================================================
Table 1: Function Approximation Comparison (Section 1.1)
================================================================================
| Dataset           | MLP test_mse | MLP arch        | KAN test_mse | KAN arch    |
|-------------------|--------------|-----------------|--------------|-------------|
| sin_freq1         | 1.23e-03     | depth=3, tanh   | 4.56e-04     | grid=20     |
| piecewise         | 5.67e-03     | depth=4, relu   | 2.34e-03     | grid=50     |
...
```

### LaTeX Table Example
The generated LaTeX can be directly included in papers:
```latex
\begin{table}[t]
    \centering
    \begin{tabular}{|l|c|c|c|c|}
    \hline
    Function & MLP MSE & KAN MSE & KAN Params & Improvement \\
    \hline
    ...
    \end{tabular}
    \caption{Function approximation comparison...}
    \label{tab:function_approximation}
\end{table}
```

## Notes

- Tables automatically find the latest results based on timestamp
- If a model type has no results, it will show 'N/A' in tables
- All MSE values use scientific notation for consistency
- Parameter counts are displayed as integers
- Summary statistics include mean, std, min, and max where applicable

## Comparison to KAN Paper Tables

| KAN Paper Table | Our Equivalent | Notes |
|-----------------|----------------|-------|
| Table 1: Special Functions | Table 1 | Adapted to our function set |
| Table 2: Feynman Dataset | Not applicable | Different dataset |
| Table 3: Signature Classification | Table 4 | Parameter efficiency focus |
| Table 5: Anderson Localization | Not applicable | Different physics problem |
| Table 6: KAN Functionalities | Not applicable | API documentation |

Additional tables (5, 6, 7) provide ablation studies not present in the original KAN paper but valuable for understanding model behavior.

## Future Enhancements

Potential additions:
- Visualization integration (connect with `../visualization/` scripts)
- Statistical significance testing
- Pareto frontier analysis (accuracy vs parameters)
- Cross-section comparison tables
- Symbolic formula extraction tables (if implemented)
- Pruning effectiveness analysis

## Support

For issues or questions about table generation:
1. Check that result pickle files exist in `../results/sec1_results/`
2. Verify all dependencies are installed
3. Ensure you've run the section experiments first
4. Check console output for specific error messages
