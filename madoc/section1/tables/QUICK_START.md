# Quick Start Guide - Section 1 Tables

## Installation

Ensure you have the required dependencies:
```bash
pip install pandas numpy tabulate
```

## Running Tables

### Option 1: Generate All Tables at Once
```bash
cd pykan/madoc/section1/tables
python generate_all_tables.py
```

This will generate all 8 tables (0-7) and display them in the console, plus save LaTeX and CSV files.

### Option 2: Generate Individual Tables
```bash
python table0_executive_summary.py      # High-level overview
python table1_function_approximation.py # Section 1.1 comparison
python table2_pde_1d_comparison.py      # Section 1.2 comparison
python table3_pde_2d_comparison.py      # Section 1.3 comparison
python table4_param_efficiency.py       # Parameter efficiency
python table5_convergence_summary.py    # Training efficiency
python table6_grid_ablation.py          # Grid size ablation
python table7_depth_ablation.py         # Depth ablation
```

### Option 3: Skip Certain Tables
```bash
python generate_all_tables.py --skip 5 6 7  # Skip tables 5, 6, 7
```

## Output Files

After running, you'll find in the `tables/` directory:

### LaTeX Files (`.tex`)
- `table0_executive_summary.tex`
- `table1_function_approximation.tex`
- `table2_pde_1d_comparison.tex`
- `table3_pde_2d_comparison.tex`
- `table4_param_efficiency.tex`
- `table5_convergence_summary.tex`
- `table6_grid_ablation.tex`
- `table7_depth_ablation.tex`

### CSV Files (`.csv`)
Same naming as LaTeX files but with `.csv` extension

## Quick Table Reference

| Table | Focus | Best For |
|-------|-------|----------|
| 0 | Executive Summary | Quick overview of all results |
| 1 | Function Approximation | Section 1.1 detailed results |
| 2 | 1D PDEs | Section 1.2 detailed results |
| 3 | 2D PDEs | Section 1.3 detailed results |
| 4 | Parameter Efficiency | Showing KAN's efficiency advantage |
| 5 | Training Speed | Computational cost comparison |
| 6 | Grid Ablation | Understanding KAN grid parameter |
| 7 | Depth Ablation | Understanding MLP/SIREN depth |

## Prerequisites

Before running table generation, ensure you have:

1. **Run the experiments** (at least once):
   ```bash
   cd pykan/madoc/section1
   python section1_1.py --epochs 100
   python section1_2.py --epochs 10
   python section1_3.py --epochs 10
   ```

2. **Check results exist**:
   ```bash
   ls results/sec1_results/*.pkl
   ```
   You should see files like `section1_1_*_mlp.pkl`, etc.

## Troubleshooting

### "No files found for X"
- Make sure you've run the corresponding section experiment
- Check that pickle files exist in `../results/sec1_results/`

### "ModuleNotFoundError: No module named 'tabulate'"
```bash
pip install tabulate
```

### "N/A" values in tables
- This is normal if certain models weren't trained
- Check which experiments you've completed

## Using Tables in Papers

The LaTeX tables can be directly included in papers:

```latex
\input{tables/table1_function_approximation.tex}
```

Or copy the table content from the `.tex` files.

## Customization

Edit the table scripts to:
- Change formatting (in `utils.py`)
- Modify which datasets to include
- Adjust metrics displayed
- Change LaTeX table styling

## Performance Tips

- Tables load the latest results automatically
- Generation is fast (< 1 minute for all tables)
- Can be run in parallel if needed
- CSV outputs good for plotting

## Next Steps

1. Generate tables
2. Review console output for insights
3. Include LaTeX tables in papers
4. Use CSV files for further analysis/plotting
5. Customize as needed for your specific use case
