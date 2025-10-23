# Section 2 Table Generation

This directory contains scripts for generating comparison tables from Section 2 experimental results.

## Available Tables

### Table 1: Optimizer Comparison (`table1_optimizer_comparison.py`)
Compares LBFGS, Adam, and Levenberg-Marquardt optimizers on 2D Poisson PDE problems.

**Columns:**
- Dataset name
- Dense MSE for each optimizer (LBFGS, Adam, LM)
- Best optimizer per dataset
- Time ratio (relative to LBFGS)

**Usage:**
```bash
python table1_optimizer_comparison.py
```

### Table 2: Adaptive Density (`table2_adaptive_density.py`)
Compares baseline, adaptive-only, and adaptive+regular densification strategies.

**Columns:**
- Dataset name
- Dense MSE for each approach
- Best approach per dataset
- Improvement over baseline (%)

**Usage:**
```bash
python table2_adaptive_density.py
```

### Table 3: Merge_KAN Analysis (`table3_merge_kan.py`)
Analyzes expert pool statistics and merge effectiveness.

**Columns:**
- Dataset name
- Number of experts trained/selected
- Unique dependency patterns
- Merged KAN MSE vs best solo expert MSE
- Merge gain (%)
- Parameters

**Usage:**
```bash
python table3_merge_kan.py
```

### Table 4: Section 2 Executive Summary (`table4_section2_summary.py`)
High-level comparison of all Section 2 approaches (one row per approach).

**Columns:**
- Approach name
- Section number
- Average Dense MSE
- Average parameters
- Average time
- Key finding

**Usage:**
```bash
python table4_section2_summary.py
```

## Generate All Tables

Use the runner script to generate all tables at once:

```bash
# Generate all tables
python generate_all_tables.py

# Generate specific tables only
python generate_all_tables.py --tables 1 3

# Show help
python generate_all_tables.py --help
```

## Output Formats

Each table script generates three outputs:

1. **CSV** (`table{N}_*.csv`) - For spreadsheets and data analysis
2. **LaTeX** (`table{N}_*.tex`) - For inclusion in papers/reports
3. **Console** - Formatted ASCII table for quick viewing

## Data Requirements

### Section 2.1 (Table 1)
- Results from: `python ../section2_1.py --epochs 100`
- Files expected: `results/sec1_results/section2_1_*_{adam,lbfgs,lm}.pkl`

### Section 2.2 (Table 2)
- Results from: `python ../section2_2.py --epochs 100`
- Files expected: `results/sec2_results/section2_2_*_{baseline,adaptive_only,adaptive_regular}.pkl`

### Section 2.3 (Table 3)
- Results from: `python ../section2_3.py --n-seeds 5`
- Files expected: `results/sec3_results/section2_3_*_{summary,experts,selected_experts,grid_history}.pkl`

### All Sections (Table 4)
- Requires results from all three sections above

## Notes

- **Iso-compute comparison**: When checkpoint metadata is available, tables use the "at_threshold" checkpoint for fair comparison (when reference approach reaches interpolation threshold)
- **Automatic timestamp**: Scripts automatically use the latest results if no timestamp is specified
- **Graceful fallbacks**: If checkpoint metadata is not available, scripts fall back to final results
- **Error handling**: Missing data is displayed as "N/A" with warnings

## Troubleshooting

### "No results found" error
Run the corresponding experiment script first:
```bash
cd /Users/main/Desktop/my_pykan/pykan/madoc/section2
python section2_1.py --epochs 100  # For table 1
python section2_2.py --epochs 100  # For table 2
python section2_3.py --n-seeds 5   # For table 3
```

### Checkpoint metadata not found
The two-checkpoint strategy requires running experiments with the checkpoint-aware code. If you have old results without checkpoints, tables will use final results instead (less fair comparison).

### Import errors
Make sure you're running from the tables directory, or the utils module is in your Python path.

## Example Workflow

```bash
# 1. Run experiments (if not already done)
cd /Users/main/Desktop/my_pykan/pykan/madoc/section2
python section2_1.py --epochs 100
python section2_2.py --epochs 100
python section2_3.py --n-seeds 5

# 2. Generate all tables
cd tables
python generate_all_tables.py

# 3. View results
ls -lh *.csv *.tex

# 4. Use in paper
# Copy *.tex files to your LaTeX project
# Include with: \input{table1_optimizer_comparison.tex}
```

## LaTeX Integration

To include tables in your LaTeX document:

```latex
% In your preamble
\usepackage{booktabs}

% In your document
\begin{table}[ht]
    \input{table1_optimizer_comparison.tex}
\end{table}
```

Tables are formatted for standard LaTeX article class. Adjust column formats in the scripts if needed for specific journal templates.
