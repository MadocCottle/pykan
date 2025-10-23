# Pipeline Setup & Management Guide

This guide explains the complete pipeline for running experiments, managing results, and generating visualizations.

## Table of Contents
- [File Naming Conventions](#file-naming-conventions)
- [Running Experiments](#running-experiments)
- [Result Management](#result-management)
- [Gadi Workflow](#gadi-workflow)
- [Cleanup & Maintenance](#cleanup--maintenance)
- [Troubleshooting](#troubleshooting)

---

## File Naming Conventions

### Result Files

All result files follow a consistent naming pattern that includes:
- Section identifier (e.g., `section1_1`, `section2_1`)
- Timestamp (`YYYYMMDD_HHMMSS`)
- Epoch count (`e100`, `e1000`, etc.) - **NEW**
- Model type (`mlp`, `siren`, `kan`, `kan_pruning`, `adam`, `lbfgs`, `lm`)

**Format:**
```
{section}_{timestamp}_e{epochs}_{model_type}.pkl
```

**Examples:**
```
section1_1_20251024_143052_e100_mlp.pkl          # Section 1.1, 100 epochs, MLP model
section1_1_20251024_143052_e100_kan.pkl          # Section 1.1, 100 epochs, KAN model
section2_1_20251024_150302_e1000_adam.pkl        # Section 2.1, 1000 epochs, Adam optimizer
```

### Checkpoint Files

KAN models create multiple checkpoint files per dataset:
```
section1_1_20251024_143052_e100_kan_0_final_state
section1_1_20251024_143052_e100_kan_0_final_config.yml
section1_1_20251024_143052_e100_kan_0_final_cache_data
section1_1_20251024_143052_e100_kan_0_at_threshold_state
section1_1_20251024_143052_e100_kan_0_at_threshold_config.yml
section1_1_20251024_143052_e100_kan_0_at_threshold_cache_data
```

Where:
- `0` is the dataset index
- `final` is the checkpoint at the end of training
- `at_threshold` is the checkpoint when KAN reaches the interpolation threshold

### Output Files

Tables and visualizations are saved in timestamped subdirectories:
```
madoc/section1/tables/outputs/20251024_150000_results/
madoc/section1/visualization/outputs/20251024_150000_results/
```

---

## Running Experiments

### Local Runs

**Section 1 (Function Approximation & PDEs):**
```bash
# Run with 100 epochs (default)
python3 pykan/madoc/section1/section1_1.py --epochs 100

# Run with 1000 epochs
python3 pykan/madoc/section1/section1_1.py --epochs 1000

# Quick test with fewer epochs per grid
python3 pykan/madoc/section1/section1_1.py --epochs 50 --steps_per_grid 100
```

**Section 2 (Optimizer Comparison):**
```bash
# Run with 10 epochs (default)
python3 pykan/madoc/section2/section2_1.py --epochs 10

# Run with 100 epochs
python3 pykan/madoc/section2/section2_1.py --epochs 100
```

### Results Location

Results are automatically saved to:
```
pykan/madoc/section1/results/sec1_results/    # For section1_1
pykan/madoc/section1/results/sec2_results/    # For section1_2
pykan/madoc/section1/results/sec3_results/    # For section1_3
pykan/madoc/section2/results/sec1_results/    # For section2_1
```

---

## Result Management

### Finding Results

**Latest results are automatically selected** by table and visualization scripts:
```python
# In tables/utils.py and visualization scripts
from utils import load_latest_results

# Automatically finds most recent run for section1_1
results = load_latest_results('section1_1')
```

**Manual loading by timestamp:**
```python
from utils import load_run

# Load specific run
results, meta = load_run('section1_1', '20251024_143052')

# Check metadata
print(f"Epochs: {meta['epochs']}")
print(f"Timestamp: {meta['timestamp']}")
print(f"Device: {meta['device']}")
```

### Identifying Runs by Epochs

The epoch count is now in the filename AND in the metadata:
```bash
# List all 100-epoch runs
ls madoc/section1/results/sec1_results/*_e100_*.pkl

# List all 1000-epoch runs
ls madoc/section1/results/sec1_results/*_e1000_*.pkl
```

---

## Gadi Workflow

### 1. Submitting Jobs

```bash
# Submit job with epoch specification
qsub -v SECTION=section1_1,EPOCHS=100 run_experiment.qsub
qsub -v SECTION=section1_2,EPOCHS=1000 run_experiment.qsub
qsub -v SECTION=section2_1,EPOCHS=50 run_experiment.qsub
```

**The qsub script:**
- Records epochs in the MANIFEST.txt
- Saves results with epoch count in filenames
- Creates timestamped `job_results_YYYYMMDD_HHMMSS/` directory

### 2. Fetching Results

```bash
# Fetch from Gadi and auto-extract
./fetch.sh

# Preview what would be extracted
./fetch.sh --dry-run

# Fetch without extraction
./fetch.sh --no-extract

# Archive processed results after extraction
./fetch.sh --archive
```

**What fetch.sh does:**
1. Copies `job_results_*/` directories from Gadi to `~/Desktop/landing/`
2. Copies PBS log files (`.o*`) to landing
3. Runs `extract_gadi_results.py` to integrate results into local directories

### 3. Extraction Process

The extraction script (`extract_gadi_results.py`):
- Reads `MANIFEST.txt` to determine section and epochs
- Copies results to appropriate `madoc/section*/results/` directory
- Maintains the timestamp-based naming
- Preserves epoch information in filenames

**Manual extraction:**
```bash
# Extract from landing directory
python3 extract_gadi_results.py

# Preview without extracting
python3 extract_gadi_results.py --dry-run

# Archive after extraction
python3 extract_gadi_results.py --archive
```

---

## Cleanup & Maintenance

### Cleaning Local Results

**Section-specific cleanup:**
```bash
# Keep only 3 most recent runs for section1_1
python3 pykan/madoc/section1/cleanup_results.py --keep-latest 3

# Keep only latest 100-epoch run, delete all others
python3 pykan/madoc/section1/cleanup_results.py --epochs 100 --keep-latest 1

# Delete all runs before Oct 20, 2025
python3 pykan/madoc/section1/cleanup_results.py --before 20251020

# Preview what would be deleted
python3 pykan/madoc/section1/cleanup_results.py --keep-latest 2 --dry-run
```

**Same for Section 2:**
```bash
python3 pykan/madoc/section2/cleanup_results.py --keep-latest 3
```

### Cleaning Landing Directory

```bash
# Keep only 5 most recent job_results
python3 pykan/cleanup_landing.py --keep-latest 5

# Archive before deleting
python3 pykan/cleanup_landing.py --keep-latest 3 --archive

# Clean PBS log files only
python3 pykan/cleanup_landing.py --clean-pbs-logs

# Preview what would be deleted
python3 pykan/cleanup_landing.py --keep-latest 3 --dry-run
```

### Git Exclusions

Results, outputs, and model files are automatically excluded from git via `.gitignore`:
- `madoc/*/results/` - All training results
- `madoc/*/visualization/outputs/` - Generated plots
- `madoc/*/tables/outputs/` - Generated tables
- `*.pkl`, `*.pth`, `*_state`, `*_config.yml`, `*_cache_data` - Model files
- `job_results_*/` - Gadi job results
- `*.o*` - PBS log files

**Never commit results to git!** They are large and change frequently.

---

## Generating Tables & Visualizations

### Tables

```bash
# Generate all tables for section1
cd pykan/madoc/section1/tables
python3 generate_all_tables.py

# Generate specific table
python3 table1_function_approximation.py
```

Tables are saved to:
```
madoc/section1/tables/outputs/{timestamp}_results/
```

### Visualizations

```bash
# Generate visualizations for section1_1
cd pykan/madoc/section1/visualization
python3 plot_best_loss.py
python3 plot_function_fit.py
python3 plot_heatmap_2d.py
```

Plots are saved to:
```
madoc/section1/visualization/outputs/{timestamp}_results/
```

**Important:** Visualization scripts automatically use the **latest** results by default.

---

## Troubleshooting

### "No results found"

**Cause:** No result files match the expected pattern.

**Solutions:**
1. Check results directory exists:
   ```bash
   ls -la pykan/madoc/section1/results/sec1_results/
   ```

2. Verify files have correct naming:
   ```bash
   ls pykan/madoc/section1/results/sec1_results/section1_1_*.pkl
   ```

3. Check if old format (without epochs):
   - Script should auto-detect and load old format
   - Consider re-running with new format

### "Can't tell which run is 100 vs 1000 epochs"

**Solution:**
- New runs include epochs in filename: `*_e100_*.pkl` vs `*_e1000_*.pkl`
- For old runs without epochs in filename, check metadata:
  ```python
  import pandas as pd
  df = pd.read_pickle('section1_1_20251024_143052_mlp.pkl')
  print(df.attrs['epochs'])  # Check stored metadata
  ```

### Git is tracking result files

**This shouldn't happen with the updated `.gitignore`.**

If it does:
```bash
# Remove from git (keeps local files)
git rm --cached madoc/section1/results/

# Check .gitignore is correct
cat .gitignore | grep results
```

### PBS logs accumulating in landing/

**Solution:**
```bash
# Clean up PBS logs
python3 pykan/cleanup_landing.py --clean-pbs-logs
```

### Can't find specific epoch run

**Find by filename:**
```bash
# Find all 100-epoch runs
find madoc/section1/results -name "*_e100_*.pkl"

# Find all 1000-epoch runs
find madoc/section1/results -name "*_e1000_*.pkl"
```

**Load specific run:**
```python
from utils import load_run
results, meta = load_run('section1_1', '20251024_143052')
print(f"This run used {meta['epochs']} epochs")
```

---

## Best Practices

### Before Running Experiments

1. ✅ Check if similar run already exists
2. ✅ Decide on epoch count (100 for quick tests, 1000+ for final)
3. ✅ Clean up old results if disk space is low

### After Running Experiments

1. ✅ Verify results were saved: `ls -lh madoc/section*/results/`
2. ✅ Check metadata is correct: Load and inspect `df.attrs`
3. ✅ Generate tables/visualizations while results are fresh
4. ✅ Clean up old runs if no longer needed

### For Gadi Workflow

1. ✅ Always specify `EPOCHS` parameter in qsub
2. ✅ Fetch results promptly after job completion
3. ✅ Clean landing directory regularly
4. ✅ Archive important runs before cleanup

### For Collaboration

1. ✅ Never commit results to git
2. ✅ Share results via separate mechanism (not git)
3. ✅ Document which epoch count was used for published results
4. ✅ Keep at least one copy of final results for paper

---

## Quick Reference

### Common Commands

```bash
# Run experiment
python3 pykan/madoc/section1/section1_1.py --epochs 100

# Fetch from Gadi
./fetch.sh

# Generate tables
python3 pykan/madoc/section1/tables/generate_all_tables.py

# Clean results (keep 3 latest)
python3 pykan/madoc/section1/cleanup_results.py --keep-latest 3

# Clean landing (keep 5 latest)
python3 pykan/cleanup_landing.py --keep-latest 5

# Preview cleanup (dry run)
python3 pykan/madoc/section1/cleanup_results.py --keep-latest 3 --dry-run
```

### File Locations

```
pykan/
├── madoc/
│   ├── section1/
│   │   ├── section1_1.py              # Training script
│   │   ├── results/
│   │   │   └── sec1_results/          # Results stored here
│   │   ├── tables/
│   │   │   ├── *.py                   # Table generation scripts
│   │   │   └── outputs/               # Generated tables (gitignored)
│   │   ├── visualization/
│   │   │   ├── *.py                   # Plot generation scripts
│   │   │   └── outputs/               # Generated plots (gitignored)
│   │   ├── utils/
│   │   │   └── io.py                  # save_run(), load_run()
│   │   └── cleanup_results.py         # Cleanup script
│   └── section2/
│       └── ...                         # Same structure
├── cleanup_landing.py                  # Landing cleanup script
├── extract_gadi_results.py             # Result extraction script
└── fetch.sh                            # Fetch from Gadi script
```

---

## Summary

The pipeline is designed to:
1. ✅ **Track epochs in filenames** - Easy to distinguish runs
2. ✅ **Auto-select latest results** - Tables/viz just work
3. ✅ **Keep git clean** - No results in repo
4. ✅ **Easy cleanup** - Scripts handle old result management
5. ✅ **Gadi integration** - Seamless fetch and extract
6. ✅ **Backward compatible** - Old results still load

**Questions?** Check the documentation in each subdirectory:
- `madoc/section1/tables/README.md`
- `madoc/section1/visualization/README.md`
- `madoc/section1/utils/` (check docstrings)
