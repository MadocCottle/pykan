# PyKAN Madoc - NCI Gadi Job Submission Guide

This guide explains how to run PyKAN experiments on NCI Gadi using the PBS job submission system.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Script Overview](#script-overview)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Resource Profiles](#resource-profiles)
- [Results](#results)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### 1. NCI Gadi Account
- Active NCI account with Gadi access
- Project allocation with compute credits
- Know your project code (e.g., `ab12`, `xy34`)

### 2. File Access
- This repository cloned/copied to your Gadi home or project directory
- Recommended location: `/home/XXX/pykan/` or `/g/data/PROJECT/pykan/`

### 3. Required Modules
- Python 3.10+ (available via `module load python3/3.10.4`)

---

## Quick Start

### 1. Initial Setup (One-time)

```bash
# Navigate to madoc directory
cd /path/to/pykan/madoc

# Run setup script to create virtual environment and install dependencies
bash setup.sh
```

The setup script will:
- Load the appropriate Python module
- Create a `.venv` virtual environment
- Install all packages from `requirements.txt`
- Verify PyKAN can be imported

### 2. Configure Project Code

**IMPORTANT**: Before submitting jobs, you must set your NCI project code.

Edit `run_experiment.qsub` and replace `p00` with your actual project code on **lines 14 and 15**:

```bash
# Lines 14-15 in run_experiment.qsub
#PBS -P p00
#PBS -l storage=scratch/p00+gdata/p00

# Change to (example):
#PBS -P ab12
#PBS -l storage=scratch/ab12+gdata/ab12
```

**Note**: PBS directives (#PBS) must have literal values and cannot use shell variables.

### 3. Submit Your First Job

```bash
# Test run (minimal resources, 10 epochs)
qsub -v SECTION=section1_1,EPOCHS=10,PROFILE=test run_experiment.qsub

# Full run (100 epochs, auto-selected resources)
qsub -v SECTION=section1_1,EPOCHS=100 run_experiment.qsub
```

### 4. Monitor Job

```bash
# Check job status
qstat -u $USER

# View output (after job starts)
tail -f pykan_experiment.o<JOBID>
```

---

## Script Overview

### `setup.sh`
**Purpose**: Initialize the Python environment

**What it does**:
- Detects if running on Gadi
- Loads Python module (on Gadi)
- Creates/updates `.venv` virtual environment
- Installs all dependencies from `requirements.txt`
- Verifies PyKAN import works

**When to run**:
- First time setup
- After updating `requirements.txt`
- If `.venv` becomes corrupted

**Usage**:
```bash
bash setup.sh
```

---

### `run_experiment.qsub`
**Purpose**: PBS job script for running experiments

**What it does**:
1. Validates parameters and checks environment
2. Loads Python module and activates `.venv`
3. Runs the specified section script with given epochs
4. Collects all results into a timestamped folder
5. Generates job summary and manifest

**Parameters**:
- **SECTION** (required): Which experiment to run
- **EPOCHS** (required): Number of training epochs
- **PROFILE** (optional): Resource allocation profile
- **PROJECT** (required): NCI project code

---

## Configuration

### Required Configuration

#### 1. Project Code

**You must edit the script** - PBS directives cannot use variables!

Edit `run_experiment.qsub` lines 14-15:
```bash
#PBS -P p00
#PBS -l storage=scratch/p00+gdata/p00

# Change to your project (example):
#PBS -P ab12
#PBS -l storage=scratch/ab12+gdata/ab12
```

**Important**: The `#PBS` directives are parsed by PBS before the script runs, so they must contain literal values, not variables.

#### 2. Email Notifications (Optional)

Edit `run_experiment.qsub` line 6:
```bash
#PBS -M your.email@example.com
```

Change to your email address to receive job status notifications.

---

## Usage Examples

### Basic Examples

```bash
# Section 1.1 with 100 epochs (auto-select resources)
qsub -v SECTION=section1_1,EPOCHS=100 run_experiment.qsub

# Section 1.2 with 50 epochs
qsub -v SECTION=section1_2,EPOCHS=50 run_experiment.qsub

# Section 2.1 with 200 epochs (auto-selects more resources)
qsub -v SECTION=section2_1,EPOCHS=200 run_experiment.qsub
```

### Test Run (Quick Validation)

```bash
# Minimal resources, 10 epochs, 30 minute limit
qsub -v SECTION=section1_1,EPOCHS=10,PROFILE=test run_experiment.qsub
```

### Override Resource Profile

```bash
# Force large profile for extended run
qsub -v SECTION=section1_1,EPOCHS=1000,PROFILE=large run_experiment.qsub

# Force test profile for quick debug
qsub -v SECTION=section2_1,EPOCHS=5,PROFILE=test run_experiment.qsub
```

### Verify Project Code is Set

```bash
# Check the PBS directives in the script
head -20 run_experiment.qsub | grep "PBS -P"

# Should show:
# #PBS -P your_project_code
```

### All Sections

To run all experiments, submit multiple jobs:

```bash
# Section 1 experiments
qsub -v SECTION=section1_1,EPOCHS=100 run_experiment.qsub
qsub -v SECTION=section1_2,EPOCHS=100 run_experiment.qsub
qsub -v SECTION=section1_3,EPOCHS=100 run_experiment.qsub

# Section 2 experiments
qsub -v SECTION=section2_1,EPOCHS=100 run_experiment.qsub
qsub -v SECTION=section2_2,EPOCHS=100 run_experiment.qsub
```

---

## Resource Profiles

The script provides four resource profiles optimized for different use cases:

| Profile | NCPUs | Memory | Walltime | Use Case |
|---------|-------|--------|----------|----------|
| **test** | 1 | 4GB | 0:30:00 | Quick verification, debugging |
| **section1** | 12 | 48GB | 4:00:00 | 1D function approximation (Section 1) |
| **section2** | 24 | 96GB | 8:00:00 | 2D PDE problems (Section 2) |
| **large** | 48 | 190GB | 24:00:00 | Extended experiments, many epochs |

### Auto-Selection

If you don't specify a `PROFILE`, it's automatically selected:
- `section1_*` → `section1` profile (12 CPUs, 48GB, 4 hours)
- `section2_*` → `section2` profile (24 CPUs, 96GB, 8 hours)

### Manual Override

Specify `PROFILE` to override auto-selection:
```bash
# Use test profile for quick validation of section2
qsub -v SECTION=section2_1,EPOCHS=10,PROFILE=test run_experiment.qsub

# Use large profile for long section1 run
qsub -v SECTION=section1_1,EPOCHS=1000,PROFILE=large run_experiment.qsub
```

---

## Results

### Results Directory Structure

Each job creates a timestamped results folder:

```
madoc/
├── job_results_20251023_143022/
│   ├── job_summary.txt              # Detailed job log
│   ├── MANIFEST.txt                 # Quick summary
│   └── section1_1_results/          # Collected results
│       ├── section1_1_20251023_142800_mlp.pkl
│       ├── section1_1_20251023_142800_mlp.parquet
│       ├── section1_1_20251023_142800_siren.pkl
│       ├── section1_1_20251023_142800_siren.parquet
│       ├── section1_1_20251023_142800_kan.pkl
│       ├── section1_1_20251023_142800_kan.parquet
│       ├── section1_1_20251023_142800_kan_pruning.pkl
│       └── section1_1_20251023_142800_kan_pruning.parquet
```

### Using Results with Visualizations

The results are collected in a format compatible with the existing visualization scripts. To use them:

```bash
# Copy results to the section's results directory
cp -r job_results_YYYYMMDD_HHMMSS/section1_1_results/* section1/results/sec1_results/

# Then run visualization scripts as normal
cd section1/visualization
python3 plot_results.py
```

### Result Files

Each experiment run produces several file types:

- **`.pkl` files**: Pickled pandas DataFrames with full results and metadata
- **`.parquet` files**: Parquet format (if available) for faster loading
- **`.pth` files**: PyTorch model state dictionaries (MLP, SIREN)
- **KAN checkpoints**: KAN model checkpoints (if models were saved)

### Accessing Results

#### View Job Summary
```bash
cat job_results_YYYYMMDD_HHMMSS/job_summary.txt
```

#### View Quick Manifest
```bash
cat job_results_YYYYMMDD_HHMMSS/MANIFEST.txt
```

#### Load Results in Python
```python
import pandas as pd
import sys
from pathlib import Path

# Add pykan to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load results
timestamp = "20251023_142800"
mlp_df = pd.read_pickle(f'section1_1_{timestamp}_mlp.pkl')
kan_df = pd.read_pickle(f'section1_1_{timestamp}_kan.pkl')

# Metadata is stored in DataFrame attributes
print(mlp_df.attrs)  # Shows epochs, device, etc.
```

---

## Troubleshooting

### Issue: "PROJECT not set"

**Problem**: You see error `PROJECT not set! Edit the script or pass via: qsub -v PROJECT=...`

**Solution**:
- Edit `run_experiment.qsub` line 24 and set your project code, OR
- Pass it via command line: `qsub -v PROJECT=ab12,...`

---

### Issue: "Virtual environment not found"

**Problem**: Error message `Virtual environment not found at .venv. Please run setup.sh first.`

**Solution**:
```bash
cd /path/to/pykan/madoc
bash setup.sh
```

---

### Issue: "Section script not found"

**Problem**: Error `Section script not found: section1/section1_X.py`

**Solution**:
- Check you're in the correct directory (`madoc/`)
- Verify the section name is correct (valid: `section1_1`, `section1_2`, `section1_3`, `section2_1`, `section2_2`)
- Ensure section scripts exist in `section1/` and `section2/` directories

---

### Issue: "Failed to load Python module"

**Problem**: Error when loading `python3/3.10.4` module

**Solution**:
```bash
# Check available Python modules
module avail python3

# Load a compatible version (3.10+)
module load python3/3.10.4  # or newer
```

---

### Issue: Job Runs Out of Memory

**Problem**: Job fails with memory errors

**Solution**:
- Use a larger profile: `PROFILE=section2` or `PROFILE=large`
- Reduce number of epochs or datasets if possible
- Check section2 experiments need more memory (use `section2` profile)

---

### Issue: Job Runs Out of Time

**Problem**: Job terminated due to walltime limit

**Solution**:
- Use `PROFILE=large` for 24-hour walltime
- Reduce number of epochs
- For section 2, walltime may need extension beyond 8 hours

---

### Issue: "Failed to import PyKAN"

**Problem**: Error when verifying PyKAN import

**Solution**:
1. Check directory structure:
   ```
   pykan/
   ├── kan/           # PyKAN package
   └── madoc/         # Experiments
       ├── setup.sh
       └── run_experiment.qsub
   ```

2. Verify parent directory contains `kan/` package
3. Re-run `setup.sh` to ensure dependencies are installed

---

### Issue: No Results Found

**Problem**: Error `No results found in section1/results/sec1_results`

**Solution**:
- Check if the experiment actually ran successfully
- Look at `job_summary.txt` for error messages
- Verify the section script completed without errors
- Check if results directory exists and has write permissions

---

### Checking Job Status

```bash
# View your jobs
qstat -u $USER

# View detailed job info
qstat -f <JOBID>

# View job output (while running)
tail -f pykan_experiment.o<JOBID>

# View job errors
tail -f pykan_experiment.e<JOBID>  # If -j oe not used
```

---

### Getting Help

1. **Check PBS output**: `pykan_experiment.o<JOBID>` contains all output
2. **Check job summary**: `job_results_*/job_summary.txt` has detailed logs
3. **Check section logs**: Original section results may have error messages
4. **NCI Help**: https://opus.nci.org.au/display/Help/

---

## Advanced Usage

### Custom Python Module

If you need a different Python version:

Edit both `setup.sh` and `run_experiment.qsub` to use a different module:
```bash
module load python3/3.11.0  # or your preferred version
```

### Custom Storage Locations

If your project uses additional storage:

Edit `run_experiment.qsub` line 18:
```bash
#PBS -l storage=scratch/${PROJECT}+gdata/${PROJECT}+gdata/ab00
```

### Running Locally (Non-Gadi)

The scripts detect if running on Gadi. To run locally:

```bash
# Setup (skips module loading)
bash setup.sh

# Run directly (without PBS)
source .venv/bin/activate
python3 section1/section1_1.py --epochs 100
```

---

## Summary of Available Sections

| Section | Description | Default Profile |
|---------|-------------|-----------------|
| `section1_1` | 1D Function Approximation - Basic | section1 (12 CPU, 4h) |
| `section1_2` | 1D Function Approximation - Variant 2 | section1 (12 CPU, 4h) |
| `section1_3` | 1D Function Approximation - Variant 3 | section1 (12 CPU, 4h) |
| `section2_1` | 2D Poisson PDE - Optimizer Comparison | section2 (24 CPU, 8h) |
| `section2_2` | 2D PDE - Extended Analysis | section2 (24 CPU, 8h) |

---

## Job Submission Checklist

Before submitting your first job:

- [ ] Run `setup.sh` to create virtual environment
- [ ] Set `PROJECT` code in `run_experiment.qsub` (or pass via `-v`)
- [ ] Optionally set email in `run_experiment.qsub` for notifications
- [ ] Test with `PROFILE=test` first
- [ ] Verify you have sufficient compute credits

---

## Additional Resources

- **NCI Gadi Documentation**: https://opus.nci.org.au/display/Help/0.+Welcome+to+Gadi
- **PBS Directives Guide**: https://opus.nci.org.au/display/Help/PBS+Directives+Explained
- **Queue Limits**: https://opus.nci.org.au/display/Help/Queue+Limits
- **Python on Gadi**: https://opus.nci.org.au/display/Help/Python

---

## Questions or Issues?

If you encounter problems not covered in this guide:
1. Check the `job_summary.txt` file in your results folder
2. Review PBS output logs
3. Consult NCI Gadi documentation
4. Contact NCI support: https://opus.nci.org.au/

---

**Happy Computing!**
