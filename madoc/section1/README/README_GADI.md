# Section 1 Experiments on NCI Gadi

This directory contains scripts for running Section 1 KAN experiments on the NCI Gadi supercomputer.

## Quick Start

### 1. Initial Setup

First time only - set up the Python environment:

```bash
cd /path/to/pykan/madoc/section1
bash setup.sh
```

This will:
- Load the Python 3.10.4 module
- Create a virtual environment in `.venv/`
- Install all dependencies from `../requirements.txt`

### 2. Configure PBS Script

Edit `run_section1.pbs` and update these lines with your details:

```bash
#PBS -P YOUR_PROJECT_CODE      # Replace with your NCI project code
#PBS -l storage=scratch/YOUR_PROJECT_CODE  # Replace with your project code
#PBS -M your.email@example.com  # Replace with your email
```

### 3. Submit Jobs

**Option A: Use wrapper scripts (recommended)**

```bash
# Run Section 1.1 (Function Approximation) with 100 epochs
bash submit_section1_1.sh 100

# Run Section 1.2 (1D Poisson PDE) with 200 epochs
bash submit_section1_2.sh 200

# Run Section 1.3 (2D Poisson PDE) with 150 epochs
bash submit_section1_3.sh 150

# Or use defaults (100, 200, 200 epochs respectively)
bash submit_section1_1.sh
bash submit_section1_2.sh
bash submit_section1_3.sh
```

**Option B: Use qsub directly**

```bash
# IMPORTANT: -v flag must come BEFORE the PBS script name
# Run Section 1.1 (Function Approximation) with 100 epochs
qsub -v SCRIPT=section1_1.py,EPOCHS=100 run_section1.pbs

# Run Section 1.2 (1D Poisson PDE) with 200 epochs
qsub -v SCRIPT=section1_2.py,EPOCHS=200 run_section1.pbs

# Run Section 1.3 (2D Poisson PDE) with 150 epochs
qsub -v SCRIPT=section1_3.py,EPOCHS=150 run_section1.pbs
```

### 4. Monitor Jobs

```bash
# Check job status
qstat -u $USER

# View job output (while running or after completion)
cat section1_experiment.o<JOBID>

# View experiment logs
ls logs/
tail -f logs/section1_1_100epochs_<JOBID>.log
```

## Available Scripts

| Script | Description | Recommended Epochs |
|--------|-------------|-------------------|
| `section1_1.py` | Function Approximation (1D functions with varying frequencies) | 100-200 |
| `section1_2.py` | 1D Poisson PDE | 200-500 |
| `section1_3.py` | 2D Poisson PDE | 200-500 |

## PBS Script Options

The PBS script accepts two variables:

- **SCRIPT**: Which Python script to run
  - Options: `section1_1.py`, `section1_2.py`, `section1_3.py`
  - Default: `section1_1.py`

- **EPOCHS**: Number of training epochs
  - Default: `10`
  - Recommended: 100-500 depending on the experiment

## Resource Allocation

Current PBS settings (in `run_section1.pbs`):

```bash
#PBS -q normal          # Queue
#PBS -l walltime=24:00:00  # 24 hours
#PBS -l mem=32GB        # 32 GB memory
#PBS -l ncpus=4         # 4 CPU cores
#PBS -l jobfs=10GB      # 10 GB local scratch
```

Adjust these based on your needs:
- For quick tests: `walltime=1:00:00`, `mem=16GB`, `ncpus=2`
- For large experiments: `walltime=48:00:00`, `mem=64GB`, `ncpus=8`

## Output Files

After running, you'll find:

```
section1/
├── results/              # Experiment results (JSON, pickle files)
│   └── section1_1_results_<timestamp>.json
├── logs/                 # Detailed execution logs
│   └── section1_1_100epochs_<JOBID>.log
├── sec1_results/        # Model checkpoints and training history
│   └── kan_models_<timestamp>/
└── section1_experiment.o<JOBID>  # PBS job output
```

## Troubleshooting

### Error: "Virtual environment not found"

Run the setup script:
```bash
bash setup.sh
```

### Error: "cannot import name 'KAN'"

PyKAN should be imported from the parent directory automatically. Check that the directory structure is:
```
pykan/
├── kan/              # PyKAN source code
└── madoc/
    └── section1/     # This directory
```

### Job fails immediately

Check the PBS output file:
```bash
cat section1_experiment.o<JOBID>
```

Look for error messages in the first few lines.

### Out of memory

Edit `run_section1.pbs` and increase memory:
```bash
#PBS -l mem=64GB
```

### Job takes too long

Reduce epochs or increase walltime:
```bash
qsub run_section1.pbs -v SCRIPT=section1_1.py,EPOCHS=50
```

Or in PBS file:
```bash
#PBS -l walltime=48:00:00
```

## Advanced Usage

### Running multiple experiments in parallel

Submit multiple jobs with different parameters:

```bash
#!/bin/bash
# Submit all section1 experiments

for script in section1_1.py section1_2.py section1_3.py; do
    for epochs in 100 200 500; do
        echo "Submitting $script with $epochs epochs"
        qsub run_section1.pbs -v SCRIPT=$script,EPOCHS=$epochs
    done
done
```

### Custom job names

Add `-N` flag to qsub:
```bash
qsub -N section1_1_100ep run_section1.pbs -v SCRIPT=section1_1.py,EPOCHS=100
```

### Email notifications

The PBS script is configured to send emails (via `#PBS -m abe`):
- `a`: When job aborts
- `b`: When job begins
- `e`: When job ends

Disable by removing or commenting out:
```bash
# #PBS -M your.email@example.com
# #PBS -m abe
```

## Environment Details

- **Python Version**: 3.10.4 (via `module load python3/3.10.4`)
- **Virtual Environment**: `.venv/` (created by setup.sh)
- **Dependencies**: Installed from `../requirements.txt`
- **PyKAN**: Imported locally via sys.path (not installed as package)

## Directory Structure

```
section1/
├── setup.sh              # Environment setup script
├── run_section1.pbs      # PBS job script
├── README_GADI.md        # This file
├── section1_1.py         # Function approximation experiments
├── section1_2.py         # 1D Poisson PDE experiments
├── section1_3.py         # 2D Poisson PDE experiments
├── utils/                # Utility modules
│   ├── data_funcs.py     # Data generation functions
│   ├── model_tests.py    # Model training utilities
│   └── ...
├── .venv/                # Virtual environment (created by setup.sh)
├── results/              # Experiment outputs (created during runs)
└── logs/                 # Execution logs (created during runs)
```

## Contact

For issues specific to Gadi or PBS, consult:
- NCI Gadi User Guide: https://opus.nci.org.au/display/Help/Gadi+User+Guide
- NCI Help: help@nci.org.au
