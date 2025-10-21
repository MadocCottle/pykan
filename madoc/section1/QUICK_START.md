# Quick Start Guide - Section 1 on Gadi

## One-Time Setup

```bash
# 1. Navigate to section1 directory
cd /path/to/pykan/madoc/section1

# 2. Run setup script
bash setup.sh

# 3. Edit PBS script with your project code
# Replace YOUR_PROJECT_CODE and your.email@example.com
nano run_section1.pbs
```

## Edit These Lines in run_section1.pbs

```bash
#PBS -P YOUR_PROJECT_CODE                    # Line 2
#PBS -l storage=scratch/YOUR_PROJECT_CODE   # Line 8
#PBS -M your.email@example.com              # Line 10
```

## Submit Jobs

```bash
# Section 1.1: Function Approximation (100 epochs)
qsub run_section1.pbs -v SCRIPT=section1_1.py,EPOCHS=100

# Section 1.2: 1D Poisson PDE (200 epochs)
qsub run_section1.pbs -v SCRIPT=section1_2.py,EPOCHS=200

# Section 1.3: 2D Poisson PDE (200 epochs)
qsub run_section1.pbs -v SCRIPT=section1_3.py,EPOCHS=200
```

## Check Status

```bash
# View job queue
qstat -u $USER

# View job output
cat section1_experiment.o<JOBID>

# View detailed log
tail -f logs/section1_1_100epochs_<JOBID>.log
```

## Results Location

- **JSON Results**: `results/section1_X_results_<timestamp>.json`
- **Execution Logs**: `logs/section1_X_<epochs>epochs_<JOBID>.log`
- **Model Files**: `sec1_results/kan_models_<timestamp>/`

## Common Issues

**"Virtual environment not found"**
→ Run `bash setup.sh`

**"Module not found"**
→ Check that you're in the correct directory and setup.sh completed successfully

**Job fails immediately**
→ Check `section1_experiment.o<JOBID>` for error messages

For detailed documentation, see [README_GADI.md](README_GADI.md)
