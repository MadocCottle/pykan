# Quick Start - NCI Gadi Job Submission

## Problem You Encountered

The error `"You have not requested a project to charge for this job"` occurred because **PBS directives cannot use shell variables**.

### What Was Wrong

```bash
# This doesn't work - PBS can't interpret ${PROJECT}
PROJECT=${PROJECT:-p00}
#PBS -P ${PROJECT}
#PBS -l storage=scratch/${PROJECT}+gdata/${PROJECT}
```

### The Fix

PBS directives are parsed **before** the shell script runs, so they need **literal values**:

```bash
# This works - PBS directives have literal values
#PBS -P p00
#PBS -l storage=scratch/p00+gdata/p00
```

---

## Setup Steps

### 1. Edit Project Code (ONE TIME)

Edit `run_experiment.qsub` lines 14-15:

```bash
#PBS -P p00                      # Change 'p00' to YOUR project code
#PBS -l storage=scratch/p00+gdata/p00  # Change 'p00' to YOUR project code
```

### 2. Run Setup Script (ONE TIME)

```bash
cd /path/to/pykan/madoc
bash setup.sh
```

This creates `.venv` and installs all dependencies.

### 3. Submit Jobs

```bash
# Test run (1 CPU, 30 min, 10 epochs)
qsub -v SECTION=section1_1,EPOCHS=10,PROFILE=test run_experiment.qsub

# Full run (auto resources, 100 epochs)
qsub -v SECTION=section1_1,EPOCHS=100 run_experiment.qsub

# All sections
qsub -v SECTION=section1_1,EPOCHS=100 run_experiment.qsub
qsub -v SECTION=section1_2,EPOCHS=100 run_experiment.qsub
qsub -v SECTION=section1_3,EPOCHS=100 run_experiment.qsub
qsub -v SECTION=section2_1,EPOCHS=100 run_experiment.qsub
qsub -v SECTION=section2_2,EPOCHS=100 run_experiment.qsub
```

---

## Monitor Jobs

```bash
# Check status
qstat -u $USER

# View output (live)
tail -f pykan_experiment.o<JOBID>

# After completion, results are in:
ls -lh job_results_*/
```

---

## Resource Profiles

| Profile | CPUs | Memory | Walltime | Auto-selected for |
|---------|------|--------|----------|-------------------|
| test | 1 | 4GB | 30 min | - (manual only) |
| section1 | 12 | 48GB | 4 hours | section1_* |
| section2 | 24 | 96GB | 8 hours | section2_* |
| large | 48 | 190GB | 24 hours | - (manual only) |

Override with: `qsub -v SECTION=...,EPOCHS=...,PROFILE=large run_experiment.qsub`

---

## Results Location

Each job creates:
```
madoc/job_results_YYYYMMDD_HHMMSS/
├── job_summary.txt           # Full logs
├── MANIFEST.txt              # Quick summary
└── section1_1_results/       # Ready to use
    ├── *.pkl
    └── *.parquet
```

To use with visualizations:
```bash
cp -r job_results_*/section1_1_results/* section1/results/sec1_results/
```

---

## Troubleshooting

### "Project not requested" error
- Edit lines 14-15 in `run_experiment.qsub` with your project code
- PBS directives **cannot** use variables!

### "Virtual environment not found"
```bash
bash setup.sh
```

### "Section not found"
- Valid sections: `section1_1`, `section1_2`, `section1_3`, `section2_1`, `section2_2`
- Check capitalization and format

### Job runs out of memory/time
- Use larger profile: `PROFILE=section2` or `PROFILE=large`

---

**For full documentation, see [README_JOB_SUBMISSION.md](README_JOB_SUBMISSION.md)**
