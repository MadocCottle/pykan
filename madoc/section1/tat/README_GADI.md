# Running madoc on NCI Gadi

This guide covers how to run madoc experiments on NCI's Gadi supercomputer.

## Quick Start

```bash
# Clone/upload madoc to Gadi
cd /g/data/p00/mc2303/
git clone <your-repo> madoc
cd madoc

# Submit a job with default settings (10 epochs, 4 hours)
qsub gadi_section1_1.pbs

# Submit with custom epochs and walltime
qsub -v EPOCHS=100,WALLTIME=8:00:00 gadi_section1_1.pbs

# Submit all three sections
qsub -v EPOCHS=100,WALLTIME=10:00:00 gadi_section1_1.pbs
qsub -v EPOCHS=100,WALLTIME=10:00:00 gadi_section1_2.pbs
qsub -v EPOCHS=100,WALLTIME=10:00:00 gadi_section1_3.pbs
```

## Available PBS Scripts

| Script | Description | Default Epochs | Default Walltime |
|--------|-------------|----------------|------------------|
| `gadi_section1_1.pbs` | Section 1.1: 1D Function Approximation | 10 | 4:00:00 |
| `gadi_section1_2.pbs` | Section 1.2: 2D Function Approximation | 10 | 4:00:00 |
| `gadi_section1_3.pbs` | Section 1.3: High-D Function Approximation | 10 | 4:00:00 |

## Job Submission Examples

### Quick Test (10 epochs, 1 hour)
Fast test to verify everything works:
```bash
qsub -v EPOCHS=10,WALLTIME=1:00:00 gadi_section1_1.pbs
```

### Medium Run (100 epochs, 8 hours)
Moderate training for development:
```bash
qsub -v EPOCHS=100,WALLTIME=8:00:00 gadi_section1_1.pbs
```

### Production Run (1000 epochs, 48 hours)
Full training for publication:
```bash
qsub -v EPOCHS=1000,WALLTIME=48:00:00 gadi_section1_1.pbs
```

### Submit All Sections
Run all three experiments in parallel:
```bash
for script in gadi_section1_{1,2,3}.pbs; do
    qsub -v EPOCHS=500,WALLTIME=24:00:00 $script
done
```

## Monitoring Jobs

### Check job status
```bash
qstat -u mc2303              # View your jobs
qstat -f <job_id>            # Detailed job info
qstat -sw                    # All jobs in queue
```

### View job output
```bash
# Output is written to section1_X_<jobid>.log
tail -f section1_1_*.log     # Follow output in real-time
less section1_1_*.log        # View completed output
```

### Check results
```bash
ls -lht section1/sec1_results/    # List results by time
```

## Resource Allocation

### Current Default (CPU)
- **Queue**: normal
- **CPUs**: 4 cores
- **Memory**: 32 GB RAM
- **Walltime**: 4 hours (customizable)
- **Project**: p00
- **Storage**: gdata/p00

### Recommended Settings by Run Type

| Run Type | Epochs | Walltime | Queue | Resources |
|----------|--------|----------|-------|-----------|
| Quick test | 10 | 1:00:00 | normal | 4 CPUs, 32GB |
| Development | 100 | 8:00:00 | normal | 4 CPUs, 32GB |
| Production | 1000 | 48:00:00 | normal | 4 CPUs, 32GB |
| GPU test | 100 | 4:00:00 | gpuvolta | 1 GPU, 4 CPUs, 32GB |
| GPU production | 1000 | 24:00:00 | gpuvolta | 1 GPU, 4 CPUs, 32GB |

## GPU Usage

To run on GPU nodes (much faster training):

### Option 1: Quick GPU Submission
Use the provided GPU queue:
```bash
# Modify the PBS script temporarily
qsub -v EPOCHS=1000,WALLTIME=12:00:00 -q gpuvolta gadi_section1_1.pbs
```

### Option 2: Edit PBS Script for GPU
Edit the `.pbs` file and change:

**From (CPU):**
```bash
#PBS -q normal
#PBS -l ncpus=4
#PBS -l mem=32GB
```

**To (GPU):**
```bash
#PBS -q gpuvolta
#PBS -l ncpus=4
#PBS -l ngpus=1
#PBS -l mem=32GB
```

Also add after `module load python3/3.10.4`:
```bash
module load cuda/11.8.0
```

And uncomment GPU PyTorch in `requirements.txt`:
```bash
# Uncomment these lines:
--extra-index-url https://download.pytorch.org/whl/cu118
torch==2.2.2+cu118
```

## Environment Setup

### Virtual Environment
The PBS scripts automatically create and manage a Python virtual environment at:
```
/g/data/p00/mc2303/madoc/.venv_gadi
```

This is shared across all jobs to save time on subsequent runs.

### Rebuild Environment
If you need to rebuild the environment (e.g., after updating requirements.txt):
```bash
rm -rf /g/data/p00/mc2303/madoc/.venv_gadi
qsub gadi_section1_1.pbs  # Will recreate on next run
```

## Results Location

Results are saved to:
```
/g/data/p00/mc2303/madoc/section1/sec1_results/
```

Files include:
- `section1_X_results_<timestamp>.json` - Main results in JSON format
- `section1_X_results_<timestamp>.pkl` - Full results with model objects (pickle)
- Model checkpoint files (if saved)

## Troubleshooting

### Job Fails Immediately
Check the log file for errors:
```bash
cat section1_1_*.log
```

Common issues:
- **PyKAN import fails**: Check that parent directory (`../`) contains pykan
- **Out of memory**: Increase `#PBS -l mem=64GB`
- **Walltime exceeded**: Increase `WALLTIME` or reduce `EPOCHS`

### Job Stuck in Queue
Check queue status:
```bash
qstat -sw                      # View queue
showq -o                       # Show queue priorities
```

Gadi queue priorities:
1. `express` (highest priority, limited SU)
2. `normal` (standard priority)
3. `copyq` (for data transfer only)

### Slow Performance
**For CPU runs**: Consider using fewer, more focused epochs
**For GPU runs**: Much faster, but limited queue availability

Benchmark (approximate):
- CPU (4 cores): ~2-5 minutes per 10 epochs
- GPU (1 V100): ~10-30 seconds per 10 epochs

### Storage Issues
If you run out of space in `/g/data/p00/`:

Use scratch storage:
```bash
# In the PBS script, change:
#PBS -l storage=gdata/p00+scratch/p00

# And save results to scratch:
mkdir -p /scratch/p00/mc2303/madoc_results
# Then copy results there in the PBS script
```

## Advanced Usage

### Custom Python Environment
If you want to use a pre-existing conda environment:

Edit the PBS script and replace the venv section with:
```bash
module load conda/analysis3
conda activate your_env_name
```

### Parallel Job Arrays
To run parameter sweeps:
```bash
# Create array job (example)
qsub -J 1-10 -v EPOCHS=100,WALLTIME=8:00:00 gadi_section1_1.pbs
```

### Email Notifications
Jobs are configured to send emails to `mc2303@nci.org.au` for:
- Job aborts (`-m a`)
- Job begins (`-m b`)
- Job ends (`-m e`)

To disable, remove or comment out:
```bash
#PBS -m abe
#PBS -M mc2303@nci.org.au
```

## Gadi-Specific Notes

### Queue Limits (as of 2024)
- **Normal queue**: Max 48 hours walltime, up to 10,000 CPUs
- **Express queue**: Max 48 hours, limited SU, higher priority
- **GPU volta**: Max 48 hours, limited GPU availability

### Service Units (SU) Usage
Approximate SU cost per job:
- CPU job (4 cores, 4 hours): ~16 SU
- GPU job (1 GPU, 4 hours): ~150 SU

Check your allocation:
```bash
nci_account -P p00
```

### Modules Available
```bash
module avail python          # Available Python versions
module avail cuda            # Available CUDA versions
module spider pytorch        # Search for PyTorch modules
```

## Best Practices

1. **Start Small**: Test with `EPOCHS=10` before long runs
2. **Monitor Early**: Check logs after 5-10 minutes to catch errors early
3. **Save Checkpoints**: For long runs, implement checkpoint saving
4. **Use Scratch**: For large temporary files, use `/scratch/p00/`
5. **Clean Up**: Remove old `.venv_gadi` if you update requirements
6. **Version Control**: Keep track of code versions with git commits

## Support

- **NCI Help**: help@nci.org.au
- **Gadi Documentation**: https://opus.nci.org.au/display/Help/Gadi+User+Guide
- **madoc Issues**: (your repository issue tracker)

## Example Complete Workflow

```bash
# 1. Login to Gadi
ssh mc2303@gadi.nci.org.au

# 2. Navigate to project directory
cd /g/data/p00/mc2303/

# 3. Clone repository (first time only)
git clone <your-repo> madoc
cd madoc

# 4. Quick test (verify everything works)
qsub -v EPOCHS=10,WALLTIME=0:30:00 gadi_section1_1.pbs

# 5. Check job status
qstat -u mc2303

# 6. Monitor output
tail -f section1_1_*.log

# 7. Check results
ls -lht section1/sec1_results/

# 8. Submit production runs
qsub -v EPOCHS=1000,WALLTIME=48:00:00 gadi_section1_1.pbs
qsub -v EPOCHS=1000,WALLTIME=48:00:00 gadi_section1_2.pbs
qsub -v EPOCHS=1000,WALLTIME=48:00:00 gadi_section1_3.pbs

# 9. Download results (from local machine)
scp mc2303@gadi.nci.org.au:/g/data/p00/mc2303/madoc/section1/sec1_results/*.json .
```

## Contact

- **User**: mc2303
- **Project**: p00
- **Email**: mc2303@nci.org.au

For questions about the madoc codebase, see the main README.md in the repository.
