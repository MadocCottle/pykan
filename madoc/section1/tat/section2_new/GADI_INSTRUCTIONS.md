# Running Evolutionary KAN on Gadi - Complete Instructions

## Prerequisites

Before starting, you need:
1. Your NCI username
2. Your NCI project code (e.g., `ab12`)
3. Access to Gadi via SSH
4. Your email for job notifications (optional)

---

## Step-by-Step Instructions

### Step 1: SSH into Gadi

```bash
ssh YOUR_USERNAME@gadi.nci.org.au
```

Enter your password when prompted.

---

### Step 2: Copy Your Code to Gadi

**Option A: Using SCP (from your local machine)**

```bash
# From your local machine (not on Gadi)
cd /Users/main/Desktop/help
scp -r KAN_Repo YOUR_USERNAME@gadi.nci.org.au:~/
```

**Option B: Using Git (on Gadi)**

```bash
# On Gadi
cd ~
git clone YOUR_REPOSITORY_URL KAN_Repo
# Or if you don't have git repo, use rsync/scp
```

**Option C: Manual copy (smallest)**

If bandwidth is limited, copy only what's needed:

```bash
# From local machine
cd /Users/main/Desktop/help/KAN_Repo
tar czf kan_minimal.tar.gz section1/models section2_new

# Upload
scp kan_minimal.tar.gz YOUR_USERNAME@gadi.nci.org.au:~/

# On Gadi
cd ~
mkdir -p KAN_Repo
cd KAN_Repo
tar xzf ../kan_minimal.tar.gz
```

---

### Step 3: Edit the PBS Script

On Gadi, edit the PBS script to add your details:

```bash
cd ~/KAN_Repo/section2_new
nano gadi_run_evolution.pbs
```

**Replace these placeholders:**
1. Line 2: `#PBS -P INSERT_YOUR_PROJECT_CODE_HERE` → `#PBS -P ab12` (use your project)
2. Line 6: `#PBS -l storage=scratch/INSERT_YOUR_PROJECT_CODE_HERE` → `#PBS -l storage=scratch/ab12`
3. Line 10: `#PBS -M INSERT_YOUR_EMAIL_HERE` → `#PBS -M your.email@domain.com`

Save and exit (Ctrl+X, then Y, then Enter)

---

### Step 4: Make Setup Script Executable

```bash
chmod +x gadi_setup.sh
```

---

### Step 5: Run Setup Script

```bash
./gadi_setup.sh
```

This will:
- Load Python 3.10
- Create virtual environment at `~/KAN_Repo/.venv`
- Install PyTorch, NumPy, SciPy, scikit-learn
- Verify installation

**Expected output:**
```
==========================================
Setup Complete!
==========================================
Virtual environment: /home/561/YOUR_USERNAME/KAN_Repo/.venv
To activate: source /home/561/YOUR_USERNAME/KAN_Repo/.venv/bin/activate

Next steps:
1. Copy your KAN_Repo code to /home/561/YOUR_USERNAME/KAN_Repo
2. Submit the PBS job: qsub gadi_run_evolution.pbs
```

---

### Step 6: Submit the Job

```bash
cd ~/KAN_Repo/section2_new
qsub gadi_run_evolution.pbs
```

**You should see:**
```
1234567.gadi-pbs
```

This is your job ID.

---

### Step 7: Monitor Your Job

**Check job status:**
```bash
qstat -u YOUR_USERNAME
```

**View job details:**
```bash
qstat -f 1234567  # Use your job ID
```

**Watch the log file (updates in real-time):**
```bash
tail -f ~/KAN_Repo/section2_new/evolutionary_kan_1234567.gadi-pbs.log
```

Press Ctrl+C to stop watching.

---

### Step 8: Check Results

After the job completes (up to 9 hours), check the results:

```bash
cd ~/KAN_Repo/section2_new/results
ls -lh
```

**You should see:**
- `best_genome_TIMESTAMP.pkl` - Best architecture found
- `evolution_history_TIMESTAMP.pkl` - Complete evolution history
- `pareto_frontier_TIMESTAMP.pkl` - Pareto-optimal solutions

**View the log:**
```bash
cd ~/KAN_Repo/section2_new
less evolutionary_kan_1234567.gadi-pbs.log  # Use your job ID
```

---

## Job Configuration

The PBS script is configured for:
- **Walltime:** 9 hours (as requested)
- **CPUs:** 8 cores
- **Memory:** 32 GB
- **Queue:** normal

**To adjust resources, edit these lines in the PBS script:**

```bash
#PBS -l walltime=9:00:00     # Change to 12:00:00 for 12 hours, etc.
#PBS -l ncpus=8              # Change to 16 for more CPUs
#PBS -l mem=32GB             # Change to 64GB for more memory
```

---

## What the Job Does

The evolutionary search will:

1. **Initialize:** Create population of 30 random KAN architectures
2. **Evolve:** Run for 50 generations with:
   - Fitness evaluation (accuracy, complexity, speed)
   - Tournament selection
   - Crossover and mutation
   - Elitism (preserve top 3)
3. **Optimize:** Find Pareto-optimal trade-offs
4. **Test:** Evaluate best genome on test set
5. **Save:** Store all results and Pareto frontier

**Expected outputs in log:**
- Generation-by-generation progress
- Best fitness per generation
- Population diversity metrics
- Pareto frontier size
- Final test performance
- Cache statistics

---

## Troubleshooting

### Issue: Job doesn't start
**Check queue:** `qstat -u YOUR_USERNAME`
**Reason:** Queue might be busy. Wait or try `express` queue (shorter walltime)

### Issue: Job fails immediately
**Check log:** Look for error messages in `evolutionary_kan_*.log`
**Common fixes:**
- Verify project code in PBS script
- Check storage allocation
- Ensure virtual environment exists

### Issue: Module not found
**Solution:** Re-run setup script:
```bash
cd ~/KAN_Repo/section2_new
./gadi_setup.sh
```

### Issue: Out of memory
**Solution:** Increase memory in PBS script:
```bash
#PBS -l mem=64GB  # or higher
```

### Issue: Job timeout
**Solution:** Reduce population or generations:
- Edit line in PBS script with `population_size=20` (instead of 30)
- Or `n_generations=30` (instead of 50)

---

## Retrieving Results to Your Local Machine

After job completes, copy results back:

```bash
# From your local machine
scp -r YOUR_USERNAME@gadi.nci.org.au:~/KAN_Repo/section2_new/results ./gadi_results
```

Then analyze locally:

```python
import pickle

# Load best genome
with open('gadi_results/best_genome_TIMESTAMP.pkl', 'rb') as f:
    best_genome = pickle.load(f)

print(f"Best architecture: {best_genome.layer_sizes}")
print(f"Fitness: {best_genome.fitness}")

# Load history
with open('gadi_results/evolution_history_TIMESTAMP.pkl', 'rb') as f:
    history = pickle.load(f)

import matplotlib.pyplot as plt
plt.plot(history['best_fitness'])
plt.xlabel('Generation')
plt.ylabel('Best Fitness')
plt.title('Evolution Progress')
plt.show()
```

---

## Quick Reference Commands

```bash
# Login
ssh YOUR_USERNAME@gadi.nci.org.au

# Submit job
cd ~/KAN_Repo/section2_new
qsub gadi_run_evolution.pbs

# Check status
qstat -u YOUR_USERNAME

# Monitor log
tail -f evolutionary_kan_*.log

# Cancel job (if needed)
qdel JOB_ID

# Copy results to local
scp -r YOUR_USERNAME@gadi.nci.org.au:~/KAN_Repo/section2_new/results ./
```

---

## Expected Runtime

With the default configuration:
- **Population:** 30 genomes
- **Generations:** 50
- **Evaluations:** ~1,500 fitness evaluations
- **Training per evaluation:** ~200 epochs
- **Estimated time:** 4-8 hours (depending on Gadi load)

The 9-hour walltime provides comfortable buffer.

---

## Contact

If you encounter issues:
1. Check the job log file for error messages
2. Verify all PBS script parameters
3. Ensure virtual environment is set up correctly
4. Check NCI help: https://opus.nci.org.au/

---

## Summary

1. ✅ SSH into Gadi
2. ✅ Copy code to `~/KAN_Repo`
3. ✅ Edit PBS script (project code, email)
4. ✅ Run `./gadi_setup.sh`
5. ✅ Submit: `qsub gadi_run_evolution.pbs`
6. ✅ Monitor: `qstat -u YOUR_USERNAME`
7. ✅ Check results in `~/KAN_Repo/section2_new/results`

**All files are ready - you just need to follow these steps!**
