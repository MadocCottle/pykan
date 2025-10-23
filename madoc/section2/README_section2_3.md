# Section 2.3: Merge_KAN Experiments

This directory contains scripts for running Merge_KAN experiments on Poisson PDEs.

## Overview

Merge_KAN trains multiple "expert" KANs with different configurations, selects the best expert for each dependency pattern, merges them into a unified model, and trains the merged model through grid refinement.

## Available Workflows

### 1. Standard (Sequential) Workflow

**File**: `section2_3.py`

Runs Merge_KAN sequentially on 2D Poisson PDE with sin forcing function only.

```bash
# Default: 5 seeds per config, 1000 epochs per expert
python section2_3.py

# Custom seeds
python section2_3.py --n-seeds 3

# Test mode (reduced computation)
python section2_3.py --test-mode
```

**Computation**:
- Trains 15 experts sequentially (with n_seeds=5)
- Each expert: 1000 epochs
- Merged model: 4 grids × 200 steps
- **Total time**: ~15-20 hours on CPU

**Cost Reduction**: Uses only `f_poisson_2d_sin` (75% reduction from original 4 datasets)

---

### 2. Parallelized Workflow (PBS Job Array)

**Files**:
- `submit_section2_3.sh` - Orchestration script
- `section2_3_experts.qsub` - Phase 1 (parallel expert training)
- `section2_3_merge.qsub` - Phase 2 (merge and train)
- `section2_3_train_expert.py` - Single expert trainer
- `section2_3_merge.py` - Expert merger and final trainer

Runs Merge_KAN with parallelized expert training using PBS job arrays.

```bash
# 4D experiment with default parameters
./submit_section2_3.sh --dim 4

# 10D with custom parameters
./submit_section2_3.sh --dim 10 --n-seeds 3 --expert-epochs 500 --merged-epochs 200

# 2D with custom grid schedule
./submit_section2_3.sh --dim 2 --grids "3,5,10,20,50"

# Dry run (show commands without submitting)
./submit_section2_3.sh --dim 4 --dry-run
```

**Workflow**:
1. **Phase 1**: Submit job array (15 parallel jobs, one per expert)
   - Each job trains 1 expert independently
   - Saves trained model to shared directory
   - Resource: 1 CPU, 4GB RAM, 2 hours per job

2. **Phase 2**: Submit merge job (depends on Phase 1 completion)
   - Loads all trained experts
   - Selects best per dependency pattern
   - Merges and trains combined model
   - Resource: 12 CPUs, 48GB RAM, 4 hours

**Speedup**: ~5× faster than sequential (1-2 hours vs 15-20 hours)

**Monitoring**:
```bash
# Check job status
qstat -u $USER

# View Phase 1 logs (per expert)
ls experts_4d_*/expert_*.log

# View Phase 2 log
ls section2/results/*.log
```

---

### 3. High-Dimensional Variant (Sequential)

**File**: `section2_3_highd.py`

Runs Merge_KAN on higher-dimensional problems (4D, 10D) without parallelization.

```bash
# 4D experiment
python section2_3_highd.py --dim 4 --n-seeds 5

# 10D with reduced seeds for faster testing
python section2_3_highd.py --dim 10 --n-seeds 3

# Test mode
python section2_3_highd.py --dim 4 --test-mode
```

**Use Cases**:
- Quick testing on local machines
- When PBS job arrays are not available
- Single-run experiments

---

## Architecture

### Expert Pool

With `n_seeds=5`, generates 15 experts:
- **Depth 2, k=3**: 5 experts (seeds 0-4)
- **Depth 3, k=3**: 5 experts (seeds 0-4)
- **Depth 2, k=2**: 5 experts (seeds 0-4)

Each expert has architecture `[n_var, 5, ..., 5, 1]` where depth determines number of hidden layers.

### Expert Selection

Groups experts by discovered dependency patterns (which input variables they use) and selects the best (lowest dense MSE) expert from each group.

### Merged Model

Combines selected experts into architecture: `[n_var, total_hidden, n_intermediate, 1]`
- `total_hidden`: Sum of all expert hidden widths
- `n_intermediate`: max(3, num_experts)

### Training Schedule

**Expert Training** (Phase 1):
- Default: 1000 epochs with LBFGS optimizer
- Configurable via `--expert-epochs`

**Merged Training** (Phase 2):
- Default grids: [3, 5, 10, 20]
- Default: 200 steps per grid with LBFGS
- Configurable via `--merged-epochs` and `--grids`
- Early stopping: stops if test loss increases for 2 consecutive grids

---

## Dimensional Variants

| Dimension | Function | Architecture (shallow) | Grid Schedule | Training Samples |
|-----------|----------|----------------------|---------------|-----------------|
| 2D | `f_poisson_2d_sin` | [2, 5, 1] | [3, 5, 10, 20] | 1000 |
| 4D | `f_poisson_4d_sin` | [4, 5, 1] | [3, 5, 10, 20] | 1000 |
| 10D | `f_poisson_10d_sin` | [10, 5, 1] | [3, 5, 10, 20] | 1000 |

**Note**: 100D is excluded due to high computational cost (~15× more expensive than 2D).

---

## Helper Modules

### `utils/expert_config.py`
Manages expert configuration generation and naming:
- `get_expert_configs(n_seeds)`: Generate all configs
- `get_expert_config(index, n_seeds)`: Get config for job array index
- `format_expert_name(config, dim)`: Generate consistent filenames

### `utils/expert_io.py`
Handles expert model I/O:
- `save_expert(expert_dict, output_dir, filename)`: Save trained expert
- `load_expert(filepath)`: Load single expert
- `load_all_experts(expert_dir)`: Load all experts from directory

---

## Output Structure

### Sequential Workflow
```
section2/results/section2_3/
├── summary.csv              # One row per dataset
├── experts.csv              # All trained experts
├── selected_experts.csv     # Best per dependency pattern
├── grid_history.csv         # Grid refinement progression
└── models.pt                # Saved merged KAN models
```

### Parallelized Workflow

**Phase 1 Output**:
```
experts_4d_20250124_120000/
├── expert_4d_depth2_k3_seed0.pkl
├── expert_4d_depth2_k3_seed1.pkl
├── ...
├── expert_4d_depth2_k2_seed4.pkl
├── expert_0.log
├── expert_1.log
├── ...
└── expert_14.log
```

**Phase 2 Output**:
```
section2/results/section2_3_4d/
├── summary.csv
├── experts.csv
├── selected_experts.csv
├── grid_history.csv
└── merged_kan.pt           # Final merged model
```

---

## Troubleshooting

### Phase 1: Expert Training Failures

**Check which experts failed**:
```bash
cd experts_4d_*/
ls expert_*.success          # Should have 15 success markers
ls expert_*.pkl | wc -l      # Count trained models
```

**View failed expert logs**:
```bash
grep -l ERROR expert_*.log
cat expert_5.log  # Example
```

**Solutions**:
- Increase walltime in `submit_section2_3.sh` if jobs timeout
- Reduce `--expert-epochs` for faster testing
- Check memory if jobs are killed (increase `PHASE1_MEM`)

### Phase 2: Merge Failures

**Check expert count**:
```bash
# Phase 2 log shows: "Found experts: N"
# Should be 15 for n_seeds=5
grep "Found experts" section2/results/*.log
```

**Solutions**:
- Re-run Phase 1 for failed experts
- Phase 2 can proceed with fewer experts (warning issued)
- Check Phase 2 walltime if training is slow

### PBS Dependency Issues

**Check if dependency was registered**:
```bash
qstat -f PHASE2_JOB_ID | grep depend
# Should show: afterokarray:PHASE1_JOB_ID
```

**If Phase 2 starts too early**:
- Check if all Phase 1 jobs completed: `qstat | grep PHASE1_JOB_ID`
- Manually hold Phase 2: `qhold PHASE2_JOB_ID`
- Release when ready: `qrls PHASE2_JOB_ID`

---

## Performance Comparison

| Workflow | Dimension | CPUs (Peak) | Walltime | Speedup |
|----------|-----------|-------------|----------|---------|
| Sequential | 2D | 1 | 15-20h | 1× |
| Parallel | 2D | 15 (Phase 1) | 2-3h | ~5-7× |
| Parallel | 4D | 15 (Phase 1) | 2-4h | ~5-6× |
| Parallel | 10D | 15 (Phase 1) | 3-5h | ~4-5× |

**Cost Reduction from Original**:
- Single dataset (sin only): 75% reduction
- Parallelization: 5-7× speedup
- **Combined**: ~93% time reduction (20 hours → 1-2 hours for 2D)

---

## Future Extensions

1. **More datasets**: Extend parallelized workflow to handle multiple datasets (poly, highfreq, spec)
2. **Adaptive epochs**: Automatically adjust expert epochs based on convergence
3. **Distributed storage**: Use shared filesystem for expert models on clusters
4. **Fault tolerance**: Checkpoint and resume for long-running expert training
5. **100D experiments**: Optimize for very high dimensions with sparse architectures
