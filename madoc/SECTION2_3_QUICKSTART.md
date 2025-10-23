# Section 2.3 Quick Start Guide

## TL;DR

**Parallelized Merge_KAN for 2D/4D/10D Poisson PDEs with 5-7× speedup**

## Quick Commands

### Test Locally (5-10 min)
```bash
cd /path/to/pykan/madoc
./test_section2_3_parallel.sh 2
```

### Run on PBS (2-3 hours)
```bash
cd /path/to/pykan/madoc

# 4D experiment with defaults
./submit_section2_3.sh --dim 4

# Monitor
qstat -u $USER
```

### Run Sequentially (15-20 hours)
```bash
cd /path/to/pykan/madoc/section2
python section2_3.py
```

## What Gets Created

### Phase 1 Output (Parallel Expert Training)
```
experts_4d_TIMESTAMP/
├── expert_4d_depth2_k3_seed0.pkl    # 15 trained expert models
├── expert_4d_depth2_k3_seed1.pkl
├── ...
└── expert_*.log                      # Individual logs
```

### Phase 2 Output (Merge and Train)
```
section2/results/section2_3_4d/
├── summary.csv              # High-level metrics
├── experts.csv              # All expert details
├── selected_experts.csv     # Best per dependency pattern
├── grid_history.csv         # Training progression
└── merged_kan.pt            # Final model
```

## Key Files

| File | Purpose |
|------|---------|
| `submit_section2_3.sh` | **Main entry point** - submit parallelized experiment |
| `test_section2_3_parallel.sh` | Local testing (no PBS) |
| `section2_3.py` | Original sequential workflow |
| `section2_3_highd.py` | Non-parallel 4D/10D variant |
| `section2/README_section2_3.md` | **Full documentation** |

## Common Options

```bash
# Dimension (required)
--dim {2,4,10}

# Number of seeds (default: 5)
--n-seeds 3

# Expert training epochs (default: 1000)
--expert-epochs 500

# Merged model epochs per grid (default: 200)
--merged-epochs 100

# Grid schedule (default: "3,5,10,20")
--grids "3,5,10"

# Preview without submitting
--dry-run
```

## Before First Run

1. **Update PBS project code** in `.qsub` files:
   ```bash
   # Edit these files:
   section2_3_experts.qsub  (line 14)
   section2_3_merge.qsub    (line 14)

   # Change: #PBS -P p00
   # To:     #PBS -P YOUR_PROJECT
   ```

2. **Test locally**:
   ```bash
   ./test_section2_3_parallel.sh 2
   ```

3. **Submit to PBS**:
   ```bash
   ./submit_section2_3.sh --dim 2
   ```

## Monitoring

```bash
# Check job status
qstat -u $USER

# View expert logs (Phase 1)
tail -f experts_4d_*/expert_0.log

# View merge log (Phase 2)
tail -f section2/results/*.log

# Count completed experts
ls experts_4d_*/*.pkl | wc -l  # Should be 15
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Expert index out of range" | Check job array range matches n_seeds |
| "No experts found" | Phase 1 failed - check logs in `experts_*/` |
| Phase 2 starts too early | PBS dependency issue - use `afterokarray` |
| Out of memory | Increase PHASE2_MEM in submit script |
| Import errors | Activate venv: `source .venv/bin/activate` |

## Performance

| Dimension | Experts | Phase 1 Time | Phase 2 Time | Total Time |
|-----------|---------|--------------|--------------|------------|
| 2D | 15 | ~1h | ~1h | **~2h** |
| 4D | 15 | ~1.5h | ~1.5h | **~3h** |
| 10D | 15 | ~2h | ~2h | **~4h** |

**vs. Sequential: 15-20 hours → 2-4 hours (5-7× speedup)**

## Full Documentation

See [`section2/README_section2_3.md`](section2/README_section2_3.md) for:
- Detailed usage instructions
- Architecture explanations
- Advanced configuration
- Troubleshooting guide
- Future extensions

## Questions?

1. Read the full README: `section2/README_section2_3.md`
2. Check the summary: `PARALLELIZED_SECTION2_3_SUMMARY.md`
3. Review PBS logs: `experts_*/expert_*.log`
4. Test locally first: `./test_section2_3_parallel.sh`
