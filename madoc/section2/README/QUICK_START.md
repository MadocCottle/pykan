# Quick Start Guide - Section 2

## Overview

Section 2 explores optimizer comparisons and adaptive training techniques for KANs on PDE problems.

## Section 2.1: Optimizer Comparison

Tests ADAM vs Levenberg-Marquardt (LM) optimizers on 2D Poisson PDE problems, using identical datasets and metrics from Section 1.3.

### Run Locally

```bash
# Navigate to section2 directory
cd /path/to/pykan/madoc/section2

# Run with default 10 epochs
python section2_1.py

# Run with custom epochs
python section2_1.py --epochs 200
```

### What It Does

- Trains KANs on 4 different 2D Poisson PDE variants
- Compares ADAM and LM optimizers across 6 grid sizes (3, 5, 10, 20, 50, 100)
- Computes dense MSE errors for accurate performance measurement
- Saves results and trained models

### Results Location

- **JSON Results**: `results/section2_1_results_<timestamp>.json`
- **Model Files**: `models/` directory

## Implementation Details

### Optimizers

1. **ADAM**: Built-in PyTorch optimizer
   - Adaptive learning rates
   - First and second moment estimates

2. **Levenberg-Marquardt (LM)**: Custom implementation
   - Combines gradient descent with Gauss-Newton method
   - Adaptive damping for stability
   - Particularly effective for small-parameter models

### Datasets

Same as Section 1.3:
- 2D Poisson with sinusoidal forcing
- 2D Poisson with polynomial forcing
- 2D Poisson with high-frequency forcing
- 2D Poisson with spectral forcing

## Performance Notes

- LM optimizer may show improved convergence for small grid sizes
- ADAM typically more stable for larger grids
- Training time varies based on optimizer characteristics
- Dense MSE computation adds overhead but provides accurate metrics

## Reference

Based on: "Optimizing the optimizer for data driven deep neural networks and physics informed neural networks"
https://arxiv.org/abs/2205.07430
