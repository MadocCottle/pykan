# Section 2: KAN Ensembles and Advanced Training

Concise implementation of advanced KAN techniques following section1 style.

## Overview

Section 2 explores ensemble methods, adaptive grid techniques, and regularization for KAN networks:

1. **section2_1**: Ensemble of KAN experts with uncertainty quantification
2. **section2_2**: Adaptive grid densification (progressive refinement)
3. **section2_3**: Multi-grid comparison study
4. **section2_4**: Ensemble uncertainty analysis
5. **section2_5**: Model pruning and regularization

## Structure

```
section2/
├── ensemble/          # Ensemble training
│   └── expert_training.py
├── utils/            # I/O utilities
│   └── io.py         # save_run()
├── analysis/         # Result loading
│   └── io.py         # load_run()
├── results/          # Saved experiments
│   ├── sec1_results/
│   ├── sec2_results/
│   └── ...
├── section2_1.py     # Ensemble experiments
├── section2_2.py     # Adaptive grids
├── section2_3.py     # Grid comparison
├── section2_4.py     # Uncertainty
└── section2_5.py     # Pruning
```

## Quick Start

```bash
# Run ensemble experiment
python section2/section2_1.py --epochs 100 --n_experts 10

# Run adaptive grid experiment
python section2/section2_2.py --epochs 100 --initial_grid 3 --max_grid 10

# Run grid comparison
python section2/section2_3.py --epochs 100

# Run uncertainty analysis
python section2/section2_4.py --epochs 100 --n_experts 5

# Run pruning experiment
python section2/section2_5.py --epochs 100
```

## Loading Results

```python
from section2.analysis import load_run

# Load latest run
results, meta, models_dir = load_run('section2_1')

# Access results
print(f"Ensemble MSE: {results['ensemble_mse']}")
print(f"Expert losses: {results['expert_losses']}")
```

## I/O System

Following section1's concise style:

**Saving** (utils/io.py):
```python
from section2.utils import save_run

save_run(results_dict, 'section2_1',
         models=model_list,
         epochs=100, n_experts=10, device='cpu')
```

**Loading** (analysis/io.py):
```python
from section2.analysis import load_run, list_runs

# Load specific timestamp
results, meta, models_dir = load_run('section2_1', timestamp='20251022_150000')

# List all runs
timestamps = list_runs('section2_1')
```

## Comparison with section2_new

section2 is a **concise rewrite** of section2_new functionality:

| Feature | section2_new | section2 |
|---------|-------------|----------|
| Ensemble training | 400+ lines | ~80 lines |
| I/O system | N/A | section1-style |
| Analysis ready | No | Yes |
| Style | Verbose, feature-rich | Concise, essential |

## Development Notes

- Uses pykan's native KAN class for simplicity
- Follows section1's concise patterns
- I/O compatible with analysis pipeline
- Models saved with timestamps
- Results in JSON + pickle format
