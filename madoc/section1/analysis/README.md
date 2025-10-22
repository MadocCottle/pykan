# Section 1 Analysis Package

Concise analysis and visualization toolkit for Section 1 experimental results.

## Structure

```
section1/
├── analysis/              # Analysis package (self-contained)
│   ├── io.py             # Load results (~110 lines)
│   ├── comparative_metrics.py
│   ├── function_fitting.py
│   ├── heatmap_2d_fits.py
│   ├── run_analysis.py
│   ├── analyze_all_section1.py
│   ├── report_utils.py
│   └── templates/
├── utils/
│   └── io.py             # Save results (~60 lines)
├── results/              # All experiment results stored here
│   ├── sec1_results/
│   ├── sec2_results/
│   └── sec3_results/
├── section1_1.py         # Training scripts
├── section1_2.py
└── section1_3.py
```

## Quick Start

### Run Experiments
```bash
cd section1
python section1_1.py --epochs 100
# Saves to: section1/results/sec1_results/section1_1_{timestamp}.*
```

### Analyze Results
```bash
cd section1/analysis

# Analyze all sections (latest results)
python analyze_all_section1.py

# Analyze specific section
python run_analysis.py ../results/sec1_results/section1_1_{timestamp}.pkl

# Or use section ID (auto-finds latest)
from analysis import io
results, meta, models = io.load_run('section1_1')
```

## API

### Loading Data

```python
from analysis import io

# Load latest run
results, metadata, models_dir = io.load_run('section1_1')

# Load specific timestamp
results, metadata, models_dir = io.load_run('section1_1', timestamp='20251022_143000')

# Check if 2D
is_2d = io.is_2d('section1_3')  # True

# Available sections
io.SECTIONS  # ['section1_1', 'section1_2', 'section1_3']
```

### Running Analysis

```python
from analysis import run_full_analysis

# Analyze results file
run_full_analysis(
    results_file='../results/sec1_results/section1_1_20251022_143000.pkl',
    models_dir='../results/sec1_results',  # Optional
    output_base_dir='my_analysis'  # Optional
)
```

### Using Analyzers Directly

```python
from analysis import MetricsAnalyzer, FunctionFittingVisualizer, Heatmap2DAnalyzer

# Comparative metrics
analyzer = MetricsAnalyzer('section1_1')  # Or pass file path
analyzer.generate_all_visualizations('output_dir/')

# Function fitting
visualizer = FunctionFittingVisualizer('section1_1')
visualizer.generate_all_function_fits('output_dir/', is_2d=False)

# 2D heatmaps (for section1_3)
heatmap = Heatmap2DAnalyzer('section1_3')
heatmap.generate_all_heatmaps('output_dir/')
```

## Data Format

Results structure (unchanged from old system):

```python
results = {
    'mlp': {
        dataset_idx: {
            depth: {
                activation: {
                    'train': [...],      # Training loss history
                    'test': [...],       # Test loss history
                    'dense_mse': [...],  # Dense MSE metrics
                    'total_time': float,
                    'time_per_epoch': float
                }
            }
        }
    },
    'siren': { ... },
    'kan': { ... },
    'kan_pruning': { ... }
}
```

## Output

Analysis generates:
- `01_comparative_metrics/` - Tables, learning curves, heatmaps
- `02_function_fitting/` - True vs predicted function plots
- `03_heatmap_analysis/` - 2D error analysis (for section1_3 only)
- `ANALYSIS_SUMMARY.md` - Comprehensive report

Batch analysis (`analyze_all_section1.py`) generates:
- Individual analysis for each section
- `SECTION1_THESIS_REPORT.md` - Combined report

## Benefits

1. **Self-contained**: All analysis code in one place
2. **Concise**: ~170 lines of IO code (vs 611 lines old system)
3. **Clear paths**: Results in `section1/results/`, analysis in `section1/analysis/`
4. **Simple**: Direct pickle/glob operations, no complex abstractions
5. **Backward compatible**: Reads old result files

## Migration

Old system archived in `section1/archive/old_io_system/`. See [MIGRATION.md](../../MIGRATION.md) for details.
