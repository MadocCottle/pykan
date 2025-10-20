# Timing Utilities Documentation

This module provides comprehensive timing functionality for tracking training performance across experiments.

## Quick Start (Minimal API)

The simplest way to add timing to your code:

```python
from utils import track_time, print_timing_summary

timers = {}
results = track_time(timers, "Model Training", train_function, arg1, arg2)
print_timing_summary(timers, "My Experiment")
```

**Code overhead in section scripts:** Only ~10 lines per file!

---

## Core Functions

### `track_time(timers, name, func, *args, **kwargs)`
Track execution time of any function call.

**Returns:** Whatever the function returns (supports multiple return values)

```python
timers = {}
result = track_time(timers, "MLP training", run_mlp_tests, datasets, depths, activations, epochs, device)
# Automatically prints: "MLP training complete: 45.32s total"
```

### `print_timing_summary(timers, section_name, num_datasets=1)`
Print a formatted summary table of all timings.

```python
print_timing_summary(timers, "Section 1.1", num_datasets=9)
```

**Output:**
```
============================================================
TIMING SUMMARY - Section 1.1
============================================================
MLP training:              125.45s ( 35.2%)
SIREN training:             89.32s ( 25.1%)
KAN training:              102.67s ( 28.8%)
------------------------------------------------------------
Total time:                317.44s
Average per dataset:        35.27s
============================================================
```

---

## Advanced Features

All 8 advanced features are optional and can be used independently. Import them as needed:

```python
from utils.timing import <feature_function>
```

### FEATURE #1: Memory Profiling
**Track peak memory usage for each model type**

```python
from utils.timing import track_time_and_memory, enable_memory_tracking

enable_memory_tracking()  # Start tracemalloc

timers = {}
result = track_time_and_memory(timers, "Model", train_fn, args)
# Prints: "Model complete: 45.32s, peak memory: 234.5 MB"

print(timers)
# {'Model': {'time': 45.32, 'peak_memory_mb': 234.5, 'current_memory_mb': 123.4}}
```

**Use case:** Understand memory vs. time tradeoffs, optimize batch sizes

---

### FEATURE #2: GPU Utilization Metrics
**Track GPU memory usage and compute utilization (CUDA only)**

```python
from utils.timing import track_time_and_gpu, get_gpu_memory_usage

timers = {}
result = track_time_and_gpu(timers, "KAN training", train_fn, args)
# Prints: "KAN training complete: 12.45s, GPU peak: 1024.3 MB"

print(timers)
# {'KAN training': {'time': 12.45, 'allocated_mb': 512.1, 'reserved_mb': 1024.0, 'max_allocated_mb': 1024.3}}

# Get current GPU stats anytime
gpu_stats = get_gpu_memory_usage()
```

**Use case:** Monitor GPU memory to prevent OOM errors, track GPU efficiency

---

### FEATURE #3: Convergence Speed Metrics
**Track time to reach specific accuracy thresholds**

```python
from utils.timing import calculate_convergence_metrics

# After training, analyze how quickly models converged
metrics = calculate_convergence_metrics(results, threshold=0.01)
# Returns: {'epochs_to_converge': 15, 'time_to_converge': 23.4, ...}
```

**Use case:** Compare which models learn faster, optimize early stopping

---

### FEATURE #4: Comparative Speedup Analysis
**Calculate speedup ratios between model types**

```python
from utils.timing import calculate_speedup_ratios, print_speedup_analysis

timers = {'MLP': 120.0, 'SIREN': 90.0, 'KAN': 150.0}

# Get speedup ratios (relative to slowest = KAN)
speedups = calculate_speedup_ratios(timers)
# Returns: {'MLP': 1.25, 'SIREN': 1.67, 'KAN': 1.0}

# Or print formatted analysis
print_speedup_analysis(timers, baseline='KAN')
```

**Output:**
```
============================================================
SPEEDUP ANALYSIS
============================================================
Baseline: KAN (150.00s)
------------------------------------------------------------
SIREN                :  1.67x faster
MLP                  :  1.25x faster
============================================================
```

**Use case:** Quickly compare model efficiency, make informed architecture choices

---

### FEATURE #5: Detailed Breakdown Timing
**Track time spent in different training phases**

```python
from utils.timing import DetailedTimer

timer = DetailedTimer("Training")

timer.start_phase("data_loading")
# ... load data ...
timer.end_phase()

timer.start_phase("forward_pass")
# ... forward pass ...
timer.end_phase()

timer.start_phase("backward_pass")
# ... backward pass ...
timer.end_phase()

timer.print_breakdown()
```

**Output:**
```
Training - Detailed Breakdown:
----------------------------------------
  backward_pass       :    12.45s ( 55.2%)
  forward_pass        :     8.34s ( 37.0%)
  data_loading        :     1.76s (  7.8%)
----------------------------------------
```

**Use case:** Identify bottlenecks in training pipeline, optimize critical sections

---

### FEATURE #6: Export Timing Data
**Save timing results to CSV/JSON for later analysis**

```python
from utils.timing import export_timing_csv, export_timing_json

timers = {'MLP': 120.0, 'SIREN': 90.0, 'KAN': 150.0}

# Export to CSV
export_timing_csv(timers, 'sec1_timing.csv', section_name='Section 1.1')
# Creates: sec1_timing.csv with columns [Section, Model, Time (s)]

# Export to JSON (supports nested dictionaries for advanced metrics)
export_timing_json(timers, 'sec1_timing.json', section_name='Section 1.1')
# Creates: sec1_timing.json with timestamp and structured data
```

**Use case:** Build timing dashboards, compare runs over time, generate reports

---

### FEATURE #7: Prediction/Inference Timing
**Time model inference/prediction separately from training**

```python
from utils.timing import benchmark_inference, print_inference_benchmark

# Benchmark a trained model
results = benchmark_inference(model, test_input, num_iterations=100, warmup=10)
print(results)
# {'mean_ms': 2.345, 'min_ms': 2.123, 'max_ms': 3.456, 'throughput_samples_per_sec': 426.3}

# Or print formatted results
print_inference_benchmark(model, test_input, model_name="KAN")
```

**Output:**
```
KAN Inference Benchmark:
----------------------------------------
  Mean latency:        2.345 ms
  Min latency:         2.123 ms
  Max latency:         3.456 ms
  Throughput:          426.3 samples/sec
----------------------------------------
```

**Use case:** Evaluate deployment performance, compare inference speed vs training speed

---

### FEATURE #8: Warmup vs Steady-State Timing
**Separate first epoch timing from remaining epochs**

```python
from utils.timing import track_warmup_timing, print_warmup_analysis

# After training, analyze warmup overhead
epoch_times = [5.2, 3.1, 3.0, 3.1, 3.0, 2.9]  # Per-epoch timings
stats = track_warmup_timing(total_epochs=6, epoch_times=epoch_times)

print(stats)
# {'first_epoch_time': 5.2, 'avg_steady_state_time': 3.02,
#  'warmup_overhead': 2.18, 'warmup_overhead_pct': 72.2}

# Or print formatted analysis
print_warmup_analysis(stats, model_name="MLP")
```

**Output:**
```
MLP - Warmup Analysis:
----------------------------------------
  First epoch:           5.200s
  Steady state:          3.020s
  Warmup overhead:       2.180s ( 72.2%)
----------------------------------------
```

**Use case:** Account for JIT compilation and cache warmup, get accurate steady-state performance

---

## Example: Using Multiple Features Together

```python
from utils import track_time, print_timing_summary
from utils.timing import (
    track_time_and_gpu,
    print_speedup_analysis,
    export_timing_json
)

# Track with GPU metrics
timers = {}
mlp_results = track_time_and_gpu(timers, "MLP", run_mlp_tests, datasets, depths, activations, epochs, device)
siren_results = track_time_and_gpu(timers, "SIREN", run_siren_tests, datasets, depths, epochs, device)
kan_results, models = track_time_and_gpu(timers, "KAN", run_kan_grid_tests, datasets, grids, epochs, device)

# Print summary
print_timing_summary(timers, "Section 1.1", num_datasets=9)

# Analyze speedups
print_speedup_analysis(timers, baseline='KAN')

# Export for later analysis
export_timing_json(timers, 'sec1_1_timings.json', section_name='Section 1.1')
```

---

## Implementation Details

### File Structure
- **Location:** `madoc/section1/utils/timing.py`
- **Lines of code:** ~450 lines (includes all 8 features + documentation)
- **Dependencies:** Standard library only (`time`, `tracemalloc`, `csv`, `json`)
- **Optional dependencies:** `torch` (for GPU metrics), `psutil` (alternative memory tracking)

### Code Impact on Section Scripts

**Before (manual timing):** ~43 lines per section script
```python
import time
script_start = time.time()
mlp_start = time.time()
# ... run training ...
mlp_time = time.time() - mlp_start
# ... repeat for each model ...
# ... 15 lines of print statements for summary table ...
```

**After (with timing.py):** ~10 lines per section script
```python
from utils import track_time, print_timing_summary
timers = {}
mlp_results = track_time(timers, "MLP", run_mlp_tests, ...)
siren_results = track_time(timers, "SIREN", run_siren_tests, ...)
kan_results, models = track_time(timers, "KAN", run_kan_grid_tests, ...)
print_timing_summary(timers, "Section 1.1", num_datasets=len(datasets))
```

**Net reduction:** -33 lines per script × 3 scripts = **-99 lines** (with base features only)

### Design Principles

1. **Minimal by default:** Core API requires only 2 functions and adds minimal code
2. **Optional advanced features:** All 8 advanced features can be imported individually
3. **Non-invasive:** Works with existing code, no need to modify training functions
4. **Consistent API:** All `track_time_*` functions have the same signature
5. **Informative output:** Prints progress messages during execution
6. **Extensible:** Easy to add new timing features without changing section scripts

---

## FAQ

**Q: Do I need to modify my training functions?**
A: No! `track_time()` wraps any function and preserves its return values.

**Q: Can I use this with functions that return multiple values?**
A: Yes! `track_time()` preserves all return values:
```python
result1, result2, result3 = track_time(timers, "name", func_that_returns_three_things, args)
```

**Q: What if I don't have a GPU?**
A: GPU features gracefully fall back to CPU-only timing if CUDA isn't available.

**Q: Can I disable the progress print statements?**
A: Currently no, but you can redirect stdout or modify the functions to add a `verbose` parameter.

**Q: How much overhead does timing add?**
A: Negligible (~0.001s per timer). The time tracking itself is not included in measurements.

**Q: Can I nest timing calls?**
A: Yes, but be aware that inner timings will be included in outer timings. Use `DetailedTimer` for breakdown analysis.

---

## Contributing

To add a new timing feature:

1. Add the function to `timing.py` under a new `ADVANCED FEATURE #X` section
2. Document with clear docstrings
3. Add example usage to this README
4. Keep the core API minimal - new features should be opt-in
