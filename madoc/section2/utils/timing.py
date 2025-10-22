"""Timing utilities for tracking training performance across experiments

This module provides clean, minimal timing functionality with optional advanced features.
"""

import time
import sys
from typing import Dict, Any, Callable, Tuple, Optional


# ============================================================================
# CORE TIMING FUNCTIONALITY (Base Implementation)
# ============================================================================

def track_time(timers: Dict[str, float], name: str, func: Callable, *args, **kwargs) -> Any:
    """Track execution time of a function call

    Args:
        timers: Dictionary to store timing results
        name: Name/label for this timing
        func: Function to execute and time
        *args, **kwargs: Arguments to pass to func

    Returns:
        Whatever func returns
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - start_time
    timers[name] = elapsed
    print(f"{name} complete: {elapsed:.2f}s total\n")
    return result


def print_timing_summary(timers: Dict[str, float], section_name: str, num_datasets: int = 1):
    """Print a formatted timing summary table

    Args:
        timers: Dictionary of timing results from track_time
        section_name: Name of the section (e.g., "Section 1.1")
        num_datasets: Number of datasets processed (for average calculation)
    """
    total_time = sum(timers.values())

    print("\n" + "="*60)
    print(f"TIMING SUMMARY - {section_name}")
    print("="*60)

    for name, elapsed in timers.items():
        percentage = (elapsed / total_time * 100) if total_time > 0 else 0
        # Adjust spacing based on name length for alignment
        spacing = max(1, 24 - len(name))
        print(f"{name}:{' ' * spacing}{elapsed:8.2f}s ({percentage:5.1f}%)")

    print(f"{'-'*60}")
    print(f"Total time:{' ' * 16}{total_time:8.2f}s")
    if num_datasets > 1:
        print(f"Average per dataset:{' ' * 8}{total_time/num_datasets:8.2f}s")
    print("="*60 + "\n")


# ============================================================================
# ADVANCED FEATURE #1: MEMORY PROFILING
# Track peak memory usage for each model type
# ============================================================================

def enable_memory_tracking() -> Optional[Any]:
    """Enable memory tracking using tracemalloc

    Returns:
        None if tracemalloc not available, otherwise starts tracking
    """
    try:
        import tracemalloc
        tracemalloc.start()
        return tracemalloc
    except ImportError:
        print("Warning: tracemalloc not available for memory tracking")
        return None


def track_time_and_memory(timers: Dict[str, Dict], name: str, func: Callable, *args, **kwargs) -> Any:
    """Track both execution time and peak memory usage

    Args:
        timers: Dictionary to store timing and memory results
        name: Name/label for this timing
        func: Function to execute and time
        *args, **kwargs: Arguments to pass to func

    Returns:
        Whatever func returns
    """
    try:
        import tracemalloc
        tracemalloc.start()

        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        timers[name] = {
            'time': elapsed,
            'peak_memory_mb': peak / 1024 / 1024,
            'current_memory_mb': current / 1024 / 1024
        }

        print(f"{name} complete: {elapsed:.2f}s, peak memory: {peak/1024/1024:.1f} MB\n")
        return result

    except ImportError:
        print("Warning: tracemalloc not available, falling back to time-only tracking")
        return track_time(timers, name, func, *args, **kwargs)


# ============================================================================
# ADVANCED FEATURE #2: GPU UTILIZATION METRICS
# Track GPU memory usage and compute utilization
# ============================================================================

def get_gpu_memory_usage() -> Optional[Dict[str, float]]:
    """Get current GPU memory usage if CUDA is available

    Returns:
        Dictionary with allocated and reserved memory in MB, or None if no GPU
    """
    try:
        import torch
        if torch.cuda.is_available():
            return {
                'allocated_mb': torch.cuda.memory_allocated() / 1024 / 1024,
                'reserved_mb': torch.cuda.memory_reserved() / 1024 / 1024,
                'max_allocated_mb': torch.cuda.max_memory_allocated() / 1024 / 1024
            }
    except:
        pass
    return None


def track_time_and_gpu(timers: Dict[str, Dict], name: str, func: Callable, *args, **kwargs) -> Any:
    """Track execution time and GPU memory usage

    Args:
        timers: Dictionary to store timing and GPU metrics
        name: Name/label for this timing
        func: Function to execute and time
        *args, **kwargs: Arguments to pass to func

    Returns:
        Whatever func returns
    """
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time

        gpu_stats = get_gpu_memory_usage()

        timers[name] = {'time': elapsed}
        if gpu_stats:
            timers[name].update(gpu_stats)
            print(f"{name} complete: {elapsed:.2f}s, GPU peak: {gpu_stats['max_allocated_mb']:.1f} MB\n")
        else:
            print(f"{name} complete: {elapsed:.2f}s (CPU mode)\n")

        return result

    except ImportError:
        return track_time(timers, name, func, *args, **kwargs)


# ============================================================================
# ADVANCED FEATURE #3: CONVERGENCE SPEED METRICS
# Track time to reach specific accuracy thresholds
# ============================================================================

def calculate_convergence_metrics(results: Dict, threshold: float = 0.01) -> Dict[str, Any]:
    """Calculate convergence speed metrics from training results

    Args:
        results: Training results dictionary containing 'test' loss arrays
        threshold: Target loss threshold

    Returns:
        Dictionary with convergence metrics (epochs to converge, etc.)
    """
    metrics = {}

    # This would be model-specific and would need the actual loss history
    # Placeholder implementation showing the concept
    for model_type, model_results in results.items():
        if isinstance(model_results, dict):
            # Find first epoch where loss drops below threshold
            # This is a simplified example - real implementation would traverse the nested structure
            pass

    return metrics


# ============================================================================
# ADVANCED FEATURE #4: COMPARATIVE SPEEDUP ANALYSIS
# Calculate speedup ratios between model types
# ============================================================================

def calculate_speedup_ratios(timers: Dict[str, float], baseline: str = None) -> Dict[str, float]:
    """Calculate speedup ratios comparing different models

    Args:
        timers: Dictionary of timing results
        baseline: Name of baseline model (if None, uses slowest)

    Returns:
        Dictionary of speedup ratios relative to baseline
    """
    if not timers:
        return {}

    if baseline is None:
        baseline = max(timers, key=timers.get)

    baseline_time = timers[baseline]
    speedups = {}

    for name, elapsed in timers.items():
        if elapsed > 0:
            speedups[name] = baseline_time / elapsed

    return speedups


def print_speedup_analysis(timers: Dict[str, float], baseline: str = None):
    """Print comparative speedup analysis

    Args:
        timers: Dictionary of timing results
        baseline: Name of baseline model for comparison
    """
    speedups = calculate_speedup_ratios(timers, baseline)

    if baseline is None:
        baseline = max(timers, key=timers.get)

    print("\n" + "="*60)
    print("SPEEDUP ANALYSIS")
    print("="*60)
    print(f"Baseline: {baseline} ({timers[baseline]:.2f}s)")
    print("-"*60)

    for name, speedup in sorted(speedups.items(), key=lambda x: x[1], reverse=True):
        if name != baseline:
            comparison = "faster" if speedup > 1 else "slower"
            print(f"{name:20s}: {speedup:5.2f}x {comparison}")

    print("="*60 + "\n")


# ============================================================================
# ADVANCED FEATURE #5: DETAILED BREAKDOWN TIMING
# Track time spent in different training phases
# ============================================================================

class DetailedTimer:
    """Context manager for tracking detailed timing breakdowns"""

    def __init__(self, name: str):
        self.name = name
        self.breakdowns = {}
        self.current_phase = None
        self.phase_start = None

    def start_phase(self, phase_name: str):
        """Start timing a specific phase"""
        if self.current_phase:
            self.end_phase()
        self.current_phase = phase_name
        self.phase_start = time.time()

    def end_phase(self):
        """End timing current phase"""
        if self.current_phase and self.phase_start:
            elapsed = time.time() - self.phase_start
            if self.current_phase in self.breakdowns:
                self.breakdowns[self.current_phase] += elapsed
            else:
                self.breakdowns[self.current_phase] = elapsed
            self.current_phase = None
            self.phase_start = None

    def print_breakdown(self):
        """Print timing breakdown by phase"""
        total = sum(self.breakdowns.values())
        print(f"\n{self.name} - Detailed Breakdown:")
        print("-" * 40)
        for phase, elapsed in sorted(self.breakdowns.items(), key=lambda x: x[1], reverse=True):
            pct = (elapsed / total * 100) if total > 0 else 0
            print(f"  {phase:20s}: {elapsed:8.2f}s ({pct:5.1f}%)")
        print("-" * 40)


# ============================================================================
# ADVANCED FEATURE #6: EXPORT TIMING DATA
# Save timing results to CSV/JSON for later analysis
# ============================================================================

def export_timing_csv(timers: Dict[str, float], filepath: str, section_name: str = ""):
    """Export timing data to CSV file

    Args:
        timers: Dictionary of timing results
        filepath: Path to save CSV file
        section_name: Optional section name to include in export
    """
    import csv

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Section', 'Model', 'Time (s)'])
        for name, elapsed in timers.items():
            writer.writerow([section_name, name, f"{elapsed:.4f}"])

    print(f"Timing data exported to: {filepath}")


def export_timing_json(timers: Dict[str, Any], filepath: str, section_name: str = ""):
    """Export timing data to JSON file

    Args:
        timers: Dictionary of timing results (can include nested dicts for advanced metrics)
        filepath: Path to save JSON file
        section_name: Optional section name to include in export
    """
    import json

    export_data = {
        'section': section_name,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'timings': timers
    }

    with open(filepath, 'w') as f:
        json.dump(export_data, f, indent=2)

    print(f"Timing data exported to: {filepath}")


# ============================================================================
# ADVANCED FEATURE #7: PREDICTION/INFERENCE TIMING
# Time model inference/prediction separately from training
# ============================================================================

def benchmark_inference(model, test_input, num_iterations: int = 100, warmup: int = 10) -> Dict[str, float]:
    """Benchmark model inference performance

    Args:
        model: Trained model to benchmark
        test_input: Test input tensor
        num_iterations: Number of inference runs to average
        warmup: Number of warmup iterations (excluded from timing)

    Returns:
        Dictionary with inference timing statistics
    """
    import torch

    model.eval()

    # Warmup iterations
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(test_input)

    # Benchmark iterations
    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start = time.time()
            _ = model(test_input)
            times.append(time.time() - start)

    return {
        'mean_ms': sum(times) / len(times) * 1000,
        'min_ms': min(times) * 1000,
        'max_ms': max(times) * 1000,
        'throughput_samples_per_sec': test_input.shape[0] * num_iterations / sum(times)
    }


def print_inference_benchmark(model, test_input, model_name: str = "Model"):
    """Run and print inference benchmark results

    Args:
        model: Trained model to benchmark
        test_input: Test input tensor
        model_name: Name of model for display
    """
    print(f"\n{model_name} Inference Benchmark:")
    print("-" * 40)

    results = benchmark_inference(model, test_input)

    print(f"  Mean latency:    {results['mean_ms']:8.3f} ms")
    print(f"  Min latency:     {results['min_ms']:8.3f} ms")
    print(f"  Max latency:     {results['max_ms']:8.3f} ms")
    print(f"  Throughput:      {results['throughput_samples_per_sec']:8.1f} samples/sec")
    print("-" * 40 + "\n")


# ============================================================================
# ADVANCED FEATURE #8: WARMUP VS STEADY-STATE TIMING
# Separate first epoch timing from remaining epochs
# ============================================================================

def track_warmup_timing(total_epochs: int, epoch_times: list) -> Dict[str, float]:
    """Analyze warmup vs steady-state timing from epoch times

    Args:
        total_epochs: Total number of epochs
        epoch_times: List of per-epoch timing measurements

    Returns:
        Dictionary with warmup and steady-state statistics
    """
    if not epoch_times or len(epoch_times) < 2:
        return {}

    first_epoch = epoch_times[0]
    remaining_epochs = epoch_times[1:]

    avg_steady = sum(remaining_epochs) / len(remaining_epochs) if remaining_epochs else 0

    return {
        'first_epoch_time': first_epoch,
        'avg_steady_state_time': avg_steady,
        'warmup_overhead': first_epoch - avg_steady,
        'warmup_overhead_pct': ((first_epoch - avg_steady) / avg_steady * 100) if avg_steady > 0 else 0
    }


def print_warmup_analysis(warmup_stats: Dict[str, float], model_name: str = "Model"):
    """Print warmup vs steady-state timing analysis

    Args:
        warmup_stats: Dictionary from track_warmup_timing
        model_name: Name of model for display
    """
    if not warmup_stats:
        return

    print(f"\n{model_name} - Warmup Analysis:")
    print("-" * 40)
    print(f"  First epoch:      {warmup_stats['first_epoch_time']:8.3f}s")
    print(f"  Steady state:     {warmup_stats['avg_steady_state_time']:8.3f}s")
    print(f"  Warmup overhead:  {warmup_stats['warmup_overhead']:8.3f}s ({warmup_stats['warmup_overhead_pct']:5.1f}%)")
    print("-" * 40 + "\n")
