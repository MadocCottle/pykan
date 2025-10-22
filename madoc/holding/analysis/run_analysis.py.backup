"""
Main Analysis Runner

This script runs all analysis and visualization scripts on a given results file.
It automatically detects whether results are 1D or 2D and generates appropriate visualizations.
"""

import sys
from pathlib import Path
import argparse
import json
from datetime import datetime

# Add analysis directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

# Import analysis modules (now in the same directory)
from comparative_metrics import MetricsAnalyzer
from function_fitting import FunctionFittingVisualizer
from heatmap_2d_fits import Heatmap2DAnalyzer


def detect_section_type(results_path: Path) -> str:
    """
    Detect which section the results are from based on filename

    Returns:
        'section1_1', 'section1_2', or 'section1_3'
    """
    filename = results_path.stem
    if 'section1_1' in filename:
        return 'section1_1'
    elif 'section1_2' in filename:
        return 'section1_2'
    elif 'section1_3' in filename:
        return 'section1_3'
    else:
        raise ValueError(f"Cannot detect section type from filename: {filename}")


def is_2d_section(section_type: str) -> bool:
    """Check if section contains 2D data"""
    return section_type == 'section1_3'


def run_full_analysis(results_file: str, models_dir: str = None, output_base_dir: str = None):
    """
    Run complete analysis pipeline

    Args:
        results_file: Path to results file (.pkl or .json)
        models_dir: Optional path to saved models directory
        output_base_dir: Base directory for all outputs (default: analysis_output_<timestamp>)
    """
    results_path = Path(results_file)

    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")

    # Detect section type
    section_type = detect_section_type(results_path)
    is_2d = is_2d_section(section_type)

    print("="*70)
    print("Section 1 Analysis Pipeline")
    print("="*70)
    print(f"Results file: {results_path}")
    print(f"Section type: {section_type}")
    print(f"Dimension: {'2D' if is_2d else '1D'}")
    if models_dir:
        print(f"Models directory: {models_dir}")
    print("="*70)
    print()

    # Create output directory
    if output_base_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_base_dir = f'analysis_output_{section_type}_{timestamp}'

    output_path = Path(output_base_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    print(f"Output directory: {output_path.absolute()}")
    print()

    # ========================================================================
    # Step 1: Comparative Metrics Analysis
    # ========================================================================
    print("\n" + "="*70)
    print("Step 1: Generating Comparative Metrics Visualizations")
    print("="*70)

    metrics_dir = output_path / '01_comparative_metrics'
    metrics_dir.mkdir(exist_ok=True)

    try:
        metrics_analyzer = MetricsAnalyzer(str(results_path))
        metrics_analyzer.generate_all_visualizations(str(metrics_dir))
        print("✓ Comparative metrics analysis complete")
    except Exception as e:
        print(f"✗ Error in comparative metrics analysis: {e}")
        import traceback
        traceback.print_exc()

    # ========================================================================
    # Step 2: Function Fitting Visualizations
    # ========================================================================
    print("\n" + "="*70)
    print("Step 2: Generating Function Fitting Visualizations")
    print("="*70)

    function_fit_dir = output_path / '02_function_fitting'
    function_fit_dir.mkdir(exist_ok=True)

    try:
        fitting_visualizer = FunctionFittingVisualizer(str(results_path), models_dir)
        fitting_visualizer.generate_all_function_fits(str(function_fit_dir), is_2d=is_2d)
        print("✓ Function fitting visualizations complete")
    except Exception as e:
        print(f"✗ Error in function fitting visualization: {e}")
        import traceback
        traceback.print_exc()

    # ========================================================================
    # Step 3: 2D Heatmap Analysis (only for 2D sections)
    # ========================================================================
    if is_2d:
        print("\n" + "="*70)
        print("Step 3: Generating 2D Heatmap Analysis")
        print("="*70)

        heatmap_dir = output_path / '03_heatmap_analysis'
        heatmap_dir.mkdir(exist_ok=True)

        try:
            heatmap_analyzer = Heatmap2DAnalyzer(str(results_path), models_dir)
            heatmap_analyzer.generate_all_heatmaps(str(heatmap_dir))
            print("✓ Heatmap analysis complete")
        except Exception as e:
            print(f"✗ Error in heatmap analysis: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n(Skipping 2D heatmap analysis for 1D data)")

    # ========================================================================
    # Generate Summary Report
    # ========================================================================
    print("\n" + "="*70)
    print("Generating Summary Report")
    print("="*70)

    summary = generate_summary_report(results_path, output_path, section_type, is_2d, models_dir)
    summary_path = output_path / 'ANALYSIS_SUMMARY.md'

    with open(summary_path, 'w') as f:
        f.write(summary)

    print(f"✓ Summary report saved to: {summary_path}")

    # ========================================================================
    # Complete
    # ========================================================================
    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)
    print(f"\nAll outputs saved to: {output_path.absolute()}")
    print(f"Summary report: {summary_path.absolute()}")
    print()


def generate_summary_report(results_path: Path, output_path: Path,
                           section_type: str, is_2d: bool, models_dir: str) -> str:
    """Generate markdown summary report"""

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    report = f"""# Analysis Summary Report

**Generated:** {timestamp}

## Input Data

- **Results File:** `{results_path.name}`
- **Section Type:** {section_type}
- **Dimensionality:** {'2D' if is_2d else '1D'}
- **Models Directory:** {models_dir if models_dir else 'Not provided'}

## Generated Outputs

### 1. Comparative Metrics Analysis

Location: `01_comparative_metrics/`

This directory contains:
- **Comparison tables** (CSV) for each dataset showing final MSE, training times, etc.
- **Learning curves** showing train/test/dense MSE over epochs
- **Training time comparisons** across all model types
- **Performance heatmaps** showing final MSE across all datasets and models

Files generated:
- `dataset_<N>_comparison_table.csv` - Detailed metrics for dataset N
- `dataset_<N>_learning_curves_<metric>.png` - Learning curves for each metric
- `dataset_<N>_training_times.png` - Training time comparisons
- `all_datasets_heatmap_<metric>.png` - Overall performance heatmaps

### 2. Function Fitting Visualizations

Location: `02_function_fitting/`

This directory contains visualizations comparing neural network predictions with true functions:

"""

    if is_2d:
        report += """
For 2D functions:
- **Surface plots** showing true function vs NN prediction
- **Contour plots** for easier comparison
- **MSE calculations** displayed on each plot

Each visualization shows all model types (MLP, SIREN, KAN, KAN with pruning) side-by-side.
"""
    else:
        report += """
For 1D functions:
- **Line plots** showing true function vs NN output
- **Point-by-point comparisons** across the domain
- **MSE calculations** for each model

Each visualization compares all model types (MLP, SIREN, KAN, KAN with pruning).
"""

    if is_2d:
        report += """
### 3. Heatmap Analysis (2D Only)

Location: `03_heatmap_analysis/`

Detailed heatmap analysis for 2D functions:
- **Comparison heatmaps** - Side-by-side views of true function, prediction, and error
- **Error analysis** - Absolute error, signed error, and relative error maps
- **Cross-section plots** - 1D slices at fixed x₁ and x₂ values
- **Error quantile maps** - Identifying high-error regions
- **Error statistics** - Quantitative breakdown by region

Files generated per function and model:
- `heatmap_<N>_<function>_<model>.png` - Detailed comparison heatmaps
- `cross_section_<N>_<function>.png` - Cross-section comparisons
- `error_quantile_<N>_<function>_<model>.png` - Error quantile analysis
"""

    report += """
## How to Use These Results

### Quick Start

1. **Start with the heatmaps** in `01_comparative_metrics/all_datasets_heatmap_test.png` to see overall model performance
2. **Review learning curves** to understand training dynamics
3. **Examine function fits** to see how well models approximate each function
4. **Check training times** if computational efficiency is important

### Detailed Analysis

For each dataset/function of interest:

1. Open the **comparison table CSV** to see exact numerical values
2. Look at **learning curves** to check for overfitting or convergence issues
3. Examine **function fitting plots** to visually assess approximation quality
"""

    if is_2d:
        report += """4. Review **heatmap analysis** to identify problematic regions in the domain
5. Check **cross-sections** to understand behavior along specific dimensions
6. Use **error quantile maps** to find where models struggle most
"""

    report += """
### Key Metrics

- **Train MSE**: Error on training data (lower is better)
- **Test MSE**: Error on test data (lower is better, indicates generalization)
- **Dense MSE**: Dense sampling MSE (better indicator of true approximation quality)
- **Total Time**: Complete training time in seconds
- **Time per Epoch**: Average time per training epoch

## Model Comparison

The analysis compares four model types:

1. **MLP**: Traditional multilayer perceptron with various depths and activations
2. **SIREN**: Sinusoidal activation networks, specialized for periodic functions
3. **KAN**: Kolmogorov-Arnold Networks with various grid sizes
4. **KAN with Pruning**: Pruned KAN models for efficiency

## Notes

- MSE values are on a log scale in learning curves for better visualization
- Relative error in heatmaps is capped at 100% for visualization
- Cross-sections are taken at x = 0.25, 0.5, and 0.75
- Error quantiles divide errors into 5 regions: Q1 (0-25%), Q2 (25-50%), Q3 (50-75%), Q4 (75-90%), Q5 (>90%)

## Next Steps

Based on this analysis, you can:

1. Identify which model type performs best for your application
2. Determine if more training epochs are needed (check learning curves)
3. Find problematic regions in the domain that need special attention
4. Balance accuracy vs computational cost using the timing data
5. Make informed decisions about hyperparameter tuning

---

*This report was automatically generated by the Section 1 Analysis Pipeline.*
"""

    return report


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Run complete analysis on Section 1 experiment results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze results with saved models
  python run_analysis.py results.pkl --models-dir ./kan_models_20251021_110446

  # Analyze results without models (limited visualizations)
  python run_analysis.py results.pkl

  # Specify custom output directory
  python run_analysis.py results.pkl --output-dir my_analysis

  # Analyze from results directory
  python run_analysis.py sec1_results/section1_1_results_20251021_110446.pkl \\
         --models-dir sec1_results/kan_models_20251021_110446
        """
    )

    parser.add_argument('results_file', type=str,
                       help='Path to results file (.pkl or .json)')
    parser.add_argument('--models-dir', type=str, default=None,
                       help='Path to saved KAN models directory (optional but recommended)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: auto-generated with timestamp)')

    args = parser.parse_args()

    try:
        run_full_analysis(args.results_file, args.models_dir, args.output_dir)
    except Exception as e:
        print(f"\n✗ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
