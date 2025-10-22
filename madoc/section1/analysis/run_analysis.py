"""
Main Analysis Runner - New System (v2.0)

This script runs the new analysis system combining:
- Pareto frontier analysis (from KAN paper methodology)
- Scaling law analysis
- Function fitting visualizations (enhanced)
- Heatmap analysis for 2D (from Section 1.3)
"""

import sys
from pathlib import Path
import argparse
from datetime import datetime
import importlib.util

# Add analysis directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

# Import analysis modules
try:
    from . import io
    from . import report_utils as ru
    from .pareto_analysis import ParetoAnalyzer
    from .scaling_laws import ScalingLawAnalyzer
    from .function_fitting import FunctionFittingVisualizer
    from .heatmap_2d_fits import Heatmap2DAnalyzer
except ImportError:
    # Allow running as script - import local modules directly using importlib
    spec = importlib.util.spec_from_file_location('io', Path(__file__).parent / 'io.py')
    io = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(io)
    import report_utils as ru
    from pareto_analysis import ParetoAnalyzer
    from scaling_laws import ScalingLawAnalyzer
    from function_fitting import FunctionFittingVisualizer
    from heatmap_2d_fits import Heatmap2DAnalyzer


def run_full_analysis(results_file: str, models_dir: str = None, output_base_dir: str = None):
    """
    Run complete analysis pipeline (new system)

    Args:
        results_file: Path to results file (.pkl or .json)
        models_dir: Optional path to saved models directory
        output_base_dir: Base directory for all outputs
    """
    results_path = Path(results_file)

    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")

    # Detect section type from filename
    filename = results_path.stem
    section_type = None
    for sec in io.SECTIONS:
        if sec in filename:
            section_type = sec
            break
    if not section_type:
        raise ValueError(f"Cannot detect section type from filename: {filename}")

    is_2d = io.is_2d(section_type)

    ru.print_separator()
    print("Section 1 Analysis Pipeline (v2.0 - KAN Paper Methodology)")
    ru.print_separator()
    print(f"Results file: {results_path}")
    print(f"Section type: {section_type}")
    print(f"Dimension: {ru.get_dimensionality_text(is_2d)}")
    if models_dir:
        print(f"Models directory: {models_dir}")
    ru.print_separator()
    print()

    # Create output directory
    if output_base_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_base_dir = f'analysis_output_{section_type}_{timestamp}'

    output_path = ru.create_output_dir(output_base_dir)
    print()

    # Load results
    import pickle
    with open(results_path, 'rb') as f:
        results = pickle.load(f)

    # Load metadata
    import json
    metadata = {}
    json_file = results_path.with_suffix('.json')
    if json_file.exists():
        with open(json_file, 'r') as f:
            data = json.load(f)
            metadata = data.get('meta', {})

    # ========================================================================
    # Step 1: Pareto Frontier Analysis (NEW)
    # ========================================================================
    ru.print_separator()
    print("Step 1: Pareto Frontier Analysis")
    ru.print_separator()

    pareto_dir = output_path / '01_pareto_analysis'
    pareto_analyzer = ParetoAnalyzer(results, metadata)
    pareto_analyzer.analyze_all_datasets(str(pareto_dir))

    print()

    # ========================================================================
    # Step 2: Scaling Law Analysis (NEW)
    # ========================================================================
    ru.print_separator()
    print("Step 2: Scaling Law Analysis")
    ru.print_separator()

    scaling_dir = output_path / '01_pareto_analysis'  # Same dir as Pareto
    scaling_analyzer = ScalingLawAnalyzer(results, metadata)
    scaling_analyzer.analyze_all_datasets(str(scaling_dir))

    print()

    # ========================================================================
    # Step 3: Function Fitting Visualizations (ENHANCED)
    # ========================================================================
    ru.print_separator()
    print("Step 3: Function Fitting Visualizations")
    ru.print_separator()

    fitting_dir = output_path / '02_function_fitting'
    try:
        fitting_viz = FunctionFittingVisualizer(str(results_path), models_dir)
        fitting_viz.analyze_all_datasets(str(fitting_dir))
    except Exception as e:
        ru.print_status(f"Error in function fitting: {e}", success=False)

    print()

    # ========================================================================
    # Step 4: Heatmap Analysis (2D only, from Section 1.3)
    # ========================================================================
    if is_2d:
        ru.print_separator()
        print("Step 4: Heatmap Analysis (2D)")
        ru.print_separator()

        heatmap_dir = output_path / '03_heatmap_analysis'
        try:
            heatmap_analyzer = Heatmap2DAnalyzer(str(results_path), models_dir)
            heatmap_analyzer.analyze_all_datasets(str(heatmap_dir))
        except Exception as e:
            ru.print_status(f"Error in heatmap analysis: {e}", success=False)

        print()

    # ========================================================================
    # Generate Summary Report
    # ========================================================================
    ru.print_separator()
    print("Generating Analysis Summary")
    ru.print_separator()

    summary_path = output_path / 'ANALYSIS_SUMMARY.md'
    generate_summary_report(
        summary_path,
        results_path,
        section_type,
        is_2d,
        models_dir,
        output_path
    )

    ru.print_status(f"Analysis summary saved: {summary_path.name}")
    print()
    ru.print_separator()
    print("Analysis Complete!")
    ru.print_separator()
    print(f"All outputs saved to: {output_path.absolute()}")
    print()


def generate_summary_report(summary_path: Path, results_path: Path, section_type: str,
                           is_2d: bool, models_dir: str, output_path: Path):
    """Generate analysis summary report"""

    report = f"""# Analysis Summary Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Input Data

- **Results File:** `{results_path.name}`
- **Section Type:** {section_type}
- **Dimensionality:** {'2D' if is_2d else '1D'}
- **Models Directory:** {models_dir if models_dir else 'N/A'}

## Generated Outputs

### 1. Pareto Frontier Analysis (NEW)

Location: `01_pareto_analysis/`

This directory contains:
- **Pareto frontier plots** - Log-log plots of test RMSE vs. parameter count
- **Best models tables** - CSV files with Pareto-optimal models per dataset
- **Scaling law plots** - Power-law fits showing architecture efficiency
- **Scaling summaries** - α exponent comparisons across architectures

**Key insights:**
- Pareto-optimal models minimize both error AND parameter count
- Higher α = architecture scales better with added parameters
- Direct comparison to Liu et al. (2024) KAN paper methodology

Files generated:
- `pareto_frontier_<N>.png` - Pareto frontier visualization
- `best_models_<N>.csv` - Pareto-optimal model list
- `scaling_laws_<N>.png` - Scaling law curves with α values
- `scaling_summary_<N>.csv` - Architecture scaling comparison

### 2. Function Fitting Visualizations

Location: `02_function_fitting/`

This directory contains visualizations comparing neural network predictions with true functions:

{'For 2D functions:' if is_2d else 'For 1D functions:'}
{'''- **Surface plots** showing true function vs NN prediction
- **Contour plots** for easier comparison''' if is_2d else '''- **Line plots** showing true function vs NN prediction
- **Residual plots** showing prediction errors'''}
- **MSE calculations** displayed on each plot
- **Parameter counts** in subplot titles (NEW)

Each visualization shows all model types (MLP, SIREN, KAN, KAN with pruning) side-by-side.

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

**This component is from Section 1.3 - kept as-is due to excellent quality!**
"""

    report += """
## How to Interpret Results

### Pareto Frontier Plots
- **X-axis**: Parameter count (log scale)
- **Y-axis**: Test RMSE (log scale)
- **Pareto-optimal models**: Highlighted with bold markers and black edges
- **Best architecture**: Lower-left = better (fewer parameters, lower error)

### Scaling Laws
- **Formula**: RMSE = A × N^(-α) where N = parameter count
- **α exponent**: Higher = better scaling efficiency
- **Interpretation**: If KAN has α=0.5 and MLP has α=0.3, KAN gains more accuracy per parameter added

### Best Models Table
- Only Pareto-optimal models (or top 3 per architecture if none)
- Sorted by test RMSE (ascending)
- Use this to identify: "What's the best model per architecture?"

## Methodology Notes

This analysis follows the comparison protocol from:
**Liu, Z., et al. (2024). KAN: Kolmogorov-Arnold Networks. arXiv:2404.19756**

Key principles:
1. Fair comparisons via parameter-aware metrics
2. Pareto optimality over exhaustive sweeps
3. Scaling laws reveal fundamental architecture efficiency
4. Focus on interpretability and actionable insights

---

*Analysis generated by Section 1 Analysis Pipeline v2.0*
"""

    with open(summary_path, 'w') as f:
        f.write(report)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Section 1 Analysis Pipeline (v2.0)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze latest results
  python run_analysis.py path/to/results.pkl

  # With models directory
  python run_analysis.py path/to/results.pkl --models-dir path/to/models

  # Custom output directory
  python run_analysis.py path/to/results.pkl --output-dir my_analysis
        """
    )

    parser.add_argument('results_file', type=str,
                       help='Path to results file (.pkl)')
    parser.add_argument('--models-dir', type=str, default=None,
                       help='Path to saved models directory')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: auto-generated with timestamp)')

    args = parser.parse_args()

    try:
        run_full_analysis(
            args.results_file,
            models_dir=args.models_dir,
            output_base_dir=args.output_dir
        )
    except Exception as e:
        print(f"\n✗ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
