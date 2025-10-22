"""
Main Analysis Runner

This script runs all analysis and visualization scripts on a given results file.
It automatically detects whether results are 1D or 2D and generates appropriate visualizations.
"""

import sys
from pathlib import Path
import argparse
from datetime import datetime

# Add analysis directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

# Import analysis modules
try:
    from . import data_io
    from . import report_utils as ru
    from .comparative_metrics import MetricsAnalyzer
    from .function_fitting import FunctionFittingVisualizer
    from .heatmap_2d_fits import Heatmap2DAnalyzer
except ImportError:
    # Allow running as script
    import data_io
    import report_utils as ru
    from comparative_metrics import MetricsAnalyzer
    from function_fitting import FunctionFittingVisualizer
    from heatmap_2d_fits import Heatmap2DAnalyzer


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

    # Detect section type using centralized function
    section_type = data_io.detect_section_type(results_path)
    is_2d = data_io.is_2d_section(section_type)
    section_config = data_io.get_section_config(section_type)

    ru.print_separator()
    print("Section 1 Analysis Pipeline")
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

    # ========================================================================
    # Step 1: Comparative Metrics Analysis
    # ========================================================================
    ru.print_section_header("Step 1: Generating Comparative Metrics Visualizations")

    metrics_dir = output_path / '01_comparative_metrics'
    metrics_dir.mkdir(exist_ok=True)

    try:
        metrics_analyzer = MetricsAnalyzer(str(results_path))
        metrics_analyzer.generate_all_visualizations(str(metrics_dir))
        ru.print_status("Comparative metrics analysis complete")
    except Exception as e:
        ru.print_status(f"Error in comparative metrics analysis: {e}", success=False)
        import traceback
        traceback.print_exc()

    # ========================================================================
    # Step 2: Function Fitting Visualizations
    # ========================================================================
    ru.print_section_header("Step 2: Generating Function Fitting Visualizations")

    function_fit_dir = output_path / '02_function_fitting'
    function_fit_dir.mkdir(exist_ok=True)

    try:
        fitting_visualizer = FunctionFittingVisualizer(str(results_path), models_dir)
        fitting_visualizer.generate_all_function_fits(str(function_fit_dir), is_2d=is_2d)
        ru.print_status("Function fitting visualizations complete")
    except Exception as e:
        ru.print_status(f"Error in function fitting visualization: {e}", success=False)
        import traceback
        traceback.print_exc()

    # ========================================================================
    # Step 3: 2D Heatmap Analysis (only for 2D sections)
    # ========================================================================
    if is_2d:
        ru.print_section_header("Step 3: Generating 2D Heatmap Analysis")

        heatmap_dir = output_path / '03_heatmap_analysis'
        heatmap_dir.mkdir(exist_ok=True)

        try:
            heatmap_analyzer = Heatmap2DAnalyzer(str(results_path), models_dir)
            heatmap_analyzer.generate_all_heatmaps(str(heatmap_dir))
            ru.print_status("Heatmap analysis complete")
        except Exception as e:
            ru.print_status(f"Error in heatmap analysis: {e}", success=False)
            import traceback
            traceback.print_exc()
    else:
        print("\n(Skipping 2D heatmap analysis for 1D data)")

    # ========================================================================
    # Generate Summary Report
    # ========================================================================
    ru.print_section_header("Generating Summary Report")

    summary = generate_summary_report(results_path, section_config, is_2d, models_dir)
    summary_path = output_path / 'ANALYSIS_SUMMARY.md'

    with open(summary_path, 'w') as f:
        f.write(summary)

    ru.print_status(f"Summary report saved to: {summary_path}")

    # ========================================================================
    # Complete
    # ========================================================================
    ru.print_section_header("Analysis Complete!")
    print(f"\nAll outputs saved to: {output_path.absolute()}")
    print(f"Summary report: {summary_path.absolute()}")
    print()


def generate_summary_report(results_path: Path, section_config: dict, is_2d: bool, models_dir: str) -> str:
    """
    Generate markdown summary report using template

    Args:
        results_path: Path to results file
        section_config: Section configuration dict
        is_2d: Whether data is 2D
        models_dir: Models directory path or None

    Returns:
        Rendered report string
    """
    template_path = ru.load_analysis_summary_template()

    variables = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'results_filename': results_path.name,
        'section_type': next(k for k, v in data_io.SECTION_CONFIG.items()
                            if v == section_config),
        'dimensionality': ru.get_dimensionality_text(is_2d),
        'models_dir': ru.format_models_dir(models_dir),
        'function_fitting_details': ru.get_function_fitting_details(is_2d),
        'heatmap_section': ru.get_heatmap_section(is_2d),
        'detailed_analysis_extra': ru.get_detailed_analysis_extra(is_2d),
    }

    return ru.render_template(template_path, **variables)


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
