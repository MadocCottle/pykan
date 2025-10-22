"""
Batch Analysis Runner for Complete Section 1

This script automatically discovers and analyzes all Section 1 experiment results:
- Section 1.1: Function Approximation (sec1_results/)
- Section 1.2: 1D Poisson PDE (sec2_results/)
- Section 1.3: 2D Poisson PDE (sec3_results/)

Usage:
    # Analyze latest results from all subsections
    python analyze_all_section1.py

    # Specify custom output directory
    python analyze_all_section1.py --output-dir my_thesis_analysis

    # Analyze specific timestamps
    python analyze_all_section1.py --timestamp 20251021_110446
"""

import sys
from pathlib import Path
import argparse
from datetime import datetime
import json
import importlib.util

# Add analysis directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from . import io
    from . import report_utils as ru
    from .run_analysis import run_full_analysis
except ImportError:
    # Allow running as script - import local modules directly using importlib
    # (regular import io conflicts with built-in io module)
    spec = importlib.util.spec_from_file_location('io', Path(__file__).parent / 'io.py')
    io = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(io)
    import report_utils as ru
    from run_analysis import run_full_analysis


def analyze_all_sections(output_base_dir: str = None, specific_timestamp: str = None):
    """
    Run analysis on all Section 1 subsections

    Args:
        output_base_dir: Base directory for all outputs
        specific_timestamp: Optional specific timestamp to analyze (instead of latest)
    """
    # Create output directory
    if output_base_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_base_dir = f'section1_complete_analysis_{timestamp}'

    output_path = Path(output_base_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    ru.print_separator('=', 80)
    print("SECTION 1 COMPLETE ANALYSIS PIPELINE")
    ru.print_separator('=', 80)
    print(f"Output directory: {output_path.absolute()}")
    print()

    analysis_results = {}

    # Analyze each subsection
    section_names = {
        'section1_1': 'Section 1.1: Function Approximation',
        'section1_2': 'Section 1.2: 1D Poisson PDE',
        'section1_3': 'Section 1.3: 2D Poisson PDE'
    }

    for section_id in io.SECTIONS:
        ru.print_separator('=', 80)
        print(f"{section_names[section_id]}")
        ru.print_separator('=', 80)

        # Find results
        try:
            results, metadata, models_dir = io.load_run(section_id, timestamp=specific_timestamp)

            # Get results file path for display
            # This file is in section1/analysis/, so parent is section1/
            results_dir = Path(__file__).parent.parent / 'results' / io.SECTION_DIRS[section_id]
            if specific_timestamp:
                ts = specific_timestamp
            else:
                # Extract timestamp: section1_1_20251022_144828.pkl -> 20251022_144828
                ts = '_'.join(sorted(results_dir.glob(f'{section_id}_*.pkl'))[-1].stem.split('_')[2:])
            results_file = results_dir / f'{section_id}_{ts}.pkl'

            print(f"Found results: {results_file.name}")
            if models_dir:
                print(f"Found models: {Path(models_dir).name}")
            else:
                print("No KAN models found (function fitting will be limited)")
            print(f"Timestamp: {ts}")
            print()

            # Create subsection output directory
            section_output_dir = output_path / f"{section_id}_analysis"

            # Run analysis
            run_full_analysis(
                str(results_file),
                models_dir,
                str(section_output_dir)
            )

            analysis_results[section_id] = {
                'status': 'success',
                'results_file': str(results_file),
                'models_dir': models_dir,
                'output_dir': str(section_output_dir),
                'timestamp': ts
            }

            ru.print_status(f"Analysis complete for {section_id}")

        except FileNotFoundError as e:
            print(f"✗ No results found")
            print(f"  {e}")
            analysis_results[section_id] = None
        except Exception as e:
            ru.print_status(f"Error analyzing {section_id}: {e}", success=False)
            import traceback
            traceback.print_exc()

            analysis_results[section_id] = {
                'status': 'failed',
                'error': str(e)
            }

    # Generate combined report
    ru.print_separator('=', 80)
    print("Generating Combined Section 1 Report")
    ru.print_separator('=', 80)

    generate_thesis_report(output_path, analysis_results)

    # Save metadata
    metadata = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'output_directory': str(output_path),
        'analysis_results': analysis_results
    }

    with open(output_path / 'analysis_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    ru.print_separator('=', 80)
    print("SECTION 1 ANALYSIS COMPLETE")
    ru.print_separator('=', 80)
    print(f"\nAll outputs saved to: {output_path.absolute()}")
    print(f"Thesis report: {output_path / 'SECTION1_THESIS_REPORT.md'}")
    print()

    # Print summary
    print("\nSummary:")
    section_names = {
        'section1_1': 'Section 1.1: Function Approximation',
        'section1_2': 'Section 1.2: 1D Poisson PDE',
        'section1_3': 'Section 1.3: 2D Poisson PDE'
    }
    for section_id, result in analysis_results.items():
        status_symbol = "✓" if result and result.get('status') == 'success' else "✗"
        print(f"  {status_symbol} {section_names[section_id]}")
    print()


def generate_thesis_report(output_path: Path, analysis_results: dict):
    """Generate combined thesis report using template"""

    section_info = {
        'section1_1': {
            'name': 'Section 1.1: Function Approximation',
            'description': 'Sinusoids, piecewise, sawtooth, polynomial, and high-frequency functions',
            'is_2d': False
        },
        'section1_2': {
            'name': 'Section 1.2: 1D Poisson PDE',
            'description': '1D Poisson equation with various forcing functions',
            'is_2d': False
        },
        'section1_3': {
            'name': 'Section 1.3: 2D Poisson PDE',
            'description': '2D Poisson equation with sin, polynomial, high-frequency, and special forcings',
            'is_2d': True
        }
    }

    # Build experimental sections text
    experimental_sections = []
    for section_id in io.SECTIONS:
        config = section_info[section_id]
        result = analysis_results.get(section_id)

        section_text = f"""
### {config['name']}

**Description:** {config['description']}

"""

        if result and result.get('status') == 'success':
            section_text += f"""**Status:** ✓ Analysis complete

**Results location:** `{section_id}_analysis/`

**Key outputs:**
- Comparative metrics and learning curves
- Function fitting visualizations
"""
            if config['is_2d']:
                section_text += "- 2D heatmap analysis with cross-sections and error quantiles\n"

            section_text += f"""
See `{section_id}_analysis/ANALYSIS_SUMMARY.md` for detailed analysis.

"""
        else:
            if result is None:
                section_text += "**Status:** ✗ No results found - experiment not yet run\n\n"
            else:
                section_text += f"**Status:** ✗ Analysis failed - {result.get('error', 'unknown error')}\n\n"

        experimental_sections.append(section_text)

    # Build file structure text
    file_structure = []
    for section_id, result in analysis_results.items():
        if result and result.get('status') == 'success':
            config = section_info[section_id]
            file_structure.append(f"""
### {section_id}

```
{section_id}_analysis/
├── 01_comparative_metrics/
│   ├── *_comparison_table.csv          # Detailed metrics tables
│   ├── *_learning_curves_*.png         # Training dynamics
│   ├── *_training_times.png            # Computational cost
│   └── all_datasets_heatmap_*.png      # Overall performance
├── 02_function_fitting/
│   └── function_fit_dataset_*.png      # Visual comparison with true functions
""")
            if config['is_2d']:
                file_structure.append("""├── 03_heatmap_analysis/
│   ├── heatmap_*.png                   # Spatial error analysis
│   ├── cross_section_*.png             # 1D slices
│   └── error_quantile_*.png            # Error distribution
""")
            file_structure.append("""└── ANALYSIS_SUMMARY.md                 # Detailed subsection report
```
""")

    # Render template
    template_path = ru.load_thesis_report_template()

    variables = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'experimental_sections': ''.join(experimental_sections),
        'file_structure': ''.join(file_structure),
    }

    report = ru.render_template(template_path, **variables)

    # Save report
    with open(output_path / 'SECTION1_THESIS_REPORT.md', 'w') as f:
        f.write(report)

    ru.print_status(f"Thesis report saved: {output_path / 'SECTION1_THESIS_REPORT.md'}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Batch analysis for all Section 1 experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze latest results from all subsections
  python analyze_all_section1.py

  # Specify output directory
  python analyze_all_section1.py --output-dir thesis_analysis

  # Analyze specific timestamp
  python analyze_all_section1.py --timestamp 20251021_110446

  # Use custom base directory
  python analyze_all_section1.py --base-dir /path/to/section1
        """
    )

    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: auto-generated with timestamp)')
    parser.add_argument('--timestamp', type=str, default=None,
                       help='Specific timestamp to analyze (default: latest)')

    args = parser.parse_args()

    try:
        analyze_all_sections(
            output_base_dir=args.output_dir,
            specific_timestamp=args.timestamp
        )
    except Exception as e:
        print(f"\n✗ Batch analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
