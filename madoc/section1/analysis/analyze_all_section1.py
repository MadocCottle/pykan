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

# Add analysis directory to path
sys.path.insert(0, str(Path(__file__).parent))

from run_analysis import run_full_analysis


class Section1BatchAnalyzer:
    """Batch analyzer for all Section 1 experiments"""

    def __init__(self, base_dir: Path = None):
        """
        Initialize batch analyzer

        Args:
            base_dir: Base directory containing sec1_results, sec2_results, sec3_results
                     (default: parent directory of analysis/)
        """
        if base_dir is None:
            base_dir = Path(__file__).parent.parent
        self.base_dir = Path(base_dir)

        self.section_configs = {
            'section1_1': {
                'name': 'Section 1.1: Function Approximation',
                'results_dir': self.base_dir / 'sec1_results',
                'is_2d': False,
                'description': 'Sinusoids, piecewise, sawtooth, polynomial, and high-frequency functions'
            },
            'section1_2': {
                'name': 'Section 1.2: 1D Poisson PDE',
                'results_dir': self.base_dir / 'sec2_results',
                'is_2d': False,
                'description': '1D Poisson equation with various forcing functions'
            },
            'section1_3': {
                'name': 'Section 1.3: 2D Poisson PDE',
                'results_dir': self.base_dir / 'sec3_results',
                'is_2d': True,
                'description': '2D Poisson equation with sin, polynomial, high-frequency, and special forcings'
            }
        }

    def find_latest_results(self, results_dir: Path, section_name: str) -> dict:
        """
        Find latest results file and models directory for a section

        Returns:
            dict with 'results_file', 'models_dir', 'timestamp' or None if not found
        """
        if not results_dir.exists():
            return None

        # Find all results files for this section
        results_files = list(results_dir.glob(f'{section_name}_results_*.pkl'))

        if not results_files:
            return None

        # Sort by timestamp and get latest
        latest_file = sorted(results_files, key=lambda p: p.stem.split('_')[-1])[-1]
        timestamp = latest_file.stem.split('_')[-1]

        # Look for corresponding models directory
        models_dir = results_dir / f'kan_models_{timestamp}'
        if not models_dir.exists():
            models_dir = None

        return {
            'results_file': latest_file,
            'models_dir': models_dir,
            'timestamp': timestamp
        }

    def analyze_all(self, output_base_dir: str = None, specific_timestamp: str = None):
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

        print("="*80)
        print("SECTION 1 COMPLETE ANALYSIS PIPELINE")
        print("="*80)
        print(f"Output directory: {output_path.absolute()}")
        print()

        analysis_results = {}

        # Analyze each subsection
        for section_id, config in self.section_configs.items():
            print("\n" + "="*80)
            print(f"{config['name']}")
            print("="*80)

            # Find results
            if specific_timestamp:
                results_file = config['results_dir'] / f'{section_id}_results_{specific_timestamp}.pkl'
                models_dir = config['results_dir'] / f'kan_models_{specific_timestamp}'
                if not results_file.exists():
                    print(f"✗ No results found for timestamp {specific_timestamp}")
                    print(f"  Looking for: {results_file}")
                    analysis_results[section_id] = None
                    continue
                results_info = {
                    'results_file': results_file,
                    'models_dir': models_dir if models_dir.exists() else None,
                    'timestamp': specific_timestamp
                }
            else:
                results_info = self.find_latest_results(config['results_dir'], section_id)

            if results_info is None:
                print(f"✗ No results found in {config['results_dir']}")
                print(f"  Expected files like: {section_id}_results_YYYYMMDD_HHMMSS.pkl")
                analysis_results[section_id] = None
                continue

            print(f"Found results: {results_info['results_file'].name}")
            if results_info['models_dir']:
                print(f"Found models: {results_info['models_dir'].name}")
            else:
                print("No KAN models found (function fitting will be limited)")
            print(f"Timestamp: {results_info['timestamp']}")
            print()

            # Create subsection output directory
            section_output_dir = output_path / f"{section_id}_analysis"

            # Run analysis
            try:
                run_full_analysis(
                    str(results_info['results_file']),
                    str(results_info['models_dir']) if results_info['models_dir'] else None,
                    str(section_output_dir)
                )

                analysis_results[section_id] = {
                    'status': 'success',
                    'results_file': str(results_info['results_file']),
                    'models_dir': str(results_info['models_dir']) if results_info['models_dir'] else None,
                    'output_dir': str(section_output_dir),
                    'timestamp': results_info['timestamp']
                }

                print(f"✓ Analysis complete for {section_id}")

            except Exception as e:
                print(f"✗ Error analyzing {section_id}: {e}")
                import traceback
                traceback.print_exc()

                analysis_results[section_id] = {
                    'status': 'failed',
                    'error': str(e)
                }

        # Generate combined report
        print("\n" + "="*80)
        print("Generating Combined Section 1 Report")
        print("="*80)

        self._generate_thesis_report(output_path, analysis_results)

        # Save metadata
        metadata = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'base_directory': str(self.base_dir),
            'output_directory': str(output_path),
            'analysis_results': analysis_results
        }

        with open(output_path / 'analysis_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print("\n" + "="*80)
        print("SECTION 1 ANALYSIS COMPLETE")
        print("="*80)
        print(f"\nAll outputs saved to: {output_path.absolute()}")
        print(f"Thesis report: {output_path / 'SECTION1_THESIS_REPORT.md'}")
        print()

        # Print summary
        print("\nSummary:")
        for section_id, result in analysis_results.items():
            status_symbol = "✓" if result and result.get('status') == 'success' else "✗"
            print(f"  {status_symbol} {self.section_configs[section_id]['name']}")
        print()

    def _generate_thesis_report(self, output_path: Path, analysis_results: dict):
        """Generate combined thesis report for all Section 1 subsections"""

        report = f"""# Section 1: Complete Analysis Report
# Kolmogorov-Arnold Networks for Function Approximation and PDE Solving

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview

This report presents comprehensive analysis of three complementary experiments comparing Kolmogorov-Arnold Networks (KAN) with traditional neural network architectures (MLP, SIREN) across different approximation tasks.

## Experimental Setup

"""

        # Add each subsection
        for section_id in ['section1_1', 'section1_2', 'section1_3']:
            config = self.section_configs[section_id]
            result = analysis_results.get(section_id)

            report += f"""
### {config['name']}

**Description:** {config['description']}

"""

            if result and result.get('status') == 'success':
                report += f"""**Status:** ✓ Analysis complete

**Results location:** `{section_id}_analysis/`

**Key outputs:**
- Comparative metrics and learning curves
- Function fitting visualizations
"""
                if config['is_2d']:
                    report += "- 2D heatmap analysis with cross-sections and error quantiles\n"

                report += f"""
See `{section_id}_analysis/ANALYSIS_SUMMARY.md` for detailed analysis.

"""
            else:
                if result is None:
                    report += "**Status:** ✗ No results found - experiment not yet run\n\n"
                else:
                    report += f"**Status:** ✗ Analysis failed - {result.get('error', 'unknown error')}\n\n"

        # Add methodology section
        report += """
## Methodology

### Models Compared

1. **MLP (Multi-Layer Perceptron)**
   - Standard feedforward neural networks
   - Tested with multiple depths (2-6 layers)
   - Activations: tanh, relu, silu
   - Baseline for comparison

2. **SIREN (Sinusoidal Representation Networks)**
   - Periodic activation functions
   - Specialized for representing periodic and smooth functions
   - Multiple depth configurations

3. **KAN (Kolmogorov-Arnold Networks)**
   - B-spline basis functions on edges
   - Grid refinement capabilities
   - Multiple grid sizes tested (3, 5, 10, 20, 50, 100)

4. **KAN with Pruning**
   - Pruned KAN models for efficiency
   - Node threshold: 1e-2, Edge threshold: 3e-2

### Evaluation Metrics

- **Train MSE**: Mean squared error on training data
- **Test MSE**: Mean squared error on held-out test data (generalization)
- **Dense MSE**: MSE on dense sampling of the function domain (true approximation quality)
- **Training Time**: Total and per-epoch training time
- **Model Complexity**: Number of parameters (where applicable)

### Training Configuration

All models trained with:
- LBFGS optimizer
- Varying epochs depending on convergence
- Same train/test data for fair comparison

## How to Use This Analysis for Thesis Writing

### Section 1.1: Function Approximation (Introduction to KAN)

**Thesis subsection goals:**
1. Introduce KAN architecture
2. Demonstrate performance on standard function approximation
3. Compare with baseline methods

**Recommended figures:**
- `section1_1_analysis/01_comparative_metrics/all_datasets_heatmap_test.png` - Overall performance comparison
- `section1_1_analysis/02_function_fitting/function_fit_dataset_*` - Select 2-3 representative functions
- `section1_1_analysis/01_comparative_metrics/dataset_*_learning_curves_test.png` - Training dynamics

**Key points to extract:**
- Which functions KAN excels at (likely: smooth, periodic)
- Where traditional methods struggle (likely: high frequency, discontinuities)
- Training efficiency comparison

### Section 1.2: 1D Poisson PDE

**Thesis subsection goals:**
1. Extend to PDE solving
2. Show KAN performance on physics-based problems
3. Analyze different forcing functions

**Recommended figures:**
- Comparative heatmap showing PDE performance
- Learning curves for convergence analysis
- Function fits showing solution quality

**Key points to extract:**
- PDE residual quality (if available)
- Comparison of approximation for different forcing functions
- Generalization to unseen test data

### Section 1.3: 2D Poisson PDE

**Thesis subsection goals:**
1. Demonstrate scalability to higher dimensions
2. Spatial error analysis
3. Identify limitations

**Recommended figures:**
- `section1_3_analysis/03_heatmap_analysis/heatmap_*` - Spatial error distribution
- `section1_3_analysis/03_heatmap_analysis/cross_section_*` - 1D slices for detailed analysis
- Surface plots from function fitting

**Key points to extract:**
- Where in the spatial domain errors concentrate
- Edge vs interior performance
- Comparison with SIREN (known to work well for PDEs)

## Cross-Subsection Comparison

### Observations Across All Experiments

*To be filled in based on actual results - look for:*

1. **Consistent patterns:**
   - Does KAN consistently outperform on certain function types?
   - Are there consistent weaknesses?
   - Training time trade-offs

2. **Scaling behavior:**
   - Performance degradation from 1D → 2D
   - Grid size requirements for 2D vs 1D

3. **Model selection insights:**
   - When to use KAN vs traditional methods
   - Optimal hyperparameters across experiments

## Recommended Thesis Structure

### Section 1: Introduction to KAN for Function Approximation

**1.1 Basic Function Approximation**
- Introduce KAN architecture
- Compare with MLP and SIREN baselines
- Demonstrate on standard test functions
- Figure: Heatmap of performance across all functions
- Figure: 2-3 representative function fits
- Table: Final MSE comparison

**1.2 Application to 1D PDEs**
- Extend to physics-based problems
- Analyze solution quality for Poisson equation
- Figure: Solution comparison for different forcings
- Table: Convergence analysis

**1.3 Scaling to 2D PDEs**
- Demonstrate higher-dimensional capability
- Spatial error analysis
- Figure: Heatmaps with error distribution
- Figure: Cross-sections showing solution quality
- Discussion: Computational challenges

**1.4 Summary and Insights**
- When KAN outperforms traditional methods
- Computational trade-offs
- Recommended use cases

## Files and Directories

"""

        # Add file structure
        for section_id in ['section1_1', 'section1_2', 'section1_3']:
            result = analysis_results.get(section_id)
            if result and result.get('status') == 'success':
                report += f"""
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
"""
                if self.section_configs[section_id]['is_2d']:
                    report += """├── 03_heatmap_analysis/
│   ├── heatmap_*.png                   # Spatial error analysis
│   ├── cross_section_*.png             # 1D slices
│   └── error_quantile_*.png            # Error distribution
"""
                report += """└── ANALYSIS_SUMMARY.md                 # Detailed subsection report
```
"""

        report += """
## Next Steps

1. **Review individual subsection reports** in each `*_analysis/ANALYSIS_SUMMARY.md`
2. **Select key figures** for thesis based on your narrative
3. **Extract numerical results** from CSV files for tables
4. **Identify interesting patterns** across all experiments
5. **Write discussion** comparing performance across experiments

## Notes for Thesis Writing

- All figures are publication-ready (300 DPI)
- CSV files contain exact numerical values for tables
- Learning curves show training dynamics (useful for methods section)
- Error quantile analysis helps identify model limitations
- Cross-section plots are excellent for detailed discussion

---

*This report was automatically generated by the Section 1 Complete Analysis Pipeline.*
*For questions or issues, see `analysis/README.md`*
"""

        # Save report
        with open(output_path / 'SECTION1_THESIS_REPORT.md', 'w') as f:
            f.write(report)

        print(f"✓ Thesis report saved: {output_path / 'SECTION1_THESIS_REPORT.md'}")


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
    parser.add_argument('--base-dir', type=str, default=None,
                       help='Base directory containing sec*_results folders (default: parent of analysis/)')

    args = parser.parse_args()

    try:
        analyzer = Section1BatchAnalyzer(base_dir=args.base_dir)
        analyzer.analyze_all(
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