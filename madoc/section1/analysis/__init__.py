"""
Section 1 Analysis Package

Comprehensive analysis and visualization tools for PyKAN Section 1 experiments.

Modules:
    data_io - Centralized I/O for loading results, metadata, and models
    comparative_metrics - Compare models across metrics, epochs, and training times
    function_fitting - Visualize how well NNs approximate true functions
    heatmap_2d_fits - Detailed spatial analysis for 2D equations
    run_analysis - Main runner to orchestrate all analyses

Usage:
    # Load results using centralized IO
    from analysis import data_io
    results, metadata = data_io.load_results('section1_1')  # or explicit path

    # Import individual analyzers
    from analysis.comparative_metrics import MetricsAnalyzer
    from analysis.function_fitting import FunctionFittingVisualizer
    from analysis.heatmap_2d_fits import Heatmap2DAnalyzer

    # Analyzers now support both explicit paths and section IDs
    analyzer = MetricsAnalyzer('section1_1')  # Auto-loads latest results
    # or
    analyzer = MetricsAnalyzer('/path/to/results.pkl')  # Explicit path

    # Or use the main runner
    from analysis.run_analysis import run_full_analysis

    run_full_analysis('results.pkl', models_dir='models/', output_dir='output/')
"""

__version__ = '1.1.0'
__author__ = 'PyKAN Analysis Tools'

# Make key classes and modules available at package level
try:
    from . import data_io
    from .comparative_metrics import MetricsAnalyzer
    from .function_fitting import FunctionFittingVisualizer
    from .heatmap_2d_fits import Heatmap2DAnalyzer
    from .run_analysis import run_full_analysis

    __all__ = [
        'data_io',
        'MetricsAnalyzer',
        'FunctionFittingVisualizer',
        'Heatmap2DAnalyzer',
        'run_full_analysis'
    ]
except ImportError:
    # Allow package to be imported even if dependencies aren't installed
    pass
