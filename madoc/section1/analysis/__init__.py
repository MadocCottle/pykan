"""
Section 1 Analysis Package

Comprehensive analysis and visualization tools for PyKAN Section 1 experiments.

Modules:
    comparative_metrics - Compare models across metrics, epochs, and training times
    function_fitting - Visualize how well NNs approximate true functions
    heatmap_2d_fits - Detailed spatial analysis for 2D equations
    run_analysis - Main runner to orchestrate all analyses

Usage:
    # Import individual analyzers
    from analysis.comparative_metrics import MetricsAnalyzer
    from analysis.function_fitting import FunctionFittingVisualizer
    from analysis.heatmap_2d_fits import Heatmap2DAnalyzer

    # Or use the main runner
    from analysis.run_analysis import run_full_analysis

    run_full_analysis('results.pkl', models_dir='models/', output_dir='output/')
"""

__version__ = '1.0.0'
__author__ = 'PyKAN Analysis Tools'

# Make key classes available at package level
try:
    from .comparative_metrics import MetricsAnalyzer
    from .function_fitting import FunctionFittingVisualizer
    from .heatmap_2d_fits import Heatmap2DAnalyzer
    from .run_analysis import run_full_analysis

    __all__ = [
        'MetricsAnalyzer',
        'FunctionFittingVisualizer',
        'Heatmap2DAnalyzer',
        'run_full_analysis'
    ]
except ImportError:
    # Allow package to be imported even if dependencies aren't installed
    pass
