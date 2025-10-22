"""Section 1 Analysis Package - New System (v2.0)"""
from .io import load_run, is_2d, SECTIONS
from .pareto_analysis import ParetoAnalyzer
from .scaling_laws import ScalingLawAnalyzer
from .function_fitting import FunctionFittingVisualizer
from .heatmap_2d_fits import Heatmap2DAnalyzer
from .run_analysis import run_full_analysis
