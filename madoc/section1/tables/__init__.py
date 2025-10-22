"""
Section 1 Tables Package

This package contains table generation scripts for Section 1 experimental results.

Available tables:
- table0_executive_summary: High-level overview of all results
- table1_function_approximation: Function approximation comparison (Section 1.1)
- table2_pde_1d_comparison: 1D PDE comparison (Section 1.2)
- table3_pde_2d_comparison: 2D PDE comparison (Section 1.3)
- table4_param_efficiency: Parameter efficiency analysis
- table5_convergence_summary: Training efficiency analysis
- table6_grid_ablation: KAN grid size ablation study
- table7_depth_ablation: Depth ablation for MLP/SIREN

Usage:
    from tables import table1_function_approximation
    table1_function_approximation.create_function_approximation_table()

Or run all:
    python generate_all_tables.py
"""

__version__ = '1.0.0'
__author__ = 'KAN Research Team'

# Import main functions for easier access
from .utils import (
    load_latest_results,
    get_best_result_per_dataset,
    format_architecture,
    format_scientific,
    create_latex_table,
    print_table,
    compare_models
)

__all__ = [
    'load_latest_results',
    'get_best_result_per_dataset',
    'format_architecture',
    'format_scientific',
    'create_latex_table',
    'print_table',
    'compare_models'
]
