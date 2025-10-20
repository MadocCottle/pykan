"""
Section 2: PDE Solving with KAN and Neural Networks

A comprehensive testing infrastructure for comparing KAN models with traditional
neural networks (MLP, SIREN) on various Partial Differential Equations.

Main modules:
- pde_data: PDE problem definitions and data generation
- models: Neural network models (MLP, SIREN, KAN wrappers)
- metrics: Evaluation metrics (MSE, H1 norm, PDE residual, etc.)
- trainer: Training utilities for supervised and physics-informed learning

Quick start:
    >>> import pde_data, models, trainer, metrics
    >>> sol_func, src_func, grad_func = pde_data.get_pde_problem('2d_poisson')
    >>> dataset = pde_data.create_pde_dataset_2d(sol_func, train_num=1000)

See README.md for detailed documentation.
"""

__version__ = '1.0.0'
__author__ = 'PyKAN Contributors'

# Make key functions easily accessible
from . import pde_data
from . import models
from . import metrics
from . import trainer

__all__ = ['pde_data', 'models', 'metrics', 'trainer']