"""Utilities for section2_new.

This module re-exports utilities from pykan and provides custom extensions
specific to section2_new's ensemble, evolution, and adaptive features.

PyKAN Reference:
    Liu, Ziming, et al. "KAN: Kolmogorov-Arnold Networks."
    arXiv preprint arXiv:2404.19756 (2024).
    https://arxiv.org/abs/2404.19756
"""

import sys
from pathlib import Path

# Add pykan to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# ============================================================
# FROM PYKAN (use these instead of custom implementations)
# ============================================================

from kan.utils import (
    create_dataset,              # Dataset generation with train/test split
    create_dataset_from_data,    # Convert numpy/torch arrays to dataset format
    sparse_mask,                  # Generate sparse connection masks
    augment_input,                # Input augmentation utilities
)

try:
    # LBFGS optimizer from pykan (more efficient than PyTorch's default)
    from kan.LBFGS import LBFGS
    _HAS_LBFGS = True
except ImportError:
    # Fallback to PyTorch's LBFGS if pykan's not available
    from torch.optim import LBFGS
    _HAS_LBFGS = False

try:
    # Interpretability and analysis tools
    from kan.hypothesis import (
        test_symmetry,                # Symmetry testing
        detect_separability,          # Separability detection
    )
    _HAS_HYPOTHESIS = True
except ImportError:
    _HAS_HYPOTHESIS = False


# ============================================================
# CUSTOM SECTION2_NEW UTILITIES (not in pykan)
# ============================================================

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple


def set_seed(seed: int):
    """Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def aggregate_metrics_over_seeds(
    results: List[Dict[str, float]]
) -> Dict[str, Dict[str, float]]:
    """Aggregate metrics across multiple random seeds.

    Args:
        results: List of metric dictionaries from different seeds

    Returns:
        Dictionary with mean, std, and all values for each metric
        {
            'metric_name': {
                'mean': float,
                'std': float,
                'all_values': List[float]
            }
        }

    Example:
        >>> results = [
        ...     {'l2_error': 0.01, 'linf_error': 0.05},
        ...     {'l2_error': 0.02, 'linf_error': 0.06}
        ... ]
        >>> agg = aggregate_metrics_over_seeds(results)
        >>> print(agg['l2_error']['mean'])  # 0.015
    """
    if not results:
        return {}

    # Get all metric names
    metric_names = results[0].keys()

    aggregated = {}
    for metric in metric_names:
        values = [r[metric] for r in results if metric in r]
        aggregated[metric] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'all_values': values
        }

    return aggregated


def compute_ensemble_diversity(
    predictions: torch.Tensor
) -> float:
    """Compute diversity of ensemble predictions.

    Diversity is measured as the average pairwise variance across ensemble members.

    Args:
        predictions: Predictions from ensemble (n_experts, n_samples, output_dim)

    Returns:
        Diversity score (higher = more diverse)

    Example:
        >>> preds = torch.randn(10, 100, 1)  # 10 experts, 100 samples
        >>> diversity = compute_ensemble_diversity(preds)
    """
    # Compute variance across experts for each sample
    variances = torch.var(predictions, dim=0)  # (n_samples, output_dim)
    # Average across samples and output dimensions
    diversity = float(variances.mean().item())
    return diversity


def get_device(device: Optional[str] = None) -> str:
    """Get compute device, auto-detecting if not specified.

    Args:
        device: Device string ('cpu', 'cuda', 'mps') or None for auto-detect

    Returns:
        Device string

    Example:
        >>> device = get_device()  # Auto-detect
        >>> device = get_device('cuda')  # Force CUDA
    """
    if device is not None:
        return device

    # Auto-detect
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    # PyKAN utilities
    'create_dataset',
    'create_dataset_from_data',
    'sparse_mask',
    'augment_input',
    'LBFGS',

    # Custom section2_new utilities
    'set_seed',
    'aggregate_metrics_over_seeds',
    'compute_ensemble_diversity',
    'get_device',
]

# Conditionally export hypothesis tools if available
if _HAS_HYPOTHESIS:
    __all__.extend([
        'test_symmetry',
        'detect_separability',
    ])
