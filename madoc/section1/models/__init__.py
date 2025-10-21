"""
Custom KAN variants with different basis functions.
"""
from .kan_variants import (
    RBF_KAN,
    ChebyshevKAN,
    FourierKAN,
    WaveletKAN,
    RBFBasis,
    ChebyshevBasis,
    FourierBasis
)

__all__ = [
    'RBF_KAN',
    'ChebyshevKAN',
    'FourierKAN',
    'WaveletKAN',
    'RBFBasis',
    'ChebyshevBasis',
    'FourierBasis'
]