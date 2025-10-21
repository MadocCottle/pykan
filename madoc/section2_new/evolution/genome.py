"""Genome representation for KAN architecture evolution.

This module defines the genome representation for evolutionary architecture
search of KAN networks. The genome encodes network architecture, basis types,
and hyperparameters.

Key Features:
- Flexible genome encoding
- Genome to model instantiation
- Mutation and crossover operators
- Validation and constraints
- Support for pykan's MultKAN (B-spline basis)

PyKAN Reference:
    Liu, Ziming, et al. "KAN: Kolmogorov-Arnold Networks."
    arXiv preprint arXiv:2404.19756 (2024).
    https://arxiv.org/abs/2404.19756

Reference:
- Plan Section: Extension 5 - Evolutionary Architecture Search for KANs
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # Add madoc/ to path
sys.path.insert(0, str(Path(__file__).parent.parent))  # Add section2_new/ to path

# Import KAN variants (use PyKAN if custom variants not available)
try:
    from section1.models.kan_variants import RBF_KAN
except ImportError:
    from models.pykan_wrapper import PyKANCompatible as RBF_KAN

# Import pykan wrapper for B-spline support
try:
    from models.pykan_wrapper import PyKANCompatible
    _HAS_PYKAN = True
except ImportError:
    _HAS_PYKAN = False


@dataclass
class KANGenome:
    """Genome representation for a KAN architecture.

    Attributes:
        layer_sizes: List of layer dimensions [input, hidden..., output]
        basis_type: Type of basis function ('bspline', 'rbf', 'fourier', 'chebyshev')
        grid_size: Grid/basis size parameter
        learning_rate: Learning rate for training
        depth: Network depth (computed from layer_sizes)
        fitness: Fitness score (set during evaluation)

    Basis Types:
        - 'bspline': B-spline basis (uses pykan's MultKAN if available)
        - 'rbf': Radial basis functions
        - 'fourier': Fourier basis
        - 'chebyshev': Chebyshev polynomials

    Example:
        >>> # Use pykan's B-spline KAN
        >>> genome = KANGenome(
        ...     layer_sizes=[3, 16, 8, 1],
        ...     basis_type='bspline',
        ...     grid_size=5,
        ...     learning_rate=0.01
        ... )
        >>> model = genome.to_model()  # Creates MultKAN

        >>> # Use custom RBF KAN
        >>> genome = KANGenome(
        ...     layer_sizes=[3, 16, 8, 1],
        ...     basis_type='rbf',
        ...     grid_size=10,
        ...     learning_rate=0.01
        ... )
        >>> model = genome.to_model()  # Creates RBF_KAN
    """

    layer_sizes: List[int]
    basis_type: str = 'rbf'
    grid_size: int = 10
    learning_rate: float = 0.01
    fitness: Optional[float] = None
    depth: int = field(init=False)

    def __post_init__(self):
        """Compute derived attributes."""
        self.depth = len(self.layer_sizes) - 1

        # Validate
        if self.depth < 1:
            raise ValueError("Network must have at least 1 layer")
        if any(s < 1 for s in self.layer_sizes):
            raise ValueError("All layer sizes must be >= 1")

    def to_model(self, device: str = 'cpu') -> nn.Module:
        """Instantiate a KAN model from this genome.

        Supports multiple basis types including pykan's B-spline KAN.

        Args:
            device: Torch device

        Returns:
            KAN model instance

        Notes:
            - 'bspline' uses pykan's MultKAN (Liu et al., 2024)
            - Other basis types use custom implementations
        """
        if self.basis_type == 'bspline' and _HAS_PYKAN:
            # Use pykan's MultKAN for B-spline basis
            # MultKAN has grid adaptation, symbolic reasoning, and pruning
            model = PyKANCompatible(
                input_dim=self.layer_sizes[0],
                hidden_dim=self.layer_sizes[1] if self.depth > 1 else self.layer_sizes[-1],
                output_dim=self.layer_sizes[-1],
                depth=self.depth,
                grid_size=self.grid_size,
                spline_order=3,  # Cubic B-splines
                device=device
            )
        elif self.basis_type == 'rbf':
            # Use custom RBF KAN
            model = RBF_KAN(
                input_dim=self.layer_sizes[0],
                hidden_dim=self.layer_sizes[1] if self.depth > 1 else self.layer_sizes[-1],
                output_dim=self.layer_sizes[-1],
                depth=self.depth,
                n_centers=self.grid_size
            ).to(device)
        else:
            # Fallback to RBF for unsupported types (fourier, chebyshev, etc.)
            # TODO: Add support for other basis types when needed
            model = RBF_KAN(
                input_dim=self.layer_sizes[0],
                hidden_dim=self.layer_sizes[1] if self.depth > 1 else self.layer_sizes[-1],
                output_dim=self.layer_sizes[-1],
                depth=self.depth,
                n_centers=self.grid_size
            ).to(device)

        return model

    def mutate(
        self,
        mutation_rate: float = 0.3,
        layer_size_range: tuple = (4, 64),
        grid_size_range: tuple = (5, 20),
        lr_range: tuple = (0.001, 0.1)
    ) -> 'KANGenome':
        """Create a mutated copy of this genome.

        Args:
            mutation_rate: Probability of mutating each component
            layer_size_range: (min, max) for layer sizes
            grid_size_range: (min, max) for grid size
            lr_range: (min, max) for learning rate

        Returns:
            Mutated genome
        """
        new_layer_sizes = self.layer_sizes.copy()

        # Mutate hidden layer sizes (keep input/output fixed)
        for i in range(1, len(new_layer_sizes) - 1):
            if np.random.rand() < mutation_rate:
                # Random adjustment
                delta = np.random.randint(-8, 9)  # -8 to +8
                new_size = max(layer_size_range[0],
                              min(layer_size_range[1],
                                  new_layer_sizes[i] + delta))
                new_layer_sizes[i] = new_size

        # Mutate grid size
        new_grid_size = self.grid_size
        if np.random.rand() < mutation_rate:
            delta = np.random.randint(-3, 4)
            new_grid_size = max(grid_size_range[0],
                               min(grid_size_range[1],
                                   self.grid_size + delta))

        # Mutate learning rate (log scale)
        new_lr = self.learning_rate
        if np.random.rand() < mutation_rate:
            factor = 10 ** np.random.uniform(-0.5, 0.5)
            new_lr = max(lr_range[0],
                        min(lr_range[1],
                            self.learning_rate * factor))

        # Mutate basis type (rare)
        new_basis = self.basis_type
        if np.random.rand() < mutation_rate * 0.3:  # Lower rate for basis change
            # Include bspline in choices if pykan is available
            if _HAS_PYKAN:
                new_basis = np.random.choice(['rbf', 'rbf', 'bspline'])  # Favor RBF, but allow bspline
            else:
                new_basis = np.random.choice(['rbf', 'rbf', 'rbf'])  # Favor RBF

        return KANGenome(
            layer_sizes=new_layer_sizes,
            basis_type=new_basis,
            grid_size=new_grid_size,
            learning_rate=new_lr
        )

    def crossover(self, other: 'KANGenome') -> tuple:
        """Create two offspring via crossover with another genome.

        Args:
            other: Other parent genome

        Returns:
            Tuple of (offspring1, offspring2)
        """
        # Ensure compatible architectures (same input/output)
        if (self.layer_sizes[0] != other.layer_sizes[0] or
            self.layer_sizes[-1] != other.layer_sizes[-1]):
            # Cannot crossover incompatible architectures
            return self.mutate(), other.mutate()

        # Crossover hidden layers
        # Use shorter depth as limit
        min_depth = min(self.depth, other.depth)

        # Take hidden layers from both parents
        hidden1 = self.layer_sizes[1:-1]
        hidden2 = other.layer_sizes[1:-1]

        # Single-point crossover on hidden layers
        if len(hidden1) > 0 and len(hidden2) > 0:
            point = np.random.randint(0, min(len(hidden1), len(hidden2)) + 1)
            new_hidden1 = hidden1[:point] + hidden2[point:]
            new_hidden2 = hidden2[:point] + hidden1[point:]
        else:
            new_hidden1 = hidden1
            new_hidden2 = hidden2

        # Combine with input/output
        new_layers1 = [self.layer_sizes[0]] + list(new_hidden1) + [self.layer_sizes[-1]]
        new_layers2 = [other.layer_sizes[0]] + list(new_hidden2) + [other.layer_sizes[-1]]

        # Crossover hyperparameters
        if np.random.rand() < 0.5:
            grid1, grid2 = self.grid_size, other.grid_size
            lr1, lr2 = self.learning_rate, other.learning_rate
            basis1, basis2 = self.basis_type, other.basis_type
        else:
            grid1, grid2 = other.grid_size, self.grid_size
            lr1, lr2 = other.learning_rate, self.learning_rate
            basis1, basis2 = other.basis_type, self.basis_type

        offspring1 = KANGenome(
            layer_sizes=new_layers1,
            basis_type=basis1,
            grid_size=grid1,
            learning_rate=lr1
        )

        offspring2 = KANGenome(
            layer_sizes=new_layers2,
            basis_type=basis2,
            grid_size=grid2,
            learning_rate=lr2
        )

        return offspring1, offspring2

    def complexity(self) -> int:
        """Compute complexity metric (total parameters approx).

        Returns:
            Complexity score
        """
        # Rough parameter count estimate
        total = 0
        for i in range(len(self.layer_sizes) - 1):
            # Each edge has grid_size parameters
            total += self.layer_sizes[i] * self.layer_sizes[i+1] * self.grid_size

        return total

    def __repr__(self) -> str:
        """String representation."""
        fitness_str = f"{self.fitness:.6f}" if self.fitness is not None else "None"
        return (f"KANGenome(layers={self.layer_sizes}, basis={self.basis_type}, "
                f"grid={self.grid_size}, lr={self.learning_rate:.4f}, "
                f"fitness={fitness_str})")


# =============================================================================
# Population Utilities
# =============================================================================

def create_random_genome(
    input_dim: int,
    output_dim: int,
    max_depth: int = 4,
    layer_size_range: tuple = (8, 64),
    grid_size_range: tuple = (5, 20)
) -> KANGenome:
    """Create a random genome.

    Args:
        input_dim: Input dimension (fixed)
        output_dim: Output dimension (fixed)
        max_depth: Maximum network depth
        layer_size_range: Range for hidden layer sizes
        grid_size_range: Range for grid size

    Returns:
        Random genome
    """
    # Random depth
    depth = np.random.randint(2, max_depth + 1)

    # Random hidden layer sizes
    layer_sizes = [input_dim]
    for _ in range(depth - 1):
        size = np.random.randint(layer_size_range[0], layer_size_range[1] + 1)
        layer_sizes.append(size)
    layer_sizes.append(output_dim)

    # Random hyperparameters
    grid_size = np.random.randint(grid_size_range[0], grid_size_range[1] + 1)
    lr = 10 ** np.random.uniform(-3, -1)  # 0.001 to 0.1

    return KANGenome(
        layer_sizes=layer_sizes,
        basis_type='rbf',
        grid_size=grid_size,
        learning_rate=lr
    )


# =============================================================================
# Example Usage and Testing
# =============================================================================

if __name__ == '__main__':
    print("="*70)
    print("Testing KAN Genome Representation")
    print("="*70)

    # Create a genome
    genome1 = KANGenome(
        layer_sizes=[3, 16, 8, 1],
        basis_type='rbf',
        grid_size=10,
        learning_rate=0.01
    )

    print(f"\nGenome 1: {genome1}")
    print(f"Complexity: {genome1.complexity()} params")

    # Test instantiation
    model = genome1.to_model()
    print(f"Model created: {type(model).__name__}")

    # Test forward pass
    X = torch.randn(5, 3)
    y = model(X)
    print(f"Forward pass: input {X.shape} -> output {y.shape}")

    # Test mutation
    print("\nTesting mutation:")
    genome2 = genome1.mutate(mutation_rate=0.5)
    print(f"Original: {genome1}")
    print(f"Mutated:  {genome2}")

    # Test crossover
    print("\nTesting crossover:")
    genome3 = create_random_genome(input_dim=3, output_dim=1)
    print(f"Parent 1: {genome1}")
    print(f"Parent 2: {genome3}")

    offspring1, offspring2 = genome1.crossover(genome3)
    print(f"Offspring 1: {offspring1}")
    print(f"Offspring 2: {offspring2}")

    # Test random population
    print("\nCreating random population:")
    population = [create_random_genome(3, 1) for _ in range(5)]
    for i, genome in enumerate(population):
        print(f"{i+1}. {genome}")

    print("\n" + "="*70)
    print("Genome test complete!")
    print("="*70)
