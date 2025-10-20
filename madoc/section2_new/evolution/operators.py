"""Selection and genetic operators for evolutionary search.

This module implements selection mechanisms and population management
for evolutionary architecture search.

Key Features:
- Tournament selection
- Roulette wheel selection
- Elitism
- Pareto frontier tracking (multi-objective)

Reference:
- Plan Section: Extension 5 - Evolutionary Architecture Search
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


def tournament_selection(
    population: list,
    fitness_scores: list,
    n_parents: int,
    tournament_size: int = 3
) -> list:
    """Select parents via tournament selection.

    Args:
        population: List of genomes
        fitness_scores: List of fitness values (higher is better)
        n_parents: Number of parents to select
        tournament_size: Size of each tournament

    Returns:
        List of selected parent genomes
    """
    parents = []
    pop_size = len(population)

    for _ in range(n_parents):
        # Random tournament
        tournament_indices = np.random.choice(pop_size, size=tournament_size, replace=False)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]

        # Select best from tournament
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        parents.append(population[winner_idx])

    return parents


def roulette_selection(
    population: list,
    fitness_scores: list,
    n_parents: int
) -> list:
    """Select parents via roulette wheel (fitness-proportional) selection.

    Args:
        population: List of genomes
        fitness_scores: List of fitness values (higher is better)
        n_parents: Number of parents to select

    Returns:
        List of selected parent genomes
    """
    # Shift fitness to be positive if needed
    min_fitness = min(fitness_scores)
    if min_fitness < 0:
        shifted_fitness = [f - min_fitness + 1e-6 for f in fitness_scores]
    else:
        shifted_fitness = [f + 1e-6 for f in fitness_scores]

    # Normalize to probabilities
    total_fitness = sum(shifted_fitness)
    probabilities = [f / total_fitness for f in shifted_fitness]

    # Select with replacement
    selected_indices = np.random.choice(
        len(population),
        size=n_parents,
        replace=True,
        p=probabilities
    )

    return [population[i] for i in selected_indices]


def elitism(
    population: list,
    fitness_scores: list,
    n_elite: int
) -> Tuple[list, list]:
    """Select top-N elite genomes.

    Args:
        population: List of genomes
        fitness_scores: List of fitness values (higher is better)
        n_elite: Number of elites to preserve

    Returns:
        Tuple of (elite_genomes, elite_fitness)
    """
    # Sort by fitness (descending)
    sorted_indices = np.argsort(fitness_scores)[::-1]

    elite_genomes = [population[i] for i in sorted_indices[:n_elite]]
    elite_fitness = [fitness_scores[i] for i in sorted_indices[:n_elite]]

    return elite_genomes, elite_fitness


def rank_selection(
    population: list,
    fitness_scores: list,
    n_parents: int,
    selection_pressure: float = 1.5
) -> list:
    """Select parents via rank-based selection.

    Args:
        population: List of genomes
        fitness_scores: List of fitness values
        n_parents: Number of parents to select
        selection_pressure: Linear ranking parameter (1.0-2.0)

    Returns:
        List of selected parent genomes
    """
    # Rank genomes by fitness
    pop_size = len(population)
    ranks = np.argsort(np.argsort(fitness_scores)) + 1  # 1 to pop_size

    # Linear ranking probabilities
    probabilities = [(2 - selection_pressure) / pop_size +
                    2 * rank * (selection_pressure - 1) / (pop_size * (pop_size - 1))
                    for rank in ranks]

    # Normalize
    probabilities = np.array(probabilities)
    probabilities /= probabilities.sum()

    # Select
    selected_indices = np.random.choice(
        pop_size,
        size=n_parents,
        replace=True,
        p=probabilities
    )

    return [population[i] for i in selected_indices]


# =============================================================================
# Pareto Frontier for Multi-Objective Optimization
# =============================================================================

@dataclass
class ParetoSolution:
    """A solution on the Pareto frontier.

    Attributes:
        genome: The KAN genome
        objectives: Tuple of objective values (accuracy, complexity, speed)
        rank: Pareto rank (0 = non-dominated frontier)
    """
    genome: any
    objectives: Tuple[float, float, float]
    rank: int = 0

    def __repr__(self):
        return f"ParetoSolution(objectives={self.objectives}, rank={self.rank})"


class ParetoFrontier:
    """Track Pareto-optimal solutions for multi-objective optimization.

    A solution A dominates solution B if:
    - A is better than B in at least one objective
    - A is not worse than B in any objective

    Example:
        >>> frontier = ParetoFrontier()
        >>> frontier.add(genome1, (0.95, -1000, -5.0))  # accuracy, -complexity, -time
        >>> frontier.add(genome2, (0.90, -500, -2.0))
        >>> non_dominated = frontier.get_frontier()
    """

    def __init__(self):
        self.solutions: List[ParetoSolution] = []

    def dominates(
        self,
        obj_a: Tuple[float, float, float],
        obj_b: Tuple[float, float, float]
    ) -> bool:
        """Check if objective set A dominates B.

        Args:
            obj_a: Objectives for solution A
            obj_b: Objectives for solution B

        Returns:
            True if A dominates B
        """
        # A dominates B if:
        # 1. A is >= B in all objectives (higher is better)
        # 2. A is > B in at least one objective

        all_greater_equal = all(a >= b for a, b in zip(obj_a, obj_b))
        at_least_one_greater = any(a > b for a, b in zip(obj_a, obj_b))

        return all_greater_equal and at_least_one_greater

    def add(
        self,
        genome,
        objectives: Tuple[float, float, float]
    ):
        """Add a solution and update the frontier.

        Args:
            genome: KAN genome
            objectives: (accuracy, complexity, speed) - all maximization
        """
        new_solution = ParetoSolution(genome, objectives)

        # Check if dominated by existing solutions
        is_dominated = False
        solutions_to_remove = []

        for i, existing in enumerate(self.solutions):
            if self.dominates(existing.objectives, objectives):
                # New solution is dominated, don't add
                is_dominated = True
                break
            elif self.dominates(objectives, existing.objectives):
                # New solution dominates existing, mark for removal
                solutions_to_remove.append(i)

        if not is_dominated:
            # Remove dominated solutions
            for i in reversed(solutions_to_remove):
                del self.solutions[i]

            # Add new solution
            self.solutions.append(new_solution)

    def get_frontier(self, rank: int = 0) -> List[ParetoSolution]:
        """Get solutions on the Pareto frontier.

        Args:
            rank: Pareto rank (0 = first frontier, 1 = second, etc.)

        Returns:
            List of solutions on the specified frontier
        """
        if rank == 0:
            # First frontier is all non-dominated solutions
            return self.solutions.copy()
        else:
            # For higher ranks, would need to implement fast non-dominated sorting
            # For simplicity, return empty for now
            return []

    def size(self) -> int:
        """Get number of solutions on the frontier."""
        return len(self.solutions)

    def get_best_by_objective(self, objective_idx: int) -> Optional[ParetoSolution]:
        """Get best solution for a specific objective.

        Args:
            objective_idx: Index of objective (0=accuracy, 1=complexity, 2=speed)

        Returns:
            Best solution for that objective
        """
        if not self.solutions:
            return None

        return max(self.solutions, key=lambda s: s.objectives[objective_idx])

    def __repr__(self):
        return f"ParetoFrontier({len(self.solutions)} solutions)"


# =============================================================================
# Example Usage and Testing
# =============================================================================

if __name__ == '__main__':
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    sys.path.insert(0, str(Path(__file__).parent))

    from genome import create_random_genome

    print("="*70)
    print("Testing Selection Operators")
    print("="*70)

    # Create population
    np.random.seed(42)
    population = [create_random_genome(3, 1) for _ in range(10)]
    fitness_scores = np.random.randn(10) + 5  # Random fitness

    print(f"\nPopulation size: {len(population)}")
    print(f"Fitness scores: {[f'{f:.2f}' for f in fitness_scores]}")

    # Tournament selection
    print("\n1. Tournament Selection (k=3, n=5):")
    parents = tournament_selection(population, fitness_scores, n_parents=5, tournament_size=3)
    print(f"   Selected {len(parents)} parents")

    # Roulette selection
    print("\n2. Roulette Selection (n=5):")
    parents = roulette_selection(population, fitness_scores, n_parents=5)
    print(f"   Selected {len(parents)} parents")

    # Elitism
    print("\n3. Elitism (top 3):")
    elite_genomes, elite_fitness = elitism(population, fitness_scores, n_elite=3)
    print(f"   Elite fitness: {[f'{f:.2f}' for f in elite_fitness]}")

    # Pareto frontier
    print("\n" + "="*70)
    print("Testing Pareto Frontier")
    print("="*70)

    frontier = ParetoFrontier()

    # Add solutions with different trade-offs
    solutions = [
        (population[0], (0.95, -1000, -5.0)),  # High accuracy, complex, slow
        (population[1], (0.90, -500, -3.0)),   # Medium accuracy, medium complexity
        (population[2], (0.85, -200, -1.0)),   # Lower accuracy, simple, fast
        (population[3], (0.92, -800, -4.0)),   # Dominated by first
        (population[4], (0.88, -400, -2.0)),   # On frontier
    ]

    print("\nAdding solutions:")
    for i, (genome, objectives) in enumerate(solutions):
        frontier.add(genome, objectives)
        print(f"  Solution {i+1}: accuracy={objectives[0]:.2f}, "
              f"complexity={objectives[1]}, time={objectives[2]:.1f}s")

    print(f"\nPareto frontier size: {frontier.size()}")
    print("\nNon-dominated solutions:")
    for i, sol in enumerate(frontier.get_frontier()):
        print(f"  {i+1}. {sol}")

    # Best by objective
    best_accuracy = frontier.get_best_by_objective(0)
    best_simplicity = frontier.get_best_by_objective(1)
    best_speed = frontier.get_best_by_objective(2)

    print("\nBest by objective:")
    print(f"  Best accuracy: {best_accuracy.objectives[0]:.3f}")
    print(f"  Best simplicity: {best_simplicity.objectives[1]}")
    print(f"  Best speed: {best_speed.objectives[2]:.1f}s")

    print("\n" + "="*70)
    print("Selection operators test complete!")
    print("="*70)
