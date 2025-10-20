"""Main evolutionary architecture search for KAN networks.

This module implements the complete evolutionary loop for discovering
optimal KAN architectures through genetic algorithms.

Key Features:
- Population initialization
- Generational evolution with selection, crossover, mutation
- Multi-objective optimization with Pareto frontier
- Progress tracking and early convergence detection
- Configurable evolution strategies

Reference:
- Plan Section: Extension 5 - Evolutionary Architecture Search for KANs
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from genome import KANGenome, create_random_genome
from fitness import FitnessEvaluator, FitnessScore
from operators import (
    tournament_selection, roulette_selection, elitism,
    rank_selection, ParetoFrontier
)


class EvolutionaryKANSearch:
    """Evolutionary search for optimal KAN architectures.

    Args:
        input_dim: Input dimension (fixed)
        output_dim: Output dimension (fixed)
        population_size: Number of genomes in population
        n_generations: Number of generations to evolve
        selection_method: 'tournament', 'roulette', or 'rank'
        tournament_size: Size of tournament (if using tournament selection)
        crossover_rate: Probability of crossover vs cloning
        mutation_rate: Probability of mutation per genome component
        n_elite: Number of elite genomes to preserve
        max_epochs_per_eval: Maximum training epochs per fitness evaluation
        objective_weights: Weights for (accuracy, complexity, speed)
        device: Torch device
        verbose: Print progress

    Example:
        >>> evolver = EvolutionaryKANSearch(
        ...     input_dim=3, output_dim=1,
        ...     population_size=20, n_generations=30
        ... )
        >>> best_genome, history = evolver.evolve(X_train, y_train, X_val, y_val)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        population_size: int = 20,
        n_generations: int = 30,
        selection_method: str = 'tournament',
        tournament_size: int = 3,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.3,
        n_elite: int = 2,
        max_epochs_per_eval: int = 150,
        objective_weights: Tuple[float, float, float] = (1.0, 0.001, 0.1),
        device: str = 'cpu',
        verbose: bool = True
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.population_size = population_size
        self.n_generations = n_generations
        self.selection_method = selection_method
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.n_elite = n_elite
        self.max_epochs_per_eval = max_epochs_per_eval
        self.objective_weights = objective_weights
        self.device = device
        self.verbose = verbose

        # Fitness evaluator
        self.evaluator = FitnessEvaluator(
            max_epochs=max_epochs_per_eval,
            early_stopping_patience=20,
            objective_weights=objective_weights,
            device=device,
            use_cache=True,
            verbose=False
        )

        # Pareto frontier for multi-objective tracking
        self.pareto_frontier = ParetoFrontier()

        # Evolution history
        self.history: Dict = {
            'best_fitness': [],
            'mean_fitness': [],
            'diversity': [],
            'pareto_size': []
        }

    def _initialize_population(self) -> List[KANGenome]:
        """Initialize random population.

        Returns:
            List of random genomes
        """
        if self.verbose:
            print(f"\nInitializing population of {self.population_size} genomes...")

        population = []
        for _ in range(self.population_size):
            genome = create_random_genome(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                max_depth=4,
                layer_size_range=(8, 64),
                grid_size_range=(5, 20)
            )
            population.append(genome)

        return population

    def _evaluate_population(
        self,
        population: List[KANGenome],
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor
    ) -> List[FitnessScore]:
        """Evaluate fitness of all genomes in population.

        Args:
            population: List of genomes
            X_train, y_train, X_val, y_val: Data tensors

        Returns:
            List of fitness scores
        """
        fitness_scores = []

        for i, genome in enumerate(population):
            if self.verbose:
                print(f"  Evaluating genome {i+1}/{len(population)}...", end='\r')

            fitness = self.evaluator.evaluate(genome, X_train, y_train, X_val, y_val)
            fitness_scores.append(fitness)

            # Update Pareto frontier
            self.pareto_frontier.add(
                genome,
                (fitness.accuracy, fitness.complexity, fitness.speed)
            )

        if self.verbose:
            print()  # New line after progress

        return fitness_scores

    def _select_parents(
        self,
        population: List[KANGenome],
        fitness_scores: List[FitnessScore],
        n_parents: int
    ) -> List[KANGenome]:
        """Select parents for next generation.

        Args:
            population: Current population
            fitness_scores: Fitness scores
            n_parents: Number of parents to select

        Returns:
            List of selected parent genomes
        """
        # Extract combined fitness values
        fitness_values = [f.combined for f in fitness_scores]

        if self.selection_method == 'tournament':
            return tournament_selection(
                population, fitness_values, n_parents, self.tournament_size
            )
        elif self.selection_method == 'roulette':
            return roulette_selection(population, fitness_values, n_parents)
        elif self.selection_method == 'rank':
            return rank_selection(population, fitness_values, n_parents)
        else:
            raise ValueError(f"Unknown selection method: {self.selection_method}")

    def _create_offspring(
        self,
        parents: List[KANGenome]
    ) -> List[KANGenome]:
        """Create offspring through crossover and mutation.

        Args:
            parents: List of parent genomes

        Returns:
            List of offspring genomes
        """
        offspring = []

        # Pair up parents for crossover
        for i in range(0, len(parents) - 1, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1]

            # Crossover
            if np.random.rand() < self.crossover_rate:
                child1, child2 = parent1.crossover(parent2)
            else:
                # Clone parents if no crossover
                child1, child2 = parent1, parent2

            # Mutation
            if np.random.rand() < self.mutation_rate:
                child1 = child1.mutate(mutation_rate=0.3)
            if np.random.rand() < self.mutation_rate:
                child2 = child2.mutate(mutation_rate=0.3)

            offspring.extend([child1, child2])

        return offspring

    def _compute_diversity(self, population: List[KANGenome]) -> float:
        """Compute diversity of population.

        Measures variety in architectures and hyperparameters.

        Args:
            population: List of genomes

        Returns:
            Diversity score
        """
        # Diversity based on architecture variety
        architectures = [tuple(g.layer_sizes) for g in population]
        unique_architectures = len(set(architectures))
        arch_diversity = unique_architectures / len(population)

        # Diversity based on hyperparameter variance
        grid_sizes = [g.grid_size for g in population]
        learning_rates = [g.learning_rate for g in population]

        grid_diversity = np.std(grid_sizes) / (np.mean(grid_sizes) + 1e-6)
        lr_diversity = np.std(learning_rates) / (np.mean(learning_rates) + 1e-6)

        # Combined diversity
        diversity = (arch_diversity + grid_diversity + lr_diversity) / 3.0

        return diversity

    def evolve(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor
    ) -> Tuple[KANGenome, Dict]:
        """Run evolutionary search.

        Args:
            X_train: Training inputs
            y_train: Training targets
            X_val: Validation inputs
            y_val: Validation targets

        Returns:
            Tuple of (best_genome, history_dict)
        """
        if self.verbose:
            print("="*70)
            print("EVOLUTIONARY KAN ARCHITECTURE SEARCH")
            print("="*70)
            print(f"Population size: {self.population_size}")
            print(f"Generations: {self.n_generations}")
            print(f"Selection: {self.selection_method}")
            print(f"Input dim: {self.input_dim}, Output dim: {self.output_dim}")

        start_time = time.time()

        # Initialize population
        population = self._initialize_population()

        # Evolution loop
        for generation in range(self.n_generations):
            gen_start = time.time()

            if self.verbose:
                print(f"\n{'='*70}")
                print(f"Generation {generation + 1}/{self.n_generations}")
                print(f"{'='*70}")

            # Evaluate fitness
            fitness_scores = self._evaluate_population(
                population, X_train, y_train, X_val, y_val
            )

            # Extract fitness values
            fitness_values = [f.combined for f in fitness_scores]
            best_fitness = max(fitness_values)
            mean_fitness = np.mean(fitness_values)
            diversity = self._compute_diversity(population)

            # Record history
            self.history['best_fitness'].append(best_fitness)
            self.history['mean_fitness'].append(mean_fitness)
            self.history['diversity'].append(diversity)
            self.history['pareto_size'].append(self.pareto_frontier.size())

            # Print statistics
            if self.verbose:
                best_idx = np.argmax(fitness_values)
                best_genome = population[best_idx]
                gen_time = time.time() - gen_start

                print(f"\nGeneration {generation + 1} Statistics:")
                print(f"  Best fitness: {best_fitness:.6f}")
                print(f"  Mean fitness: {mean_fitness:.6f}")
                print(f"  Diversity: {diversity:.4f}")
                print(f"  Pareto frontier size: {self.pareto_frontier.size()}")
                print(f"  Best genome: layers={best_genome.layer_sizes}, "
                      f"grid={best_genome.grid_size}")
                print(f"  Generation time: {gen_time:.1f}s")

            # Selection
            n_offspring = self.population_size - self.n_elite
            parents = self._select_parents(population, fitness_scores, n_offspring)

            # Create offspring
            offspring = self._create_offspring(parents)[:n_offspring]

            # Elitism: preserve best genomes
            elite_genomes, elite_fitness_values = elitism(
                population, fitness_values, self.n_elite
            )

            # New population = elites + offspring
            population = elite_genomes + offspring

        # Final evaluation
        if self.verbose:
            print(f"\n{'='*70}")
            print("EVOLUTION COMPLETE")
            print(f"{'='*70}")

        final_fitness = self._evaluate_population(
            population, X_train, y_train, X_val, y_val
        )
        final_fitness_values = [f.combined for f in final_fitness]
        best_idx = np.argmax(final_fitness_values)
        best_genome = population[best_idx]

        total_time = time.time() - start_time

        if self.verbose:
            print(f"\nBest genome found:")
            print(f"  Architecture: {best_genome.layer_sizes}")
            print(f"  Basis: {best_genome.basis_type}")
            print(f"  Grid size: {best_genome.grid_size}")
            print(f"  Learning rate: {best_genome.learning_rate:.4f}")
            if best_genome.fitness is not None:
                print(f"  Fitness: {best_genome.fitness:.6f}")
            else:
                print(f"  Fitness: {final_fitness_values[best_idx]:.6f}")
            print(f"\nPareto frontier size: {self.pareto_frontier.size()}")
            print(f"Total evolution time: {total_time:.1f}s")

            # Cache statistics
            cache_stats = self.evaluator.get_cache_stats()
            print(f"\nEvaluation cache statistics:")
            print(f"  Total evaluations: {cache_stats['total_evaluations']}")
            print(f"  Cache hits: {cache_stats['cache_hits']}")
            print(f"  Hit rate: {cache_stats['hit_rate']:.1%}")

        return best_genome, self.history

    def get_pareto_frontier(self) -> List:
        """Get current Pareto frontier.

        Returns:
            List of Pareto-optimal solutions
        """
        return self.pareto_frontier.get_frontier()


# =============================================================================
# Example Usage and Testing
# =============================================================================

if __name__ == '__main__':
    print("="*70)
    print("Testing Evolutionary KAN Search")
    print("="*70)

    # Generate synthetic data
    torch.manual_seed(42)
    X_train = torch.randn(100, 3)
    y_train = (2.0 * X_train[:, 0] + torch.sin(X_train[:, 1])).reshape(-1, 1)

    X_val = torch.randn(30, 3)
    y_val = (2.0 * X_val[:, 0] + torch.sin(X_val[:, 1])).reshape(-1, 1)

    X_test = torch.randn(20, 3)
    y_test = (2.0 * X_test[:, 0] + torch.sin(X_test[:, 1])).reshape(-1, 1)

    # Create evolutionary search
    evolver = EvolutionaryKANSearch(
        input_dim=3,
        output_dim=1,
        population_size=10,  # Small for quick test
        n_generations=5,     # Few generations for quick test
        selection_method='tournament',
        n_elite=2,
        max_epochs_per_eval=100,
        verbose=True
    )

    # Run evolution
    best_genome, history = evolver.evolve(X_train, y_train, X_val, y_val)

    # Test best genome
    print(f"\n{'='*70}")
    print("Testing Best Genome")
    print(f"{'='*70}")

    best_model = best_genome.to_model()
    import torch.nn as nn

    best_model.eval()
    with torch.no_grad():
        y_pred = best_model(X_test)
        test_mse = nn.MSELoss()(y_pred, y_test).item()

    print(f"Test MSE: {test_mse:.6f}")

    # Show Pareto frontier
    print(f"\nPareto Frontier ({evolver.pareto_frontier.size()} solutions):")
    for i, sol in enumerate(evolver.get_pareto_frontier()[:3]):
        print(f"  {i+1}. {sol}")

    print("\n" + "="*70)
    print("Evolutionary search test complete!")
    print("="*70)
