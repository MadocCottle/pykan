"""Fitness evaluation for evolutionary architecture search.

This module implements fitness evaluation for KAN genomes, including:
- Single-objective fitness (validation error)
- Multi-objective fitness (accuracy, complexity, speed)
- Efficient training with early stopping
- Fitness caching to avoid redundant evaluations

Reference:
- Plan Section: Extension 5 - Evolutionary Architecture Search
"""

import torch
import torch.nn as nn
import numpy as np
import time
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import hashlib
import pickle


@dataclass
class FitnessScore:
    """Multi-objective fitness score for a genome.

    Attributes:
        accuracy: Negative validation MSE (higher is better)
        complexity: Negative parameter count (higher = simpler)
        speed: Negative training time in seconds (higher = faster)
        combined: Weighted combination of objectives
    """
    accuracy: float
    complexity: float
    speed: float
    combined: float

    def __repr__(self):
        return (f"Fitness(acc={self.accuracy:.6f}, "
                f"complex={self.complexity:.1f}, "
                f"speed={self.speed:.3f}, "
                f"combined={self.combined:.6f})")


class FitnessEvaluator:
    """Evaluate fitness of KAN genomes.

    Args:
        max_epochs: Maximum training epochs per genome
        early_stopping_patience: Stop if no improvement for N epochs
        objective_weights: Weights for (accuracy, complexity, speed)
        device: Torch device
        use_cache: Cache fitness scores for identical genomes
        verbose: Print evaluation progress

    Example:
        >>> evaluator = FitnessEvaluator(max_epochs=200)
        >>> fitness = evaluator.evaluate(genome, X_train, y_train, X_val, y_val)
        >>> print(fitness.combined)
    """

    def __init__(
        self,
        max_epochs: int = 200,
        early_stopping_patience: int = 20,
        objective_weights: Tuple[float, float, float] = (1.0, 0.001, 0.1),
        device: str = 'cpu',
        use_cache: bool = True,
        verbose: bool = False
    ):
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.objective_weights = objective_weights
        self.device = torch.device(device)
        self.use_cache = use_cache
        self.verbose = verbose

        # Cache: genome_hash -> FitnessScore
        self.fitness_cache: Dict[str, FitnessScore] = {}
        self.evaluations_count = 0
        self.cache_hits = 0

    def _genome_hash(self, genome) -> str:
        """Compute hash of genome for caching.

        Args:
            genome: KANGenome instance

        Returns:
            Hash string
        """
        # Hash the genome's key attributes
        genome_str = f"{genome.layer_sizes}_{genome.basis_type}_{genome.grid_size}_{genome.learning_rate:.6f}"
        return hashlib.md5(genome_str.encode()).hexdigest()

    def evaluate(
        self,
        genome,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor
    ) -> FitnessScore:
        """Evaluate fitness of a genome.

        Args:
            genome: KANGenome to evaluate
            X_train: Training inputs
            y_train: Training targets
            X_val: Validation inputs
            y_val: Validation targets

        Returns:
            FitnessScore with multi-objective metrics
        """
        self.evaluations_count += 1

        # Check cache
        if self.use_cache:
            genome_hash = self._genome_hash(genome)
            if genome_hash in self.fitness_cache:
                self.cache_hits += 1
                if self.verbose:
                    print(f"  Cache hit! ({self.cache_hits}/{self.evaluations_count})")
                return self.fitness_cache[genome_hash]

        # Move data to device
        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)
        X_val = X_val.to(self.device)
        y_val = y_val.to(self.device)

        # Instantiate model
        model = genome.to_model(device=str(self.device))

        # Count parameters (complexity metric)
        n_params = sum(p.numel() for p in model.parameters())

        # Setup training
        optimizer = torch.optim.Adam(model.parameters(), lr=genome.learning_rate)
        loss_fn = nn.MSELoss()

        # Training loop with early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        start_time = time.time()

        for epoch in range(self.max_epochs):
            # Train
            model.train()
            optimizer.zero_grad()
            pred_train = model(X_train)
            train_loss = loss_fn(pred_train, y_train)
            train_loss.backward()
            optimizer.step()

            # Validate
            model.eval()
            with torch.no_grad():
                pred_val = model(X_val)
                val_loss = loss_fn(pred_val, y_val).item()

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.early_stopping_patience:
                if self.verbose:
                    print(f"  Early stopping at epoch {epoch+1}")
                break

        training_time = time.time() - start_time

        # Compute multi-objective fitness
        # Note: All objectives are maximization (negative for minimization)
        accuracy_score = -best_val_loss  # Higher is better
        complexity_score = -n_params / 1000.0  # Simpler is better, scaled
        speed_score = -training_time  # Faster is better

        # Combined weighted fitness
        combined_score = (
            self.objective_weights[0] * accuracy_score +
            self.objective_weights[1] * complexity_score +
            self.objective_weights[2] * speed_score
        )

        fitness = FitnessScore(
            accuracy=accuracy_score,
            complexity=complexity_score,
            speed=speed_score,
            combined=combined_score
        )

        # Cache result
        if self.use_cache:
            self.fitness_cache[genome_hash] = fitness

        # Store fitness in genome
        genome.fitness = combined_score

        if self.verbose:
            print(f"  Evaluated genome: {fitness}")
            print(f"    Params: {n_params}, Time: {training_time:.2f}s, Val Loss: {best_val_loss:.6f}")

        return fitness

    def evaluate_batch(
        self,
        genomes: list,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        parallel: bool = False,
        n_workers: int = 4
    ) -> list:
        """Evaluate fitness of multiple genomes.

        Args:
            genomes: List of KANGenome instances
            X_train, y_train, X_val, y_val: Data tensors
            parallel: Use multiprocessing for parallel evaluation
            n_workers: Number of parallel workers

        Returns:
            List of FitnessScore instances
        """
        if not parallel:
            # Sequential evaluation
            return [self.evaluate(genome, X_train, y_train, X_val, y_val)
                   for genome in genomes]
        else:
            # Parallel evaluation
            return self._evaluate_parallel(genomes, X_train, y_train, X_val, y_val, n_workers)

    def _evaluate_parallel(
        self,
        genomes: list,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        n_workers: int
    ) -> list:
        """Parallel fitness evaluation using multiprocessing.

        Note: This is a simplified version. Full implementation would use
        multiprocessing.Pool with proper data serialization.
        """
        # For now, fall back to sequential (multiprocessing with PyTorch is complex)
        if self.verbose:
            print(f"  Parallel evaluation not fully implemented, using sequential")
        return [self.evaluate(genome, X_train, y_train, X_val, y_val)
               for genome in genomes]

    def reset_cache(self):
        """Clear fitness cache."""
        self.fitness_cache.clear()
        self.evaluations_count = 0
        self.cache_hits = 0

    def get_cache_stats(self) -> Dict:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        hit_rate = self.cache_hits / self.evaluations_count if self.evaluations_count > 0 else 0.0
        return {
            'total_evaluations': self.evaluations_count,
            'cache_hits': self.cache_hits,
            'cache_size': len(self.fitness_cache),
            'hit_rate': hit_rate
        }


# =============================================================================
# Example Usage and Testing
# =============================================================================

if __name__ == '__main__':
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    sys.path.insert(0, str(Path(__file__).parent))

    from genome import KANGenome, create_random_genome

    print("="*70)
    print("Testing Fitness Evaluator")
    print("="*70)

    # Generate synthetic data
    torch.manual_seed(42)
    X_train = torch.randn(100, 3)
    y_train = (2.0 * X_train[:, 0] + torch.sin(X_train[:, 1])).reshape(-1, 1)

    X_val = torch.randn(30, 3)
    y_val = (2.0 * X_val[:, 0] + torch.sin(X_val[:, 1])).reshape(-1, 1)

    # Create evaluator
    evaluator = FitnessEvaluator(
        max_epochs=100,
        early_stopping_patience=10,
        objective_weights=(1.0, 0.001, 0.1),
        verbose=True
    )

    print("\n1. Evaluating single genome...")
    genome1 = KANGenome(
        layer_sizes=[3, 16, 8, 1],
        basis_type='rbf',
        grid_size=10,
        learning_rate=0.01
    )

    fitness1 = evaluator.evaluate(genome1, X_train, y_train, X_val, y_val)
    print(f"\nFitness: {fitness1}")

    print("\n2. Evaluating same genome again (should hit cache)...")
    fitness1_cached = evaluator.evaluate(genome1, X_train, y_train, X_val, y_val)
    print(f"Fitness: {fitness1_cached}")

    print("\n3. Evaluating different genome...")
    genome2 = create_random_genome(3, 1)
    fitness2 = evaluator.evaluate(genome2, X_train, y_train, X_val, y_val)
    print(f"\nFitness: {fitness2}")

    print("\n4. Batch evaluation...")
    genomes = [create_random_genome(3, 1) for _ in range(3)]
    fitness_scores = evaluator.evaluate_batch(genomes, X_train, y_train, X_val, y_val)

    for i, fitness in enumerate(fitness_scores):
        print(f"  Genome {i+1}: {fitness}")

    # Cache statistics
    print("\n5. Cache statistics:")
    stats = evaluator.get_cache_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n" + "="*70)
    print("Fitness evaluator test complete!")
    print("="*70)
