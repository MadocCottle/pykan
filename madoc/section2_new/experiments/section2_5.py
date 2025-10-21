import sys
from pathlib import Path

# Add pykan to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "evolution"))

import torch
import numpy as np
import argparse

from evolutionary_search import EvolutionaryKANSearch

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Section 2.5: Evolutionary Architecture Search')
parser.add_argument('--n_generations', type=int, default=10, help='Number of generations (default: 10)')
parser.add_argument('--population_size', type=int, default=10, help='Population size (default: 10)')
parser.add_argument('--max_epochs', type=int, default=100, help='Max epochs per evaluation (default: 100)')
args = parser.parse_args()

n_generations = args.n_generations
population_size = args.population_size
max_epochs = args.max_epochs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print(f"Running evolution: {n_generations} generations, pop={population_size}, max_epochs={max_epochs}")

# Section 2.5: Evolutionary Architecture Search
# ============= Create Dataset =============
print("\n" + "="*60)
print("Creating synthetic dataset...")
print("="*60)

X_train = torch.randn(500, 3).to(device)
y_train = (2*X_train[:, 0] + torch.sin(X_train[:, 1]) + X_train[:, 2]**2).reshape(-1, 1)

X_val = torch.randn(200, 3).to(device)
y_val = (2*X_val[:, 0] + torch.sin(X_val[:, 1]) + X_val[:, 2]**2).reshape(-1, 1)

X_test = torch.randn(200, 3).to(device)
y_test = (2*X_test[:, 0] + torch.sin(X_test[:, 1]) + X_test[:, 2]**2).reshape(-1, 1)

# ============= Evolutionary Search =============
print("\n" + "="*60)
print("Starting Evolutionary Architecture Search...")
print("="*60)

evolver = EvolutionaryKANSearch(
    input_dim=3,
    output_dim=1,
    population_size=population_size,
    n_generations=n_generations,
    selection_method='tournament',
    tournament_size=3,
    crossover_rate=0.8,
    mutation_rate=0.3,
    n_elite=2,
    max_epochs_per_eval=max_epochs,
    objective_weights=(1.0, 0.001, 0.1),  # (accuracy, complexity, speed)
    device=device,
    verbose=True
)

best_genome, history = evolver.evolve(X_train, y_train, X_val, y_val)

# ============= Evolution Analysis =============
print("\n" + "="*60)
print("Evolution Analysis")
print("="*60)

print(f"\nFitness progression:")
print(f"  Initial best fitness: {history['best_fitness_per_gen'][0]:.6f}")
print(f"  Final best fitness: {history['best_fitness_per_gen'][-1]:.6f}")
print(f"  Improvement: {((history['best_fitness_per_gen'][0] - history['best_fitness_per_gen'][-1]) / history['best_fitness_per_gen'][0] * 100):.2f}%")

print(f"\nPopulation diversity:")
print(f"  Initial diversity: {history['diversity_per_gen'][0]:.6f}")
print(f"  Final diversity: {history['diversity_per_gen'][-1]:.6f}")

print(f"\nPareto frontier:")
print(f"  Initial size: {history['pareto_size_per_gen'][0]}")
print(f"  Final size: {history['pareto_size_per_gen'][-1]}")

# ============= Best Architecture =============
print("\n" + "="*60)
print("Best Architecture Found")
print("="*60)

print(f"\nGenome:")
print(f"  Layer sizes: {best_genome.layer_sizes}")
print(f"  Basis type: {best_genome.basis_type}")
print(f"  Grid size: {best_genome.grid_size}")
print(f"  Learning rate: {best_genome.learning_rate:.4f}")
print(f"  Complexity: {best_genome.complexity()} parameters")
print(f"  Fitness: {best_genome.fitness:.6f}")

# ============= Final Evaluation =============
print("\n" + "="*60)
print("Final Evaluation on Test Set")
print("="*60)

best_model = best_genome.to_model(device=device)
best_model.eval()

with torch.no_grad():
    y_pred = best_model(X_test)
    test_mse = torch.nn.functional.mse_loss(y_pred, y_test).item()

print(f"\nTest MSE: {test_mse:.6f}")

# ============= Pareto Frontier =============
print("\n" + "="*60)
print("Pareto Frontier Solutions")
print("="*60)

pareto_solutions = evolver.get_pareto_frontier()
print(f"\nFound {len(pareto_solutions)} Pareto-optimal solutions:\n")

for i, (genome, fitness) in enumerate(pareto_solutions[:5]):  # Show top 5
    print(f"Solution {i+1}:")
    print(f"  Architecture: {genome.layer_sizes}")
    print(f"  Accuracy: {fitness.accuracy:.6f}")
    print(f"  Complexity: {fitness.complexity} params")
    print(f"  Training time: {fitness.training_time:.1f}s")
    print()

print("="*60)
print("Section 2.5 Complete")
print("="*60)
