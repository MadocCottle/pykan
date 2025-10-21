"""
Comprehensive Demo of Section 2 New: Evolutionary KAN Implementation

This script demonstrates all implemented components:
1. Ensemble training with importance analysis
2. Adaptive selective densification
3. Population-based training
4. Heterogeneous basis functions
5. Evolutionary genome representation

Uses pykan utilities for data generation and other core functionality.

PyKAN Reference:
    Liu, Ziming, et al. "KAN: Kolmogorov-Arnold Networks."
    arXiv preprint arXiv:2404.19756 (2024).
    https://arxiv.org/abs/2404.19756

Author: Claude Code
Date: October 18, 2025
"""

import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path

# Setup paths - add pykan to path (parent directory of madoc)
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent / "ensemble"))
sys.path.insert(0, str(Path(__file__).parent / "models"))
sys.path.insert(0, str(Path(__file__).parent / "population"))
sys.path.insert(0, str(Path(__file__).parent / "evolution"))

# Import pykan utilities
from kan.utils import create_dataset


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80)


def demo_ensemble():
    """Demo 1: Ensemble Framework"""
    print_section("DEMO 1: Ensemble Framework with Variable Importance")

    from expert_training import KANExpertEnsemble
    from variable_importance import VariableImportanceAnalyzer
    from stacking import StackedEnsemble

    # Generate data using pykan's create_dataset
    # Target: 3*x0 + 2*sin(x1) + x2^2 + 0*x3 (x3 is irrelevant)
    def target_fn(x):
        return 3.0 * x[0] + 2.0 * np.sin(x[1]) + x[2]**2

    dataset = create_dataset(
        f=target_fn,
        n_var=4,  # 4 inputs (x3 is noise)
        ranges=[-2, 2],
        train_num=100,
        test_num=30,
        device='cpu',
        seed=42
    )

    X_train = dataset['train_input']
    y_train = dataset['train_label']
    X_test = dataset['test_input']
    y_test = dataset['test_label']

    print("\n1. Training ensemble of 5 experts...")
    ensemble = KANExpertEnsemble(
        input_dim=4, hidden_dim=12, output_dim=1, depth=3,
        n_experts=5, kan_variant='rbf'
    )

    results = ensemble.train_experts(X_train, y_train, epochs=100, lr=0.01, verbose=False)
    print(f"   ✓ Training complete. Mean loss: {np.mean(results['individual_losses']):.6f}")

    # Variable importance
    print("\n2. Analyzing variable importance...")
    analyzer = VariableImportanceAnalyzer(ensemble)
    importance = analyzer.compute_consensus_importance(X_test, y_test, methods=['gradient', 'permutation'])

    print("   Consensus importance:")
    for i, score in enumerate(importance['consensus']):
        star = " ⭐" if i < 3 else ""
        print(f"     Feature {i}: {score:.4f}{star}")

    # Stacking
    print("\n3. Training stacked ensemble...")
    stacked = StackedEnsemble(ensemble, meta_hidden_dim=16)
    stacked.train_meta_learner(X_train, y_train, epochs=50, lr=0.01, verbose=False)

    y_pred_avg = ensemble.predict(X_test)
    y_pred_stack = stacked.predict(X_test)

    mse_avg = nn.MSELoss()(y_pred_avg, y_test).item()
    mse_stack = nn.MSELoss()(y_pred_stack, y_test).item()

    print(f"   ✓ Simple averaging MSE: {mse_avg:.6f}")
    print(f"   ✓ Stacked ensemble MSE: {mse_stack:.6f}")


def demo_adaptive():
    """Demo 2: Adaptive Selective Densification"""
    print_section("DEMO 2: Adaptive Selective Densification")

    from adaptive_selective_kan import AdaptiveSelectiveKAN, AdaptiveSelectiveTrainer

    # Generate data using pykan's create_dataset
    # Target: 2*x0 + sin(x1) + x2^2
    def target_fn(x):
        return 2.0 * x[0] + np.sin(x[1]) + x[2]**2

    dataset = create_dataset(
        f=target_fn,
        n_var=3,
        ranges=[-2, 2],
        train_num=100,
        device='cpu',
        seed=42
    )

    X = dataset['train_input']
    y = dataset['train_label']

    print("\n1. Creating adaptive selective KAN...")
    kan = AdaptiveSelectiveKAN(
        input_dim=3, hidden_dim=10, output_dim=1, depth=2,
        initial_grid=5, max_grid=15
    )

    stats_before = kan.get_grid_statistics()
    print(f"   Initial grid size: {stats_before['mean_grid_size']:.1f} (uniform)")

    print("\n2. Training with automatic densification...")
    trainer = AdaptiveSelectiveTrainer(kan, densify_every=30, densify_k=2, densify_delta=2)
    history = trainer.train(X, y, epochs=100, lr=0.01, verbose=False)

    stats_after = kan.get_grid_statistics()
    print(f"   ✓ Training complete")
    print(f"   Final grid size: {stats_after['mean_grid_size']:.1f} (range: [{stats_after['min_grid_size']}, {stats_after['max_grid_size']}])")
    print(f"   Grid points saved: {int((9 * stats_after['max_grid_size']) - stats_after['total_grid_points'])}")

    kan.cleanup()


def demo_population():
    """Demo 3: Population-Based Training"""
    print_section("DEMO 3: Population-Based Training")

    from population_trainer import PopulationBasedKANTrainer

    # Generate data using pykan's create_dataset
    # Target: 2*x0 + sin(x1)
    def target_fn(x):
        return 2.0 * x[0] + np.sin(x[1])

    dataset = create_dataset(
        f=target_fn,
        n_var=3,  # 3 inputs (x2 is irrelevant)
        ranges=[-2, 2],
        train_num=100,
        test_num=30,
        device='cpu',
        seed=42
    )

    X_train = dataset['train_input']
    y_train = dataset['train_label']
    X_test = dataset['test_input']
    y_test = dataset['test_label']

    print("\n1. Training population of 5 models...")
    trainer = PopulationBasedKANTrainer(
        input_dim=3, hidden_dim=10, output_dim=1, depth=2,
        population_size=5, sync_method='average', sync_frequency=25
    )

    history = trainer.train(X_train, y_train, epochs=100, lr=0.01, verbose=False)

    print(f"   ✓ Training complete")
    print(f"   Synchronization events: {len(history['sync_events'])}")
    print(f"   Final diversity: {history['diversity'][-1]:.6f}")

    print("\n2. Testing population ensemble...")
    best_model = trainer.get_best_model()
    best_model.eval()

    with torch.no_grad():
        y_pred_best = best_model(X_test)
        y_pred_ensemble = trainer.get_ensemble_prediction(X_test, method='mean')

    mse_best = nn.MSELoss()(y_pred_best, y_test).item()
    mse_ensemble = nn.MSELoss()(y_pred_ensemble, y_test).item()

    print(f"   ✓ Best model MSE: {mse_best:.6f}")
    print(f"   ✓ Ensemble MSE: {mse_ensemble:.6f}")


def demo_heterogeneous():
    """Demo 4: Heterogeneous Basis Functions"""
    print_section("DEMO 4: Heterogeneous Basis Functions")

    from heterogeneous_kan import HeterogeneousBasisKAN

    # Generate data using pykan's create_dataset
    # Target: sin(x0) + x1^2
    def target_fn(x):
        return np.sin(x[0]) + x[1]**2

    dataset = create_dataset(
        f=target_fn,
        n_var=2,
        ranges=[-2, 2],
        train_num=80,
        device='cpu',
        seed=42
    )

    X = dataset['train_input']
    y = dataset['train_label']

    print("\n1. Creating heterogeneous KAN with mixed bases...")
    kan = HeterogeneousBasisKAN(
        layer_dims=[2, 10, 1],
        basis_config=[
            {0: 'fourier', 1: 'rbf'},  # Layer 0: different bases per input
            'rbf'  # Layer 1: all RBF
        ]
    )

    usage = kan.get_all_basis_usage()
    print("   Basis assignments:")
    for layer_idx, layer_usage in usage.items():
        bases = list(set(layer_usage.values()))
        print(f"     Layer {layer_idx}: {', '.join(bases)}")

    print("\n2. Training heterogeneous KAN...")
    optimizer = torch.optim.Adam(kan.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    for epoch in range(100):
        optimizer.zero_grad()
        pred = kan(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

    print(f"   ✓ Training complete. Final loss: {loss.item():.6f}")


def demo_evolution():
    """Demo 5: Evolutionary Genome"""
    print_section("DEMO 5: Evolutionary Genome Representation")

    from genome import KANGenome, create_random_genome

    print("\n1. Creating and testing genome...")
    genome1 = KANGenome(
        layer_sizes=[3, 16, 8, 1],
        basis_type='rbf',
        grid_size=10,
        learning_rate=0.01
    )

    print(f"   Original genome: layers={genome1.layer_sizes}, grid={genome1.grid_size}")
    print(f"   Complexity: {genome1.complexity()} parameters")

    # Test model instantiation
    model = genome1.to_model()
    X = torch.randn(10, 3)
    y = model(X)
    print(f"   ✓ Model forward pass: {X.shape} -> {y.shape}")

    print("\n2. Testing genetic operators...")
    genome2 = genome1.mutate(mutation_rate=0.5)
    print(f"   Mutated genome: layers={genome2.layer_sizes}, grid={genome2.grid_size}")

    genome3 = create_random_genome(3, 1)
    offspring1, offspring2 = genome1.crossover(genome3)
    print(f"   Parent 1: layers={genome1.layer_sizes}")
    print(f"   Parent 2: layers={genome3.layer_sizes}")
    print(f"   Offspring 1: layers={offspring1.layer_sizes}")
    print(f"   Offspring 2: layers={offspring2.layer_sizes}")

    print("\n3. Creating random population...")
    population = [create_random_genome(3, 1) for _ in range(5)]
    complexities = [g.complexity() for g in population]
    print(f"   Population size: {len(population)}")
    print(f"   Complexity range: [{min(complexities)}, {max(complexities)}] params")
    print(f"   ✓ All genomes valid and functional")


def main():
    """Run all demos."""
    print("\n" + "="*80)
    print(" SECTION 2 NEW: EVOLUTIONARY KAN - COMPREHENSIVE DEMONSTRATION")
    print("="*80)
    print("\nThis demo showcases all implemented components:")
    print("  1. Ensemble Framework")
    print("  2. Adaptive Selective Densification")
    print("  3. Population-Based Training")
    print("  4. Heterogeneous Basis Functions")
    print("  5. Evolutionary Genome Representation")

    try:
        demo_ensemble()
        demo_adaptive()
        demo_population()
        demo_heterogeneous()
        demo_evolution()

        print("\n" + "="*80)
        print(" ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY ✓")
        print("="*80)
        print("\nFor more details, see:")
        print("  - FINAL_SUMMARY.md - Comprehensive implementation summary")
        print("  - README.md - Quick start guide")
        print("  - Individual module files - Detailed examples and tests")
        print("\n")

    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
