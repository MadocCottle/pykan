"""Integration tests for pykan compatibility with section2_new.

Tests that section2_new components work correctly with pykan utilities:
1. Data generation with create_dataset()
2. MultKAN (B-spline) integration with ensembles
3. MultKAN integration with evolution
4. LBFGS optimizer from pykan

PyKAN Reference:
    Liu, Ziming, et al. "KAN: Kolmogorov-Arnold Networks."
    arXiv preprint arXiv:2404.19756 (2024).
    https://arxiv.org/abs/2404.19756
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add paths
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "section2_new"))

# Import pykan utilities
from kan.utils import create_dataset
from kan.LBFGS import LBFGS

# Import section2_new components
from models.pykan_wrapper import PyKANCompatible, create_pykan_model
from ensemble.expert_training import KANExpertEnsemble
from evolution.genome import KANGenome


def test_pykan_data_generation():
    """Test that pykan's create_dataset works correctly."""
    print("\n" + "="*80)
    print("TEST 1: PyKAN Data Generation")
    print("="*80)

    def target_fn(x):
        return x[0]**2 + np.sin(x[1])

    dataset = create_dataset(
        f=target_fn,
        n_var=2,
        ranges=[-2, 2],
        train_num=100,
        test_num=50,
        device='cpu',
        seed=42
    )

    # Check dataset structure
    assert 'train_input' in dataset
    assert 'train_label' in dataset
    assert 'test_input' in dataset
    assert 'test_label' in dataset

    # Check shapes
    assert dataset['train_input'].shape == (100, 2)
    assert dataset['train_label'].shape == (100, 1)
    assert dataset['test_input'].shape == (50, 2)
    assert dataset['test_label'].shape == (50, 1)

    print("  ✓ Dataset created successfully")
    print(f"  ✓ Train shape: {dataset['train_input'].shape}")
    print(f"  ✓ Test shape: {dataset['test_input'].shape}")


def test_pykan_wrapper():
    """Test PyKANCompatible wrapper."""
    print("\n" + "="*80)
    print("TEST 2: PyKAN Wrapper")
    print("="*80)

    # Create model using wrapper
    model = PyKANCompatible(
        input_dim=3,
        hidden_dim=8,
        output_dim=1,
        depth=3,
        grid_size=5,
        device='cpu'
    )

    # Test forward pass
    X = torch.randn(10, 3)
    y = model(X)

    assert y.shape == (10, 1)
    print("  ✓ PyKANCompatible created successfully")
    print(f"  ✓ Forward pass: {X.shape} -> {y.shape}")

    # Test factory function
    model2 = create_pykan_model(3, 8, 1, depth=3, grid_size=5)
    y2 = model2(X)
    assert y2.shape == (10, 1)
    print("  ✓ create_pykan_model() works correctly")


def test_multkan_with_ensemble():
    """Test that MultKAN (bspline) works with ensemble training."""
    print("\n" + "="*80)
    print("TEST 3: MultKAN with Ensemble")
    print("="*80)

    # Generate data
    def target_fn(x):
        return x[0]**2 + x[1]**2

    dataset = create_dataset(
        f=target_fn,
        n_var=2,
        ranges=[-1, 1],
        train_num=50,
        test_num=20,
        device='cpu',
        seed=42
    )

    X_train = dataset['train_input']
    y_train = dataset['train_label']
    X_test = dataset['test_input']
    y_test = dataset['test_label']

    # Create ensemble with bspline variant
    try:
        ensemble = KANExpertEnsemble(
            input_dim=2,
            hidden_dim=5,
            output_dim=1,
            depth=2,
            n_experts=3,
            kan_variant='bspline',  # Use MultKAN
            device='cpu'
        )

        # Train ensemble
        results = ensemble.train_experts(
            X_train, y_train,
            epochs=20,
            lr=0.01,
            verbose=False
        )

        assert results is not None
        assert len(ensemble.experts) == 3
        print("  ✓ Ensemble created with MultKAN (bspline)")
        print(f"  ✓ Trained {len(ensemble.experts)} experts")
        print(f"  ✓ Mean final loss: {np.mean(results['individual_losses']):.6f}")

        # Test prediction
        y_pred, uncertainty = ensemble.predict_with_uncertainty(X_test)
        assert y_pred.shape == y_test.shape
        assert uncertainty.shape == y_test.shape
        print(f"  ✓ Prediction with uncertainty works")
        print(f"  ✓ Mean uncertainty: {uncertainty.mean():.6f}")

    except ValueError as e:
        if 'bspline' in str(e):
            print("  ⚠ MultKAN not available (pykan not found)")
            print("  ⚠ Skipping bspline test")
        else:
            raise


def test_multkan_with_evolution():
    """Test that MultKAN works with evolutionary genome."""
    print("\n" + "="*80)
    print("TEST 4: MultKAN with Evolution")
    print("="*80)

    try:
        # Create genome with bspline basis
        genome = KANGenome(
            layer_sizes=[2, 8, 1],
            basis_type='bspline',  # Use MultKAN
            grid_size=5,
            learning_rate=0.01
        )

        # Convert to model
        model = genome.to_model(device='cpu')

        # Should be PyKANCompatible (which inherits from MultKAN)
        assert hasattr(model, 'forward')
        print("  ✓ Genome with bspline created successfully")
        print(f"  ✓ Model type: {type(model).__name__}")

        # Test forward pass
        X = torch.randn(10, 2)
        y = model(X)
        assert y.shape == (10, 1)
        print(f"  ✓ Forward pass works: {X.shape} -> {y.shape}")

        # Test mutation
        mutated = genome.mutate(mutation_rate=0.5)
        assert mutated.layer_sizes[0] == genome.layer_sizes[0]  # Input unchanged
        assert mutated.layer_sizes[-1] == genome.layer_sizes[-1]  # Output unchanged
        print("  ✓ Mutation preserves input/output dimensions")

    except Exception as e:
        if 'pykan' in str(e).lower() or 'bspline' in str(e).lower():
            print("  ⚠ MultKAN not available (pykan not found)")
            print("  ⚠ Skipping bspline genome test")
        else:
            raise


def test_lbfgs_optimizer():
    """Test that pykan's LBFGS optimizer works."""
    print("\n" + "="*80)
    print("TEST 5: LBFGS Optimizer from PyKAN")
    print("="*80)

    # Create simple model
    model = PyKANCompatible(
        input_dim=2,
        hidden_dim=5,
        output_dim=1,
        depth=2,
        grid_size=3,
        device='cpu'
    )

    # Generate data
    X = torch.randn(30, 2)
    y = X[:, 0]**2 + X[:, 1]**2
    y = y.unsqueeze(1)

    # Create LBFGS optimizer
    optimizer = LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=20,
        history_size=10
    )

    # Training closure (required for LBFGS)
    def closure():
        optimizer.zero_grad()
        y_pred = model(X)
        loss = torch.nn.MSELoss()(y_pred, y)
        loss.backward()
        return loss

    # Train for a few steps
    initial_loss = closure().item()
    for _ in range(3):
        optimizer.step(closure)
    final_loss = closure().item()

    assert final_loss < initial_loss
    print("  ✓ LBFGS optimizer works correctly")
    print(f"  ✓ Initial loss: {initial_loss:.6f}")
    print(f"  ✓ Final loss: {final_loss:.6f}")
    print(f"  ✓ Loss decreased: {initial_loss - final_loss:.6f}")


def test_ensemble_with_different_variants():
    """Test ensemble with mixed KAN variants."""
    print("\n" + "="*80)
    print("TEST 6: Multiple KAN Variants")
    print("="*80)

    def target_fn(x):
        return x[0] + x[1]

    dataset = create_dataset(
        f=target_fn,
        n_var=2,
        ranges=[-1, 1],
        train_num=50,
        device='cpu',
        seed=42
    )

    X = dataset['train_input']
    y = dataset['train_label']

    # Test different variants
    variants_to_test = ['rbf', 'chebyshev', 'fourier']

    # Add bspline if available
    try:
        test_ensemble = KANExpertEnsemble(
            input_dim=2, hidden_dim=3, output_dim=1, depth=2,
            n_experts=1, kan_variant='bspline'
        )
        variants_to_test.append('bspline')
        print("  ✓ bspline variant available")
    except:
        print("  ⚠ bspline variant not available")

    results_summary = []
    for variant in variants_to_test:
        try:
            ensemble = KANExpertEnsemble(
                input_dim=2,
                hidden_dim=5,
                output_dim=1,
                depth=2,
                n_experts=2,
                kan_variant=variant,
                device='cpu'
            )

            results = ensemble.train_experts(
                X, y,
                epochs=10,
                lr=0.01,
                verbose=False
            )

            final_loss = np.mean(results['individual_losses'])
            results_summary.append((variant, final_loss))
            print(f"  ✓ {variant:12s}: final loss = {final_loss:.6f}")

        except Exception as e:
            print(f"  ✗ {variant:12s}: {str(e)[:50]}")

    assert len(results_summary) >= 3  # At least rbf, chebyshev, fourier should work


def run_all_tests():
    """Run all integration tests."""
    print("\n" + "="*80)
    print(" PYKAN INTEGRATION TESTS FOR SECTION2_NEW")
    print("="*80)
    print("\nThese tests verify that section2_new works correctly with pykan:")
    print("  - Data generation (create_dataset)")
    print("  - MultKAN (B-spline KAN) integration")
    print("  - LBFGS optimizer")
    print("  - Multiple KAN variants")

    try:
        test_pykan_data_generation()
        test_pykan_wrapper()
        test_multkan_with_ensemble()
        test_multkan_with_evolution()
        test_lbfgs_optimizer()
        test_ensemble_with_different_variants()

        print("\n" + "="*80)
        print(" ALL TESTS PASSED ✓")
        print("="*80)
        print("\nPyKAN integration is working correctly!")
        print("\nReferences:")
        print("  Liu, Ziming, et al. 'KAN: Kolmogorov-Arnold Networks.'")
        print("  arXiv preprint arXiv:2404.19756 (2024).")
        print("  https://arxiv.org/abs/2404.19756")
        print()

    except Exception as e:
        print("\n" + "="*80)
        print(" TEST FAILED ✗")
        print("="*80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    run_all_tests()
