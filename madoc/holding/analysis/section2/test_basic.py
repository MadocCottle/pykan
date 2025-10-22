"""
Basic test to verify all modules work correctly.
"""

import sys
from pathlib import Path
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

print("Testing Section 2 PDE Infrastructure")
print("=" * 60)

# Test imports
print("\n1. Testing imports...")
try:
    import pde_data
    import models
    import metrics
    import trainer
    from kan import KAN
    print("   ✓ All imports successful")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

# Test PDE data
print("\n2. Testing PDE data generation...")
try:
    sol_func, source_func, grad_func = pde_data.get_pde_problem('2d_poisson')
    dataset = pde_data.create_pde_dataset_2d(sol_func, train_num=100, test_num=100)
    print(f"   ✓ Created dataset with {dataset['train_input'].shape[0]} train samples")

    x_interior = pde_data.create_interior_points_2d([-1, 1], 21, mode='mesh')
    x_boundary = pde_data.create_boundary_points_2d([-1, 1], 21)
    print(f"   ✓ Created {x_interior.shape[0]} interior and {x_boundary.shape[0]} boundary points")
except Exception as e:
    print(f"   ✗ PDE data failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test models
print("\n3. Testing model creation...")
try:
    device = torch.device('cpu')

    kan = KAN(width=[2, 3, 1], grid=3, k=3, device=device)
    kan = kan.speed()
    print(f"   ✓ Created KAN with {models.count_parameters(kan):,} parameters")

    mlp = models.create_model('mlp', 2, 32, 3, 1, device=device)
    print(f"   ✓ Created MLP with {models.count_parameters(mlp):,} parameters")

    siren = models.create_model('siren', 2, 32, 2, 1, device=device)
    print(f"   ✓ Created SIREN with {models.count_parameters(siren):,} parameters")
except Exception as e:
    print(f"   ✗ Model creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test metrics
print("\n4. Testing metrics computation...")
try:
    import numpy as np
    ranges_2d = np.array([[-1, 1], [-1, 1]])
    x_test = metrics.create_dense_test_set(ranges_2d, 11, device=device)
    y_test = sol_func(x_test)

    dense_dataset = {'test_input': x_test, 'test_label': y_test}

    tracker = metrics.MetricsTracker(
        kan,
        dense_dataset,
        solution_func=sol_func,
        gradient_func=grad_func,
        source_func=source_func
    )

    metrics_dict = tracker.compute_all_metrics()
    print(f"   ✓ Computed metrics: {list(metrics_dict.keys())}")

    # Test individual metric functions
    mse = metrics.compute_mse_error(kan, x_test, y_test)
    print(f"   ✓ MSE error: {mse.item():.6e}")

    h1_norm = metrics.compute_h1_norm(kan, x_test, y_test, grad_func(x_test))
    print(f"   ✓ H1 norm: {h1_norm.item():.6e}")

    laplacian = metrics.compute_laplacian(kan, x_test[:10])
    print(f"   ✓ Laplacian computed, shape: {laplacian.shape}")

except Exception as e:
    print(f"   ✗ Metrics failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test training (just a few epochs)
print("\n5. Testing training...")
try:
    kan_small = KAN(width=[2, 3, 1], grid=3, k=3, device=device)
    kan_small = kan_small.speed()

    pde_trainer = trainer.PDETrainer(kan_small, device)

    history = pde_trainer.train_supervised(
        dataset,
        epochs=3,
        lr=1e-3,
        optimizer_type='adam'
    )

    print(f"   ✓ Training completed")
    print(f"   ✓ Final train loss: {history['train_loss'][-1]:.6e}")
    print(f"   ✓ Final test loss: {history['test_loss'][-1]:.6e}")

except Exception as e:
    print(f"   ✗ Training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test PDE residual training
print("\n6. Testing PDE residual training...")
try:
    kan_pde = KAN(width=[2, 3, 1], grid=3, k=3, device=device)
    kan_pde = kan_pde.speed()

    pde_trainer = trainer.PDETrainer(kan_pde, device)

    x_interior_small = pde_data.create_interior_points_2d([-1, 1], 11, mode='mesh')
    x_boundary_small = pde_data.create_boundary_points_2d([-1, 1], 11)

    history = pde_trainer.train_pde_residual(
        x_interior_small,
        x_boundary_small,
        sol_func,
        source_func,
        epochs=3,
        alpha=0.01,
        lr=1.0
    )

    print(f"   ✓ PDE training completed")
    print(f"   ✓ Final PDE loss: {history['pde_loss'][-1]:.6e}")
    print(f"   ✓ Final BC loss: {history['bc_loss'][-1]:.6e}")

except Exception as e:
    print(f"   ✗ PDE training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ All tests passed!")
print("=" * 60)
print("\nThe infrastructure is ready to use. Try running:")
print("  python example_2d_poisson.py")
print("or")
print("  python run_pde_tests.py --pde 2d_poisson --models kan mlp --epochs 50")