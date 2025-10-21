import sys
from pathlib import Path

# Add pykan to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "population"))

import torch
import numpy as np
import argparse

from population_trainer import PopulationBasedKANTrainer

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Section 2.4: Population-Based Training')
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs for training (default: 500)')
parser.add_argument('--population_size', type=int, default=10, help='Population size (default: 10)')
parser.add_argument('--sync_frequency', type=int, default=50, help='Sync frequency (default: 50)')
args = parser.parse_args()

epochs = args.epochs
population_size = args.population_size
sync_frequency = args.sync_frequency

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print(f"Running with {epochs} epochs, population={population_size}, sync_freq={sync_frequency}")

# Section 2.4: Population-Based Training
# ============= Create Dataset =============
print("\n" + "="*60)
print("Creating synthetic dataset...")
print("="*60)

X_train = torch.randn(500, 3).to(device)
y_train = (2*X_train[:, 0] + torch.sin(X_train[:, 1]) + X_train[:, 2]**2).reshape(-1, 1)

X_test = torch.randn(200, 3).to(device)
y_test = (2*X_test[:, 0] + torch.sin(X_test[:, 1]) + X_test[:, 2]**2).reshape(-1, 1)

# ============= Population Training: Average =============
print("\n" + "="*60)
print("Training Population with 'average' synchronization...")
print("="*60)

trainer_avg = PopulationBasedKANTrainer(
    input_dim=3,
    hidden_dim=10,
    output_dim=1,
    depth=2,
    population_size=population_size,
    sync_method='average',
    sync_frequency=sync_frequency,
    diversity_weight=0.1,
    device=device
)

history_avg = trainer_avg.train(X_train, y_train, epochs=epochs, lr=0.01, verbose=True)

print(f"\nAverage sync results:")
print(f"  Synchronization events: {len(history_avg['sync_events'])}")
print(f"  Final diversity: {history_avg['diversity'][-1]:.6f}")

# ============= Population Training: Best =============
print("\n" + "="*60)
print("Training Population with 'best' synchronization...")
print("="*60)

trainer_best = PopulationBasedKANTrainer(
    input_dim=3,
    hidden_dim=10,
    output_dim=1,
    depth=2,
    population_size=population_size,
    sync_method='best',
    sync_frequency=sync_frequency,
    diversity_weight=0.1,
    device=device
)

history_best = trainer_best.train(X_train, y_train, epochs=epochs, lr=0.01, verbose=True)

print(f"\nBest sync results:")
print(f"  Synchronization events: {len(history_best['sync_events'])}")
print(f"  Final diversity: {history_best['diversity'][-1]:.6f}")

# ============= Evaluation =============
print("\n" + "="*60)
print("Evaluation Results")
print("="*60)

# Get best models
best_model_avg = trainer_avg.get_best_model()
best_model_best = trainer_best.get_best_model()

# Evaluate best individual models
best_model_avg.eval()
best_model_best.eval()

with torch.no_grad():
    y_pred_avg_single = best_model_avg(X_test)
    mse_avg_single = torch.nn.functional.mse_loss(y_pred_avg_single, y_test).item()

    y_pred_best_single = best_model_best(X_test)
    mse_best_single = torch.nn.functional.mse_loss(y_pred_best_single, y_test).item()

# Ensemble predictions
y_pred_avg_ensemble = trainer_avg.get_ensemble_prediction(X_test, method='mean')
mse_avg_ensemble = torch.nn.functional.mse_loss(y_pred_avg_ensemble, y_test).item()

y_pred_best_ensemble = trainer_best.get_ensemble_prediction(X_test, method='mean')
mse_best_ensemble = torch.nn.functional.mse_loss(y_pred_best_ensemble, y_test).item()

print(f"\n'Average' synchronization:")
print(f"  Best individual MSE: {mse_avg_single:.6f}")
print(f"  Ensemble MSE: {mse_avg_ensemble:.6f}")
print(f"  Ensemble improvement: {((mse_avg_single - mse_avg_ensemble) / mse_avg_single * 100):.2f}%")

print(f"\n'Best' synchronization:")
print(f"  Best individual MSE: {mse_best_single:.6f}")
print(f"  Ensemble MSE: {mse_best_ensemble:.6f}")
print(f"  Ensemble improvement: {((mse_best_single - mse_best_ensemble) / mse_best_single * 100):.2f}%")

print("\n" + "="*60)
print("Section 2.4 Complete")
print("="*60)
