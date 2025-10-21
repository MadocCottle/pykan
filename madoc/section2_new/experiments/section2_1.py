import sys
from pathlib import Path

# Add pykan to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "ensemble"))

import torch
import numpy as np
import argparse

from expert_training import KANExpertEnsemble
from variable_importance import VariableImportanceAnalyzer
from clustering import ExpertClusterer
from stacking import StackedEnsemble

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Section 2.1: Hierarchical Ensemble of KAN Experts')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs for training (default: 200)')
parser.add_argument('--n_experts', type=int, default=10, help='Number of experts (default: 10)')
args = parser.parse_args()

epochs = args.epochs
n_experts = args.n_experts

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print(f"Running with {epochs} epochs and {n_experts} experts")

# Section 2.1: Hierarchical Ensemble of KAN Experts
# ============= Create Dataset =============
print("\n" + "="*60)
print("Creating synthetic dataset...")
print("="*60)

X_train = torch.randn(500, 3).to(device)
y_train = (2*X_train[:, 0] + torch.sin(X_train[:, 1]) + X_train[:, 2]**2).reshape(-1, 1)

X_test = torch.randn(200, 3).to(device)
y_test = (2*X_test[:, 0] + torch.sin(X_test[:, 1]) + X_test[:, 2]**2).reshape(-1, 1)

# ============= Train Ensemble =============
print("\n" + "="*60)
print("Training KAN Expert Ensemble...")
print("="*60)

ensemble = KANExpertEnsemble(
    input_dim=3,
    hidden_dim=16,
    output_dim=1,
    depth=3,
    n_experts=n_experts,
    kan_variant='bspline',  # Use pykan's B-spline KAN
    device=device
)

train_results = ensemble.train_experts(X_train, y_train, epochs=epochs, lr=0.01, verbose=True)

print(f"\nTraining complete:")
print(f"  Individual losses: {train_results['individual_losses']}")
print(f"  Mean loss: {np.mean(train_results['individual_losses']):.6f}")
print(f"  Training time: {train_results['training_time']:.2f}s")

# ============= Variable Importance =============
print("\n" + "="*60)
print("Computing Variable Importance...")
print("="*60)

analyzer = VariableImportanceAnalyzer(ensemble)
importance = analyzer.compute_consensus_importance(X_test, y_test, methods=['gradient', 'permutation'])

print("\nFeature importance (consensus):")
for i, score in enumerate(importance['consensus']):
    print(f"  Feature {i}: {score:.4f}")
print(f"Expert agreement: {importance['expert_agreement']:.3f}")

# ============= Expert Clustering =============
print("\n" + "="*60)
print("Clustering Experts...")
print("="*60)

clusterer = ExpertClusterer(ensemble, method='kmeans')
labels = clusterer.cluster_by_importance(X_test, y_test, n_clusters=3, importance_method='permutation')

print(f"Cluster assignments: {labels}")
for cluster_id in range(3):
    experts_in_cluster = np.where(labels == cluster_id)[0]
    print(f"Cluster {cluster_id}: Experts {list(experts_in_cluster)}")

# ============= Stacking Ensemble =============
print("\n" + "="*60)
print("Training Stacked Ensemble...")
print("="*60)

stacked = StackedEnsemble(
    base_ensemble=ensemble,
    meta_hidden_dim=16,
    use_input_features=False,
    cluster_labels=labels
)

meta_results = stacked.train_meta_learner(X_train, y_train, epochs=50, lr=0.01, verbose=True)

# ============= Evaluation =============
print("\n" + "="*60)
print("Evaluation Results")
print("="*60)

# Individual predictions
y_pred_mean, uncertainty = ensemble.predict_with_uncertainty(X_test)
mse_ensemble = torch.nn.functional.mse_loss(y_pred_mean, y_test).item()

# Stacked predictions
y_pred_stacked = stacked.predict(X_test)
mse_stacked = torch.nn.functional.mse_loss(y_pred_stacked, y_test).item()

print(f"\nEnsemble (averaging) MSE: {mse_ensemble:.6f}")
print(f"Mean uncertainty: {uncertainty.mean():.6f}")
print(f"Stacked ensemble MSE: {mse_stacked:.6f}")
print(f"Improvement: {((mse_ensemble - mse_stacked) / mse_ensemble * 100):.2f}%")

print("\n" + "="*60)
print("Section 2.1 Complete")
print("="*60)
