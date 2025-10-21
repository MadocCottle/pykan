"""Complete demonstration of Extension 1: Hierarchical Ensemble of KAN Experts.

This experiment demonstrates the full ensemble pipeline:
1. Train multiple KAN experts with different seeds
2. Analyze variable importance across the ensemble
3. Cluster experts by specialization
4. Stack experts with meta-learner for optimal combination

Uses pykan utilities for data generation.

PyKAN Reference:
    Liu, Ziming, et al. "KAN: Kolmogorov-Arnold Networks."
    arXiv preprint arXiv:2404.19756 (2024).
    https://arxiv.org/abs/2404.19756

Author: Claude Code
Date: 2025-10-18
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add paths - add pykan to path (parent directory of madoc)
pykan_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(pykan_root))
sys.path.insert(0, str(Path(__file__).parent.parent / "ensemble"))

from expert_training import KANExpertEnsemble
from variable_importance import VariableImportanceAnalyzer
from clustering import ExpertClusterer
from stacking import StackedEnsemble

# Import pykan utilities
from kan.utils import create_dataset


def generate_synthetic_data(n_samples: int = 500, noise: float = 0.1):
    """Generate synthetic data with known variable importances.

    Function: y = 3*x0 + 2*sin(x1) + x2^2 + 0.5*x3 + noise
    Variable importance: x0 > x1 > x2 > x3 > x4 (x4 is irrelevant)

    Uses pykan's create_dataset for consistency.
    """
    def target_fn(x):
        # Note: create_dataset adds noise internally if specified
        return (3.0 * x[0] +
                2.0 * np.sin(x[1]) +
                x[2] ** 2 +
                0.5 * x[3] +
                0.0 * x[4])  # x4 is irrelevant

    # Generate dataset using pykan
    dataset = create_dataset(
        f=target_fn,
        n_var=5,
        ranges=[-2, 2],
        train_num=n_samples,
        device='cpu',
        seed=42
    )

    X = dataset['train_input']
    y = dataset['train_label']

    # Add noise if specified
    if noise > 0:
        y = y + noise * torch.randn_like(y)

    return X, y


def run_complete_ensemble_experiment():
    """Run complete ensemble experiment."""

    print("="*80)
    print("EXTENSION 1: Hierarchical Ensemble of KAN Experts - Complete Demo")
    print("="*80)

    # Generate data
    print("\n" + "="*80)
    print("1. Data Generation")
    print("="*80)

    X, y = generate_synthetic_data(n_samples=500, noise=0.1)

    # Split data
    train_size = 300
    val_size = 100
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]

    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")
    print(f"Test samples: {len(X_test)}")
    print(f"Input features: {X.shape[1]}")
    print(f"True variable importance order: x0 > x1 > x2 > x3 > x4 (x4 irrelevant)")

    # Train ensemble
    print("\n" + "="*80)
    print("2. Training Expert Ensemble")
    print("="*80)

    ensemble = KANExpertEnsemble(
        input_dim=5,
        hidden_dim=16,
        output_dim=1,
        depth=3,
        n_experts=10,
        kan_variant='rbf',
        device='cpu'
    )

    print(f"Training {ensemble.n_experts} experts with RBF-KAN...")
    print(f"Architecture: [{ensemble.input_dim}, {ensemble.hidden_dim}, {ensemble.output_dim}]")

    results = ensemble.train_experts(
        X_train, y_train,
        epochs=200,
        lr=0.01,
        verbose=False,
        validation_data=(X_val, y_val)
    )

    print(f"\nTraining complete!")
    print(f"Mean training loss: {np.mean(results['individual_losses']):.6f}")
    print(f"Std training loss:  {np.std(results['individual_losses']):.6f}")
    print(f"Mean validation loss: {np.mean(results['validation_losses']):.6f}")

    # Test simple averaging
    y_pred_avg, uncertainty = ensemble.predict_with_uncertainty(X_test)
    mse_avg = nn.MSELoss()(y_pred_avg, y_test).item()
    print(f"\nSimple averaging test MSE: {mse_avg:.6f}")
    print(f"Mean prediction uncertainty: {uncertainty.mean().item():.6f}")

    # Diversity analysis
    diversity = ensemble.get_expert_diversity(X_test)
    print(f"\nDiversity metrics:")
    for key, value in diversity.items():
        print(f"  {key}: {value:.6f}")

    # Variable importance analysis
    print("\n" + "="*80)
    print("3. Variable Importance Analysis")
    print("="*80)

    analyzer = VariableImportanceAnalyzer(ensemble)

    importance_scores = analyzer.compute_consensus_importance(
        X_val, y_val,
        methods=['weight', 'gradient', 'permutation']
    )

    print("\nImportance scores by method:")
    for method in ['weight', 'gradient', 'permutation', 'consensus']:
        scores = importance_scores[method]
        print(f"\n{method.capitalize()}:")
        for i, score in enumerate(scores):
            print(f"  x{i}: {score:.4f}")

    top_features = analyzer.get_top_features(importance_scores['consensus'], k=5)
    print(f"\nTop features (consensus):")
    for rank, (feat_idx, score) in enumerate(top_features, 1):
        print(f"  {rank}. Feature x{feat_idx}: {score:.4f}")

    # Expert clustering
    print("\n" + "="*80)
    print("4. Expert Clustering by Specialization")
    print("="*80)

    clusterer = ExpertClusterer(ensemble, method='kmeans')

    # Cluster by importance patterns
    labels_importance = clusterer.cluster_by_importance(
        X_val, y_val, n_clusters=3, importance_method='gradient'
    )

    summary = clusterer.get_cluster_summary(labels_importance, X_val, y_val)
    print(f"\nClustering by variable importance patterns:")
    print(f"Number of clusters: {summary['n_clusters']}")
    for cluster_id in range(summary['n_clusters']):
        members = summary['cluster_members'][cluster_id]
        perf = summary['cluster_performance'][cluster_id]
        print(f"  Cluster {cluster_id}: {len(members)} experts, MSE = {perf:.6f}")
        print(f"    Members: {members}")

    # Cluster by predictions
    labels_pred = clusterer.cluster_by_predictions(X_val, n_clusters=3)
    summary_pred = clusterer.get_cluster_summary(labels_pred, X_val, y_val)
    print(f"\nClustering by prediction similarity:")
    for cluster_id in range(summary_pred['n_clusters']):
        members = summary_pred['cluster_members'][cluster_id]
        perf = summary_pred['cluster_performance'][cluster_id]
        print(f"  Cluster {cluster_id}: {len(members)} experts, MSE = {perf:.6f}")

    # Stacked ensemble
    print("\n" + "="*80)
    print("5. Stacked Ensemble with Meta-Learner")
    print("="*80)

    # Linear stacking
    print("\nA. Linear Stacking (simple weighted combination)")
    stacked_linear = StackedEnsemble(ensemble, meta_hidden_dim=None, use_input=False)
    stacked_linear.train_meta_learner(
        X_train, y_train,
        epochs=150,
        lr=0.01,
        freeze_experts=True,
        validation_data=(X_val, y_val),
        verbose=False
    )

    y_pred_linear = stacked_linear.predict(X_test)
    mse_linear = nn.MSELoss()(y_pred_linear, y_test).item()
    print(f"Linear stacking test MSE: {mse_linear:.6f}")
    improvement_linear = (mse_avg - mse_linear) / mse_avg * 100
    print(f"Improvement over averaging: {improvement_linear:.1f}%")

    # Nonlinear stacking with input features
    print("\nB. Nonlinear Stacking (MLP meta-learner + input features)")
    stacked_nonlinear = StackedEnsemble(ensemble, meta_hidden_dim=32, use_input=True)
    stacked_nonlinear.train_meta_learner(
        X_train, y_train,
        epochs=150,
        lr=0.01,
        freeze_experts=True,
        validation_data=(X_val, y_val),
        verbose=False
    )

    y_pred_nonlinear = stacked_nonlinear.predict(X_test)
    mse_nonlinear = nn.MSELoss()(y_pred_nonlinear, y_test).item()
    print(f"Nonlinear stacking test MSE: {mse_nonlinear:.6f}")
    improvement_nonlinear = (mse_avg - mse_nonlinear) / mse_avg * 100
    print(f"Improvement over averaging: {improvement_nonlinear:.1f}%")

    # Summary
    print("\n" + "="*80)
    print("6. Final Summary")
    print("="*80)

    print(f"\nTest MSE comparison:")
    print(f"  Simple averaging:     {mse_avg:.6f}")
    print(f"  Linear stacking:      {mse_linear:.6f} ({improvement_linear:+.1f}%)")
    print(f"  Nonlinear stacking:   {mse_nonlinear:.6f} ({improvement_nonlinear:+.1f}%)")

    best_mse = min(mse_avg, mse_linear, mse_nonlinear)
    best_method = ['Simple averaging', 'Linear stacking', 'Nonlinear stacking'][
        [mse_avg, mse_linear, mse_nonlinear].index(best_mse)
    ]
    print(f"\nBest method: {best_method} (MSE = {best_mse:.6f})")

    print("\nVariable importance correctly identified:")
    expected_order = [0, 1, 2, 3, 4]  # Expected importance order
    actual_order = [feat_idx for feat_idx, _ in top_features]
    print(f"  Expected: {expected_order}")
    print(f"  Detected: {actual_order}")

    match = actual_order[:3] == expected_order[:3]
    print(f"  Top 3 match: {'✓' if match else '✗'}")

    print("\n" + "="*80)
    print("Experiment Complete!")
    print("="*80)

    return {
        'ensemble': ensemble,
        'mse_avg': mse_avg,
        'mse_linear': mse_linear,
        'mse_nonlinear': mse_nonlinear,
        'importance_scores': importance_scores,
        'cluster_labels': labels_importance
    }


if __name__ == '__main__':
    results = run_complete_ensemble_experiment()
