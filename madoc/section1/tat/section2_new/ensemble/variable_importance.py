"""Variable importance analysis for KAN ensembles.

This module extracts and analyzes variable usage patterns across an ensemble
of KAN experts to identify which input features are most important for predictions.

Key Features:
- Weight-based importance (magnitude of edge weights connecting to each input)
- Gradient-based importance (sensitivity of output to input changes)
- Permutation importance (performance drop when feature is shuffled)
- Consensus importance across ensemble

Reference:
- Plan Section: Extension 1 - Variable Importance and Pruning-based Analysis
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path


class VariableImportanceAnalyzer:
    """Analyze variable importance in KAN ensembles.

    This class provides multiple methods to quantify which input variables
    are most important for the model's predictions, with consensus estimates
    across the ensemble.

    Args:
        ensemble: KANExpertEnsemble instance with trained experts

    Example:
        >>> analyzer = VariableImportanceAnalyzer(ensemble)
        >>> importance = analyzer.compute_consensus_importance(X_val, y_val)
        >>> print(f"Feature 0 importance: {importance[0]:.4f}")
    """

    def __init__(self, ensemble):
        """Initialize analyzer with a trained ensemble.

        Args:
            ensemble: KANExpertEnsemble instance
        """
        self.ensemble = ensemble
        self.device = ensemble.device

        if len(ensemble.experts) == 0:
            raise ValueError("Ensemble has no trained experts")

    def compute_weight_importance(self, expert_idx: Optional[int] = None) -> np.ndarray:
        """Compute importance based on input layer weight magnitudes.

        For each input variable, compute the L2 norm of weights connecting
        that input to the first hidden layer.

        Args:
            expert_idx: Index of expert to analyze (None = average over all)

        Returns:
            Importance scores (input_dim,)
        """
        if expert_idx is not None:
            expert = self.ensemble.experts[expert_idx]
            return self._expert_weight_importance(expert)
        else:
            # Average importance across all experts
            importances = []
            for expert in self.ensemble.experts:
                importances.append(self._expert_weight_importance(expert))
            return np.mean(importances, axis=0)

    def _expert_weight_importance(self, expert: nn.Module) -> np.ndarray:
        """Compute weight importance for a single expert.

        Args:
            expert: KAN model

        Returns:
            Importance scores (input_dim,)
        """
        # Get first layer parameters
        first_layer = expert.layers[0]

        importance = np.zeros(self.ensemble.input_dim)

        # Aggregate weight magnitudes from first layer
        for param_name, param in first_layer.named_parameters():
            if 'weight' in param_name or 'coef' in param_name:
                # Param shape varies by basis type, but first dim is typically related to input
                weights = param.detach().cpu().numpy()

                # Compute L2 norm per input dimension
                # Handle different weight shapes
                if len(weights.shape) == 2:
                    # Shape: (output, input)
                    per_input = np.linalg.norm(weights, axis=0)
                elif len(weights.shape) == 3:
                    # Shape: (output, input, basis)
                    per_input = np.linalg.norm(weights, axis=(0, 2))
                else:
                    continue

                if len(per_input) == self.ensemble.input_dim:
                    importance += per_input

        # Normalize
        if importance.sum() > 0:
            importance /= importance.sum()

        return importance

    def compute_gradient_importance(
        self,
        X: torch.Tensor,
        expert_idx: Optional[int] = None
    ) -> np.ndarray:
        """Compute importance based on gradient magnitudes.

        Measures how much the output changes when each input changes.

        Args:
            X: Input data (N, input_dim)
            expert_idx: Index of expert to analyze (None = average over all)

        Returns:
            Importance scores (input_dim,)
        """
        X = X.to(self.device)

        if expert_idx is not None:
            return self._expert_gradient_importance(self.ensemble.experts[expert_idx], X)
        else:
            # Average over all experts
            importances = []
            for expert in self.ensemble.experts:
                importances.append(self._expert_gradient_importance(expert, X))
            return np.mean(importances, axis=0)

    def _expert_gradient_importance(self, expert: nn.Module, X: torch.Tensor) -> np.ndarray:
        """Compute gradient importance for a single expert.

        Args:
            expert: KAN model
            X: Input data (N, input_dim)

        Returns:
            Importance scores (input_dim,)
        """
        expert.eval()
        X_grad = X.clone().requires_grad_(True)

        # Forward pass
        output = expert(X_grad)

        # Compute gradients
        importance = np.zeros(self.ensemble.input_dim)

        for out_idx in range(output.shape[1]):
            # Backward for each output dimension
            if output.shape[1] > 1:
                grad_outputs = torch.zeros_like(output)
                grad_outputs[:, out_idx] = 1.0
                grads = torch.autograd.grad(
                    outputs=output,
                    inputs=X_grad,
                    grad_outputs=grad_outputs,
                    create_graph=False,
                    retain_graph=True
                )[0]
            else:
                grads = torch.autograd.grad(
                    outputs=output,
                    inputs=X_grad,
                    grad_outputs=torch.ones_like(output),
                    create_graph=False
                )[0]

            # Compute mean absolute gradient per input dimension
            importance += torch.abs(grads).mean(dim=0).detach().cpu().numpy()

        # Normalize
        if importance.sum() > 0:
            importance /= importance.sum()

        return importance

    def compute_permutation_importance(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        n_repeats: int = 10,
        metric: str = 'mse',
        expert_idx: Optional[int] = None
    ) -> np.ndarray:
        """Compute permutation importance.

        Measures performance drop when each feature is randomly shuffled.

        Args:
            X: Input data (N, input_dim)
            y: Target data (N, output_dim)
            n_repeats: Number of permutation repeats
            metric: Evaluation metric ('mse', 'mae')
            expert_idx: Index of expert to analyze (None = average over all)

        Returns:
            Importance scores (input_dim,)
        """
        X = X.to(self.device)
        y = y.to(self.device)

        if expert_idx is not None:
            return self._expert_permutation_importance(
                self.ensemble.experts[expert_idx], X, y, n_repeats, metric
            )
        else:
            # Average over all experts
            importances = []
            for expert in self.ensemble.experts:
                importances.append(
                    self._expert_permutation_importance(expert, X, y, n_repeats, metric)
                )
            return np.mean(importances, axis=0)

    def _expert_permutation_importance(
        self,
        expert: nn.Module,
        X: torch.Tensor,
        y: torch.Tensor,
        n_repeats: int,
        metric: str
    ) -> np.ndarray:
        """Compute permutation importance for a single expert.

        Args:
            expert: KAN model
            X: Input data
            y: Target data
            n_repeats: Number of repeats
            metric: Evaluation metric

        Returns:
            Importance scores (input_dim,)
        """
        expert.eval()

        # Baseline performance
        with torch.no_grad():
            y_pred = expert(X)
            baseline_score = self._compute_metric(y_pred, y, metric)

        importance = np.zeros(self.ensemble.input_dim)

        # Permute each feature
        for feat_idx in range(self.ensemble.input_dim):
            scores = []
            for _ in range(n_repeats):
                # Permute feature
                X_perm = X.clone()
                perm_indices = torch.randperm(X.shape[0])
                X_perm[:, feat_idx] = X[perm_indices, feat_idx]

                # Evaluate
                with torch.no_grad():
                    y_pred_perm = expert(X_perm)
                    perm_score = self._compute_metric(y_pred_perm, y, metric)

                # Importance = performance drop
                scores.append(perm_score - baseline_score)

            importance[feat_idx] = np.mean(scores)

        # Normalize (larger drop = more important)
        if importance.sum() > 0:
            importance /= importance.sum()

        return importance

    def _compute_metric(self, y_pred: torch.Tensor, y_true: torch.Tensor, metric: str) -> float:
        """Compute evaluation metric.

        Args:
            y_pred: Predictions
            y_true: Ground truth
            metric: Metric name

        Returns:
            Metric value (lower is better for MSE/MAE)
        """
        if metric == 'mse':
            return nn.MSELoss()(y_pred, y_true).item()
        elif metric == 'mae':
            return nn.L1Loss()(y_pred, y_true).item()
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def compute_consensus_importance(
        self,
        X: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        methods: List[str] = ['weight', 'gradient']
    ) -> Dict[str, np.ndarray]:
        """Compute consensus importance across multiple methods.

        Args:
            X: Input data (N, input_dim)
            y: Target data (N, output_dim) - required for permutation method
            methods: List of methods to use ('weight', 'gradient', 'permutation')

        Returns:
            Dictionary mapping method names to importance scores
        """
        results = {}

        for method in methods:
            if method == 'weight':
                results['weight'] = self.compute_weight_importance()
            elif method == 'gradient':
                results['gradient'] = self.compute_gradient_importance(X)
            elif method == 'permutation':
                if y is None:
                    raise ValueError("Target data y is required for permutation importance")
                results['permutation'] = self.compute_permutation_importance(X, y)
            else:
                raise ValueError(f"Unknown method: {method}")

        # Compute average consensus
        all_scores = np.array(list(results.values()))
        results['consensus'] = np.mean(all_scores, axis=0)

        # Normalize consensus
        results['consensus'] /= results['consensus'].sum()

        return results

    def get_top_features(
        self,
        importance: np.ndarray,
        k: int = 5
    ) -> List[tuple]:
        """Get top-k most important features.

        Args:
            importance: Importance scores (input_dim,)
            k: Number of top features to return

        Returns:
            List of (feature_index, importance_score) tuples, sorted by importance
        """
        indices = np.argsort(importance)[::-1][:k]
        return [(int(idx), float(importance[idx])) for idx in indices]


# =============================================================================
# Example Usage and Testing
# =============================================================================

if __name__ == '__main__':
    import sys
    from pathlib import Path

    # Add parent directories to path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "section1"))
    sys.path.insert(0, str(Path(__file__).parent))

    from expert_training import KANExpertEnsemble

    print("="*70)
    print("Testing Variable Importance Analysis")
    print("="*70)

    # Generate synthetic data with different feature importances
    torch.manual_seed(42)
    n_samples = 200
    n_features = 5

    # Create features with different importances
    X = torch.randn(n_samples, n_features)

    # y depends strongly on features 0 and 2, weakly on 1, not at all on 3 and 4
    y = (2.0 * X[:, 0] +
         0.5 * X[:, 1] +
         1.5 * torch.sin(X[:, 2]) +
         0.0 * X[:, 3] +
         0.0 * X[:, 4]).reshape(-1, 1)

    # Split data
    train_size = 150
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    # Train ensemble
    ensemble = KANExpertEnsemble(
        input_dim=n_features,
        hidden_dim=10,
        output_dim=1,
        depth=3,
        n_experts=3,
        kan_variant='rbf'
    )

    print("\nTraining ensemble...")
    results = ensemble.train_experts(
        X_train, y_train,
        epochs=100,
        lr=0.01,
        verbose=False
    )
    print(f"Training complete. Mean final loss: {np.mean(results['individual_losses']):.6f}")

    # Analyze variable importance
    analyzer = VariableImportanceAnalyzer(ensemble)

    print("\n" + "="*70)
    print("Variable Importance Analysis")
    print("="*70)

    # Compute all importance metrics
    importance_scores = analyzer.compute_consensus_importance(
        X_val, y_val,
        methods=['weight', 'gradient', 'permutation']
    )

    # Display results
    for method, scores in importance_scores.items():
        print(f"\n{method.capitalize()} Importance:")
        for i, score in enumerate(scores):
            print(f"  Feature {i}: {score:.4f}")

    print("\n" + "="*70)
    print("Top Features (Consensus):")
    print("="*70)

    top_features = analyzer.get_top_features(importance_scores['consensus'], k=3)
    for rank, (feat_idx, score) in enumerate(top_features, 1):
        print(f"  {rank}. Feature {feat_idx}: {score:.4f}")

    print("\nExpected: Features 0, 2, and 1 should be most important (in that order)")
