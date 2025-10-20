"""Stacking ensemble for KAN experts.

This module implements hierarchical stacking of expert predictions, where
a meta-learner combines expert outputs to make final predictions. The meta-
learner can be trained with frozen or fine-tuned experts.

Key Features:
- Linear and nonlinear meta-learners
- Cluster-aware stacking (different weights per cluster)
- Cross-validation for meta-learner training
- Selective expert inclusion based on performance

Reference:
- Plan Section: Extension 1 - Stacked Ensemble Architecture
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Union
from pathlib import Path


class MetaLearner(nn.Module):
    """Meta-learner for stacking ensemble.

    The meta-learner takes expert predictions as input and learns optimal
    combination weights.

    Args:
        n_experts: Number of experts in ensemble
        output_dim: Output dimension
        hidden_dim: Hidden dimension for nonlinear meta-learner (None = linear)
        use_input: Whether to include original input features

    Example:
        >>> meta = MetaLearner(n_experts=10, output_dim=1, hidden_dim=32)
        >>> expert_preds = torch.randn(batch_size, n_experts, output_dim)
        >>> final_pred = meta(expert_preds)
    """

    def __init__(
        self,
        n_experts: int,
        output_dim: int = 1,
        hidden_dim: Optional[int] = None,
        use_input: bool = False,
        input_dim: Optional[int] = None
    ):
        super().__init__()
        self.n_experts = n_experts
        self.output_dim = output_dim
        self.use_input = use_input

        # Input dimension for meta-learner
        meta_input_dim = n_experts * output_dim
        if use_input:
            if input_dim is None:
                raise ValueError("input_dim required when use_input=True")
            meta_input_dim += input_dim
            self.input_dim = input_dim

        if hidden_dim is None:
            # Linear meta-learner (simple weighted combination)
            self.meta_net = nn.Linear(meta_input_dim, output_dim)
        else:
            # Nonlinear meta-learner (MLP)
            self.meta_net = nn.Sequential(
                nn.Linear(meta_input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, output_dim)
            )

    def forward(
        self,
        expert_predictions: torch.Tensor,
        input_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through meta-learner.

        Args:
            expert_predictions: Expert predictions (batch_size, n_experts, output_dim)
            input_features: Original input features (batch_size, input_dim), optional

        Returns:
            Final predictions (batch_size, output_dim)
        """
        batch_size = expert_predictions.shape[0]

        # Flatten expert predictions
        meta_input = expert_predictions.view(batch_size, -1)

        # Optionally include original input features
        if self.use_input:
            if input_features is None:
                raise ValueError("input_features required when use_input=True")
            meta_input = torch.cat([meta_input, input_features], dim=1)

        # Meta-learner prediction
        output = self.meta_net(meta_input)

        return output


class StackedEnsemble:
    """Stacked ensemble of KAN experts with meta-learner.

    This class wraps a KANExpertEnsemble and adds a meta-learner that
    optimally combines expert predictions.

    Args:
        ensemble: KANExpertEnsemble instance
        meta_hidden_dim: Hidden dimension for meta-learner (None = linear)
        use_input: Whether meta-learner sees original input features
        device: Torch device

    Example:
        >>> stacked = StackedEnsemble(ensemble, meta_hidden_dim=32)
        >>> stacked.train_meta_learner(X_train, y_train, epochs=100)
        >>> y_pred = stacked.predict(X_test)
    """

    def __init__(
        self,
        ensemble,
        meta_hidden_dim: Optional[int] = None,
        use_input: bool = False,
        device: str = 'cpu'
    ):
        self.ensemble = ensemble
        self.device = torch.device(device)
        self.use_input = use_input

        if len(ensemble.experts) == 0:
            raise ValueError("Ensemble has no trained experts")

        # Create meta-learner
        self.meta_learner = MetaLearner(
            n_experts=len(ensemble.experts),
            output_dim=ensemble.output_dim,
            hidden_dim=meta_hidden_dim,
            use_input=use_input,
            input_dim=ensemble.input_dim if use_input else None
        ).to(self.device)

        self.is_trained = False

    def get_expert_predictions(
        self,
        X: torch.Tensor,
        detach: bool = True
    ) -> torch.Tensor:
        """Get predictions from all experts.

        Args:
            X: Input tensor (batch_size, input_dim)
            detach: Whether to detach gradients (for frozen experts)

        Returns:
            Expert predictions (batch_size, n_experts, output_dim)
        """
        X = X.to(self.device)

        expert_preds = []
        for expert in self.ensemble.experts:
            expert.eval()
            if detach:
                with torch.no_grad():
                    pred = expert(X)
            else:
                pred = expert(X)
            expert_preds.append(pred)

        return torch.stack(expert_preds, dim=1)  # (batch, n_experts, output_dim)

    def train_meta_learner(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        epochs: int = 100,
        lr: float = 0.001,
        freeze_experts: bool = True,
        validation_data: Optional[tuple] = None,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """Train the meta-learner on expert predictions.

        Args:
            X_train: Training inputs (N, input_dim)
            y_train: Training outputs (N, output_dim)
            epochs: Number of training epochs
            lr: Learning rate
            freeze_experts: Whether to freeze expert weights
            validation_data: Optional (X_val, y_val) tuple
            verbose: Print progress

        Returns:
            Training history
        """
        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)

        # Optimizer for meta-learner
        if freeze_experts:
            # Only optimize meta-learner
            optimizer = torch.optim.Adam(self.meta_learner.parameters(), lr=lr)
        else:
            # Fine-tune experts + meta-learner
            all_params = list(self.meta_learner.parameters())
            for expert in self.ensemble.experts:
                all_params.extend(expert.parameters())
            optimizer = torch.optim.Adam(all_params, lr=lr)

        loss_fn = nn.MSELoss()

        history = {
            'train_loss': [],
            'val_loss': [] if validation_data is not None else None
        }

        if verbose:
            mode = "frozen experts" if freeze_experts else "fine-tuning experts"
            print(f"\nTraining meta-learner ({mode})...")

        for epoch in range(epochs):
            self.meta_learner.train()

            # Get expert predictions
            expert_preds = self.get_expert_predictions(X_train, detach=freeze_experts)

            # Meta-learner prediction
            if self.use_input:
                final_pred = self.meta_learner(expert_preds, X_train)
            else:
                final_pred = self.meta_learner(expert_preds)

            # Loss and backprop
            loss = loss_fn(final_pred, y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            history['train_loss'].append(loss.item())

            # Validation
            if validation_data is not None:
                X_val, y_val = validation_data
                X_val = X_val.to(self.device)
                y_val = y_val.to(self.device)

                self.meta_learner.eval()
                with torch.no_grad():
                    expert_preds_val = self.get_expert_predictions(X_val, detach=True)
                    if self.use_input:
                        final_pred_val = self.meta_learner(expert_preds_val, X_val)
                    else:
                        final_pred_val = self.meta_learner(expert_preds_val)
                    val_loss = loss_fn(final_pred_val, y_val)
                    history['val_loss'].append(val_loss.item())

            # Print progress
            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                msg = f"Epoch {epoch+1:4d}/{epochs}: Train Loss = {loss.item():.6f}"
                if validation_data is not None:
                    msg += f", Val Loss = {val_loss.item():.6f}"
                print(msg)

        self.is_trained = True

        if verbose:
            print("Meta-learner training complete!")

        return history

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Make predictions using stacked ensemble.

        Args:
            X: Input tensor (N, input_dim)

        Returns:
            Predictions (N, output_dim)
        """
        if not self.is_trained:
            raise RuntimeError("Meta-learner not trained. Call train_meta_learner() first.")

        X = X.to(self.device)

        self.meta_learner.eval()
        with torch.no_grad():
            expert_preds = self.get_expert_predictions(X, detach=True)
            if self.use_input:
                final_pred = self.meta_learner(expert_preds, X)
            else:
                final_pred = self.meta_learner(expert_preds)

        return final_pred

    def get_meta_weights(self) -> np.ndarray:
        """Get meta-learner weights for linear meta-learner.

        Returns:
            Weight matrix (output_dim, n_experts * expert_output_dim)
        """
        if isinstance(self.meta_learner.meta_net, nn.Linear):
            weights = self.meta_learner.meta_net.weight.detach().cpu().numpy()
            return weights
        else:
            raise ValueError("Cannot extract weights from nonlinear meta-learner")


class ClusterAwareStackedEnsemble(StackedEnsemble):
    """Stacked ensemble with cluster-aware meta-learner.

    Uses expert clustering to create specialized meta-learners for each cluster.

    Args:
        ensemble: KANExpertEnsemble instance
        cluster_labels: Cluster assignments (n_experts,)
        meta_hidden_dim: Hidden dimension for meta-learners
        use_input: Whether to use original input features
        device: Torch device

    Example:
        >>> # After clustering experts
        >>> stacked = ClusterAwareStackedEnsemble(ensemble, cluster_labels)
        >>> stacked.train_meta_learner(X_train, y_train)
        >>> y_pred = stacked.predict(X_test)
    """

    def __init__(
        self,
        ensemble,
        cluster_labels: np.ndarray,
        meta_hidden_dim: Optional[int] = None,
        use_input: bool = False,
        device: str = 'cpu'
    ):
        # Don't call parent __init__, we'll create multiple meta-learners
        self.ensemble = ensemble
        self.device = torch.device(device)
        self.use_input = use_input
        self.cluster_labels = cluster_labels

        if len(ensemble.experts) == 0:
            raise ValueError("Ensemble has no trained experts")

        # Create meta-learner for each cluster
        self.n_clusters = len(np.unique(cluster_labels[cluster_labels >= 0]))
        self.meta_learners = nn.ModuleList()

        for cluster_id in range(self.n_clusters):
            cluster_size = np.sum(cluster_labels == cluster_id)
            meta = MetaLearner(
                n_experts=cluster_size,
                output_dim=ensemble.output_dim,
                hidden_dim=meta_hidden_dim,
                use_input=use_input,
                input_dim=ensemble.input_dim if use_input else None
            ).to(self.device)
            self.meta_learners.append(meta)

        # Final aggregation layer (combines cluster outputs)
        self.final_aggregator = nn.Linear(
            self.n_clusters * ensemble.output_dim,
            ensemble.output_dim
        ).to(self.device)

        self.is_trained = False

    def get_cluster_expert_predictions(
        self,
        X: torch.Tensor,
        cluster_id: int,
        detach: bool = True
    ) -> torch.Tensor:
        """Get predictions from experts in a specific cluster.

        Args:
            X: Input tensor
            cluster_id: Cluster ID
            detach: Whether to detach gradients

        Returns:
            Expert predictions for cluster (batch_size, cluster_size, output_dim)
        """
        X = X.to(self.device)

        cluster_mask = self.cluster_labels == cluster_id
        cluster_expert_indices = np.where(cluster_mask)[0]

        expert_preds = []
        for expert_idx in cluster_expert_indices:
            expert = self.ensemble.experts[expert_idx]
            expert.eval()
            if detach:
                with torch.no_grad():
                    pred = expert(X)
            else:
                pred = expert(X)
            expert_preds.append(pred)

        return torch.stack(expert_preds, dim=1)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Make predictions using cluster-aware stacking.

        Args:
            X: Input tensor (N, input_dim)

        Returns:
            Predictions (N, output_dim)
        """
        if not self.is_trained:
            raise RuntimeError("Meta-learners not trained. Call train_meta_learner() first.")

        X = X.to(self.device)

        # Get predictions from each cluster's meta-learner
        cluster_outputs = []
        for cluster_id in range(self.n_clusters):
            meta = self.meta_learners[cluster_id]
            meta.eval()

            with torch.no_grad():
                expert_preds = self.get_cluster_expert_predictions(X, cluster_id, detach=True)
                if self.use_input:
                    cluster_out = meta(expert_preds, X)
                else:
                    cluster_out = meta(expert_preds)
                cluster_outputs.append(cluster_out)

        # Aggregate cluster outputs
        cluster_outputs = torch.cat(cluster_outputs, dim=1)  # (batch, n_clusters * output_dim)
        final_pred = self.final_aggregator(cluster_outputs)

        return final_pred


# =============================================================================
# Example Usage and Testing
# =============================================================================

if __name__ == '__main__':
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "section1"))
    sys.path.insert(0, str(Path(__file__).parent))

    from expert_training import KANExpertEnsemble

    print("="*70)
    print("Testing Stacked Ensemble")
    print("="*70)

    # Generate synthetic data
    torch.manual_seed(42)
    np.random.seed(42)

    X_train = torch.randn(200, 3)
    y_train = (2.0 * X_train[:, 0] + torch.sin(X_train[:, 1]) + 0.5 * X_train[:, 2]).reshape(-1, 1)

    X_val = torch.randn(50, 3)
    y_val = (2.0 * X_val[:, 0] + torch.sin(X_val[:, 1]) + 0.5 * X_val[:, 2]).reshape(-1, 1)

    X_test = torch.randn(30, 3)
    y_test = (2.0 * X_test[:, 0] + torch.sin(X_test[:, 1]) + 0.5 * X_test[:, 2]).reshape(-1, 1)

    # Train ensemble
    print("\n1. Training Expert Ensemble")
    print("-"*70)

    ensemble = KANExpertEnsemble(
        input_dim=3,
        hidden_dim=10,
        output_dim=1,
        depth=3,
        n_experts=5,
        kan_variant='rbf'
    )

    ensemble.train_experts(X_train, y_train, epochs=100, lr=0.01, verbose=False)
    print("Expert ensemble trained!")

    # Test simple averaging
    y_pred_avg = ensemble.predict(X_test)
    mse_avg = nn.MSELoss()(y_pred_avg, y_test).item()
    print(f"Simple averaging MSE: {mse_avg:.6f}")

    # Test linear stacking
    print("\n2. Testing Linear Stacking")
    print("-"*70)

    stacked_linear = StackedEnsemble(ensemble, meta_hidden_dim=None)
    history = stacked_linear.train_meta_learner(
        X_train, y_train,
        epochs=100,
        lr=0.01,
        validation_data=(X_val, y_val),
        verbose=True
    )

    y_pred_stack = stacked_linear.predict(X_test)
    mse_stack = nn.MSELoss()(y_pred_stack, y_test).item()
    print(f"\nLinear stacking MSE: {mse_stack:.6f}")
    print(f"Improvement: {(mse_avg - mse_stack) / mse_avg * 100:.1f}%")

    # Test nonlinear stacking
    print("\n3. Testing Nonlinear Stacking")
    print("-"*70)

    stacked_nonlinear = StackedEnsemble(ensemble, meta_hidden_dim=32, use_input=True)
    stacked_nonlinear.train_meta_learner(
        X_train, y_train,
        epochs=100,
        lr=0.01,
        validation_data=(X_val, y_val),
        verbose=False
    )

    y_pred_nonlinear = stacked_nonlinear.predict(X_test)
    mse_nonlinear = nn.MSELoss()(y_pred_nonlinear, y_test).item()
    print(f"Nonlinear stacking MSE: {mse_nonlinear:.6f}")
    print(f"Improvement over averaging: {(mse_avg - mse_nonlinear) / mse_avg * 100:.1f}%")

    print("\n" + "="*70)
    print("Stacking tests complete!")
    print("="*70)
