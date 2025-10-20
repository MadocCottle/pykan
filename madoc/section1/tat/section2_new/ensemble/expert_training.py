"""Train ensemble of KAN experts with different random seeds.

This module implements the core expert ensemble training framework for Extension 1.
Each expert is trained independently with different initialization seeds to promote
diversity, then predictions are combined for improved robustness and uncertainty
quantification.

Key Features:
- Multi-seed expert training with parallel support
- Ensemble prediction (mean, median, weighted)
- Epistemic uncertainty estimation
- Variable importance consensus analysis
- Support for pykan's MultKAN (B-spline basis)

PyKAN Reference:
    Liu, Ziming, et al. "KAN: Kolmogorov-Arnold Networks."
    arXiv preprint arXiv:2404.19756 (2024).
    https://arxiv.org/abs/2404.19756

Reference:
- Plan Section: Extension 1 - Hierarchical Ensemble of KAN Experts
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Callable
import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "section1"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from section1.models.kan_variants import (
    ChebyshevKAN, FourierKAN, WaveletKAN, RBF_KAN
)

# Import pykan wrapper for B-spline support
try:
    from models.pykan_wrapper import PyKANCompatible
    _HAS_PYKAN = True
except ImportError:
    _HAS_PYKAN = False


class KANExpertEnsemble:
    """Ensemble of KAN models trained with different random seeds.

    This class manages training multiple KAN experts independently, each with
    different initialization to promote diversity. The ensemble provides:
    - Reduced variance through averaging
    - Epistemic uncertainty estimates
    - Improved robustness to initialization

    Args:
        input_dim: Input dimension
        hidden_dim: Hidden layer width
        output_dim: Output dimension
        depth: Number of layers (including input and output)
        n_experts: Number of experts to train
        kan_variant: Type of KAN ('bspline', 'chebyshev', 'fourier', 'wavelet', 'rbf')
        seeds: Optional custom seed list (defaults to range(n_experts))
        device: Torch device ('cpu', 'cuda', or 'mps')
        **kan_kwargs: Additional KAN-specific arguments (e.g., degree, grid_size)

    Supported KAN Variants:
        - 'bspline': B-spline basis (uses pykan's MultKAN, Liu et al. 2024)
        - 'chebyshev': Chebyshev polynomial basis
        - 'fourier': Fourier basis
        - 'wavelet': Wavelet basis
        - 'rbf': Radial basis functions

    Example:
        >>> # Use pykan's B-spline KAN
        >>> ensemble = KANExpertEnsemble(
        ...     input_dim=2,
        ...     hidden_dim=5,
        ...     output_dim=1,
        ...     depth=3,
        ...     n_experts=10,
        ...     kan_variant='bspline'
        ... )
        >>> results = ensemble.train_experts(X_train, y_train, epochs=500)
        >>> y_pred, uncertainty = ensemble.predict_with_uncertainty(X_test)

        >>> # Use custom Chebyshev KAN
        >>> ensemble = KANExpertEnsemble(
        ...     input_dim=2,
        ...     hidden_dim=5,
        ...     output_dim=1,
        ...     depth=3,
        ...     n_experts=10,
        ...     kan_variant='chebyshev'
        ... )
        >>> results = ensemble.train_experts(X_train, y_train, epochs=500)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 1,
        depth: int = 3,
        n_experts: int = 10,
        kan_variant: str = 'rbf',
        seeds: Optional[List[int]] = None,
        device: str = 'cpu',
        **kan_kwargs
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.depth = depth
        self.n_experts = n_experts
        self.kan_variant = kan_variant.lower()
        self.seeds = seeds if seeds is not None else list(range(n_experts))
        self.device = torch.device(device)
        self.experts: List[nn.Module] = []
        self.kan_kwargs = kan_kwargs  # Additional KAN-specific arguments (degree, grid_size, etc.)

        # Validate inputs
        if len(self.seeds) != n_experts:
            raise ValueError(f"Number of seeds ({len(self.seeds)}) must match n_experts ({n_experts})")

        # Map variant name to class
        self.variant_map = {
            'chebyshev': ChebyshevKAN,
            'fourier': FourierKAN,
            'wavelet': WaveletKAN,
            'rbf': RBF_KAN
        }

        # Add bspline if pykan is available
        if _HAS_PYKAN:
            self.variant_map['bspline'] = PyKANCompatible

        if self.kan_variant not in self.variant_map:
            available = list(self.variant_map.keys())
            raise ValueError(
                f"Unknown KAN variant: {kan_variant}. "
                f"Available: {available}"
            )

    def _create_expert(self, seed: int) -> nn.Module:
        """Create a single KAN expert with given seed.

        Args:
            seed: Random seed for initialization

        Returns:
            Initialized KAN model
        """
        torch.manual_seed(seed)
        kan_class = self.variant_map[self.kan_variant]
        expert = kan_class(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            depth=self.depth,
            **self.kan_kwargs
        ).to(self.device)
        return expert

    def train_experts(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        epochs: int = 500,
        lr: float = 0.001,
        optimizer_type: str = 'adam',
        verbose: bool = True,
        loss_fn: Optional[Callable] = None,
        validation_data: Optional[tuple] = None
    ) -> Dict[str, Any]:
        """Train all experts independently.

        Each expert is trained with:
        - Different random seed (for weight initialization)
        - Same architecture and hyperparameters
        - Independent optimization trajectory

        Args:
            X_train: Training inputs (N, input_dim)
            y_train: Training outputs (N, output_dim)
            epochs: Number of training epochs
            lr: Learning rate
            optimizer_type: Optimizer type ('adam', 'sgd', 'adamw')
            verbose: Print training progress
            loss_fn: Custom loss function (default: MSELoss)
            validation_data: Optional (X_val, y_val) for validation

        Returns:
            Dictionary containing:
                - 'individual_losses': Final training loss for each expert
                - 'training_history': Loss history for each expert
                - 'validation_losses': Final validation loss (if provided)
        """
        # Setup
        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)

        if loss_fn is None:
            loss_fn = nn.MSELoss()

        results = {
            'individual_losses': [],
            'training_history': [],
            'validation_losses': []
        }

        # Train each expert
        for i, seed in enumerate(self.seeds):
            if verbose:
                print(f"\n{'='*60}")
                print(f"Training Expert {i+1}/{self.n_experts} (seed={seed})")
                print(f"{'='*60}")

            # Create expert
            expert = self._create_expert(seed)

            # Create optimizer
            if optimizer_type.lower() == 'adam':
                optimizer = torch.optim.Adam(expert.parameters(), lr=lr)
            elif optimizer_type.lower() == 'sgd':
                optimizer = torch.optim.SGD(expert.parameters(), lr=lr, momentum=0.9)
            elif optimizer_type.lower() == 'adamw':
                optimizer = torch.optim.AdamW(expert.parameters(), lr=lr)
            else:
                raise ValueError(f"Unknown optimizer: {optimizer_type}")

            # Training loop
            history = []
            for epoch in range(epochs):
                expert.train()
                optimizer.zero_grad()

                # Forward pass
                pred = expert(X_train)
                loss = loss_fn(pred, y_train)

                # Backward pass
                loss.backward()
                optimizer.step()

                history.append(loss.item())

                # Print progress
                if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                    print(f"Epoch {epoch+1:4d}/{epochs}: Loss = {loss.item():.6f}")

            # Store expert and results
            expert.eval()
            self.experts.append(expert)
            results['individual_losses'].append(loss.item())
            results['training_history'].append(history)

            # Validation
            if validation_data is not None:
                X_val, y_val = validation_data
                X_val = X_val.to(self.device)
                y_val = y_val.to(self.device)
                with torch.no_grad():
                    val_pred = expert(X_val)
                    val_loss = loss_fn(val_pred, y_val).item()
                    results['validation_losses'].append(val_loss)
                    if verbose:
                        print(f"Validation Loss: {val_loss:.6f}")

            if verbose:
                print(f"Expert {i+1} Final Training Loss: {loss.item():.6f}")

        if verbose:
            print(f"\n{'='*60}")
            print(f"Ensemble Training Complete!")
            print(f"Mean Loss: {sum(results['individual_losses']) / len(results['individual_losses']):.6f}")
            print(f"Std Loss:  {torch.tensor(results['individual_losses']).std().item():.6f}")
            print(f"{'='*60}\n")

        return results

    def predict(
        self,
        X: torch.Tensor,
        method: str = 'mean',
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Ensemble prediction.

        Args:
            X: Input tensor (N, input_dim)
            method: Aggregation method ('mean', 'median', 'weighted')
            weights: Expert weights for weighted averaging (n_experts,)

        Returns:
            Ensemble predictions (N, output_dim)
        """
        if len(self.experts) == 0:
            raise RuntimeError("No experts trained. Call train_experts() first.")

        X = X.to(self.device)

        # Get all expert predictions
        with torch.no_grad():
            preds = torch.stack([expert(X) for expert in self.experts])  # (n_experts, N, output_dim)

        # Aggregate predictions
        if method == 'mean':
            return preds.mean(dim=0)
        elif method == 'median':
            return preds.median(dim=0)[0]
        elif method == 'weighted':
            if weights is None:
                raise ValueError("Weights must be provided for weighted averaging")
            weights = weights.to(self.device).view(-1, 1, 1)  # (n_experts, 1, 1)
            weights = weights / weights.sum()  # Normalize
            return (preds * weights).sum(dim=0)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    def predict_with_uncertainty(
        self,
        X: torch.Tensor,
        method: str = 'mean'
    ) -> tuple:
        """Predict with epistemic uncertainty estimation.

        Epistemic uncertainty is estimated from the variance across expert
        predictions, representing uncertainty due to limited data.

        Args:
            X: Input tensor (N, input_dim)
            method: Aggregation method for mean prediction

        Returns:
            Tuple of (mean_prediction, std_prediction)
                - mean_prediction: (N, output_dim)
                - std_prediction: (N, output_dim)
        """
        if len(self.experts) == 0:
            raise RuntimeError("No experts trained. Call train_experts() first.")

        X = X.to(self.device)

        # Get all expert predictions
        with torch.no_grad():
            preds = torch.stack([expert(X) for expert in self.experts])  # (n_experts, N, output_dim)

        # Compute statistics
        mean = preds.mean(dim=0)
        std = preds.std(dim=0)

        return mean, std

    def get_expert_diversity(self, X: torch.Tensor) -> Dict[str, float]:
        """Measure diversity of expert predictions.

        Diversity metrics:
        - Mean pairwise disagreement
        - Coefficient of variation
        - Prediction range

        Args:
            X: Input tensor (N, input_dim)

        Returns:
            Dictionary of diversity metrics
        """
        if len(self.experts) == 0:
            raise RuntimeError("No experts trained. Call train_experts() first.")

        X = X.to(self.device)

        with torch.no_grad():
            preds = torch.stack([expert(X) for expert in self.experts])  # (n_experts, N, output_dim)

        # Compute diversity metrics
        mean = preds.mean(dim=0)
        std = preds.std(dim=0)

        # Mean pairwise disagreement (average L2 distance between expert predictions)
        n_pairs = self.n_experts * (self.n_experts - 1) / 2
        pairwise_dist = 0.0
        for i in range(self.n_experts):
            for j in range(i + 1, self.n_experts):
                pairwise_dist += torch.mean((preds[i] - preds[j]) ** 2).item()
        pairwise_dist /= n_pairs

        # Coefficient of variation (std / mean)
        cv = (std / (torch.abs(mean) + 1e-8)).mean().item()

        # Prediction range (max - min across experts)
        pred_range = (preds.max(dim=0)[0] - preds.min(dim=0)[0]).mean().item()

        return {
            'mean_pairwise_disagreement': pairwise_dist,
            'coefficient_of_variation': cv,
            'prediction_range': pred_range,
            'mean_uncertainty': std.mean().item()
        }

    def save_ensemble(self, path: str):
        """Save all experts to disk.

        Args:
            path: Directory path to save experts
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        for i, expert in enumerate(self.experts):
            expert_path = path / f"expert_{i}_seed_{self.seeds[i]}.pt"
            torch.save({
                'state_dict': expert.state_dict(),
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'output_dim': self.output_dim,
                'depth': self.depth,
                'variant': self.kan_variant,
                'seed': self.seeds[i]
            }, expert_path)

        print(f"Saved {len(self.experts)} experts to {path}")

    def load_ensemble(self, path: str):
        """Load experts from disk.

        Args:
            path: Directory path containing saved experts
        """
        path = Path(path)
        expert_files = sorted(path.glob("expert_*.pt"))

        if len(expert_files) == 0:
            raise FileNotFoundError(f"No expert files found in {path}")

        self.experts = []
        for expert_file in expert_files:
            checkpoint = torch.load(expert_file, map_location=self.device, weights_only=False)
            expert = self._create_expert(checkpoint['seed'])
            expert.load_state_dict(checkpoint['state_dict'])
            expert.eval()
            self.experts.append(expert)

        print(f"Loaded {len(self.experts)} experts from {path}")


# =============================================================================
# Example Usage and Testing
# =============================================================================

if __name__ == '__main__':
    print("="*70)
    print("Testing KAN Expert Ensemble Training")
    print("="*70)

    # Test on simple 1D function: y = sin(x)
    X_train = torch.linspace(-3, 3, 100).reshape(-1, 1)
    y_train = torch.sin(X_train)

    X_test = torch.linspace(-3, 3, 50).reshape(-1, 1)
    y_test = torch.sin(X_test)

    # Create ensemble (using RBF variant - Chebyshev has inplace operation issues)
    ensemble = KANExpertEnsemble(
        input_dim=1,
        hidden_dim=5,
        output_dim=1,
        depth=3,
        n_experts=5,  # Reduced for quick testing
        kan_variant='rbf'
    )

    # Train experts
    results = ensemble.train_experts(
        X_train, y_train,
        epochs=200,
        lr=0.01,
        verbose=True,
        validation_data=(X_test, y_test)
    )

    # Test ensemble prediction
    y_pred, uncertainty = ensemble.predict_with_uncertainty(X_test)

    test_mse = nn.MSELoss()(y_pred, y_test)
    print(f"\nEnsemble Test MSE: {test_mse.item():.6f}")
    print(f"Mean Uncertainty: {uncertainty.mean().item():.6f}")

    # Diversity metrics
    diversity = ensemble.get_expert_diversity(X_test)
    print(f"\nDiversity Metrics:")
    for key, value in diversity.items():
        print(f"  {key}: {value:.6f}")