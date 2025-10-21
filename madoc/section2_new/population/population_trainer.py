"""Population-based training for KAN networks.

This module implements Extension 4: Population-Based Training where multiple
KAN models train in parallel with periodic synchronization to share knowledge
and maintain diversity.

Key Features:
- Parallel training of population
- Periodic parameter averaging/sharing
- Gradient aggregation across population
- Diversity maintenance metrics
- Cross-pollination of successful variants

Reference:
- Plan Section: Extension 4 - Population-Based Training (Multi-Seed Coordination)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Callable
import copy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # Add madoc/ to path
sys.path.insert(0, str(Path(__file__).parent.parent))  # Add section2_new/ to path

# Import KAN variants (use PyKAN if custom variants not available)
try:
    from section1.models.kan_variants import RBF_KAN
except ImportError:
    from models.pykan_wrapper import PyKANCompatible as RBF_KAN


class PopulationBasedKANTrainer:
    """Population-based trainer for KAN networks.

    Trains multiple KAN instances in parallel with periodic synchronization
    to share knowledge while maintaining diversity.

    Args:
        input_dim: Input dimension
        hidden_dim: Hidden layer width
        output_dim: Output dimension
        depth: Number of layers
        population_size: Number of models in population
        sync_method: Synchronization method ('average', 'best', 'tournament')
        sync_frequency: Synchronize every N epochs
        diversity_weight: Weight for diversity maintenance (0-1)
        device: Torch device

    Example:
        >>> trainer = PopulationBasedKANTrainer(
        ...     input_dim=3, hidden_dim=10, output_dim=1,
        ...     population_size=10, sync_frequency=50
        ... )
        >>> trainer.train(X_train, y_train, epochs=500)
        >>> best_model = trainer.get_best_model()
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 1,
        depth: int = 3,
        population_size: int = 10,
        sync_method: str = 'average',
        sync_frequency: int = 50,
        diversity_weight: float = 0.1,
        device: str = 'cpu'
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.depth = depth
        self.population_size = population_size
        self.sync_method = sync_method
        self.sync_frequency = sync_frequency
        self.diversity_weight = diversity_weight
        self.device = torch.device(device)

        # Initialize population
        self.population: List[nn.Module] = []
        self.optimizers: List[torch.optim.Optimizer] = []
        self.performance_history: List[List[float]] = []

        for i in range(population_size):
            # Create model with different seed
            torch.manual_seed(i)
            model = RBF_KAN(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                depth=depth
            ).to(self.device)

            self.population.append(model)
            self.performance_history.append([])

    def train(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        epochs: int = 500,
        lr: float = 0.001,
        validation_data: Optional[tuple] = None,
        verbose: bool = True
    ) -> Dict:
        """Train population with periodic synchronization.

        Args:
            X_train: Training inputs
            y_train: Training targets
            epochs: Number of epochs
            lr: Learning rate
            validation_data: Optional (X_val, y_val) tuple
            verbose: Print progress

        Returns:
            Training history
        """
        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)

        # Create optimizers
        self.optimizers = [
            torch.optim.Adam(model.parameters(), lr=lr)
            for model in self.population
        ]

        loss_fn = nn.MSELoss()

        history = {
            'population_losses': [[] for _ in range(self.population_size)],
            'mean_loss': [],
            'best_loss': [],
            'diversity': [],
            'sync_events': []
        }

        if verbose:
            print("="*70)
            print(f"Population-Based Training ({self.population_size} models)")
            print("="*70)
            print(f"Sync method: {self.sync_method}, Frequency: {self.sync_frequency}")

        for epoch in range(epochs):
            epoch_losses = []

            # Train each model in population
            for model_idx, (model, optimizer) in enumerate(zip(self.population, self.optimizers)):
                model.train()
                optimizer.zero_grad()

                # Forward
                pred = model(X_train)
                loss = loss_fn(pred, y_train)

                # Backward
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())
                history['population_losses'][model_idx].append(loss.item())
                self.performance_history[model_idx].append(loss.item())

            # Record statistics
            mean_loss = np.mean(epoch_losses)
            best_loss = np.min(epoch_losses)
            history['mean_loss'].append(mean_loss)
            history['best_loss'].append(best_loss)

            # Compute diversity
            diversity = self._compute_diversity(X_train)
            history['diversity'].append(diversity)

            # Periodic synchronization
            if (epoch + 1) % self.sync_frequency == 0 and epoch > 0:
                self._synchronize()
                history['sync_events'].append(epoch + 1)

                if verbose:
                    print(f"\nEpoch {epoch+1}: Synchronized population")
                    print(f"  Mean loss: {mean_loss:.6f}")
                    print(f"  Best loss: {best_loss:.6f}")
                    print(f"  Diversity: {diversity:.6f}")

            # Print progress
            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch+1:4d}/{epochs}: Mean={mean_loss:.6f}, Best={best_loss:.6f}, Div={diversity:.6f}")

        if verbose:
            print("\n" + "="*70)
            print("Training complete!")
            print("="*70)

            # Final statistics
            final_losses = [self.performance_history[i][-1] for i in range(self.population_size)]
            print(f"\nFinal population statistics:")
            print(f"  Mean loss: {np.mean(final_losses):.6f}")
            print(f"  Best loss: {np.min(final_losses):.6f}")
            print(f"  Worst loss: {np.max(final_losses):.6f}")
            print(f"  Std loss: {np.std(final_losses):.6f}")

        return history

    def _synchronize(self):
        """Synchronize population models."""

        if self.sync_method == 'average':
            # Average parameters across population
            self._parameter_averaging()

        elif self.sync_method == 'best':
            # Share parameters from best model
            self._best_sharing()

        elif self.sync_method == 'tournament':
            # Tournament-based parameter sharing
            self._tournament_sharing()

        else:
            raise ValueError(f"Unknown sync method: {self.sync_method}")

    def _parameter_averaging(self):
        """Average parameters across all models in population."""

        # Get average parameters
        avg_params = {}
        for name, param in self.population[0].named_parameters():
            param_sum = torch.zeros_like(param.data)
            for model in self.population:
                param_sum += dict(model.named_parameters())[name].data
            avg_params[name] = param_sum / self.population_size

        # Update each model with weighted average
        for model in self.population:
            for name, param in model.named_parameters():
                # Blend: keep some individuality, adopt some average
                blend_weight = 1.0 - self.diversity_weight
                param.data = (blend_weight * avg_params[name] +
                             self.diversity_weight * param.data)

    def _best_sharing(self):
        """Share parameters from best-performing model."""

        # Find best model
        recent_losses = [hist[-10:] if len(hist) >= 10 else hist
                        for hist in self.performance_history]
        mean_recent = [np.mean(losses) for losses in recent_losses]
        best_idx = np.argmin(mean_recent)
        best_model = self.population[best_idx]

        # Share with other models
        for model_idx, model in enumerate(self.population):
            if model_idx == best_idx:
                continue

            for name, param in model.named_parameters():
                best_param = dict(best_model.named_parameters())[name]
                # Blend with best model
                blend_weight = 1.0 - self.diversity_weight
                param.data = (blend_weight * best_param.data +
                             self.diversity_weight * param.data)

    def _tournament_sharing(self):
        """Tournament-based parameter sharing."""

        # Rank models by recent performance
        recent_losses = [hist[-10:] if len(hist) >= 10 else hist
                        for hist in self.performance_history]
        mean_recent = [np.mean(losses) for losses in recent_losses]
        rankings = np.argsort(mean_recent)  # Best to worst

        # Top half shares with bottom half
        n_winners = self.population_size // 2

        for i in range(n_winners):
            winner_idx = rankings[i]
            loser_idx = rankings[-(i+1)]

            winner_model = self.population[winner_idx]
            loser_model = self.population[loser_idx]

            # Loser adopts parameters from winner
            for name, param in loser_model.named_parameters():
                winner_param = dict(winner_model.named_parameters())[name]
                blend_weight = 0.7  # Stronger influence from winner
                param.data = (blend_weight * winner_param.data +
                             (1 - blend_weight) * param.data)

    def _compute_diversity(self, X: torch.Tensor) -> float:
        """Compute diversity of predictions across population.

        Args:
            X: Input data

        Returns:
            Diversity score (higher = more diverse)
        """
        predictions = []
        with torch.no_grad():
            for model in self.population:
                model.eval()
                pred = model(X)
                predictions.append(pred)

        predictions = torch.stack(predictions)  # (pop_size, batch, output_dim)

        # Compute variance across population
        variance = torch.var(predictions, dim=0).mean().item()

        return variance

    def get_best_model(self) -> nn.Module:
        """Get the best-performing model from population.

        Returns:
            Best model
        """
        # Find best based on recent performance
        recent_losses = [hist[-20:] if len(hist) >= 20 else hist
                        for hist in self.performance_history]
        mean_recent = [np.mean(losses) for losses in recent_losses]
        best_idx = np.argmin(mean_recent)

        return self.population[best_idx]

    def get_ensemble_prediction(
        self,
        X: torch.Tensor,
        method: str = 'mean'
    ) -> torch.Tensor:
        """Get ensemble prediction from entire population.

        Args:
            X: Input tensor
            method: Aggregation method ('mean', 'median', 'best')

        Returns:
            Ensemble prediction
        """
        X = X.to(self.device)

        predictions = []
        with torch.no_grad():
            for model in self.population:
                model.eval()
                pred = model(X)
                predictions.append(pred)

        predictions = torch.stack(predictions)

        if method == 'mean':
            return predictions.mean(dim=0)
        elif method == 'median':
            return predictions.median(dim=0)[0]
        elif method == 'best':
            best_idx = np.argmin([np.mean(hist[-20:]) for hist in self.performance_history])
            return predictions[best_idx]
        else:
            raise ValueError(f"Unknown method: {method}")


# =============================================================================
# Example Usage and Testing
# =============================================================================

if __name__ == '__main__':
    print("="*70)
    print("Testing Population-Based Training")
    print("="*70)

    # Generate synthetic data
    torch.manual_seed(42)
    X_train = torch.randn(100, 3)
    y_train = (2.0 * X_train[:, 0] + torch.sin(X_train[:, 1]) + 0.5 * X_train[:, 2]).reshape(-1, 1)

    X_test = torch.randn(30, 3)
    y_test = (2.0 * X_test[:, 0] + torch.sin(X_test[:, 1]) + 0.5 * X_test[:, 2]).reshape(-1, 1)

    # Create population trainer
    trainer = PopulationBasedKANTrainer(
        input_dim=3,
        hidden_dim=10,
        output_dim=1,
        depth=2,
        population_size=5,
        sync_method='average',
        sync_frequency=30,
        diversity_weight=0.1,
        device='cpu'
    )

    # Train
    history = trainer.train(
        X_train, y_train,
        epochs=150,
        lr=0.01,
        verbose=True
    )

    # Test best model
    print("\nTesting best model:")
    best_model = trainer.get_best_model()
    best_model.eval()
    with torch.no_grad():
        y_pred = best_model(X_test)
        test_mse = nn.MSELoss()(y_pred, y_test).item()
    print(f"  Best model test MSE: {test_mse:.6f}")

    # Test ensemble
    print("\nTesting ensemble prediction:")
    y_pred_ensemble = trainer.get_ensemble_prediction(X_test, method='mean')
    test_mse_ensemble = nn.MSELoss()(y_pred_ensemble, y_test).item()
    print(f"  Ensemble test MSE: {test_mse_ensemble:.6f}")

    print("\n" + "="*70)
    print("Population-based training test complete!")
    print("="*70)
