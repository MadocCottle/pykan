"""Concise KAN ensemble training"""
import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from kan import *


class KANEnsemble:
    """Train N KAN experts with different seeds"""

    def __init__(self, n_experts, width, grid, depth=3, device='cpu'):
        self.n_experts = n_experts
        self.width = width
        self.grid = grid
        self.depth = depth
        self.device = torch.device(device)
        self.experts = []

    def train(self, dataset, epochs=100, lr=0.001, verbose=True):
        """Train all experts independently

        Returns:
            Dict with 'losses', 'models', 'predictions'
        """
        losses = []
        X_test = dataset['test_input'].to(self.device)
        y_test = dataset['test_label'].to(self.device)

        for i in range(self.n_experts):
            if verbose:
                print(f"\nExpert {i+1}/{self.n_experts} (seed={i})")

            # Create model with seed
            torch.manual_seed(i)
            np.random.seed(i)

            width_list = [dataset['train_input'].shape[1]] + [self.width] * (self.depth - 1) + [dataset['train_label'].shape[1]]
            model = KAN(width=width_list, grid=self.grid, k=3, device=self.device)

            # Train
            model.fit(dataset, opt='LBFGS', steps=epochs, lr=lr, lamb=0.01, lamb_entropy=10.0)

            # Test
            with torch.no_grad():
                pred = model(X_test)
                loss = torch.nn.functional.mse_loss(pred, y_test).item()

            losses.append(loss)
            self.experts.append(model)

            if verbose:
                print(f"  Test MSE: {loss:.6f}")

        return {
            'losses': losses,
            'models': self.experts,
            'mean_loss': np.mean(losses),
            'std_loss': np.std(losses)
        }

    def predict(self, X, uncertainty=False):
        """Ensemble prediction

        Args:
            X: Input tensor
            uncertainty: Return epistemic uncertainty

        Returns:
            (mean_pred, std_pred) if uncertainty else mean_pred
        """
        X = X.to(self.device)
        preds = []

        with torch.no_grad():
            for model in self.experts:
                preds.append(model(X))

        preds = torch.stack(preds)
        mean = preds.mean(dim=0)

        if uncertainty:
            std = preds.std(dim=0)
            return mean, std
        return mean
