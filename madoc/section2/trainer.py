"""
Training utilities for PDE solving with KAN and traditional neural networks.
"""

import torch
import torch.nn as nn
from tqdm import tqdm
from kan import LBFGS
import metrics


class PDETrainer:
    """
    Trainer for PDE solving with physics-informed loss.
    """

    def __init__(self, model, device='cpu'):
        """
        Initialize trainer.

        Args:
            model: Neural network model
            device: Device to train on
        """
        self.model = model
        self.device = device

    def train_supervised(self, dataset, epochs=100, lr=1e-3, optimizer_type='adam',
                         metrics_tracker=None, update_grid_every=None, x_interior=None):
        """
        Train model using supervised learning (fitting to solution values).

        Args:
            dataset: Dataset dict with train_input, train_label, test_input, test_label
            epochs: Number of epochs
            lr: Learning rate
            optimizer_type: 'adam', 'lbfgs', or 'sgd'
            metrics_tracker: Optional MetricsTracker instance
            update_grid_every: For KAN models, update grid every N epochs
            x_interior: Interior points for grid updates (KAN only)

        Returns:
            Dictionary of training history
        """
        x_train = dataset['train_input']
        y_train = dataset['train_label']

        criterion = nn.MSELoss()

        if optimizer_type == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        elif optimizer_type == 'lbfgs':
            optimizer = LBFGS(self.model.parameters(), lr=lr, history_size=10,
                             line_search_fn="strong_wolfe", tolerance_grad=1e-32,
                             tolerance_change=1e-32, tolerance_ys=1e-32)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")

        history = {
            'train_loss': [],
            'test_loss': []
        }

        pbar = tqdm(range(epochs), desc='Training', ncols=100)

        for epoch in pbar:
            # Update grid for KAN models
            if update_grid_every is not None and epoch % update_grid_every == 0 and epoch < epochs // 2:
                if hasattr(self.model, 'update_grid_from_samples') and x_interior is not None:
                    self.model.update_grid_from_samples(x_interior)

            if optimizer_type == 'lbfgs':
                def closure():
                    optimizer.zero_grad()
                    y_pred = self.model(x_train)
                    loss = criterion(y_pred, y_train)
                    loss.backward()
                    return loss

                optimizer.step(closure)

                # Evaluate
                with torch.no_grad():
                    y_pred = self.model(x_train)
                    train_loss = criterion(y_pred, y_train).item()
                    y_test_pred = self.model(dataset['test_input'])
                    test_loss = criterion(y_test_pred, dataset['test_label']).item()
            else:
                optimizer.zero_grad()
                y_pred = self.model(x_train)
                loss = criterion(y_pred, y_train)
                loss.backward()
                optimizer.step()

                train_loss = loss.item()
                with torch.no_grad():
                    y_test_pred = self.model(dataset['test_input'])
                    test_loss = criterion(y_test_pred, dataset['test_label']).item()

            history['train_loss'].append(train_loss)
            history['test_loss'].append(test_loss)

            # Log metrics if tracker provided
            if metrics_tracker is not None and epoch % 10 == 0:
                metrics_dict = metrics_tracker.log_metrics()
                pbar.set_description(f"Train: {train_loss:.2e} | Test: {test_loss:.2e} | "
                                   f"H1: {metrics_dict.get('h1_norm', 0):.2e}")
            else:
                pbar.set_description(f"Train: {train_loss:.2e} | Test: {test_loss:.2e}")

        return history

    def train_pde_residual(self, x_interior, x_boundary, solution_func, source_func,
                          epochs=100, alpha=0.01, lr=1, metrics_tracker=None,
                          update_grid_every=5):
        """
        Train using PDE residual loss (physics-informed neural networks approach).

        Args:
            x_interior: Interior points (batch, n_dims)
            x_boundary: Boundary points (batch, n_dims)
            solution_func: Ground truth solution function (for boundary conditions)
            source_func: Source function for PDE
            epochs: Number of epochs
            alpha: Weight for PDE residual loss vs boundary loss
            lr: Learning rate
            metrics_tracker: Optional MetricsTracker
            update_grid_every: Update grid every N epochs (for KAN)

        Returns:
            Dictionary of training history
        """
        optimizer = LBFGS(self.model.parameters(), lr=lr, history_size=10,
                         line_search_fn="strong_wolfe", tolerance_grad=1e-32,
                         tolerance_change=1e-32, tolerance_ys=1e-32)

        history = {
            'pde_loss': [],
            'bc_loss': [],
            'total_loss': []
        }

        pbar = tqdm(range(epochs), desc='PDE Training', ncols=100)

        for epoch in pbar:
            # Update grid for KAN
            if update_grid_every is not None and epoch % update_grid_every == 0 and epoch < epochs // 2:
                if hasattr(self.model, 'update_grid_from_samples'):
                    self.model.update_grid_from_samples(x_interior)

            def closure():
                optimizer.zero_grad()

                # Interior loss (PDE residual)
                lap = metrics.compute_laplacian(self.model, x_interior)
                source = source_func(x_interior)
                pde_loss = torch.mean((lap - source) ** 2)

                # Boundary loss
                bc_true = solution_func(x_boundary)
                bc_pred = self.model(x_boundary)
                bc_loss = torch.mean((bc_pred - bc_true) ** 2)

                # Total loss
                loss = alpha * pde_loss + bc_loss
                loss.backward()

                # Store for logging
                closure.pde_loss = pde_loss
                closure.bc_loss = bc_loss

                return loss

            optimizer.step(closure)

            pde_loss = closure.pde_loss.item()
            bc_loss = closure.bc_loss.item()
            total_loss = alpha * pde_loss + bc_loss

            history['pde_loss'].append(pde_loss)
            history['bc_loss'].append(bc_loss)
            history['total_loss'].append(total_loss)

            # Log metrics
            if metrics_tracker is not None and epoch % 10 == 0:
                metrics_dict = metrics_tracker.log_metrics()
                pbar.set_description(f"PDE: {pde_loss:.2e} | BC: {bc_loss:.2e} | "
                                   f"MSE: {metrics_dict.get('mse_error', 0):.2e}")
            else:
                pbar.set_description(f"PDE: {pde_loss:.2e} | BC: {bc_loss:.2e}")

        return history


class KANProgressiveTrainer:
    """
    Trainer for KAN with progressive grid refinement.
    """

    def __init__(self, model, grids, device='cpu'):
        """
        Initialize progressive trainer.

        Args:
            model: Initial KAN model
            grids: List of grid sizes for progressive refinement
            device: Device
        """
        self.model = model
        self.grids = grids
        self.device = device

    def train_progressive(self, dataset, x_interior, solution_func, source_func,
                         steps_per_grid=50, alpha=0.01, metrics_tracker=None):
        """
        Train with progressive grid refinement.

        Args:
            dataset: Dataset for evaluation
            x_interior: Interior points
            solution_func: Solution function
            source_func: Source function
            steps_per_grid: Training steps per grid size
            alpha: Weight for PDE loss
            metrics_tracker: Optional MetricsTracker

        Returns:
            Dictionary of training history
        """
        history = {
            'pde_loss': [],
            'bc_loss': [],
            'total_loss': [],
            'grid_changes': []
        }

        x_boundary = metrics.create_dense_test_set([-1, 1], 51, device=self.device)
        # Reshape boundary for 2D
        if x_interior.shape[1] == 2:
            x_boundary = self._create_2d_boundary(51)

        for grid_idx, grid in enumerate(self.grids):
            print(f"\n=== Grid {grid} ({grid_idx + 1}/{len(self.grids)}) ===")

            if grid_idx > 0:
                # Refine grid
                self.model.save_act = True
                self.model.get_act(x_interior)
                self.model = self.model.refine(grid)

            # Speed up model
            if hasattr(self.model, 'speed'):
                self.model = self.model.speed()

            # Train on this grid
            trainer = PDETrainer(self.model, self.device)
            grid_history = trainer.train_pde_residual(
                x_interior, x_boundary, solution_func, source_func,
                epochs=steps_per_grid, alpha=alpha, lr=1,
                metrics_tracker=metrics_tracker,
                update_grid_every=5
            )

            # Append history
            history['pde_loss'].extend(grid_history['pde_loss'])
            history['bc_loss'].extend(grid_history['bc_loss'])
            history['total_loss'].extend(grid_history['total_loss'])
            history['grid_changes'].append(len(history['total_loss']))

        return history

    def _create_2d_boundary(self, n_points):
        """Helper to create 2D boundary points."""
        x_mesh = torch.linspace(-1, 1, steps=n_points)
        y_mesh = torch.linspace(-1, 1, steps=n_points)
        X, Y = torch.meshgrid(x_mesh, y_mesh, indexing="ij")

        helper = lambda X, Y: torch.stack([X.reshape(-1, ), Y.reshape(-1, )]).permute(1, 0)

        xb1 = helper(X[0], Y[0])
        xb2 = helper(X[-1], Y[0])
        xb3 = helper(X[:, 0], Y[:, 0])
        xb4 = helper(X[:, 0], Y[:, -1])

        x_b = torch.cat([xb1, xb2, xb3, xb4], dim=0)
        return x_b.to(self.device)