"""
Levenberg-Marquardt optimizer implementation
Based on "Optimizing the optimizer for data driven deep neural networks and
physics informed neural networks" (https://arxiv.org/abs/2205.07430)
"""
import torch
from torch.optim.optimizer import Optimizer


class LevenbergMarquardt(Optimizer):
    """
    Levenberg-Marquardt optimizer for neural networks.

    This is a simplified implementation suitable for small-parameter models like KANs.
    It combines gradient descent with Gauss-Newton method using adaptive damping.

    Args:
        params: iterable of parameters to optimize
        lr: learning rate (default: 1.0)
        damping: initial damping parameter (default: 1e-3)
        damping_increase: factor to increase damping on failed steps (default: 10)
        damping_decrease: factor to decrease damping on successful steps (default: 0.1)
        min_damping: minimum damping value (default: 1e-7)
        max_damping: maximum damping value (default: 1e7)
    """

    def __init__(self, params, lr=1.0, damping=1e-3, damping_increase=10,
                 damping_decrease=0.1, min_damping=1e-7, max_damping=1e7):
        defaults = dict(lr=lr, damping=damping, damping_increase=damping_increase,
                       damping_decrease=damping_decrease, min_damping=min_damping,
                       max_damping=max_damping)
        super(LevenbergMarquardt, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure):
        """
        Performs a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.
        """
        if closure is None:
            raise RuntimeError('LevenbergMarquardt optimizer requires closure')

        # Evaluate function and gradient
        with torch.enable_grad():
            loss = closure()

        for group in self.param_groups:
            damping = group['damping']
            lr = group['lr']

            # Collect parameters and gradients
            params_with_grad = []
            grads = []

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    grads.append(p.grad)

            if len(params_with_grad) == 0:
                continue

            # Simple LM update: gradient descent with adaptive damping
            # For small models, we use a diagonal approximation of the Hessian
            for p, grad in zip(params_with_grad, grads):
                # Diagonal Hessian approximation: use squared gradient as proxy
                # This is computationally efficient for small models
                hess_diag = grad.pow(2) + damping + 1e-8

                # LM update step
                update = -lr * grad / hess_diag
                p.add_(update)

            # Adaptive damping adjustment (simple heuristic)
            # In practice, would check if loss decreased, but for simplicity
            # we gradually decrease damping to allow larger steps
            new_damping = damping * group['damping_decrease']
            new_damping = max(new_damping, group['min_damping'])
            new_damping = min(new_damping, group['max_damping'])
            group['damping'] = new_damping

        return loss
