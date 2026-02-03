"""Adam optimizer implementations."""

import numpy as np

from .types import FloatArray, ParameterArray

__all__ = [
    "AdamOptimizer",
    "AdamWithRestarts",
]


class AdamOptimizer:
    """
    Adam optimizer for variational quantum circuits.

    Adam is effective for:
    - Noisy/stochastic gradients
    - Escaping shallow local minima via momentum
    - Adaptive learning rates per parameter

    For VQCs, we combine Adam with exact parameter-shift gradients.
    """

    def __init__(
        self,
        n_params: int,
        learning_rate: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        amsgrad: bool = False,
    ) -> None:
        self.n_params = n_params
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.amsgrad = amsgrad

        # Initialize moment estimates
        self.m = np.zeros(n_params)  # First moment (momentum)
        self.v = np.zeros(n_params)  # Second moment (RMSprop)
        self.v_hat_max = np.zeros(n_params)  # For AMSGrad
        self.t = 0  # Timestep

    def step(self, params: ParameterArray, gradient: FloatArray) -> ParameterArray:
        """
        Perform one Adam update step.

        Args:
            params: Current parameters
            gradient: Gradient of loss w.r.t. parameters

        Returns:
            Updated parameters
        """
        self.t += 1

        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient

        # Update biased second moment estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient**2)

        # Compute bias-corrected estimates
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)

        if self.amsgrad:
            # AMSGrad: use maximum of past v_hat values
            self.v_hat_max = np.maximum(self.v_hat_max, v_hat)
            v_hat = self.v_hat_max

        # Compute update
        update = self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return params - update

    def reset(self) -> None:
        """Reset optimizer state (for restarts)"""
        self.m = np.zeros(self.n_params)
        self.v = np.zeros(self.n_params)
        self.v_hat_max = np.zeros(self.n_params)
        self.t = 0


class AdamWithRestarts:
    """
    Adam optimizer with warm restarts (cosine annealing).

    Learning rate follows cosine schedule and resets periodically,
    which helps escape local minima.
    """

    def __init__(
        self,
        n_params: int,
        lr_max: float = 0.05,
        lr_min: float = 0.001,
        restart_period: int = 200,  # Steps between restarts
        restart_mult: float = 1.5,  # Multiply period after each restart
    ):
        self.adam = AdamOptimizer(n_params, learning_rate=lr_max)
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.restart_period = restart_period
        self.restart_mult = restart_mult
        self.current_period = restart_period
        self.steps_since_restart = 0

    def get_lr(self) -> float:
        """Cosine annealing learning rate"""
        progress = self.steps_since_restart / self.current_period
        return self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + np.cos(np.pi * progress))

    def step(self, params: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """Perform Adam step with cosine LR schedule"""
        self.steps_since_restart += 1

        # Update learning rate
        self.adam.lr = self.get_lr()

        # Check for restart
        if self.steps_since_restart >= self.current_period:
            self.steps_since_restart = 0
            self.current_period = int(self.current_period * self.restart_mult)
            # Partial momentum reset (keep some history)
            self.adam.m *= 0.5
            self.adam.v *= 0.8

        return self.adam.step(params, gradient)
