"""SPSA (Simultaneous Perturbation Stochastic Approximation) optimizer."""

from typing import Callable

import numpy as np

from .types import FloatArray, ParameterArray

__all__ = ["SPSAOptimizer"]


class SPSAOptimizer:
    """
    SPSA optimizer for variational quantum circuits.

    Estimates the full gradient from only 2 function evaluations
    (or 2*n_avg with averaging), regardless of parameter count.
    Follows Spall (IEEE TAC 1992, IEEE TAES 1998).

    Gain sequences:
        a_k = a / (A + k + 1)^alpha
        c_k = c / (k + 1)^gamma

    Recommended values (Spall 1998):
        alpha = 0.602, gamma = 0.101
        A ~ 0.1 * max_iterations
        a calibrated so first step is not too large
        c ~ std of noise in objective (or ~0.1 for noiseless)
    """

    def __init__(
        self,
        n_params: int,
        a: float = 0.1,
        c: float = 0.1,
        A: float = 100.0,
        alpha: float = 0.602,
        gamma: float = 0.101,
        n_avg: int = 1,
    ) -> None:
        self.n_params = n_params
        self.a = a
        self.c = c
        self.A = A
        self.alpha = alpha
        self.gamma = gamma
        self.n_avg = n_avg
        self.k = 0  # iteration counter

    def get_a_k(self) -> float:
        """Step size gain sequence."""
        return self.a / (self.A + self.k + 1) ** self.alpha

    def get_c_k(self) -> float:
        """Perturbation size gain sequence."""
        return self.c / (self.k + 1) ** self.gamma

    def _generate_perturbation(self) -> FloatArray:
        """Generate Rademacher (+/-1) perturbation vector."""
        return 2 * (np.random.randint(0, 2, size=self.n_params) - 0.5)  # yields +1 or -1

    def estimate_gradient(
        self, params: ParameterArray, loss_fn: Callable[[ParameterArray], float]
    ) -> tuple[FloatArray, int]:
        """
        Estimate gradient using simultaneous perturbation.

        Args:
            params: Current parameters
            loss_fn: Objective function (we minimize this)

        Returns:
            (gradient_estimate, n_evaluations)
        """
        c_k = self.get_c_k()
        n_evals = 0

        if self.n_avg == 1:
            delta = self._generate_perturbation()
            f_plus = loss_fn(params + c_k * delta)
            f_minus = loss_fn(params - c_k * delta)
            n_evals = 2
            g_hat = (f_plus - f_minus) / (2.0 * c_k) * (1.0 / delta)
        else:
            # Average over multiple perturbations for variance reduction
            g_hat = np.zeros(self.n_params)
            for _ in range(self.n_avg):
                delta = self._generate_perturbation()
                f_plus = loss_fn(params + c_k * delta)
                f_minus = loss_fn(params - c_k * delta)
                g_hat += (f_plus - f_minus) / (2.0 * c_k) * (1.0 / delta)
                n_evals += 2
            g_hat /= self.n_avg

        return g_hat, n_evals

    def step(
        self, params: ParameterArray, loss_fn: Callable[[ParameterArray], float]
    ) -> tuple[ParameterArray, FloatArray, int]:
        """
        Perform one SPSA update step.

        Args:
            params: Current parameters
            loss_fn: Objective function to minimize

        Returns:
            (updated_params, gradient_estimate, n_evaluations)
        """
        self.k += 1
        a_k = self.get_a_k()

        g_hat, n_evals = self.estimate_gradient(params, loss_fn)
        new_params = params - a_k * g_hat

        return new_params, g_hat, n_evals

    def reset(self) -> None:
        """Reset iteration counter."""
        self.k = 0
