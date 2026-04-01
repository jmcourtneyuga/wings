"""Natural gradient descent with diagonal Quantum Fisher Information Matrix."""

import numpy as np
from numpy.typing import NDArray

__all__ = ["compute_qfim_diagonal", "compute_natural_gradient"]


def compute_qfim_diagonal(
    optimizer: "GaussianOptimizer",
    params: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Compute the diagonal of the Quantum Fisher Information Matrix.

    Uses the parameter-shift rule to estimate each diagonal element:
        F_ii = 4 * (Re<dpsi|dpsi> - |<psi_0|dpsi>|^2)

    where dpsi = (psi_plus - psi_minus) / 2 with shift = pi/2.

    Args:
        optimizer: GaussianOptimizer instance (provides get_statevector).
        params: Current variational parameters of shape (n_params,).

    Returns:
        Diagonal of the QFIM, shape (n_params,). Clamped to >= 0.
    """
    n_params = len(params)
    shift = np.pi / 2

    # Reference state
    psi_0 = optimizer.get_statevector(params)

    diag = np.zeros(n_params)

    for i in range(n_params):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[i] += shift
        params_minus[i] -= shift

        psi_plus = optimizer.get_statevector(params_plus)
        psi_minus = optimizer.get_statevector(params_minus)

        # Finite-difference approximation to the state derivative
        dpsi = (psi_plus - psi_minus) / 2.0

        # F_ii = 4 * (Re<dpsi|dpsi> - |<psi_0|dpsi>|^2)
        inner_dpsi = np.real(np.vdot(dpsi, dpsi))
        overlap = np.vdot(psi_0, dpsi)
        diag[i] = 4.0 * (inner_dpsi - np.abs(overlap) ** 2)

    # Clamp numerical noise to non-negative
    return np.maximum(diag, 0.0)


def compute_natural_gradient(
    optimizer: "GaussianOptimizer",
    params: NDArray[np.float64],
    regularization: float = 0.001,
) -> NDArray[np.float64]:
    """
    Compute the natural gradient using diagonal QFIM with Tikhonov regularization.

    natural_grad_i = euclidean_grad_i / (F_ii + regularization)

    The regularization prevents division by zero when QFIM diagonal entries
    are small or vanish (e.g., in barren plateau regions).

    Args:
        optimizer: GaussianOptimizer instance.
        params: Current variational parameters of shape (n_params,).
        regularization: Tikhonov regularization constant (default 0.001).

    Returns:
        Natural gradient array of shape (n_params,).
    """
    euclidean_grad = optimizer.compute_gradient(params)
    qfim_diag = compute_qfim_diagonal(optimizer, params)
    natural_grad = euclidean_grad / (qfim_diag + regularization)
    return natural_grad
