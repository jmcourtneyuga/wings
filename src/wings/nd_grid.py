"""
Multi-dimensional grid support for WINGS.

Extends WINGS to support 2D and 3D spatial wavefunctions
by partitioning qubits across dimensions.
"""

import numpy as np
from typing import Optional

__all__ = ["NDGrid", "gaussian_nd"]


class NDGrid:
    """
    N-dimensional position grid for quantum state preparation.

    Qubits are partitioned across dimensions. For a 2D grid:
    n_qubits_per_dim=[4, 4] gives a 16x16 grid on 8 total qubits.

    Args:
        n_qubits_per_dim: List of qubit counts per dimension
        box_sizes: List of box half-widths per dimension
        centers: List of grid centers per dimension (default: all zeros)
    """

    def __init__(
        self,
        n_qubits_per_dim: list[int],
        box_sizes: list[float],
        centers: Optional[list[float]] = None,
    ) -> None:
        assert len(n_qubits_per_dim) == len(box_sizes)
        self._n_qubits_per_dim = list(n_qubits_per_dim)
        self._box_sizes = list(box_sizes)
        self._centers = centers or [0.0] * len(n_qubits_per_dim)

    @property
    def n_dimensions(self) -> int:
        return len(self._n_qubits_per_dim)

    @property
    def total_qubits(self) -> int:
        return sum(self._n_qubits_per_dim)

    @property
    def total_states(self) -> int:
        return 2 ** self.total_qubits

    @property
    def states_per_dim(self) -> list[int]:
        return [2 ** n for n in self._n_qubits_per_dim]

    def positions(self) -> list[np.ndarray]:
        """Return list of 1D position arrays, one per dimension."""
        return [
            np.linspace(-L + c, L + c, 2**n)
            for n, L, c in zip(self._n_qubits_per_dim, self._box_sizes, self._centers)
        ]

    def meshgrid(self) -> list[np.ndarray]:
        """Return meshgrid arrays for all dimensions."""
        pos = self.positions()
        return np.meshgrid(*pos, indexing='ij')


def gaussian_nd(
    grid: NDGrid,
    sigmas: list[float],
    centers: Optional[list[float]] = None,
    momenta: Optional[list[float]] = None,
) -> np.ndarray:
    """
    N-dimensional Gaussian wavefunction on the given grid.

    psi(x1, ..., xd) = prod_i exp(-(x_i - c_i)^2 / (2*sigma_i^2)) * exp(i*k_i*x_i)

    Returns a normalized, flattened statevector of length grid.total_states.
    """
    if centers is None:
        centers = [0.0] * grid.n_dimensions
    if momenta is None:
        momenta = [0.0] * grid.n_dimensions

    pos = grid.positions()

    # Build separable Gaussian as outer product
    components = []
    for i in range(grid.n_dimensions):
        x = pos[i]
        psi_1d = np.exp(-(x - centers[i])**2 / (2 * sigmas[i]**2))
        if momenta[i] != 0:
            psi_1d = psi_1d * np.exp(1j * momenta[i] * x)
        components.append(psi_1d.astype(np.complex128))

    # Tensor product
    if len(components) == 1:
        psi = components[0]
    else:
        psi = components[0]
        for c in components[1:]:
            psi = np.outer(psi, c).reshape(-1)

    # Normalize
    psi = psi / np.linalg.norm(psi)

    return psi
