"""Warm-start transfer learning for variational quantum circuits.

Enables transferring optimized parameters from a smaller qubit system
to initialize a larger one, preserving learned structure while adding
small random values for new qubit positions.
"""

import numpy as np
from numpy.typing import NDArray

__all__ = ["transfer_params"]


def transfer_params(
    source_params: NDArray[np.floating],
    n_source: int,
    n_target: int,
    init_scale: float = 0.01,
) -> NDArray[np.floating]:
    """Transfer optimized parameters from a smaller to a larger circuit.

    Assumes DefaultAnsatz structure where depth = n_qubits and
    n_params = n_qubits * depth = n_qubits^2.

    For each shared layer, existing parameter values are preserved and
    new qubit positions are initialized with small random values.
    New layers (beyond the source depth) are fully random.

    Parameters
    ----------
    source_params : ndarray
        Optimized parameters from the source (smaller) circuit.
    n_source : int
        Number of qubits in the source circuit.
    n_target : int
        Number of qubits in the target (larger) circuit.
    init_scale : float, optional
        Scale for random initialization of new parameters (default 0.01).

    Returns
    -------
    ndarray
        Parameter array of size n_target^2 for the target circuit.

    Raises
    ------
    ValueError
        If n_source > n_target.
    """
    if n_source > n_target:
        raise ValueError(
            f"Source qubit count ({n_source}) cannot exceed "
            f"target qubit count ({n_target})."
        )

    if n_source == n_target:
        return source_params.copy()

    source_depth = n_source
    target_depth = n_target
    n_new_qubits = n_target - n_source

    target_params = np.zeros(n_target * target_depth)

    # Shared layers: copy existing params + pad new qubit positions
    shared_layers = min(source_depth, target_depth)
    for layer in range(shared_layers):
        src_offset = layer * n_source
        tgt_offset = layer * n_target

        # Copy existing source params for this layer
        target_params[tgt_offset : tgt_offset + n_source] = (
            source_params[src_offset : src_offset + n_source]
        )
        # Small random init for new qubit positions
        target_params[tgt_offset + n_source : tgt_offset + n_target] = (
            np.random.randn(n_new_qubits) * init_scale
        )

    # New layers beyond source depth: all small random
    for layer in range(shared_layers, target_depth):
        tgt_offset = layer * n_target
        target_params[tgt_offset : tgt_offset + n_target] = (
            np.random.randn(n_target) * init_scale
        )

    return target_params
