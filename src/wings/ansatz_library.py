"""
Ansatz library for WINGS.

Provides reusable ansatz circuits with various entanglement patterns
for variational quantum state preparation.
"""

from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

if TYPE_CHECKING:
    from .evaluators.custatevec import CuStateVecSimulator

__all__ = ["EfficientSU2Ansatz", "generate_entanglement_map"]


def generate_entanglement_map(n_qubits: int, pattern: str) -> list[tuple[int, int]]:
    """
    Generate an entanglement map (list of qubit pairs) for a given pattern.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    pattern : str
        Entanglement pattern. One of: "linear", "circular", "reverse_linear",
        "parity", "log_distance", "full".

    Returns
    -------
    List[Tuple[int, int]]
        List of (control, target) qubit pairs.

    Raises
    ------
    ValueError
        If the pattern is not recognized.
    """
    if pattern == "linear":
        return [(i, i + 1) for i in range(n_qubits - 1)]

    elif pattern == "circular":
        pairs = [(i, i + 1) for i in range(n_qubits - 1)]
        pairs.append((n_qubits - 1, 0))
        return pairs

    elif pattern == "reverse_linear":
        return [(i + 1, i) for i in range(n_qubits - 2, -1, -1)]

    elif pattern == "parity":
        pairs = []
        # Even-odd pairs: (0,1), (2,3), (4,5), ...
        for i in range(0, n_qubits - 1, 2):
            pairs.append((i, i + 1))
        # Odd-even pairs: (1,2), (3,4), (5,6), ...
        for i in range(1, n_qubits - 1, 2):
            pairs.append((i, i + 1))
        return pairs

    elif pattern == "log_distance":
        pairs = []
        distance = 1
        while distance < n_qubits:
            for i in range(n_qubits - distance):
                pairs.append((i, i + distance))
            distance *= 2
        return pairs

    elif pattern == "full":
        return [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]

    else:
        raise ValueError(
            f"Unknown entanglement pattern: {pattern!r}. "
            f"Supported patterns: linear, circular, reverse_linear, "
            f"parity, log_distance, full."
        )


class EfficientSU2Ansatz:
    """
    Efficient SU(2) ansatz with RY-RZ rotation layers and configurable entanglement.

    Each layer consists of:
    1. RY rotation on every qubit
    2. RZ rotation on every qubit
    3. CNOT entanglement (skipped on the last layer)

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    layers : int, optional
        Number of layers (default: 4).
    entanglement : str, optional
        Entanglement pattern (default: "linear").
        See :func:`generate_entanglement_map` for supported patterns.

    Examples
    --------
    >>> ansatz = EfficientSU2Ansatz(n_qubits=4, layers=3)
    >>> params = np.random.randn(ansatz.n_params)
    >>> circuit = ansatz(params, n_qubits=4)
    """

    def __init__(
        self,
        n_qubits: int,
        layers: int = 4,
        entanglement: str = "linear",
    ) -> None:
        self._n_qubits = n_qubits
        self._layers = layers
        self._entanglement = entanglement
        self._n_params = 2 * n_qubits * layers  # RY + RZ per qubit per layer
        self._entanglement_map = generate_entanglement_map(n_qubits, entanglement)

    @property
    def n_params(self) -> int:
        """Number of variational parameters."""
        return self._n_params

    @property
    def depth(self) -> int:
        """Number of layers."""
        return self._layers

    @property
    def n_qubits(self) -> int:
        """Number of qubits."""
        return self._n_qubits

    @property
    def entanglement(self) -> str:
        """Entanglement pattern name."""
        return self._entanglement

    def __call__(
        self,
        params: Union[NDArray[np.float64], ParameterVector],
        n_qubits: Optional[int] = None,
        **_kwargs: Any,
    ) -> QuantumCircuit:
        """
        Build the parameterized quantum circuit.

        Parameters
        ----------
        params : array-like or ParameterVector
            Parameter values or symbolic parameters.
        n_qubits : int, optional
            Number of qubits (defaults to instance n_qubits).
        **_kwargs
            Additional keyword arguments (ignored).

        Returns
        -------
        QuantumCircuit
            The constructed circuit.
        """
        n = n_qubits if n_qubits is not None else self._n_qubits
        qc = QuantumCircuit(n)

        param_idx = 0
        for layer in range(self._layers):
            # RY on all qubits
            for qubit in range(n):
                qc.ry(params[param_idx], qubit)
                param_idx += 1

            # RZ on all qubits
            for qubit in range(n):
                qc.rz(params[param_idx], qubit)
                param_idx += 1

            # CNOT entanglement (skip on last layer)
            if layer < self._layers - 1:
                for ctrl, tgt in self._entanglement_map:
                    qc.cx(ctrl, tgt)

        return qc

    def apply_custatevec(
        self, simulator: "CuStateVecSimulator", params: NDArray[np.float64]
    ) -> None:
        """
        Native cuStateVec implementation for maximum performance.

        Parameters
        ----------
        simulator : CuStateVecSimulator
            The cuStateVec simulator instance.
        params : NDArray[np.float64]
            Parameter values.
        """
        n = self._n_qubits
        param_idx = 0

        simulator.reset_state()

        for layer in range(self._layers):
            # RY on all qubits
            for qubit in range(n):
                simulator.apply_ry(float(params[param_idx]), qubit)
                param_idx += 1

            # RZ on all qubits
            for qubit in range(n):
                if hasattr(simulator, "apply_rz"):
                    simulator.apply_rz(float(params[param_idx]), qubit)
                param_idx += 1

            # CNOT entanglement (skip on last layer)
            if layer < self._layers - 1:
                for ctrl, tgt in self._entanglement_map:
                    simulator.apply_cnot(ctrl, tgt)
