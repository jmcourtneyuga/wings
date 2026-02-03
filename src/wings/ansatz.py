from typing import TYPE_CHECKING, Any, Optional, Protocol, Union, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import ParameterVector

if TYPE_CHECKING:
    from .evaluators.custatevec import CuStateVecSimulator

__all__ = [
    "AnsatzProtocol",
    "DefaultAnsatz",
    "CustomHardwareEfficientAnsatz",
    "get_full_variational_quantum_circuit",  # Deprecated, for backward compat
]


@runtime_checkable
class AnsatzProtocol(Protocol):
    """Protocol for custom ansatz implementations."""

    def __call__(
        self, params: Union[np.ndarray, ParameterVector], n_qubits: int, **_kwargs
    ) -> QuantumCircuit:
        """
        Build parameterized quantum circuit.

        Args:
            params: Parameter values or ParameterVector
            n_qubits: Number of qubits
            **_kwargs: Additional ansatz-specific arguments

        Returns:
            QuantumCircuit with parameters bound or parameterized
        """
        ...

    @property
    def n_params(self) -> int:
        """Return number of parameters for given qubit count."""
        ...


class DefaultAnsatz:
    def __init__(self, n_qubits: int, depth: Optional[int] = None) -> None:
        self._n_qubits = n_qubits
        self._depth = depth if depth is not None else n_qubits
        self._n_params = n_qubits * self._depth

    @property
    def n_params(self) -> int:
        return self._n_params

    @property
    def n_qubits(self) -> int:
        return self._n_qubits

    @property
    def layers(self) -> int:
        return self._depth

    @property
    def depth(self) -> int:
        return self._depth

    def __call__(
        self,
        params: Union[NDArray[np.float64], ParameterVector],
        n_qubits: Optional[int] = None,
        **_kwargs: Any,
    ) -> QuantumCircuit:
        """Build the ansatz circuit."""
        n = n_qubits if n_qubits is not None else self._n_qubits
        D2 = _kwargs.get("depth", self._depth)

        qr = QuantumRegister(n, name="q")
        qc = QuantumCircuit(qr)

        # Initial state: |0...01⟩
        qc.x(n - 1)
        for i in range(n):
            qc.ry(params[i], i)
        for d in range(D2 - 1):
            for i in range(n - 1):
                qc.cx(i, i + 1)
            qc.barrier()
            for i in range(n):
                param_idx = n + n * d + i
                qc.ry(params[param_idx], i)

        return qc

    def get_param_count(self, n_qubits: Optional[int] = None, depth: Optional[int] = None) -> int:
        """Calculate parameter count for given configuration."""
        n = n_qubits if n_qubits is not None else self._n_qubits
        d = depth if depth is not None else self._depth
        return n * d


# Keep old function for backward compatibility
def get_full_variational_quantum_circuit(thetas, D2, qubits_num, input_state=None):
    """
    DEPRECATED: Use DefaultAnsatz class instead.

    Kept for backward compatibility.
    """
    ansatz = DefaultAnsatz(qubits_num, depth=D2)
    return ansatz(thetas, qubits_num, depth=D2)


class CustomHardwareEfficientAnsatz:
    """
    Example custom ansatz with configurable entanglement pattern.

    Usage:
        ansatz = CustomHardwareEfficientAnsatz(n_qubits=8, layers=4, entanglement='circular')
        config = OptimizerConfig(n_qubits=8, ansatz=ansatz)
        optimizer = GaussianOptimizer(config)
    """

    def __init__(
        self,
        n_qubits: int,
        layers: int = 4,
        entanglement: str = "linear",
        rotation_gates: Optional[list[str]] = None,
    ) -> None:
        self._n_qubits = n_qubits
        self._layers = layers
        self._entanglement = entanglement
        self._rotation_gates = rotation_gates or ["ry"]

        # Calculate parameter count
        rotations_per_layer = n_qubits * len(self._rotation_gates)
        self._n_params = rotations_per_layer * layers

    @property
    def n_params(self) -> int:
        return self._n_params

    @property
    def depth(self) -> int:
        return self._layers

    def __call__(
        self, params: Union[np.ndarray, ParameterVector], n_qubits: int = None, **_kwargs
    ) -> QuantumCircuit:
        n = n_qubits or self._n_qubits
        qc = QuantumCircuit(n)

        param_idx = 0
        for layer in range(self._layers):
            # Rotation block
            for qubit in range(n):
                for gate_name in self._rotation_gates:
                    if gate_name == "ry":
                        qc.ry(params[param_idx], qubit)
                    elif gate_name == "rx":
                        qc.rx(params[param_idx], qubit)
                    elif gate_name == "rz":
                        qc.rz(params[param_idx], qubit)
                    param_idx += 1

            # Entanglement block (skip on last layer)
            if layer < self._layers - 1:
                if self._entanglement == "linear":
                    for i in range(n - 1):
                        qc.cx(i, i + 1)
                elif self._entanglement == "circular":
                    for i in range(n - 1):
                        qc.cx(i, i + 1)
                    qc.cx(n - 1, 0)
                elif self._entanglement == "full":
                    for i in range(n):
                        for j in range(i + 1, n):
                            qc.cx(i, j)

        return qc

    def apply_custatevec(
        self, simulator: "CuStateVecSimulator", params: NDArray[np.float64]
    ) -> None:
        """Native cuStateVec implementation for speed."""
        n = self._n_qubits
        param_idx = 0

        simulator.reset_state()

        for layer in range(self._layers):
            # Rotations
            for qubit in range(n):
                for gate_name in self._rotation_gates:
                    if gate_name == "ry":
                        simulator.apply_ry(float(params[param_idx]), qubit)
                    # Add rx, rz if simulator supports them
                    param_idx += 1

            # Entanglement
            if layer < self._layers - 1:
                if self._entanglement == "linear":
                    for i in range(n - 1):
                        simulator.apply_cnot(i, i + 1)
                elif self._entanglement == "circular":
                    for i in range(n - 1):
                        simulator.apply_cnot(i, i + 1)
                    simulator.apply_cnot(n - 1, 0)
