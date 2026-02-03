"""Thread-safe CPU circuit evaluator."""

import threading
from typing import TYPE_CHECKING, Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector

from ..ansatz import get_full_variational_quantum_circuit
from ..types import ComplexArray, ParameterArray

if TYPE_CHECKING:
    from ..config import OptimizerConfig

__all__ = ["ThreadSafeCircuitEvaluator"]


class ThreadSafeCircuitEvaluator:
    """
    Thread-safe circuit evaluation for parallel gradient computation.

    Each thread gets its own circuit copy to avoid race conditions.
    """

    def __init__(self, config: "OptimizerConfig", target: ComplexArray) -> None:
        self.config: OptimizerConfig = config
        self.target: ComplexArray = target
        self._target_conj: ComplexArray = np.conj(target)
        self.n_params: int = config.n_params

        # Thread-local storage for circuit copies
        self._local = threading.local()

    def _get_circuit(self) -> tuple[QuantumCircuit, list[Any]]:
        """Get or create thread-local circuit"""
        if not hasattr(self._local, "circuit"):
            # Create fresh circuit for this thread
            from qiskit import transpile

            param_vector = ParameterVector("theta", self.n_params)
            circuit = get_full_variational_quantum_circuit(
                thetas=param_vector,
                D2=self.config.n_qubits,
                qubits_num=self.config.n_qubits,
                input_state=None,
            )
            self._local.circuit = transpile(
                circuit, basis_gates=["ry", "cx", "x"], optimization_level=1
            )
            self._local.param_vector = param_vector
            self._local.param_list = list(param_vector)

        return self._local.circuit, self._local.param_list

    def get_statevector(self, params: ParameterArray) -> ComplexArray:
        """Thread-safe statevector computation"""
        circuit, param_list = self._get_circuit()
        bound_circuit = circuit.assign_parameters(dict(zip(param_list, params)))
        return Statevector(bound_circuit).data

    def compute_fidelity(self, params: ParameterArray) -> float:
        """Thread-safe fidelity computation"""
        psi = self.get_statevector(params)
        overlap = np.dot(self._target_conj, psi)
        return overlap.real**2 + overlap.imag**2

    def compute_fidelity_from_psi(self, psi: ComplexArray) -> float:
        """Compute fidelity from statevector"""
        overlap = np.dot(self._target_conj, psi)
        return overlap.real**2 + overlap.imag**2
