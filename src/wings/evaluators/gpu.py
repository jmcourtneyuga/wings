"""GPU-accelerated circuit evaluator using Qiskit Aer."""

import logging
from typing import TYPE_CHECKING

import numpy as np
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector

from ..ansatz import get_full_variational_quantum_circuit

# Conditional import
try:
    from qiskit_aer import AerSimulator

    HAS_AER_GPU = True
except ImportError:
    HAS_AER_GPU = False
    AerSimulator = None

if TYPE_CHECKING:
    from ..config import OptimizerConfig

logger = logging.getLogger(__name__)

__all__ = ["GPUCircuitEvaluator", "HAS_AER_GPU"]


class GPUCircuitEvaluator:
    """
    GPU-accelerated circuit evaluation using Qiskit Aer.

    Features:
    - Double precision for high-accuracy results
    - Batched execution for efficiency
    - Automatic CPU fallback
    """

    def __init__(self, config: "OptimizerConfig", target: np.ndarray):
        self.config = config
        self.target = target
        self._target_conj = np.conj(target)
        self.n_params = config.n_params
        self.n_qubits = config.n_qubits

        # Build circuit template
        self.param_vector = ParameterVector("theta", self.n_params)
        self.circuit = get_full_variational_quantum_circuit(
            thetas=self.param_vector,
            D2=config.n_qubits,
            qubits_num=config.n_qubits,
            input_state=None,
        )
        self._param_list = list(self.param_vector)

        # Initialize GPU backend
        self._init_gpu_backend()

        # Statistics
        self.n_gpu_calls = 0
        self.n_circuits_evaluated = 0

    def _init_gpu_backend(self):
        """Initialize GPU backend with appropriate settings"""
        self.gpu_available = False
        self.backend = None

        if not HAS_AER_GPU:
            print("  Aer not available, using CPU statevector")
            self._use_cpu_fallback()
            return

        try:
            # Try to create GPU backend
            self.backend = AerSimulator(
                method="statevector",
                device="GPU",
                precision=self.config.gpu_precision,  # 'double' for high precision
                blocking_enable=self.config.gpu_blocking,
            )

            # Verify GPU is actually available
            available_devices = self.backend.available_devices()

            if "GPU" in available_devices:
                self.gpu_available = True
                print("  ✓ GPU backend initialized")
                print(f"    Precision: {self.config.gpu_precision}")
                print(f"    Available devices: {available_devices}")
            else:
                print(f"  GPU not in available devices: {available_devices}")
                self._use_cpu_fallback()

        except ImportError as e:
            logger.info(f"GPU backend not available (missing dependency): {e}")
            self._use_cpu_fallback()
        except RuntimeError as e:
            logger.warning(f"GPU initialization failed (runtime error): {e}")
            self._use_cpu_fallback()
        except Exception as e:
            logger.warning(f"Unexpected GPU initialization error ({type(e).__name__}): {e}")
            self._use_cpu_fallback()

    def _use_cpu_fallback(self):
        """Fall back to CPU Aer backend"""
        print("  Using CPU Aer backend (fallback)")
        try:
            self.backend = AerSimulator(
                method="statevector",
                device="CPU",
                precision="double",
            )
            self.gpu_available = False
        except Exception as e:
            print(f"  CPU Aer also failed: {e}")
            self.backend = None

    def get_statevector(self, params: np.ndarray) -> np.ndarray:
        """
        Get statevector for single parameter set.

        Uses GPU if available, otherwise falls back to standard Qiskit.
        """
        if self.backend is None:
            # Ultimate fallback: use standard Qiskit Statevector
            bound_circuit = self.circuit.assign_parameters(dict(zip(self._param_list, params)))
            return Statevector(bound_circuit).data

        # Bind parameters
        bound_circuit = self.circuit.assign_parameters(dict(zip(self._param_list, params)))

        # Add save_statevector instruction
        bound_circuit.save_statevector()

        # Execute on GPU/CPU Aer
        job = self.backend.run(bound_circuit, shots=1)
        result = job.result()
        statevector = result.get_statevector()

        self.n_gpu_calls += 1
        self.n_circuits_evaluated += 1

        return np.array(statevector.data, dtype=np.complex128)

    def get_statevectors_batched(self, params_batch: np.ndarray) -> list[np.ndarray]:
        """
        Get statevectors for multiple parameter sets in a single GPU call.

        This is much more efficient than individual calls because:
        1. Single data transfer to GPU
        2. GPU parallelism across circuits
        3. Single result retrieval

        Args:
            params_batch: Array of shape (batch_size, n_params)

        Returns:
            List of statevectors
        """
        batch_size = len(params_batch)

        if self.backend is None:
            # Fallback: sequential evaluation
            return [self.get_statevector(p) for p in params_batch]

        # Build all circuits
        circuits = []
        for params in params_batch:
            bound_circuit = self.circuit.assign_parameters(dict(zip(self._param_list, params)))
            bound_circuit.save_statevector()
            circuits.append(bound_circuit)

        # Single batched GPU execution
        job = self.backend.run(circuits, shots=1)
        result = job.result()

        # Extract all statevectors
        statevectors = []
        for i in range(batch_size):
            sv = result.get_statevector(i)
            statevectors.append(np.array(sv.data, dtype=np.complex128))

        self.n_gpu_calls += 1
        self.n_circuits_evaluated += batch_size

        return statevectors

    def compute_fidelity(self, params: np.ndarray) -> float:
        """Compute fidelity for single parameter set"""
        psi = self.get_statevector(params)
        overlap = np.dot(self._target_conj, psi)
        return overlap.real**2 + overlap.imag**2

    def compute_fidelities_batched(self, params_batch: np.ndarray) -> np.ndarray:
        """
        Compute fidelities for batch of parameter sets.

        Optimized for GPU: single batched circuit execution,
        then vectorized fidelity computation on CPU.
        """
        statevectors = self.get_statevectors_batched(params_batch)

        # Vectorized fidelity computation
        fidelities = np.zeros(len(params_batch))
        for i, psi in enumerate(statevectors):
            overlap = np.dot(self._target_conj, psi)
            fidelities[i] = overlap.real**2 + overlap.imag**2

        return fidelities

    def get_stats(self) -> dict:
        """Return GPU usage statistics"""
        return {
            "gpu_available": self.gpu_available,
            "n_gpu_calls": self.n_gpu_calls,
            "n_circuits_evaluated": self.n_circuits_evaluated,
            "circuits_per_call": (
                self.n_circuits_evaluated / self.n_gpu_calls if self.n_gpu_calls > 0 else 0
            ),
        }
