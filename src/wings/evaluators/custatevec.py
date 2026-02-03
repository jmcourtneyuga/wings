"""cuStateVec-based circuit evaluators."""

import concurrent.futures
import logging
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from numpy.typing import NDArray

from ..compat import HAS_CUSTATEVEC, get_compute_type, get_cuda_dtype
from ..types import ComplexArray, FloatArray, ParameterArray

# Conditional imports
if HAS_CUSTATEVEC:
    import cupy as cp
    from cuquantum.bindings import custatevec as cusv
else:
    cp = None
    cusv = None

if TYPE_CHECKING:
    from ..config import OptimizerConfig

logger = logging.getLogger(__name__)

__all__ = [
    "CuStateVecSimulator",
    "CuStateVecEvaluator",
    "BatchedCuStateVecEvaluator",
    "MultiGPUBatchEvaluator",
]


class CuStateVecSimulator:
    """
    High-performance statevector simulator using NVIDIA cuStateVec.

    This provides the fastest possible GPU simulation by:
    1. Direct GPU memory management with CuPy
    2. Optimized gate application via cuStateVec
    3. Minimal Python overhead
    4. Double precision for high accuracy

    Parameters
    ----------
    n_qubits : int
        Number of qubits
    precision : str
        'double' or 'single' precision
    device_id : int
        GPU device ID (default: 0)

    For VQC optimization, this can be 5-20x faster than Qiskit Aer GPU.
    """

    def __init__(
        self,
        n_qubits: int,
        precision: str = "double",
        device_id: int = 0,
    ) -> None:
        self.n_qubits = n_qubits
        self.n_states = 2**n_qubits
        self.device_id = device_id
        self.handle = None

        # Set precision FIRST - before any GPU operations
        if precision == "double":
            self.dtype = cp.complex128
            self.cuda_dtype = get_cuda_dtype()
            self.compute_type = get_compute_type()
        else:
            self.dtype = cp.complex64
            self.cuda_dtype = cusv.cudaDataType.CUDA_C_32F
            self.compute_type = cusv.ComputeType.COMPUTE_32F

        # Initialize on the specified GPU device
        with cp.cuda.Device(device_id):
            # Initialize cuStateVec handle
            self.handle = cusv.create()

            # Pre-allocate GPU memory for statevector
            self.d_sv = cp.zeros(self.n_states, dtype=self.dtype)

            # Pre-allocate workspace (reused across operations)
            self._workspace_size = 0
            self._d_workspace = None

            # Pre-compute and cache gate matrices on GPU
            self._precompute_gates()

        # Statistics
        self.n_gate_applications = 0
        self.n_resets = 0

    def _precompute_gates(self):
        """Pre-compute and cache gate matrices on GPU."""
        x_np = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        self.x_gate = cp.asarray(x_np)

        # Pre-allocate RY matrix on GPU (will be updated in-place)
        self._ry_matrix = cp.zeros((2, 2), dtype=self.dtype)

    def _get_ry_matrix(self, theta: float) -> "cp.ndarray":
        """Compute RY(theta) matrix on GPU - optimized."""
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        self._ry_matrix[0, 0] = c
        self._ry_matrix[0, 1] = -s
        self._ry_matrix[1, 0] = s
        self._ry_matrix[1, 1] = c
        return self._ry_matrix

    def reset_state(self) -> None:
        """Reset to |0...0⟩ state."""
        with cp.cuda.Device(self.device_id):
            self.d_sv.fill(0)
            self.d_sv[0] = 1.0 + 0j
        self.n_resets += 1

    def apply_x(self, target: int) -> None:
        """Apply Pauli-X gate to target qubit."""
        targets = np.array([target], dtype=np.int32)

        with cp.cuda.Device(self.device_id):
            cusv.apply_matrix(
                self.handle,
                self.d_sv.data.ptr,
                self.cuda_dtype,
                self.n_qubits,
                self.x_gate.data.ptr,
                self.cuda_dtype,
                cusv.MatrixLayout.ROW,
                0,
                targets.ctypes.data,
                1,
                0,
                0,
                0,
                self.compute_type,
                0,
                0,
            )
        self.n_gate_applications += 1

    def apply_ry(self, theta: float, target: int) -> None:
        """Apply RY(theta) gate - optimized to minimize transfers."""
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)

        targets = np.array([target], dtype=np.int32)

        with cp.cuda.Device(self.device_id):
            # Build matrix on CPU and copy once
            ry_cpu = np.array([[c, -s], [s, c]], dtype=np.complex128)
            cp.copyto(self._ry_matrix, cp.asarray(ry_cpu))

            cusv.apply_matrix(
                self.handle,
                self.d_sv.data.ptr,
                self.cuda_dtype,
                self.n_qubits,
                self._ry_matrix.data.ptr,
                self.cuda_dtype,
                cusv.MatrixLayout.ROW,
                0,
                targets.ctypes.data,
                1,
                0,
                0,
                0,
                self.compute_type,
                0,
                0,
            )
        self.n_gate_applications += 1

    def apply_cnot(self, control: int, target: int) -> None:
        """Apply CNOT gate."""
        targets = np.array([target], dtype=np.int32)
        controls = np.array([control], dtype=np.int32)
        control_bits = np.array([1], dtype=np.int32)

        with cp.cuda.Device(self.device_id):
            cusv.apply_matrix(
                self.handle,
                self.d_sv.data.ptr,
                self.cuda_dtype,
                self.n_qubits,
                self.x_gate.data.ptr,
                self.cuda_dtype,
                cusv.MatrixLayout.ROW,
                0,
                targets.ctypes.data,
                1,
                controls.ctypes.data,
                control_bits.ctypes.data,
                1,
                self.compute_type,
                0,
                0,
            )
        self.n_gate_applications += 1

    def get_statevector_gpu(self) -> "cp.ndarray":
        """Return current statevector (stays on GPU)."""
        with cp.cuda.Device(self.device_id):
            return self.d_sv.copy()

    def get_statevector_cpu(self) -> ComplexArray:
        """Return statevector copied to CPU."""
        with cp.cuda.Device(self.device_id):
            return cp.asnumpy(self.d_sv)

    def compute_overlap_gpu(self, target_conj: "cp.ndarray") -> complex:
        """Compute ⟨target|current⟩ entirely on GPU."""
        with cp.cuda.Device(self.device_id):
            return cp.vdot(target_conj, self.d_sv)

    def compute_fidelity_gpu(self, target_conj: "cp.ndarray") -> float:
        """Compute |⟨target|current⟩|² on GPU."""
        with cp.cuda.Device(self.device_id):
            overlap = cp.vdot(target_conj, self.d_sv)
            fidelity = float(overlap.real**2 + overlap.imag**2)
        return fidelity

    def get_stats(self) -> dict[str, Any]:
        """Return simulator statistics."""
        return {
            "n_qubits": self.n_qubits,
            "n_states": self.n_states,
            "device_id": self.device_id,
            "n_gate_applications": self.n_gate_applications,
            "n_resets": self.n_resets,
            "gates_per_circuit": (
                self.n_gate_applications / self.n_resets if self.n_resets > 0 else 0
            ),
        }

    def destroy(self) -> None:
        """Clean up cuStateVec resources."""
        if self.handle is not None:
            with cp.cuda.Device(self.device_id):
                cusv.destroy(self.handle)
            self.handle = None

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.destroy()


class CuStateVecEvaluator:
    """
    High-performance circuit evaluator using cuStateVec.

    Implements the default ansatz directly with cuStateVec gates,
    avoiding Qiskit overhead entirely.
    """

    def __init__(
        self,
        config: "OptimizerConfig",
        target: np.ndarray,
        device_id: int = 0,
    ):
        if not HAS_CUSTATEVEC:
            raise RuntimeError("cuStateVec not available")

        self.config = config
        self.n_qubits = config.n_qubits
        self.n_params = config.n_params
        self.ansatz = config.ansatz
        self.device_id = device_id

        # Check if ansatz has custatevec-native implementation
        self._has_native_cusv = hasattr(self.ansatz, "apply_custatevec") if self.ansatz else False

        # Initialize cuStateVec simulator
        precision = "double" if config.gpu_precision == "double" else "single"
        self.simulator = CuStateVecSimulator(
            config.n_qubits,
            precision=precision,
            device_id=device_id,
        )

        target_reversed = self._bit_reverse_statevector(target, config.n_qubits)

        with cp.cuda.Device(device_id):
            self.target_gpu = cp.array(target_reversed, dtype=cp.complex128)
            self.target_conj_gpu = cp.conj(self.target_gpu)

        # Also keep original target for plotting reference
        self._target_original = target

        # Statistics
        self.n_circuits_evaluated = 0
        self.n_fidelity_computations = 0

    def _bit_reverse_statevector(self, sv: np.ndarray, n_qubits: int) -> np.ndarray:
        """Reverse bit order of statevector indices to match cuStateVec convention."""
        n_states = len(sv)
        result = np.zeros_like(sv)
        for i in range(n_states):
            reversed_i = int(format(i, f"0{n_qubits}b")[::-1], 2)
            result[reversed_i] = sv[i]
        return result

    def apply_ansatz(self, params: np.ndarray) -> None:
        """Apply the ansatz circuit to the simulator."""
        if self._has_native_cusv:
            self.ansatz.apply_custatevec(self.simulator, params)
        else:
            self._apply_default_ansatz(params)
        self.n_circuits_evaluated += 1

    def _apply_default_ansatz(self, params: np.ndarray) -> None:
        """Apply default ansatz (Ollitrault-style) directly with cuStateVec."""
        n = self.n_qubits
        D2 = n

        self.simulator.reset_state()
        self.simulator.apply_x(n - 1)

        for i in range(n):
            self.simulator.apply_ry(float(params[i]), i)

        for d in range(D2 - 1):
            for i in range(n - 1):
                self.simulator.apply_cnot(i, i + 1)
            for i in range(n):
                param_idx = n + n * d + i
                self.simulator.apply_ry(float(params[param_idx]), i)

    def get_statevector(self, params: np.ndarray) -> np.ndarray:
        """Apply ansatz and return statevector on CPU."""
        self.apply_ansatz(params)
        return self.simulator.get_statevector_cpu()

    def get_statevector_gpu(self, params: np.ndarray) -> "cp.ndarray":
        """Apply ansatz and return statevector on GPU."""
        self.apply_ansatz(params)
        return self.simulator.get_statevector_gpu()

    def compute_fidelity(self, params: np.ndarray) -> float:
        """Compute fidelity entirely on GPU."""
        self.apply_ansatz(params)
        fidelity = self.simulator.compute_fidelity_gpu(self.target_conj_gpu)
        self.n_fidelity_computations += 1
        return fidelity

    def compute_fidelity_from_statevector(self, psi_gpu: "cp.ndarray") -> float:
        """Compute fidelity from GPU statevector."""
        with cp.cuda.Device(self.device_id):
            overlap = cp.vdot(self.target_conj_gpu, psi_gpu)
            return float(overlap.real**2 + overlap.imag**2)

    def get_statevector_qiskit_order(self, params: np.ndarray) -> np.ndarray:
        self.apply_ansatz(params)
        sv_cusv = self.simulator.get_statevector_cpu()
        # Convert from cuStateVec convention back to Qiskit convention
        return self._bit_reverse_statevector(sv_cusv, self.n_qubits)

    def get_stats(self) -> dict:
        """Return evaluator statistics."""
        sim_stats = self.simulator.get_stats()
        return {
            **sim_stats,
            "n_circuits_evaluated": self.n_circuits_evaluated,
            "n_fidelity_computations": self.n_fidelity_computations,
        }

    def cleanup(self) -> None:
        """Release GPU resources."""
        if self.simulator is not None:
            self.simulator.destroy()
            self.simulator = None


class BatchedCuStateVecEvaluator:
    """
    Batched evaluation using multiple cuStateVec simulators on a single GPU.

    For population-based methods (CMA-ES, etc.), we can evaluate
    multiple circuits by:
    1. Using multiple simulators with reused pre-allocated GPU memory
    2. Round-robin distribution across simulators
    3. Batched fidelity computations
    """

    def __init__(
        self,
        config: "OptimizerConfig",
        target: np.ndarray,
        n_simulators: int = 4,
        device_id: int = 0,
    ):
        if not HAS_CUSTATEVEC:
            raise RuntimeError("cuStateVec not available")

        self.config = config
        self.n_qubits = config.n_qubits
        self.n_params = config.n_params
        self.n_simulators = n_simulators
        self.device_id = device_id

        precision = "double" if config.gpu_precision == "double" else "single"
        self.simulators = [
            CuStateVecSimulator(config.n_qubits, precision=precision, device_id=device_id)
            for _ in range(n_simulators)
        ]

        # Bit-reverse target to match cuStateVec convention
        target_reversed = self._bit_reverse_statevector(target, config.n_qubits)

        with cp.cuda.Device(device_id):
            self.target_gpu = cp.array(target_reversed, dtype=cp.complex128)
            self.target_conj_gpu = cp.conj(self.target_gpu)
            self.batch_fidelities = cp.zeros(config.custatevec_batch_size, dtype=cp.float64)

        self.n_batches = 0
        self.n_circuits_total = 0

    def _bit_reverse_statevector(self, sv: np.ndarray, n_qubits: int) -> np.ndarray:
        """Reverse bit order of statevector indices to match cuStateVec convention."""
        n_states = len(sv)
        result = np.zeros_like(sv)
        for i in range(n_states):
            reversed_i = int(format(i, f"0{n_qubits}b")[::-1], 2)
            result[reversed_i] = sv[i]
        return result

    def _apply_ansatz_to_simulator(
        self,
        simulator: CuStateVecSimulator,
        params: ParameterArray,
    ) -> None:
        """Apply ansatz to a specific simulator."""
        n = self.n_qubits
        D2 = n

        simulator.reset_state()
        simulator.apply_x(n - 1)

        for i in range(n):
            simulator.apply_ry(float(params[i]), i)

        for d in range(D2 - 1):
            for i in range(n - 1):
                simulator.apply_cnot(i, i + 1)
            for i in range(n):
                param_idx = n + n * d + i
                simulator.apply_ry(float(params[param_idx]), i)

    def get_statevector_qiskit_order(self, params: np.ndarray) -> np.ndarray:
        """Get statevector in Qiskit ordering (bit-reversed from cuStateVec)."""
        self._apply_ansatz_to_simulator(self.simulators[0], params)
        sv_cusv = self.simulators[0].get_statevector_cpu()
        return self._bit_reverse_statevector(sv_cusv, self.n_qubits)

    def evaluate_batch(self, params_batch: NDArray[np.float64]) -> FloatArray:
        """
        Evaluate fidelities for a batch of parameter sets.

        Uses round-robin distribution across simulators.
        """
        batch_size = len(params_batch)
        fidelities = np.zeros(batch_size, dtype=np.float64)

        for i, params in enumerate(params_batch):
            sim_idx = i % self.n_simulators
            simulator = self.simulators[sim_idx]

            self._apply_ansatz_to_simulator(simulator, params)
            fidelities[i] = simulator.compute_fidelity_gpu(self.target_conj_gpu)

        self.n_batches += 1
        self.n_circuits_total += batch_size

        return fidelities

    def evaluate_batch_chunked(
        self,
        params_batch: NDArray[np.float64],
        chunk_size: Optional[int] = None,
    ) -> FloatArray:
        """Evaluate large batch in chunks for memory efficiency."""
        if chunk_size is None:
            chunk_size = self.config.custatevec_batch_size

        batch_size = len(params_batch)
        fidelities = np.zeros(batch_size, dtype=np.float64)

        for start in range(0, batch_size, chunk_size):
            end = min(start + chunk_size, batch_size)
            fidelities[start:end] = self.evaluate_batch(params_batch[start:end])

        return fidelities

    def compute_gradient_batched(self, params: ParameterArray) -> FloatArray:
        """
        Compute gradient using batched parameter-shift evaluation.

        All 2*n_params shifted circuits evaluated in batch.
        """
        shift = np.pi / 2
        n_params = self.n_params

        params_shifted = np.zeros((2 * n_params, n_params), dtype=np.float64)

        for i in range(n_params):
            params_shifted[2 * i] = params.copy()
            params_shifted[2 * i, i] += shift

            params_shifted[2 * i + 1] = params.copy()
            params_shifted[2 * i + 1, i] -= shift

        fidelities = self.evaluate_batch(params_shifted)

        gradient = np.zeros(n_params, dtype=np.float64)
        for i in range(n_params):
            gradient[i] = (fidelities[2 * i] - fidelities[2 * i + 1]) / 2

        return -gradient  # Negative for minimization of -fidelity

    def get_stats(self) -> dict:
        """Return batch evaluator statistics."""
        total_gate_apps = sum(s.n_gate_applications for s in self.simulators)
        return {
            "n_simulators": self.n_simulators,
            "device_id": self.device_id,
            "n_batches": self.n_batches,
            "n_circuits_total": self.n_circuits_total,
            "total_gate_applications": total_gate_apps,
            "avg_circuits_per_batch": (
                self.n_circuits_total / self.n_batches if self.n_batches > 0 else 0
            ),
        }

    def cleanup(self) -> None:
        """Release all GPU resources."""
        for sim in self.simulators:
            sim.destroy()
        self.simulators = []


class MultiGPUBatchEvaluator:
    """
    Distribute circuit evaluations across multiple GPUs.

    Each GPU gets its own set of CuStateVecSimulators, and work is
    distributed across GPUs in chunks.

    Parameters
    ----------
    config : OptimizerConfig
        Optimizer configuration
    target : np.ndarray
        Target wavefunction
    device_ids : list of int, optional
        GPU device IDs to use. If None, auto-detects all available GPUs.
    simulators_per_gpu : int
        Number of simulators per GPU for round-robin evaluation

    Example
    -------
    >>> evaluator = MultiGPUBatchEvaluator(
    ...     config, target,
    ...     device_ids=[0, 1, 2, 3],
    ...     simulators_per_gpu=2,
    ... )
    >>> fidelities = evaluator.evaluate_batch_parallel(params_batch)
    """

    def __init__(
        self,
        config: "OptimizerConfig",
        target: np.ndarray,
        device_ids: Optional[list[int]] = None,
        simulators_per_gpu: int = 2,
    ):
        if not HAS_CUSTATEVEC:
            raise RuntimeError("cuStateVec not available")

        self.config = config
        self.n_qubits = config.n_qubits
        self.n_params = config.n_params

        # Auto-detect GPUs if not specified
        if device_ids is None:
            device_ids = list(range(cp.cuda.runtime.getDeviceCount()))

        self.device_ids = device_ids
        self.n_gpus = len(device_ids)
        self.simulators_per_gpu = simulators_per_gpu

        if self.n_gpus == 0:
            raise RuntimeError("No GPUs available")

        logger.info(f"Initializing Multi-GPU evaluator with {self.n_gpus} GPUs: {device_ids}")

        precision = "double" if config.gpu_precision == "double" else "single"

        # Bit-reverse target to match cuStateVec convention
        target_reversed = self._bit_reverse_statevector(target, config.n_qubits)

        # Create simulators and target arrays on each GPU
        self.simulators: list[list[CuStateVecSimulator]] = []
        self.target_conj_gpu: list[cp.ndarray] = []

        for gpu_id in device_ids:
            with cp.cuda.Device(gpu_id):
                # Create simulators for this GPU
                gpu_sims = [
                    CuStateVecSimulator(self.n_qubits, precision, device_id=gpu_id)
                    for _ in range(simulators_per_gpu)
                ]
                self.simulators.append(gpu_sims)

                # Copy target to this GPU
                target_gpu = cp.array(target_reversed, dtype=cp.complex128)
                self.target_conj_gpu.append(cp.conj(target_gpu))

        self.total_simulators = self.n_gpus * simulators_per_gpu

        # Statistics
        self.n_batches = 0
        self.n_circuits_evaluated = 0

    def _bit_reverse_statevector(self, sv: np.ndarray, n_qubits: int) -> np.ndarray:
        """Reverse bit order of statevector indices."""
        n_states = len(sv)
        result = np.zeros_like(sv)
        for i in range(n_states):
            reversed_i = int(format(i, f"0{n_qubits}b")[::-1], 2)
            result[reversed_i] = sv[i]
        return result

    def _apply_ansatz_to_simulator(
        self,
        simulator: CuStateVecSimulator,
        params: np.ndarray,
    ) -> None:
        """Apply default ansatz to a simulator."""
        n = self.n_qubits
        D2 = n

        simulator.reset_state()
        simulator.apply_x(n - 1)

        for i in range(n):
            simulator.apply_ry(float(params[i]), i)

        for d in range(D2 - 1):
            for i in range(n - 1):
                simulator.apply_cnot(i, i + 1)
            for i in range(n):
                param_idx = n + n * d + i
                simulator.apply_ry(float(params[param_idx]), i)

    def evaluate_batch(self, params_batch: np.ndarray) -> np.ndarray:
        """
        Evaluate batch distributed across GPUs sequentially.

        For parallel execution, use evaluate_batch_parallel().
        """
        batch_size = len(params_batch)
        fidelities = np.zeros(batch_size, dtype=np.float64)

        # Split batch across GPUs
        chunk_size = (batch_size + self.n_gpus - 1) // self.n_gpus

        for gpu_idx, gpu_id in enumerate(self.device_ids):
            start = gpu_idx * chunk_size
            end = min(start + chunk_size, batch_size)

            if start >= batch_size:
                break

            with cp.cuda.Device(gpu_id):
                gpu_sims = self.simulators[gpu_idx]
                target_conj = self.target_conj_gpu[gpu_idx]

                for i, params in enumerate(params_batch[start:end]):
                    sim_idx = i % len(gpu_sims)
                    sim = gpu_sims[sim_idx]

                    self._apply_ansatz_to_simulator(sim, params)
                    fidelities[start + i] = sim.compute_fidelity_gpu(target_conj)

        self.n_batches += 1
        self.n_circuits_evaluated += batch_size
        return fidelities

    def evaluate_batch_parallel(self, params_batch: np.ndarray) -> np.ndarray:
        """
        Evaluate batch with parallel GPU execution using threads.

        Each GPU processes its chunk in a separate thread for maximum throughput.
        """
        batch_size = len(params_batch)
        fidelities = np.zeros(batch_size, dtype=np.float64)
        chunk_size = (batch_size + self.n_gpus - 1) // self.n_gpus

        def process_gpu_chunk(gpu_idx: int) -> None:
            gpu_id = self.device_ids[gpu_idx]
            start = gpu_idx * chunk_size
            end = min(start + chunk_size, batch_size)

            if start >= batch_size:
                return

            with cp.cuda.Device(gpu_id):
                gpu_sims = self.simulators[gpu_idx]
                target_conj = self.target_conj_gpu[gpu_idx]

                for i, params in enumerate(params_batch[start:end]):
                    sim_idx = i % len(gpu_sims)
                    sim = gpu_sims[sim_idx]

                    self._apply_ansatz_to_simulator(sim, params)
                    fidelities[start + i] = sim.compute_fidelity_gpu(target_conj)

        # Process all GPUs in parallel using thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_gpus) as executor:
            futures = [executor.submit(process_gpu_chunk, i) for i in range(self.n_gpus)]
            concurrent.futures.wait(futures)

            # Check for exceptions
            for future in futures:
                if future.exception() is not None:
                    raise future.exception()

        self.n_batches += 1
        self.n_circuits_evaluated += batch_size
        return fidelities

    def compute_gradient_parallel(self, params: np.ndarray) -> np.ndarray:
        """Compute gradient using parallel multi-GPU evaluation."""
        shift = np.pi / 2
        n_params = self.n_params

        # Build all shifted parameter sets
        params_shifted = np.zeros((2 * n_params, n_params), dtype=np.float64)

        for i in range(n_params):
            params_shifted[2 * i] = params.copy()
            params_shifted[2 * i, i] += shift

            params_shifted[2 * i + 1] = params.copy()
            params_shifted[2 * i + 1, i] -= shift

        # Parallel evaluation across GPUs
        fidelities = self.evaluate_batch_parallel(params_shifted)

        # Compute gradients
        gradient = np.zeros(n_params, dtype=np.float64)
        for i in range(n_params):
            gradient[i] = (fidelities[2 * i] - fidelities[2 * i + 1]) / 2

        return -gradient

    def get_stats(self) -> dict:
        """Return multi-GPU evaluator statistics."""
        total_gate_apps = sum(
            sim.n_gate_applications for gpu_sims in self.simulators for sim in gpu_sims
        )
        return {
            "n_gpus": self.n_gpus,
            "device_ids": self.device_ids,
            "simulators_per_gpu": self.simulators_per_gpu,
            "total_simulators": self.total_simulators,
            "n_batches": self.n_batches,
            "n_circuits_evaluated": self.n_circuits_evaluated,
            "total_gate_applications": total_gate_apps,
        }

    def cleanup(self) -> None:
        """Release all GPU resources."""
        for gpu_sims in self.simulators:
            for sim in gpu_sims:
                sim.destroy()
        self.simulators = []
        self.target_conj_gpu = []
