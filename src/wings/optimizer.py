"""Main Gaussian state optimizer."""

from __future__ import annotations

import copy
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import pipeline as pipeline_mod

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from scipy.optimize import basinhopping, differential_evolution, minimize

from .adam import AdamOptimizer, AdamWithRestarts
from .ansatz import DefaultAnsatz
from .barren_plateau import BarrenPlateauDetector
from .compat import HAS_CUSTATEVEC
from .config import (
    _TRACY_WIDOM_TARGETS,
    _TW_BETA_MAP,
    OptimizationPipeline,
    OptimizerConfig,
    TargetFunction,
)
from .evaluators.cpu import ThreadSafeCircuitEvaluator
from .evaluators.custatevec import (
    BatchedCuStateVecEvaluator,
    CuStateVecEvaluator,
    MultiGPUBatchEvaluator,
)
from .evaluators.gpu import GPUCircuitEvaluator
from .types import ComplexArray, FloatArray, ParameterArray

__all__ = ["GaussianOptimizer"]


class GaussianOptimizer:
    """High-precision optimizer with enhanced convergence capabilities"""

    def __init__(self, config: OptimizerConfig) -> None:
        self.config = config
        self.n_params = config.n_params

        # Get ansatz (use config's ansatz or create default)
        self.ansatz = config.ansatz
        if self.ansatz is None:
            self.ansatz = DefaultAnsatz(config.n_qubits)

        # Build circuit template using ansatz
        self.param_vector = ParameterVector("theta", self.n_params)
        self.circuit = self.ansatz(
            self.param_vector, config.n_qubits, **(config.ansatz_kwargs or {})
        )

        # Compute target Gaussian with high precision
        self.positions = config.positions
        self.target = self._compute_target_wavefunction()
        # Pre-conjugate target for faster overlap
        self._target_conj = np.conj(self.target)

        # Tracking
        self.n_evals = 0
        self.best_fidelity = 0
        self.best_params = None
        self.history = {"fidelity": [], "iteration": [], "gradient_norm": []}
        self.last_gradient = None
        self._log_interval = 500  # Log every 500 evals instead of 100
        self._last_log_time = time.time()
        self._min_log_interval_sec = 2.0

        self._circuit_transpiled = transpile(
            self.circuit, basis_gates=["ry", "rz", "cx", "x"], optimization_level=1
        )
        # Pre-store parameter vector as list for faster zip
        self._param_list = list(self.param_vector)

        self._gpu_evaluator = None
        if self.config.use_gpu:
            print("\nInitializing GPU acceleration...")
            self._gpu_evaluator = GPUCircuitEvaluator(self.config, self.target)
            if self._gpu_evaluator.gpu_available:
                print("  GPU acceleration enabled")
            else:
                print("  GPU not available, using CPU")

        # === Multi-GPU Acceleration ===
        self._multi_gpu_evaluator = None

        if self.config.use_multi_gpu and HAS_CUSTATEVEC:
            try:
                import cupy as cp

                n_gpus = cp.cuda.runtime.getDeviceCount()

                if n_gpus > 1:
                    print(f"\nInitializing Multi-GPU acceleration ({n_gpus} GPUs available)...")
                    self._multi_gpu_evaluator = MultiGPUBatchEvaluator(
                        self.config,
                        self.target,
                        device_ids=self.config.gpu_device_ids,
                        simulators_per_gpu=self.config.simulators_per_gpu,
                    )
                    print("    Multi-GPU initialized")
                    print(f"    GPUs: {self._multi_gpu_evaluator.device_ids}")
                    print(f"    Simulators per GPU: {self.config.simulators_per_gpu}")
                else:
                    print("\\nMulti-GPU requested but only 1 GPU available")
            except Exception as e:
                print(f"  Multi-GPU initialization failed: {e}")
                self._multi_gpu_evaluator = None
        elif self.config.use_multi_gpu and not HAS_CUSTATEVEC:
            print("\\nNote: Multi-GPU requires cuStateVec which is not available.")

        # Add after the GPU evaluator initialization block

        # === Stage 5: cuStateVec Acceleration ===
        self._custatevec_evaluator = None
        self._custatevec_batch_evaluator = None

        if self.config.use_custatevec and HAS_CUSTATEVEC:
            print("\nInitializing cuStateVec acceleration...")
            try:
                self._custatevec_evaluator = CuStateVecEvaluator(self.config, self.target)
                self._custatevec_batch_evaluator = BatchedCuStateVecEvaluator(
                    self.config, self.target, n_simulators=4
                )
                print("    cuStateVec initialized")
                print(f"    Precision: {self.config.gpu_precision}")
                print("    Batch simulators: 4")
            except (RuntimeError, MemoryError) as e:
                print(f"  cuStateVec initialization failed (GPU issue): {e}")
                self._custatevec_evaluator = None
                self._custatevec_batch_evaluator = None
            except ImportError as e:
                print(f"  cuStateVec initialization failed (missing library): {e}")
                self._custatevec_evaluator = None
                self._custatevec_batch_evaluator = None
        elif self.config.use_custatevec and not HAS_CUSTATEVEC:
            print("\nNote: cuStateVec requested but not available. Using Aer GPU fallback.")

    def _compute_target_wavefunction(self) -> ComplexArray:
        """Compute normalized target wavefunction based on config."""
        # Warn if complex target is used with RY-only ansatz
        if self.config.momentum != 0.0:
            ansatz_name = (
                type(self.config.ansatz).__name__ if self.config.ansatz else "DefaultAnsatz"
            )
            if ansatz_name == "DefaultAnsatz":
                import warnings

                warnings.warn(
                    f"Complex target (momentum={self.config.momentum}) requires RZ gates "
                    f"for phase encoding. The DefaultAnsatz (RY+CNOT only) cannot produce "
                    f"complex amplitudes. Use CustomHardwareEfficientAnsatz with "
                    f"rotation_gates=['ry', 'rz'] for best results.",
                    UserWarning,
                    stacklevel=2,
                )

        x = self.positions
        dx = self.config.delta_x

        if self.config.target_function == TargetFunction.GAUSSIAN:
            psi = self._gaussian(x)
        elif self.config.target_function == TargetFunction.LORENTZIAN:
            psi = self._lorentzian(x)
        elif self.config.target_function == TargetFunction.SECH:
            psi = self._sech(x)  # ADD THIS CASE
        elif self.config.target_function == TargetFunction.GAUSSIAN_WAVEPACKET:
            sigma = self.config.sigma
            x0 = self.config.x0
            k = self.config.momentum
            psi = np.exp(-((x - x0) ** 2) / (2 * sigma**2)) * np.exp(1j * k * x)
        elif self.config.target_function in _TRACY_WIDOM_TARGETS:
            from .tracy_widom import tracy_widom_wavefunction

            beta = _TW_BETA_MAP[self.config.target_function]
            psi = tracy_widom_wavefunction(x, beta=beta)
        elif self.config.target_function == TargetFunction.CUSTOM:
            if self.config.custom_target_fn is None:
                raise ValueError("custom_target_fn required for CUSTOM target")
            psi = self.config.custom_target_fn(x)
        else:
            raise ValueError(f"Unknown target function: {self.config.target_function}")

        # Apply momentum phase if specified (for non-wavepacket targets)
        if (
            self.config.momentum != 0.0
            and self.config.target_function != TargetFunction.GAUSSIAN_WAVEPACKET
        ):
            k = self.config.momentum
            psi = psi * np.exp(1j * k * x)

        # Normalize
        psi = psi.astype(np.complex128)
        norm = np.sqrt(np.sum(np.abs(psi) ** 2) * dx)
        psi = psi / norm

        # Ensure unit norm
        psi = psi / np.linalg.norm(psi)

        return psi

    def _gaussian(self, x: np.ndarray) -> np.ndarray:
        """Gaussian wavefunction."""
        return np.exp(-((x - self.config.x0) ** 2) / (2 * self.config.sigma**2))

    def _lorentzian(self, x: np.ndarray) -> np.ndarray:
        """Lorentzian (Cauchy) wavefunction."""
        gamma = self.config.gamma if self.config.gamma else self.config.sigma
        return gamma / ((x - self.config.x0) ** 2 + gamma**2)

    def _sech(self, x: np.ndarray) -> np.ndarray:
        """Hyperbolic secant wavefunction (soliton-like)."""
        return 1.0 / np.cosh((x - self.config.x0) / self.config.sigma)

    def get_statevector(self, params: ParameterArray, backend: str = None) -> ComplexArray:
        """
        Get statevector with automatic backend selection.

        Args:
            params: Circuit parameters
            backend: None (use config), 'auto', 'jax', 'custatevec', 'gpu', or 'cpu'

        Returns:
            Statevector as numpy array
        """
        if backend is None:
            backend = self.config.backend if hasattr(self.config, "backend") else "auto"

        if backend == "jax":
            from .evaluators.jax_backend import HAS_JAX

            if not HAS_JAX:
                raise ImportError("JAX not installed. Use backend='qiskit' or install jax.")
            # JAX statevector path would go here
            # For now, fall through to Qiskit (JAX integration is structural prep)
            backend = "auto"

        if backend == "auto":
            if self.config.use_custatevec and self._custatevec_evaluator is not None:
                backend = "custatevec"
            elif (
                self.config.use_gpu
                and self._gpu_evaluator is not None
                and self._gpu_evaluator.gpu_available
            ):
                backend = "gpu"
            else:
                backend = "cpu"

        if backend == "custatevec":
            return self._custatevec_evaluator.get_statevector(params)
        elif backend == "gpu":
            return self._gpu_evaluator.get_statevector(params)
        else:
            # CPU path using Qiskit
            bound_circuit = self._circuit_transpiled.assign_parameters(
                dict(zip(self._param_list, params))
            )
            return Statevector(bound_circuit).data

    def compute_fidelity(
        self,
        params: ParameterArray | None = None,
        psi: ComplexArray | None = None,
        backend: str = "auto",
    ) -> float:
        """
        Compute fidelity with automatic backend selection.

        Args:
            params: Circuit parameters (provide this OR psi)
            psi: Pre-computed statevector (provide this OR params)
            backend: 'auto', 'custatevec', 'gpu', or 'cpu'

        Returns:
            Fidelity value
        """
        if psi is not None:
            # Direct computation from provided statevector
            return self._compute_fidelity_fast(psi)

        if params is None:
            raise ValueError("Must provide either params or psi")

        if backend == "auto":
            if self.config.use_custatevec and self._custatevec_evaluator is not None:
                backend = "custatevec"
            elif (
                self.config.use_gpu
                and self._gpu_evaluator is not None
                and self._gpu_evaluator.gpu_available
            ):
                backend = "gpu"
            else:
                backend = "cpu"

        if backend == "custatevec":
            return self._custatevec_evaluator.compute_fidelity(params)
        elif backend == "gpu":
            return self._gpu_evaluator.compute_fidelity(params)
        else:
            psi = self.get_statevector(params, backend="cpu")
            return self._compute_fidelity_fast(psi)

    def evaluate_population(
        self, population: NDArray[np.float64], backend: str = "auto"
    ) -> FloatArray:
        """
        Evaluate fidelities for population with automatic backend selection.

        Args:
            population: Array of shape (pop_size, n_params)
            backend: 'auto', 'multi_gpu', 'custatevec', 'gpu', or 'cpu'

        Returns:
            Array of fidelities
        """
        pop_size = len(population)

        if backend == "auto":
            # Priority: Multi-GPU > cuStateVec > GPU > CPU
            if self.config.use_multi_gpu and self._multi_gpu_evaluator is not None:
                backend = "multi_gpu"
            elif self.config.use_custatevec and self._custatevec_batch_evaluator is not None:
                backend = "custatevec"
            elif (
                self.config.use_gpu
                and self._gpu_evaluator is not None
                and self._gpu_evaluator.gpu_available
            ):
                backend = "gpu"
            else:
                backend = "cpu"

        if backend == "multi_gpu":
            # Use parallel multi-GPU evaluation
            fidelities = self._multi_gpu_evaluator.evaluate_batch_parallel(population)
        elif backend == "custatevec":
            fidelities = self._custatevec_batch_evaluator.evaluate_batch_chunked(population)
        elif backend == "gpu":
            batch_size = self.config.gpu_batch_size
            fidelities = np.zeros(pop_size)
            for start in range(0, pop_size, batch_size):
                end = min(start + batch_size, pop_size)
                fidelities[start:end] = self._gpu_evaluator.compute_fidelities_batched(
                    population[start:end]
                )
        else:
            # CPU path
            if self.config.parallel_gradients and self.config.n_workers > 1:
                fidelities = self._evaluate_population_parallel_cpu(population)
            else:
                fidelities = np.array(
                    [
                        self._compute_fidelity_fast(self.get_statevector(p, backend="cpu"))
                        for p in population
                    ]
                )

        # Update tracking
        self.n_evals += pop_size
        best_idx = np.argmax(fidelities)
        if fidelities[best_idx] > self.best_fidelity:
            self.best_fidelity = fidelities[best_idx]
            self.best_params = population[best_idx].copy()

        return fidelities

    def _evaluate_population_parallel_cpu(self, population: np.ndarray) -> np.ndarray:
        """CPU parallel population evaluation helper."""
        pop_size = len(population)
        chunk_size = max(1, pop_size // (self.config.n_workers * 2))

        if not hasattr(self, "_parallel_evaluator"):
            self._parallel_evaluator = ThreadSafeCircuitEvaluator(self.config, self.target)

        def evaluate_chunk(indices: list[int]) -> list[tuple[int, float]]:
            results = []
            for idx in indices:
                fid = self._parallel_evaluator.compute_fidelity(population[idx])
                results.append((idx, fid))
            return results

        indices = list(range(pop_size))
        chunks = [indices[i : i + chunk_size] for i in range(0, len(indices), chunk_size)]

        fidelities = np.zeros(pop_size)

        with ThreadPoolExecutor(max_workers=self.config.n_workers) as executor:
            chunk_results = list(executor.map(evaluate_chunk, chunks))

        for chunk_result in chunk_results:
            for idx, fid in chunk_result:
                fidelities[idx] = fid

        return fidelities

    def _compute_fidelity_fast(self, psi_circuit: ComplexArray) -> float:
        """Optimized fidelity using pre-conjugated target"""
        from .fidelity import compute_fidelity_fast

        return compute_fidelity_fast(self._target_conj, psi_circuit)

    def _compute_infidelity_direct(self, psi_circuit: ComplexArray) -> float:
        """Compute infidelity without catastrophic cancellation."""
        from .fidelity import compute_infidelity_direct

        return compute_infidelity_direct(self._target_conj, self.target, psi_circuit)

    def compute_gradient(self, params: np.ndarray, method: str = "auto") -> np.ndarray:
        """
        Unified gradient computation with automatic backend selection.

        Args:
            params: Current parameters
            method: 'auto', 'custatevec', 'gpu', 'parallel', or 'sequential'

        Returns:
            Gradient array (for minimizing -fidelity)
        """
        if method == "auto":
            # Priority: Multi-GPU > cuStateVec > GPU > Parallel CPU > Sequential CPU
            if self.config.use_multi_gpu and self._multi_gpu_evaluator is not None:
                method = "multi_gpu"
            elif self.config.use_custatevec and self._custatevec_batch_evaluator is not None:
                method = "custatevec"
            elif (
                self.config.use_gpu
                and self._gpu_evaluator is not None
                and self._gpu_evaluator.gpu_available
            ):
                method = "gpu"
            elif self.config.parallel_gradients and self.config.n_workers > 1:
                method = "parallel"
            else:
                method = "sequential"

        if method == "multi_gpu":
            return self._multi_gpu_evaluator.compute_gradient_parallel(params)
        elif method == "custatevec":
            return self._compute_gradient_custatevec_impl(params)
        elif method == "gpu":
            return self._compute_gradient_gpu_impl(params)
        elif method == "parallel":
            return self._compute_gradient_parallel_impl(params)
        else:
            return self._compute_gradient_sequential_impl(params)

    def compute_gradient_stochastic(
        self,
        params: np.ndarray,
        fraction: float = 0.5,
        rng: np.random.Generator = None,
    ) -> np.ndarray:
        """
        Stochastic parameter-shift gradient: sample a random subset of coordinates.

        Only computes gradient for k = max(1, int(n_params * fraction)) randomly
        chosen parameters. Unsampled components are set to 0.

        With Adam's momentum, the missing components are interpolated from history,
        providing convergence despite the sparse gradient signal.

        Args:
            params: Current parameters
            fraction: Fraction of parameters to sample (0, 1]
            rng: Optional random number generator for reproducibility

        Returns:
            Sparse gradient array (unsampled components are 0)
        """
        if rng is None:
            rng = np.random.default_rng()

        n_params = len(params)
        k = max(1, int(n_params * fraction))

        if k >= n_params:
            # Full gradient
            return self.compute_gradient(params)

        # Sample random subset of parameter indices
        sampled_indices = rng.choice(n_params, size=k, replace=False)

        gradient = np.zeros(n_params)
        shift = np.pi / 2

        for idx in sampled_indices:
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[idx] += shift
            params_minus[idx] -= shift

            psi_plus = self.get_statevector(params_plus)
            psi_minus = self.get_statevector(params_minus)

            f_plus = self._compute_fidelity_fast(psi_plus)
            f_minus = self._compute_fidelity_fast(psi_minus)

            gradient[idx] = (f_plus - f_minus) / 2.0

        # Return negative gradient (we minimize -fidelity), consistent with compute_gradient()
        return -gradient

    def compute_hessian_diagonal(self, params: np.ndarray) -> np.ndarray:
        """
        Compute the diagonal of the Hessian via second-order parameter-shift rule.

        Uses Mari et al. (PRA 2021) Eq. 13:
        d^2(-F)/dtheta_j^2 = -[F(theta+pi/2*e_j) - 2*F(theta) + F(theta-pi/2*e_j)] / 2

        Cost: 2*n_params + 1 evaluations (same shifted states as gradient, plus f(theta)).
        If gradient was already computed, only 1 extra evaluation needed.

        Returns:
            Diagonal Hessian of the -fidelity objective, shape (n_params,)
        """
        shift = np.pi / 2
        n_params = len(params)

        f_0 = self.compute_fidelity(params=params)

        f_plus = np.zeros(n_params)
        f_minus = np.zeros(n_params)

        for i in range(n_params):
            p_plus = params.copy()
            p_plus[i] += shift
            f_plus[i] = self.compute_fidelity(params=p_plus)

            p_minus = params.copy()
            p_minus[i] -= shift
            f_minus[i] = self.compute_fidelity(params=p_minus)

        # Hessian of fidelity: (f+ - 2*f0 + f-) / 2
        # Hessian of -fidelity (our objective): negate
        hess_diag = -(f_plus - 2 * f_0 + f_minus) / 2

        return hess_diag

    def newton_refinement_step(
        self,
        params: np.ndarray,
        lr: float = 0.5,
        epsilon: float = 1e-6,
    ) -> np.ndarray:
        """
        One Newton-like refinement step using diagonal Hessian preconditioning.

        delta = -lr * gradient / (|hess_diag| + epsilon)

        Args:
            params: Current parameters
            lr: Learning rate (< 1 for stability)
            epsilon: Regularization to prevent division by zero

        Returns:
            Updated parameters
        """
        gradient = self.compute_gradient(params)
        hess_diag = self.compute_hessian_diagonal(params)

        # Preconditioned step: scale each gradient component by inverse curvature
        step = lr * gradient / (np.abs(hess_diag) + epsilon)

        new_params = params - step
        return np.clip(new_params, -2 * np.pi, 2 * np.pi)

    def _compute_gradient_sequential_impl(self, params: np.ndarray) -> np.ndarray:
        """
        Compute gradient analytically via parameter-shift rule.
        For RY gates: ∂f/∂θ = (f(θ+π/2) - f(θ-π/2)) / 2

        This replaces finite-difference gradients (n+1 evals) with
        exact gradients (2n evals, but more accurate).
        """
        gradient = np.zeros(self.n_params)
        shift = np.pi / 2

        for i in range(self.n_params):
            # Forward shift: θ_i + π/2
            params_plus = params.copy()
            params_plus[i] += shift
            psi_plus = self.get_statevector(params_plus)
            fid_plus = self._compute_fidelity_fast(psi_plus)

            # Backward shift: θ_i - π/2
            params_minus = params.copy()
            params_minus[i] -= shift
            psi_minus = self.get_statevector(params_minus)
            fid_minus = self._compute_fidelity_fast(psi_minus)

            # Parameter-shift gradient formula
            gradient[i] = (fid_plus - fid_minus) / 2

        # Return negative gradient (we minimize -fidelity)
        return -gradient

    def compute_gradient_parallel(self, params: np.ndarray) -> np.ndarray:
        return self.compute_gradient(params, method="parallel")

    def _compute_gradient_parallel_impl(self, params: np.ndarray) -> np.ndarray:
        """
        Chunked parallel gradient computation for better load balancing.

        Groups parameters into chunks to reduce thread overhead.
        More efficient when n_params >> n_workers.
        """
        if not self.config.parallel_gradients or self.config.n_workers <= 1:
            return self._compute_gradient_sequential_impl(params)

        shift = np.pi / 2
        n_workers = self.config.n_workers
        chunk_size = self.config.gradient_chunk_size

        # Create thread-safe evaluator if not exists
        if not hasattr(self, "_parallel_evaluator"):
            self._parallel_evaluator = ThreadSafeCircuitEvaluator(self.config, self.target)

        def evaluate_chunk(param_indices: list[int]) -> list[tuple[int, float]]:
            """Evaluate gradient for a chunk of parameters"""
            results = []
            for idx in param_indices:
                # Forward shift
                params_plus = params.copy()
                params_plus[idx] += shift
                fid_plus = self._parallel_evaluator.compute_fidelity(params_plus)

                # Backward shift
                params_minus = params.copy()
                params_minus[idx] -= shift
                fid_minus = self._parallel_evaluator.compute_fidelity(params_minus)

                grad_i = (fid_plus - fid_minus) / 2
                results.append((idx, grad_i))

            return results

        # Create chunks
        indices = list(range(self.n_params))
        chunks = [indices[i : i + chunk_size] for i in range(0, len(indices), chunk_size)]

        # Parallel execution
        gradient = np.zeros(self.n_params)

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            chunk_results = list(executor.map(evaluate_chunk, chunks))

        # Collect results
        for chunk_result in chunk_results:
            for idx, grad_val in chunk_result:
                gradient[idx] = grad_val

        return -gradient

    def _compute_gradient_custatevec_impl(self, params: np.ndarray) -> np.ndarray:
        """
        Compute gradient using cuStateVec batched evaluation.

        This is the fastest gradient computation method.
        """
        if self._custatevec_batch_evaluator is not None:
            return self._custatevec_batch_evaluator.compute_gradient_batched(params)
        elif self._gpu_evaluator is not None and self._gpu_evaluator.gpu_available:
            return self._compute_gradient_gpu_impl(params)
        else:
            return self._compute_gradient_sequential_impl(params)

    def run_optimization(
        self, pipeline: OptimizationPipeline = None, initial_params: np.ndarray = None, **kwargs
    ) -> dict:
        """
        Unified optimization entry point.

        Replaces: optimize(), optimize_ultra_precision(), optimize_hybrid()

        Args:
            pipeline: OptimizationPipeline config (or use kwargs for quick setup)
            initial_params: Starting parameters (None = auto-select)
            **kwargs: Override pipeline settings

        Returns:
            Results dictionary with optimal_params, fidelity, etc.
        """
        # Build pipeline config
        if pipeline is None:
            pipeline = OptimizationPipeline(**kwargs)
        else:
            # Apply any kwargs overrides
            for key, value in kwargs.items():
                if hasattr(pipeline, key):
                    setattr(pipeline, key, value)

        start_time = time.time()

        if pipeline.verbose:
            print(f"\n{'=' * 80}")
            print(f"OPTIMIZATION PIPELINE: {pipeline.mode.upper()}")
            print(f"{'=' * 80}")
            print(f"  Target fidelity:   {pipeline.target_fidelity}")
            print(f"  Target infidelity: {pipeline.target_infidelity:.0e}")
            print(f"  Max time:          {pipeline.max_total_time:.0f}s")

        current_params = initial_params

        # === STAGE: Initialization Search ===
        if pipeline.use_init_search:
            current_params = self._pipeline_init_search(pipeline, current_params, start_time)
        elif current_params is None:
            current_params = self.get_initial_params("smart")

        # === STAGE: Adam Exploration ===
        if pipeline.use_adam_stage:
            elapsed = time.time() - start_time
            time_limit = pipeline.max_total_time * pipeline.adam_time_fraction
            if self.best_fidelity < pipeline.target_fidelity and elapsed < time_limit:
                current_params = self._pipeline_adam_stage(pipeline, current_params, start_time)

        # === STAGE: Basin Hopping (if stuck) ===
        if pipeline.use_basin_hopping:
            elapsed = time.time() - start_time
            if (
                self.best_fidelity < pipeline.basin_hopping_threshold
                and elapsed < pipeline.max_total_time * 0.8
            ):
                current_params = self._pipeline_basin_hopping(pipeline, current_params, start_time)

        # === STAGE: L-BFGS-B Refinement ===
        if pipeline.use_lbfgs_refinement:
            elapsed = time.time() - start_time
            time_limit = pipeline.max_total_time * pipeline.lbfgs_time_fraction
            if self.best_fidelity < pipeline.target_fidelity and elapsed < time_limit:
                current_params = self._pipeline_lbfgs_refinement(
                    pipeline, current_params, start_time
                )

        # === STAGE: Fine Tuning ===
        if pipeline.use_fine_tuning:
            elapsed = time.time() - start_time
            if (
                self.best_fidelity > pipeline.fine_tuning_threshold
                and self.best_fidelity < pipeline.target_fidelity
                and elapsed < pipeline.max_total_time
            ):
                current_params = self._pipeline_fine_tuning(pipeline, current_params, start_time)

        # === Final Results ===
        return self._pipeline_finalize(pipeline, start_time)

    def run_pipeline(
        self, pipeline: pipeline_mod.Pipeline = None, initial_params: np.ndarray = None
    ) -> dict:
        """
        Execute a composable optimization pipeline.

        Each stage runs in order, receiving the best parameters from the
        previous stage. Stops early if target_fidelity is reached or
        max_total_time is exceeded.

        Args:
            pipeline: A Pipeline object with ordered stages.
                      If None, uses Pipeline.standard().
            initial_params: Starting parameters (None = determined by first stage)

        Returns:
            Results dictionary with optimal_params, fidelity, infidelity, time, etc.
        """
        from .pipeline import (
            LBFGSB,
            SPSA,
            Adam,
            BasinHopping,
            GrowCircuit,
            InitSearch,
            NaturalGradient,
            Newton,
            Pipeline,
        )

        if pipeline is None:
            pipeline = Pipeline.standard()

        start_time = time.time()
        current_params = initial_params

        if pipeline.verbose:
            print(f"\n{'=' * 80}")
            print("COMPOSABLE OPTIMIZATION PIPELINE")
            print(f"{'=' * 80}")
            print(pipeline.summary())
            print(f"{'=' * 80}\n")

        for i, stage in enumerate(pipeline.stages):
            elapsed = time.time() - start_time

            # Early stopping: target reached
            if self.best_fidelity >= pipeline.target_fidelity:
                if pipeline.verbose:
                    print(f"\n  Target fidelity reached after stage {i}. Stopping.")
                break

            # Early stopping: time limit
            if elapsed >= pipeline.max_total_time:
                if pipeline.verbose:
                    print(f"\n  Time limit reached ({elapsed:.0f}s). Stopping.")
                break

            stage_name = f"[{i + 1}/{len(pipeline.stages)}] {stage.describe()}"
            if pipeline.verbose:
                print(f"\n{'=' * 60}")
                print(f"STAGE: {stage_name}")
                print(f"{'=' * 60}")

            remaining_time = pipeline.max_total_time - elapsed

            # ----------------------------------------------------------
            # Dispatch by stage type
            # ----------------------------------------------------------
            if isinstance(stage, InitSearch):
                best_fid = 0
                best_p = current_params
                for j, strategy in enumerate(stage.strategies):
                    np.random.seed(42 + j)
                    p = self.get_initial_params(strategy)
                    fid = self.compute_fidelity(params=p)
                    if pipeline.verbose:
                        print(f"  {strategy:20s}: F = {fid:.8f}")
                    if fid > best_fid:
                        best_fid = fid
                        best_p = p.copy()
                if current_params is None or best_fid > self.best_fidelity:
                    current_params = best_p

            elif isinstance(stage, Adam):
                if current_params is None:
                    current_params = self.get_initial_params("smart")
                max_t = stage.max_time if stage.max_time else remaining_time * 0.8
                old_fraction = self.config.gradient_sample_fraction
                if stage.gradient_fraction < 1.0:
                    self.config.gradient_sample_fraction = stage.gradient_fraction
                result = self.optimize_adam(
                    current_params,
                    max_steps=stage.max_steps,
                    lr=stage.lr,
                    max_time=max_t,
                    convergence_window=stage.convergence_window,
                    convergence_threshold=stage.convergence_threshold,
                )
                self.config.gradient_sample_fraction = old_fraction
                current_params = result["params"]

            elif isinstance(stage, SPSA):
                if current_params is None:
                    current_params = self.get_initial_params("smart")
                max_t = stage.max_time if stage.max_time else remaining_time * 0.8
                result = self.optimize_spsa(
                    current_params,
                    max_steps=stage.max_steps,
                    a=stage.a,
                    c=stage.c,
                    A=stage.A,
                    n_avg=stage.n_avg,
                    max_time=max_t,
                )
                current_params = result["params"]

            elif isinstance(stage, NaturalGradient):
                if current_params is None:
                    current_params = self.get_initial_params("smart")
                max_t = stage.max_time if stage.max_time else remaining_time * 0.8
                result = self.optimize_natural_gradient(
                    current_params,
                    max_steps=stage.max_steps,
                    lr=stage.lr,
                    regularization=stage.regularization,
                    max_time=max_t,
                )
                current_params = result["params"]

            elif isinstance(stage, LBFGSB):
                if current_params is None:
                    current_params = (
                        self.best_params
                        if self.best_params is not None
                        else self.get_initial_params("smart")
                    )
                for tol in stage.tolerances:
                    if self.best_fidelity >= pipeline.target_fidelity:
                        break
                    if time.time() - start_time > pipeline.max_total_time * 0.95:
                        break
                    if stage.use_log_objective and self.best_fidelity > stage.log_threshold:
                        if pipeline.verbose:
                            print(f"  L-BFGS-B (tol={tol:.0e}, log objective)...")
                        lbfgs_opts = {
                            "maxiter": stage.max_iter,
                            "ftol": tol,
                            "gtol": tol,
                            "disp": False,
                        }
                        if self.config.high_precision:
                            lbfgs_opts["maxcor"] = 30
                            lbfgs_opts["maxls"] = 40
                        result = minimize(
                            self.objective_and_gradient_log_infidelity,
                            self.best_params,
                            method="L-BFGS-B",
                            jac=True,
                            bounds=[(-2 * np.pi, 2 * np.pi)] * self.n_params,
                            options=lbfgs_opts,
                        )
                        psi = self.get_statevector(result.x)
                        fid = self._compute_fidelity_fast(psi)
                        if fid > self.best_fidelity:
                            self.best_fidelity = fid
                            self.best_params = result.x.copy()
                    else:
                        if pipeline.verbose:
                            print(f"  L-BFGS-B (tol={tol:.0e})...")
                        self.config.gtol = tol
                        self.optimize_stage(
                            self.best_params,
                            f"L-BFGS-B (tol={tol:.0e})",
                            max_iter=stage.max_iter,
                            tolerance=tol,
                        )
                    if pipeline.verbose:
                        print(f"    F = {self.best_fidelity:.15f}")
                current_params = self.best_params

            elif isinstance(stage, BasinHopping):
                if current_params is None:
                    current_params = (
                        self.best_params
                        if self.best_params is not None
                        else self.get_initial_params("smart")
                    )
                self.optimize_basin_hopping(
                    current_params,
                    n_iterations=stage.n_iterations,
                    temperature=stage.temperature,
                    step_size=stage.step_size,
                )
                current_params = self.best_params

            elif isinstance(stage, Newton):
                if current_params is None:
                    current_params = (
                        self.best_params
                        if self.best_params is not None
                        else self.get_initial_params("smart")
                    )
                for step in range(stage.max_steps):
                    current_params = self.newton_refinement_step(
                        current_params,
                        lr=stage.lr,
                        epsilon=stage.epsilon,
                    )
                    fid = self.compute_fidelity(params=current_params)
                    if fid > self.best_fidelity:
                        self.best_fidelity = fid
                        self.best_params = current_params.copy()
                    if pipeline.verbose and step % 10 == 0:
                        print(f"  Newton step {step}: F={fid:.12f}")
                current_params = self.best_params

            elif isinstance(stage, GrowCircuit):
                if current_params is None:
                    current_params = (
                        self.best_params
                        if self.best_params is not None
                        else self.get_initial_params("smart")
                    )
                current_params = self.grow_circuit(current_params)
                if pipeline.verbose:
                    print(f"  Circuit grown to depth {self.ansatz.depth}, n_params={self.n_params}")

            if pipeline.verbose and self.best_fidelity > 0:
                print(
                    f"  -> Best F = {self.best_fidelity:.12f} (1-F = {1 - self.best_fidelity:.3e})"
                )

        # Finalize
        total_time = time.time() - start_time
        if self.best_params is not None:
            final_psi = self.get_statevector(self.best_params)
            final_fidelity = self._compute_fidelity_fast(final_psi)
            final_infidelity = self._compute_infidelity_direct(final_psi)
            circuit_stats = self.compute_statistics(final_psi)
        else:
            final_psi = None
            final_fidelity = 0.0
            final_infidelity = 1.0
            circuit_stats = {"mean": 0.0, "std": 0.0, "variance": 0.0}

        results = {
            "optimal_params": self.best_params,
            "fidelity": final_fidelity,
            "infidelity": final_infidelity,
            "circuit_mean": circuit_stats["mean"],
            "circuit_std": circuit_stats["std"],
            "target_mean": self.config.x0,
            "target_std": self.config.sigma,
            "time": total_time,
            "n_evaluations": self.n_evals,
            "success": final_fidelity >= pipeline.target_fidelity,
            "final_statevector": final_psi,
            "circuit_stats": circuit_stats,
            "n_stages_completed": min(i + 1, len(pipeline.stages)),
        }

        if pipeline.verbose:
            print(f"\n{'=' * 80}")
            print("PIPELINE COMPLETE")
            print(f"{'=' * 80}")
            print(f"  Final fidelity:      {final_fidelity:.15f}")
            print(f"  Infidelity:          {final_infidelity:.3e}")
            print(f"  Target:              {pipeline.target_infidelity:.3e}")
            print(f"  Success:             {'Yes' if results['success'] else 'No'}")
            print(f"  Time:                {total_time:.1f}s")
            print(f"  Stages completed:    {results['n_stages_completed']}/{len(pipeline.stages)}")
            print(f"  Total evaluations:   {self.n_evals}")

        return results

    def _pipeline_init_search(
        self, pipeline: OptimizationPipeline, initial_params: np.ndarray, _start_time: float
    ) -> np.ndarray:
        """Pipeline stage: initialization search."""
        if pipeline.verbose:
            print(f"\n{'=' * 60}")
            print("STAGE: Initialization Search")
            print("=" * 60)

        best_init_fid = 0
        best_init_params = initial_params
        best_init_strategy = None

        for i, strategy in enumerate(pipeline.init_strategies):
            np.random.seed(42 + i)
            params = self.get_initial_params(strategy)

            # Use fastest available evaluator
            if self.config.use_custatevec and self._custatevec_evaluator is not None:
                fid = self._custatevec_evaluator.compute_fidelity(params)
            elif self._gpu_evaluator is not None and self._gpu_evaluator.gpu_available:
                fid = self._gpu_evaluator.compute_fidelity(params)
            else:
                psi = self.get_statevector(params)
                fid = self._compute_fidelity_fast(psi)

            if pipeline.verbose:
                print(f"  {strategy:20s}: F = {fid:.8f}")

            if fid > best_init_fid:
                best_init_fid = fid
                best_init_params = params.copy()
                best_init_strategy = strategy

        if pipeline.verbose:
            print(f"\nBest initialization: '{best_init_strategy}' with F = {best_init_fid:.8f}")

        return best_init_params if initial_params is None else initial_params

    def _pipeline_adam_stage(
        self, pipeline: OptimizationPipeline, current_params: np.ndarray, start_time: float
    ) -> np.ndarray:
        """Pipeline stage: Adam exploration."""
        if pipeline.verbose:
            print(f"\n{'=' * 60}")
            print("STAGE: Adam Exploration")
            print("=" * 60)

        # Use explicit max_time if provided, otherwise calculate from fraction
        if pipeline.adam_max_time is not None:
            adam_time_budget = pipeline.adam_max_time
        else:
            elapsed = time.time() - start_time
            adam_time_budget = pipeline.max_total_time * pipeline.adam_time_fraction - elapsed
            adam_time_budget = max(1.0, adam_time_budget)

        if pipeline.verbose:
            print(f"  Time budget: {adam_time_budget:.0f}s")

        self.optimize_adam(
            current_params,
            max_steps=pipeline.adam_max_steps,
            lr=pipeline.adam_lr,
            max_time=adam_time_budget,
        )

        if pipeline.verbose:
            print(f"\nAfter Adam: F = {self.best_fidelity:.12f}")
            print(f"  Infidelity: {1 - self.best_fidelity:.3e}")

        return self.best_params

    def _pipeline_basin_hopping(
        self, pipeline: OptimizationPipeline, _current_params: np.ndarray, _start_time: float
    ) -> np.ndarray:
        """Pipeline stage: Basin hopping for escaping local minima."""
        if pipeline.verbose:
            print(f"\n{'=' * 60}")
            print("STAGE: Basin Hopping (escaping local minimum)")
            print("=" * 60)

        self.optimize_basin_hopping(
            self.best_params,
            n_iterations=pipeline.basin_hopping_iterations,
            temperature=0.5,
            local_optimizer="lbfgs",
        )

        return self.best_params

    def _pipeline_lbfgs_refinement(
        self, pipeline: OptimizationPipeline, current_params: np.ndarray, start_time: float
    ) -> np.ndarray:
        """Pipeline stage: L-BFGS-B high-precision refinement."""
        if pipeline.verbose:
            print(f"\n{'=' * 60}")
            print("STAGE: L-BFGS-B High-Precision Refinement")
            print("=" * 60)

        if self.best_params is None or len(self.best_params) != self.n_params:
            self.best_params = current_params

        for tol in pipeline.lbfgs_tolerances:
            if self.best_fidelity >= pipeline.target_fidelity:
                break

            elapsed = time.time() - start_time
            if elapsed > pipeline.max_total_time * 0.9:
                break

            if pipeline.verbose:
                print(f"\n  Refinement pass (tol={tol:.0e})...")

            self.config.gtol = tol

            # Use log-infidelity objective when enabled and fidelity is high enough
            if pipeline.use_log_objective and self.best_fidelity > pipeline.log_objective_threshold:
                if pipeline.verbose:
                    print("    Using log-infidelity objective")
                lbfgs_options = {
                    "maxiter": 3000,
                    "maxfun": self.config.max_fun,
                    "ftol": tol,
                    "gtol": tol,
                    "disp": self.config.verbose,
                }
                if self.config.high_precision:
                    lbfgs_options["maxcor"] = 30
                    lbfgs_options["maxls"] = 40
                result = minimize(
                    self.objective_and_gradient_log_infidelity,
                    self.best_params,
                    method="L-BFGS-B",
                    jac=True,
                    bounds=[(-2 * np.pi, 2 * np.pi)] * self.n_params,
                    options=lbfgs_options,
                )
                # Update best from result
                psi = self.get_statevector(result.x)
                fidelity = self._compute_fidelity_fast(psi)
                if fidelity > self.best_fidelity:
                    self.best_fidelity = fidelity
                    self.best_params = result.x.copy()
            else:
                self.optimize_stage(
                    self.best_params,
                    f"Refinement (tol={tol:.0e})",
                    max_iter=3000,
                    tolerance=tol,
                )

            if pipeline.verbose:
                print(f"    F = {self.best_fidelity:.15f}")
                print(f"    Infidelity = {1 - self.best_fidelity:.3e}")

        return self.best_params

    def _pipeline_fine_tuning(
        self, pipeline: OptimizationPipeline, _current_params: np.ndarray, _start_time: float
    ) -> np.ndarray:
        """Pipeline stage: Ultra-fine tuning for near-target fidelities."""
        if pipeline.verbose:
            print(f"\n{'=' * 60}")
            print("STAGE: Ultra-Fine Tuning")
            print("=" * 60)

        # Small Adam steps
        self.optimize_adam(
            self.best_params,
            max_steps=1000,
            lr=0.0001,
            convergence_threshold=pipeline.target_infidelity / 10,
        )

        # Final polish
        self.config.gtol = 1e-15
        self.optimize_stage(self.best_params, "Final Polish", max_iter=5000, tolerance=1e-15)

        return self.best_params

    def _pipeline_finalize(self, pipeline: OptimizationPipeline, start_time: float) -> dict:
        """Finalize pipeline and return results."""
        total_time = time.time() - start_time

        # Get statevector for plotting - must match target ordering
        if self.config.use_custatevec and self._custatevec_evaluator is not None:
            final_psi = self._custatevec_evaluator.get_statevector_qiskit_order(self.best_params)
            final_fidelity = self._custatevec_evaluator.compute_fidelity(self.best_params)
        elif (
            self.config.use_gpu
            and self._gpu_evaluator is not None
            and self._gpu_evaluator.gpu_available
        ):
            final_psi = self._gpu_evaluator.get_statevector(self.best_params)
            final_fidelity = self._gpu_evaluator.compute_fidelity(self.best_params)
        else:
            final_psi = self.get_statevector(self.best_params)
            final_fidelity = self._compute_fidelity_fast(final_psi)

        circuit_stats = self.compute_statistics(final_psi)

        results = {
            "optimal_params": self.best_params,
            "fidelity": final_fidelity,
            "infidelity": self._compute_infidelity_direct(final_psi),
            "circuit_mean": circuit_stats["mean"],
            "circuit_std": circuit_stats["std"],
            "target_mean": self.config.x0,
            "target_std": self.config.sigma,
            "mean_error": abs(circuit_stats["mean"] - self.config.x0),
            "std_error": abs(circuit_stats["std"] - self.config.sigma),
            "relative_std_error": abs(circuit_stats["std"] - self.config.sigma) / self.config.sigma,
            "time": total_time,
            "n_evaluations": self.n_evals,
            "success": final_fidelity >= pipeline.target_fidelity,
            "final_statevector": final_psi,
            "circuit_stats": circuit_stats,
        }

        if pipeline.verbose:
            print(f"\n{'=' * 80}")
            print("OPTIMIZATION COMPLETE")
            print(f"{'=' * 80}")
            print(f"Final fidelity:      {final_fidelity:.15f}")
            print(f"Infidelity:          {1 - final_fidelity:.3e}")
            print(f"Target infidelity:   {pipeline.target_infidelity:.3e}")
            print(f"Success:             {' ' if results['success'] else ''}")
            print(f"Circuit σ:           {circuit_stats['std']:.10f}")
            print(f"Target σ:            {self.config.sigma:.10f}")
            print(f"σ relative error:    {results['relative_std_error'] * 100:.6f}%")
            print(f"Total time:          {total_time:.1f}s")
            print(f"Total evaluations:   {self.n_evals}")

        return results

    def objective_and_gradient(self, params: np.ndarray) -> tuple:
        """
        Combined objective and gradient computation for scipy.
        Using jac=True in minimize() avoids redundant evaluations.
        """
        # Compute objective (also updates tracking)
        obj = self.objective(params)
        grad = self.compute_gradient(params)

        # Store for diagnostics
        self.last_gradient = grad
        grad_norm = np.linalg.norm(grad)
        if self.n_evals % self._log_interval == 0:
            self.history["gradient_norm"].append(grad_norm)

        return obj, grad

    def objective(self, params: np.ndarray) -> float:
        """Objective function for minimization"""
        self.n_evals += 1

        # Get circuit output
        psi_circuit = self.get_statevector(params)

        # Compute fidelity with high precision
        fidelity = self._compute_fidelity_fast(psi_circuit)

        # Track best with high precision comparison
        if fidelity > self.best_fidelity:
            self.best_fidelity = fidelity
            self.best_params = params.copy()

        if self.n_evals % 10 == 0:  # Only store every 10th evaluation
            self.history["fidelity"].append(fidelity)
            self.history["iteration"].append(self.n_evals)

        # Progress updates - show more precision
        if self.config.verbose and self.n_evals % self._log_interval == 0:
            current_time = time.time()
            if current_time - self._last_log_time >= self._min_log_interval_sec:
                print(f"Eval {self.n_evals:6d}: F={fidelity:.10f} (best={self.best_fidelity:.10f})")
                self._last_log_time = current_time
                if self.config.verbose and self.n_evals % 100 == 0:
                    print(
                        f"Eval {self.n_evals:6d}: Fidelity = {fidelity:.12f} (best = {self.best_fidelity:.12f})"
                    )

        # Return negative for minimization
        return -fidelity

    def objective_log_infidelity(self, params: np.ndarray) -> float:
        """
        Log-infidelity objective: log(1-F).

        This objective has well-scaled gradients even at extreme fidelities:
        d/dtheta log(1-F) = -1/(1-F) * dF/dtheta

        The 1/(1-F) amplification keeps gradients informative as F -> 1.
        Uses np.log for numerical stability.
        """
        self.n_evals += 1
        psi_circuit = self.get_statevector(params)
        infidelity = self._compute_infidelity_direct(psi_circuit)

        # Track best
        fidelity = 1.0 - infidelity
        if fidelity > self.best_fidelity:
            self.best_fidelity = fidelity
            self.best_params = params.copy()

        if infidelity <= 0.0:
            return -40.0  # log(~1e-17), floor to avoid -inf

        return np.log(infidelity)  # = log(1-F), negative and decreasing is good

    def objective_and_gradient_log_infidelity(self, params: np.ndarray) -> tuple:
        """
        Compute log-infidelity and its gradient simultaneously.

        Gradient: d/dtheta log(1-F) = -1/(1-F) * dF/dtheta
        where dF/dtheta comes from the parameter-shift rule.
        """
        # First compute infidelity at current params
        psi = self.get_statevector(params)
        infidelity = self._compute_infidelity_direct(psi)
        fidelity = 1.0 - infidelity

        # Track best
        if fidelity > self.best_fidelity:
            self.best_fidelity = fidelity
            self.best_params = params.copy()

        if infidelity <= 0.0:
            return -40.0, np.zeros_like(params)

        log_infidelity = np.log(infidelity)

        # Compute standard fidelity gradient via parameter-shift
        fidelity_gradient = self.compute_gradient(params)

        # Chain rule: d/dtheta log(1-F) = -1/(1-F) * dF/dtheta
        # compute_gradient returns -dF/dtheta (gradient of -F for minimization),
        # so: d/dtheta log(1-F) = (1/(1-F)) * (-dF/dtheta) = amplification * fidelity_gradient
        # Clip the amplification to prevent numerical explosion
        amplification = min(1.0 / infidelity, 1e12)
        log_gradient = amplification * fidelity_gradient

        return log_infidelity, log_gradient

    def grow_circuit(self, current_params: np.ndarray) -> np.ndarray:
        """Grow the circuit by one layer, preserving existing parameters."""
        from .ansatz import DefaultAnsatz

        if not isinstance(self.ansatz, DefaultAnsatz):
            raise TypeError(
                f"Adaptive depth only supports DefaultAnsatz. Got {type(self.ansatz).__name__}"
            )

        old_depth = self.ansatz.depth
        new_depth = old_depth + 1
        n = self.config.n_qubits
        entanglement = getattr(self.ansatz, "_entanglement", "linear")

        self.ansatz = DefaultAnsatz(n, depth=new_depth, entanglement=entanglement)
        self.n_params = self.ansatz.n_params

        self.param_vector = ParameterVector("theta", self.n_params)
        self.circuit = self.ansatz(self.param_vector, n, **(self.config.ansatz_kwargs or {}))
        self._circuit_transpiled = transpile(
            self.circuit, basis_gates=["ry", "rz", "cx", "x"], optimization_level=1
        )
        self._param_list = list(self.param_vector)

        new_layer_params = np.random.randn(n) * 0.01
        return np.concatenate([current_params, new_layer_params])

    def get_initial_params(self, strategy="smart", scale_factor=1.0):
        """
        Generate initial parameters with physics-informed strategies.

        Strategies:
        - 'smart': Physics-informed initialization based on target Gaussian
        - 'gaussian_product': Approximate Gaussian as product state
        - 'random': Uniform random in [-π, π]
        - 'small_random': Small perturbations (for refinement)
        - 'zero': All zeros
        """
        n = self.config.n_qubits
        params = np.zeros(self.n_params)

        if strategy == "smart":
            # Physics-informed initialization
            # Key insight: For Gaussian, we want smooth amplitude distribution

            # Compute effective width in grid units
            sigma_grid = self.config.sigma / self.config.delta_x
            width_ratio = sigma_grid / self.config.n_states

            # First layer: set up approximate Gaussian envelope
            # Smaller angles for narrower Gaussians (less superposition needed)
            base_angle = np.pi * min(0.3, width_ratio * 2) * scale_factor

            for i in range(n):
                # Higher-order qubits (larger 2^i) need smaller rotations
                # for narrow Gaussians to avoid high-frequency components
                bit_weight = 2**i / self.config.n_states
                damping = np.exp(-bit_weight / (4 * width_ratio + 0.1))
                params[i] = base_angle * damping * (1 + 0.1 * np.random.randn())

            # Subsequent layers: entangling layers need small initial values
            # to allow optimization to find correlations
            remaining_params = self.n_params - n
            if remaining_params > 0:
                params[n:] = 0.1 * scale_factor * np.random.randn(remaining_params)

        elif strategy == "gaussian_product":
            # Approximate Gaussian as product state (no entanglement initially)
            # Good starting point that optimization can refine

            # For each computational basis state |x⟩, we want amplitude ~ exp(-x²/2σ²)
            # With product state, amplitude of |x⟩ = ∏ᵢ amplitude of qubit i

            sigma_grid = self.config.sigma / self.config.delta_x

            for i in range(n):
                # Contribution of qubit i to position
                pos_contribution = 2**i - self.config.n_states / 2

                # Desired probability for this qubit being |1⟩
                # Based on Gaussian weight at this position contribution
                gauss_weight = np.exp(-(pos_contribution**2) / (2 * sigma_grid**2 * n))

                # RY(θ)|0⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩
                # P(|1⟩) = sin²(θ/2), so θ = 2*arcsin(sqrt(p))
                prob_one = np.clip(gauss_weight, 0.01, 0.99)
                params[i] = 2 * np.arcsin(np.sqrt(prob_one)) * scale_factor

            # Small random for entangling layers
            params[n:] = 0.05 * scale_factor * np.random.randn(self.n_params - n)

        elif strategy == "random":
            params = np.random.uniform(-np.pi, np.pi, self.n_params)

        elif strategy == "small_random":
            # Small perturbations for refinement from current best
            params = scale_factor * np.random.randn(self.n_params)

        elif strategy == "mps":
            from .mps_init import mps_initial_params

            params = mps_initial_params(self.target, self.config.n_qubits)
            # Ensure correct length
            if len(params) != self.n_params:
                params = np.resize(params, self.n_params)

        elif strategy == "perturb_best":
            # Perturb from current best (if available)
            if self.best_params is not None:
                params = self.best_params + scale_factor * 0.1 * np.random.randn(self.n_params)
            else:
                params = self.get_initial_params("smart", scale_factor)

        else:  # 'zero'
            pass  # Already zeros

        return params

    def cleanup(self) -> None:
        """Release all GPU resources."""
        if hasattr(self, "_multi_gpu_evaluator") and self._multi_gpu_evaluator is not None:
            self._multi_gpu_evaluator.cleanup()
            self._multi_gpu_evaluator = None

        if hasattr(self, "_custatevec_evaluator") and self._custatevec_evaluator is not None:
            self._custatevec_evaluator.cleanup()
            self._custatevec_evaluator = None

        if (
            hasattr(self, "_custatevec_batch_evaluator")
            and self._custatevec_batch_evaluator is not None
        ):
            self._custatevec_batch_evaluator.cleanup()
            self._custatevec_batch_evaluator = None

    def compute_statistics(self, psi: np.ndarray) -> dict:
        """Compute wavefunction statistics with high precision"""
        x = self.positions
        dx = self.config.delta_x

        # Probability density with high precision
        prob = np.abs(psi) ** 2
        prob_sum = np.sum(prob) * dx
        prob = prob / prob_sum

        # Moments with high precision
        mean_x = np.sum(x * prob) * dx
        variance = np.sum((x - mean_x) ** 2 * prob) * dx
        std_x = np.sqrt(max(variance, 0))  # Ensure non-negative

        return {"mean": mean_x, "std": std_x, "variance": variance}

    def _compute_gradient_gpu_impl(self, params: np.ndarray) -> np.ndarray:
        """
        Compute gradient using batched GPU evaluation.

        Instead of 2*n_params individual calls, we make:
        - 1 call with 2*n_params circuits (all shifts at once)

        This is much faster on GPU due to parallelism.
        """
        if self._gpu_evaluator is None or not self._gpu_evaluator.gpu_available:
            # Fall back to sequential analytic gradient
            return self._compute_gradient_sequential_impl(params)

        shift = np.pi / 2
        n_params = self.n_params

        # Build all shifted parameter sets at once (vectorized)
        # Shape: (2 * n_params, n_params)
        params_shifted = np.tile(params, (2 * n_params, 1))
        idx = np.arange(n_params)
        params_shifted[2 * idx, idx] += shift
        params_shifted[2 * idx + 1, idx] -= shift

        # Single batched GPU call for all shifts
        fidelities = self._gpu_evaluator.compute_fidelities_batched(params_shifted)

        # Compute gradients from shift results (vectorized)
        gradient = (fidelities[0::2] - fidelities[1::2]) / 2

        # Return negative gradient (we minimize -fidelity)
        return -gradient

    def optimize_stage(
        self, initial_params: np.ndarray, stage_name: str, max_iter: int, tolerance: float
    ) -> dict:
        """Run a single optimization stage"""
        print(f"\n{stage_name}...")
        print(f"  Max iterations: {max_iter}")
        print(f"  Tolerance: {tolerance:.2e}")

        if self.config.method == "differential_evolution":
            bounds = [(-2 * np.pi, 2 * np.pi)] * self.n_params
            result = differential_evolution(
                self.objective,
                bounds,
                maxiter=max_iter // 15,
                tol=tolerance,
                disp=self.config.verbose,
                polish=True,
                workers=1,
                atol=tolerance / 10,
            )
        else:
            # High-precision optimization options
            options = {
                "maxiter": max_iter,
                "maxfun": self.config.max_fun,
                "ftol": tolerance,
                "gtol": self.config.gtol,
                "disp": self.config.verbose,
            }

            # For very high precision, use tighter convergence
            if self.config.high_precision:
                options["maxcor"] = 30  # More corrections with exact gradients
                options["maxls"] = 40  # Line search steps

            # Use analytic gradients if enabled
            if getattr(self.config, "use_analytic_gradients", True):
                result = minimize(
                    self.objective_and_gradient,
                    initial_params,
                    method=self.config.method,
                    jac=True,  # We provide gradients
                    bounds=[(-2 * np.pi, 2 * np.pi)] * self.n_params,
                    options=options,
                )
            else:
                result = minimize(
                    self.objective,
                    initial_params,
                    method=self.config.method,
                    bounds=[(-2 * np.pi, 2 * np.pi)] * self.n_params,
                    options=options,
                )

        return result

    def optimize_adam(
        self,
        initial_params: np.ndarray,
        max_steps: int = 2000,
        lr: float = 0.02,
        max_time: float = None,
        convergence_window: int = 100,
        convergence_threshold: float = 1e-8,
        verbose_interval: int = 100,
    ) -> dict:
        """
        Adam optimization with parameter-shift gradients.

        Effective for escaping local minima and plateaus where
        L-BFGS-B gets stuck.

        Args:
            initial_params: Starting parameters
            max_steps: Maximum Adam steps
            lr: Initial learning rate
            convergence_window: Steps to check for convergence
            convergence_threshold: Minimum improvement to continue
            verbose_interval: Print progress every N steps

        Returns:
            Dictionary with optimization results
        """
        start_time = time.time()
        print(f"\nAdam Optimization (lr={lr}, max_steps={max_steps})")
        print("-" * 50)

        params = initial_params.copy()
        optimizer = AdamWithRestarts(self.n_params, lr_max=lr, lr_min=lr / 50, restart_period=200)

        # Barren plateau detection
        bp_detector = BarrenPlateauDetector(self.n_params)

        # Tracking
        fidelity_history = []
        best_fidelity = 0
        best_params = params.copy()

        start_time = time.time()

        for step in range(max_steps):
            if max_time is not None and (time.time() - start_time) > max_time:
                print(f"  Time limit reached at step {step}")
                break
            # Compute fidelity and gradient
            # Priority: cuStateVec > Aer GPU > CPU
            if self.config.use_custatevec and self._custatevec_evaluator is not None:
                fidelity = self._custatevec_evaluator.compute_fidelity(params)
                gradient = self._compute_gradient_custatevec_impl(params)
            elif (
                self.config.use_gpu
                and self._gpu_evaluator is not None
                and self._gpu_evaluator.gpu_available
            ):
                fidelity = self._gpu_evaluator.compute_fidelity(params)
                gradient = self._compute_gradient_gpu_impl(params)
            else:
                psi = self.get_statevector(params)
                fidelity = self._compute_fidelity_fast(psi)
                # Use stochastic gradient when configured
                if self.config.gradient_sample_fraction < 1.0:
                    gradient = self.compute_gradient_stochastic(
                        params, fraction=self.config.gradient_sample_fraction
                    )
                else:
                    gradient = self._compute_gradient_sequential_impl(params)

            # Track best
            if fidelity > best_fidelity:
                best_fidelity = fidelity
                best_params = params.copy()

            fidelity_history.append(fidelity)

            # Barren plateau check
            bp_detector.update(gradient, fidelity)
            if bp_detector.is_barren_plateau():
                print(f"  Step {step}: Barren plateau detected, performing random restart")
                params = self.get_initial_params("random")
                optimizer = AdamWithRestarts(
                    self.n_params, lr_max=lr, lr_min=lr / 50, restart_period=200
                )
                bp_detector.reset()

            # Convergence check
            if len(fidelity_history) > convergence_window:
                recent_improvement = max(fidelity_history[-convergence_window:]) - min(
                    fidelity_history[-convergence_window:]
                )
                if recent_improvement < convergence_threshold and fidelity > 0.99:
                    print(
                        f"  Converged at step {step} (improvement {recent_improvement:.2e} < {convergence_threshold:.2e})"
                    )
                    break

            # Progress logging
            if step % verbose_interval == 0:
                grad_norm = np.linalg.norm(gradient)
                current_lr = optimizer.get_lr()
                print(
                    f"  Step {step:5d}: F={fidelity:.10f}, |∇|={grad_norm:.2e}, lr={current_lr:.4f}"
                )

            # Adam update (gradient points toward increasing fidelity,
            # but Adam minimizes, so we negate)
            params = optimizer.step(params, gradient)

            # Keep parameters bounded
            params = np.clip(params, -2 * np.pi, 2 * np.pi)

        self.n_evals += step
        elapsed = time.time() - start_time

        # Update instance tracking
        if best_fidelity > self.best_fidelity:
            self.best_fidelity = best_fidelity
            self.best_params = best_params

        print(f"\nAdam complete: F={best_fidelity:.12f} in {elapsed:.1f}s ({step + 1} steps)")

        return {
            "params": best_params,
            "fidelity": best_fidelity,
            "history": fidelity_history,
            "steps": step + 1,
            "time": elapsed,
        }

    def optimize_spsa(
        self,
        initial_params: np.ndarray,
        max_steps: int = 5000,
        a: float = 0.1,
        c: float = 0.1,
        A: float = None,
        n_avg: int = 1,
        max_time: float = None,
        convergence_window: int = 200,
        convergence_threshold: float = 1e-8,
        verbose_interval: int = 100,
    ) -> dict:
        """
        SPSA optimization -- estimates gradient from only 2 evaluations per step.

        Particularly effective for large parameter counts where parameter-shift
        gradients (2*n_params evaluations) are expensive.

        Args:
            initial_params: Starting parameters
            max_steps: Maximum SPSA steps
            a: Step size parameter
            c: Perturbation size parameter
            A: Stability constant (default: 0.1 * max_steps)
            n_avg: Number of perturbation averages per step (default: 1)
            max_time: Time limit in seconds
            convergence_window: Steps to check for convergence
            convergence_threshold: Minimum improvement to continue
            verbose_interval: Print progress every N steps

        Returns:
            Dictionary with optimization results
        """
        from .spsa import SPSAOptimizer

        start_time = time.time()

        if A is None:
            A = 0.1 * max_steps

        print(f"\nSPSA Optimization (a={a}, c={c}, max_steps={max_steps}, n_avg={n_avg})")
        print("-" * 50)

        params = initial_params.copy()
        spsa = SPSAOptimizer(self.n_params, a=a, c=c, A=A, n_avg=n_avg)

        fidelity_history = []
        best_fidelity = 0.0
        best_params = params.copy()
        total_evals = 0

        for step in range(max_steps):
            if max_time is not None and (time.time() - start_time) > max_time:
                print(f"  Time limit reached at step {step}")
                break

            # SPSA step (minimizes -fidelity)
            params, g_hat, n_evals = spsa.step(params, self.objective)
            total_evals += n_evals

            # Keep parameters bounded
            params = np.clip(params, -2 * np.pi, 2 * np.pi)

            # Evaluate current fidelity (not counted in total_evals;
            # total_evals only tracks SPSA gradient evaluations)
            psi = self.get_statevector(params)
            fidelity = self._compute_fidelity_fast(psi)

            if fidelity > best_fidelity:
                best_fidelity = fidelity
                best_params = params.copy()

            fidelity_history.append(fidelity)

            # Convergence check
            if len(fidelity_history) > convergence_window:
                recent_improvement = max(fidelity_history[-convergence_window:]) - min(
                    fidelity_history[-convergence_window:]
                )
                if recent_improvement < convergence_threshold and fidelity > 0.99:
                    print(f"  Converged at step {step}")
                    break

            # Progress
            if step % verbose_interval == 0:
                grad_norm = np.linalg.norm(g_hat)
                print(f"  Step {step:5d}: F={fidelity:.10f}, |g|={grad_norm:.2e}")

        elapsed = time.time() - start_time

        # Update instance tracking
        # Note: self.objective already increments self.n_evals for each
        # SPSA perturbation evaluation. Only add the extra fidelity checks.
        self.n_evals += step + 1  # one get_statevector+fidelity check per step
        if best_fidelity > self.best_fidelity:
            self.best_fidelity = best_fidelity
            self.best_params = best_params

        print(
            f"\nSPSA complete: F={best_fidelity:.12f} in {elapsed:.1f}s "
            f"({step + 1} steps, {total_evals} evals)"
        )

        return {
            "params": best_params,
            "fidelity": best_fidelity,
            "history": fidelity_history,
            "steps": step + 1,
            "total_evals": total_evals,
            "time": elapsed,
        }

    def optimize_natural_gradient(
        self,
        initial_params: np.ndarray,
        max_steps: int = 500,
        lr: float = 0.01,
        regularization: float = 0.001,
        max_time: float = None,
        verbose_interval: int = 50,
    ) -> dict:
        """
        Natural gradient descent using diagonal Quantum Fisher Information.

        Uses Adam with the natural gradient (Euclidean gradient rescaled by
        the inverse diagonal QFIM) for geometry-aware optimization on the
        quantum state manifold.

        Args:
            initial_params: Starting parameters
            max_steps: Maximum optimization steps
            lr: Adam learning rate
            regularization: Tikhonov regularization for QFIM diagonal
            max_time: Time limit in seconds (None = no limit)
            verbose_interval: Print progress every N steps

        Returns:
            Dictionary with optimization results
        """
        from .natural_gradient import compute_natural_gradient

        start_time = time.time()
        print(
            f"\nNatural Gradient Optimization (lr={lr}, reg={regularization}, max_steps={max_steps})"
        )
        print("-" * 50)

        params = initial_params.copy()
        adam = AdamOptimizer(self.n_params, learning_rate=lr)

        best_fidelity = 0.0
        best_params = params.copy()
        fidelity_history = []

        for step in range(max_steps):
            if max_time is not None and (time.time() - start_time) > max_time:
                print(f"  Time limit reached at step {step}")
                break

            # Compute fidelity
            psi = self.get_statevector(params)
            fidelity = self._compute_fidelity_fast(psi)

            if fidelity > best_fidelity:
                best_fidelity = fidelity
                best_params = params.copy()

            fidelity_history.append(fidelity)

            # Compute natural gradient
            nat_grad = compute_natural_gradient(self, params, regularization=regularization)

            # Progress logging
            if step % verbose_interval == 0:
                grad_norm = np.linalg.norm(nat_grad)
                print(f"  Step {step:5d}: F={fidelity:.10f}, |nat_grad|={grad_norm:.2e}")

            # Adam update with natural gradient
            params = adam.step(params, nat_grad)
            params = np.clip(params, -2 * np.pi, 2 * np.pi)

        elapsed = time.time() - start_time
        self.n_evals += step + 1

        if best_fidelity > self.best_fidelity:
            self.best_fidelity = best_fidelity
            self.best_params = best_params

        print(
            f"\nNatural gradient complete: F={best_fidelity:.12f} in {elapsed:.1f}s ({step + 1} steps)"
        )

        return {
            "params": best_params,
            "fidelity": best_fidelity,
            "history": fidelity_history,
            "steps": step + 1,
            "time": elapsed,
        }

    def optimize(self, initial_params: np.ndarray | None = None) -> dict:
        """
        Multi-stage adaptive optimization.

        DEPRECATED: Use run_optimization(mode='adaptive') instead.
        """
        pipeline = OptimizationPipeline(
            mode="adaptive",
            target_fidelity=getattr(self.config, "target_fidelity", 0.9999),
            use_basin_hopping=False,
            verbose=self.config.verbose,
        )
        return self.run_optimization(pipeline, initial_params)

    def optimize_ultra_precision(
        self,
        target_infidelity: float = 1e-10,
        max_total_time: float = 3600,
        initial_params: np.ndarray = None,
    ) -> dict:
        """
        Ultra-high precision optimization pipeline.

        DEPRECATED: Use run_optimization(mode='ultra', ...) instead.
        """
        pipeline = OptimizationPipeline(
            mode="ultra",
            target_fidelity=1 - target_infidelity,
            max_total_time=max_total_time,
            use_basin_hopping=True,
            basin_hopping_threshold=0.9999,
            use_fine_tuning=True,
            verbose=self.config.verbose,
        )
        return self.run_optimization(pipeline, initial_params)

    def optimize_hybrid(
        self, initial_params: np.ndarray = None, adam_steps: int = 5000, _lbfgs_iter: int = 2000
    ) -> dict:
        """
        Hybrid Adam + L-BFGS-B optimization.

        DEPRECATED: Use run_optimization(mode='hybrid', ...) instead.
        """
        pipeline = OptimizationPipeline(
            mode="hybrid",
            target_fidelity=getattr(self.config, "target_fidelity", 0.9999),
            use_init_search=True,
            use_adam_stage=True,
            adam_max_steps=adam_steps,
            use_basin_hopping=False,
            use_lbfgs_refinement=True,
            lbfgs_tolerances=[1e-10, 1e-12],
            use_fine_tuning=False,
            verbose=self.config.verbose,
        )
        return self.run_optimization(pipeline, initial_params)

    def optimize_basin_hopping(
        self,
        initial_params: np.ndarray = None,
        n_iterations: int = 50,
        temperature: float = 1.0,
        step_size: float = 0.5,
        local_optimizer: str = "adam",  # 'adam' or 'lbfgs'
    ) -> dict:
        """
        Basin hopping global optimization.

        Combines random jumps with local optimization to explore
        multiple basins and find global minimum.

        Effective for escaping deep local minima that Adam cannot escape.
        """

        print(f"\n{'=' * 80}")
        print("BASIN HOPPING GLOBAL OPTIMIZATION")
        print(f"{'=' * 80}")
        print(f"  Iterations: {n_iterations}")
        print(f"  Temperature: {temperature}")
        print(f"  Step size: {step_size}")
        print(f"  Local optimizer: {local_optimizer}")

        if initial_params is None:
            initial_params = self.get_initial_params("smart")

        # Custom local minimizer using Adam
        def local_adam_minimizer(fun, x0, args=(), **kwargs):
            """Local minimizer using Adam for a fixed number of steps"""
            params = x0.copy()
            adam = AdamOptimizer(len(params), learning_rate=0.02)

            best_f = fun(params)
            best_params = params.copy()

            for _ in range(200):  # Short Adam run
                # Numerical gradient (fast approximation for basin hopping)
                grad = np.zeros_like(params)
                eps = 1e-5
                f0 = fun(params)
                for i in range(len(params)):
                    params[i] += eps
                    grad[i] = (fun(params) - f0) / eps
                    params[i] -= eps

                params = adam.step(params, grad)
                params = np.clip(params, -2 * np.pi, 2 * np.pi)

                f = fun(params)
                if f < best_f:
                    best_f = f
                    best_params = params.copy()

            class Result:
                x = best_params
                fun = best_f
                success = True

            return Result()

        # Minimizer options
        if local_optimizer == "adam":
            minimizer_kwargs = {
                "method": local_adam_minimizer,
            }
        else:
            minimizer_kwargs = {
                "method": "L-BFGS-B",
                "bounds": [(-2 * np.pi, 2 * np.pi)] * self.n_params,
                "options": {"maxiter": 500, "ftol": 1e-10},
            }

        # Callback to track progress
        best_fidelities = []

        def callback(x, f, accept):
            fid = -f
            best_fidelities.append(fid)
            if len(best_fidelities) % 10 == 0:
                print(f"  Iteration {len(best_fidelities)}: F={fid:.10f}, accepted={accept}")

        start_time = time.time()

        result = basinhopping(
            self.objective,
            initial_params,
            niter=n_iterations,
            T=temperature,
            stepsize=step_size,
            minimizer_kwargs=minimizer_kwargs,
            callback=callback,
            seed=42,
        )

        elapsed = time.time() - start_time

        # Update best
        final_fidelity = -result.fun
        if final_fidelity > self.best_fidelity:
            self.best_fidelity = final_fidelity
            self.best_params = result.x

        print(f"\nBasin hopping complete: F={self.best_fidelity:.12f} in {elapsed:.1f}s")

        return {
            "params": self.best_params,
            "fidelity": self.best_fidelity,
            "history": best_fidelities,
            "time": elapsed,
        }

    def optimize_multistart_parallel(
        self,
        n_starts: int = 10,
        strategies: list[str] = None,
        max_iter_per_start: int = 2000,
        tolerance: float = 1e-10,
        return_all: bool = False,
    ) -> dict:
        """
        Parallel multi-start optimization for robust global minimum search.

        Runs multiple independent optimizations in parallel with different
        initializations, then returns the best result.

        Args:
            n_starts: Number of independent optimization runs
            strategies: List of initialization strategies to cycle through
            max_iter_per_start: Max iterations per individual optimization
            tolerance: Convergence tolerance for each run
            return_all: If True, return all results (not just best)

        Returns:
            Best optimization result (or all results if return_all=True)
        """
        if strategies is None:
            strategies = ["smart", "gaussian_product", "random"]

        n_workers = min(self.config.n_workers, n_starts)

        print(f"\n{'=' * 80}")
        print("PARALLEL MULTI-START OPTIMIZATION")
        print(f"{'=' * 80}")
        print(f"  Starts: {n_starts}")
        print(f"  Workers: {n_workers}")
        print(f"  Strategies: {strategies}")
        print(f"  Max iter/start: {max_iter_per_start}")

        start_time = time.time()

        # Prepare configurations for each start
        start_configs = []
        for i in range(n_starts):
            strategy = strategies[i % len(strategies)]
            seed = 42 + i  # Reproducible seeds
            start_configs.append(
                {
                    "start_id": i,
                    "strategy": strategy,
                    "seed": seed,
                    "max_iter": max_iter_per_start,
                    "tolerance": tolerance,
                }
            )

        def run_single_start(start_config: dict) -> dict:
            """Run a single optimization start (for parallel execution)"""
            start_id = start_config["start_id"]
            strategy = start_config["strategy"]
            seed = start_config["seed"]

            # Set seed for reproducibility
            np.random.seed(seed)

            # Create fresh optimizer for this process
            # (necessary for ProcessPoolExecutor, optional for ThreadPoolExecutor)
            config_copy = copy.copy(self.config)
            config_copy.verbose = False  # Suppress output in parallel runs
            config_copy.parallel_gradients = False  # Avoid nested parallelism
            config_copy.use_custatevec = (
                False  # ADD THIS - cuStateVec doesn't work well with multiprocessing
            )
            config_copy.use_gpu = False  # ADD THIS - Aer GPU also has issues

            optimizer = GaussianOptimizer(config_copy)

            # Get initial parameters
            initial_params = optimizer.get_initial_params(strategy)

            # Run optimization
            try:
                optimizer.optimize_stage(
                    initial_params,
                    f"Start {start_id}",
                    start_config["max_iter"],
                    start_config["tolerance"],
                )

                return {
                    "start_id": start_id,
                    "strategy": strategy,
                    "seed": seed,
                    "fidelity": optimizer.best_fidelity,
                    "params": optimizer.best_params,
                    "success": True,
                    "n_evals": optimizer.n_evals,
                }
            except Exception as e:
                return {
                    "start_id": start_id,
                    "strategy": strategy,
                    "seed": seed,
                    "fidelity": 0.0,
                    "params": None,
                    "success": False,
                    "error": str(e),
                    "n_evals": 0,
                }

        # Run in parallel
        # Note: ProcessPoolExecutor is safer but has overhead
        # ThreadPoolExecutor is faster but requires thread-safe code
        all_results = []

        if self.config.parallel_backend == "process":
            # Process-based parallelism (safer, more overhead)
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                all_results = list(executor.map(run_single_start, start_configs))
        else:
            # Thread-based parallelism (faster, requires thread-safety)
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                all_results = list(executor.map(run_single_start, start_configs))

        elapsed = time.time() - start_time

        # Find best result
        successful_results = [r for r in all_results if r["success"]]

        if not successful_results:
            print("WARNING: All optimization starts failed!")
            return {"fidelity": 0, "params": None, "success": False}

        best_result = max(successful_results, key=lambda x: x["fidelity"])

        # Update instance state with best result
        if best_result["fidelity"] > self.best_fidelity:
            self.best_fidelity = best_result["fidelity"]
            self.best_params = best_result["params"]

        # Summary
        fidelities = [r["fidelity"] for r in successful_results]
        total_evals = sum(r["n_evals"] for r in all_results)

        print(f"\n{'=' * 60}")
        print("Multi-start Results Summary")
        print(f"{'=' * 60}")
        print(f"  Successful starts: {len(successful_results)}/{n_starts}")
        print(f"  Best fidelity:     {best_result['fidelity']:.15f}")
        print(f"  Best infidelity:   {1 - best_result['fidelity']:.3e}")
        print(f"  Best strategy:     {best_result['strategy']} (start {best_result['start_id']})")
        print(f"  Fidelity range:    [{min(fidelities):.10f}, {max(fidelities):.10f}]")
        print(f"  Total time:        {elapsed:.1f}s")
        print(f"  Total evaluations: {total_evals}")
        print(f"  Avg time/start:    {elapsed / n_starts:.2f}s")

        # Print all results sorted by fidelity
        print("\n  All results (sorted by fidelity):")
        for r in sorted(successful_results, key=lambda x: -x["fidelity"])[:10]:
            print(f"    Start {r['start_id']:2d} ({r['strategy']:18s}): F = {r['fidelity']:.12f}")

        if return_all:
            return {
                "best": best_result,
                "all_results": all_results,
                "time": elapsed,
                "total_evals": total_evals,
            }

        return best_result

    def evaluate_population_parallel(
        self, population: np.ndarray, _chunk_size: int = None
    ) -> np.ndarray:
        """
        DEPRECATED: Use evaluate_population() instead.
        Evaluate fidelity for a population of parameter sets.
        """
        # Just delegate to the unified method
        return self.evaluate_population(population)

    def optimize_cmaes_parallel(
        self,
        initial_params: np.ndarray = None,
        sigma0: float = 0.5,
        population_size: int = None,
        max_generations: int = 200,
        target_fidelity: float = 0.9999,
        ftol: float = 1e-12,
    ) -> dict:
        """
        CMA-ES optimization with parallel population evaluation.

        CMA-ES (Covariance Matrix Adaptation Evolution Strategy) is highly
        effective for:
        - Non-convex optimization landscapes
        - Escaping local minima
        - High-dimensional parameter spaces

        Combined with parallel evaluation, this provides robust global
        optimization with good speedup.

        Args:
            initial_params: Starting point (None = smart init)
            sigma0: Initial step size
            population_size: Population size (None = auto)
            max_generations: Maximum generations
            target_fidelity: Stop early if achieved
            ftol: Function tolerance for convergence

        Returns:
            Optimization results dictionary
        """
        try:
            import cma
        except ImportError:
            print("CMA-ES requires the 'cma' package. Install with: pip install cma")
            print("Falling back to standard optimization...")
            return self.optimize(initial_params)

        if initial_params is None:
            initial_params = self.get_initial_params("smart")

        if population_size is None:
            # CMA-ES default: 4 + floor(3 * ln(n))
            population_size = 4 + int(3 * np.log(self.n_params))
            # Round up to multiple of n_workers for efficiency
            population_size = (
                (population_size + self.config.n_workers - 1)
                // self.config.n_workers
                * self.config.n_workers
            )

        print(f"\n{'=' * 80}")
        print("CMA-ES OPTIMIZATION (Parallel)")
        print(f"{'=' * 80}")
        print(f"  Population size:  {population_size}")
        print(f"  Workers:          {self.config.n_workers}")
        print(f"  Max generations:  {max_generations}")
        print(f"  Initial sigma:    {sigma0}")
        print(f"  Target fidelity:  {target_fidelity}")

        start_time = time.time()

        # CMA-ES options
        opts = {
            "popsize": population_size,
            "maxiter": max_generations,
            "ftarget": -target_fidelity,  # We minimize -fidelity
            "tolfun": ftol,
            "verb_disp": 1 if self.config.verbose else 0,
            "verb_log": 0,
            "bounds": [-2 * np.pi, 2 * np.pi],
        }

        # Initialize CMA-ES
        es = cma.CMAEvolutionStrategy(initial_params, sigma0, opts)

        generation = 0
        history = {"fidelity": [], "generation": []}

        while not es.stop():
            generation += 1

            # Get population
            population = np.array(es.ask())

            # Parallel evaluation
            fidelities = self.evaluate_population_parallel(population)

            # CMA-ES minimizes, so negate fidelities
            es.tell(population, -fidelities)

            # Track progress
            best_gen_fid = np.max(fidelities)
            history["fidelity"].append(best_gen_fid)
            history["generation"].append(generation)

            # Progress output
            if self.config.verbose and generation % 10 == 0:
                print(
                    f"  Gen {generation:4d}: best F = {self.best_fidelity:.12f}, "
                    f"gen best = {best_gen_fid:.10f}, sigma = {es.sigma:.4f}"
                )

            # Early stopping if target achieved
            if self.best_fidelity >= target_fidelity:
                print(f"\n  Target fidelity {target_fidelity} achieved at generation {generation}")
                break

        elapsed = time.time() - start_time

        # Get final result
        final_params = es.result.xbest
        final_psi = self.get_statevector(final_params)
        final_fidelity = self._compute_fidelity_fast(final_psi)

        # Ensure we have the true best
        if final_fidelity > self.best_fidelity:
            self.best_fidelity = final_fidelity
            self.best_params = final_params

        print(f"\n{'=' * 60}")
        print("CMA-ES Complete")
        print(f"{'=' * 60}")
        print(f"  Final fidelity:   {self.best_fidelity:.15f}")
        print(f"  Infidelity:       {1 - self.best_fidelity:.3e}")
        print(f"  Generations:      {generation}")
        print(f"  Total time:       {elapsed:.1f}s")
        print(f"  Time/generation:  {elapsed / generation:.2f}s")

        return {
            "params": self.best_params,
            "fidelity": self.best_fidelity,
            "infidelity": 1 - self.best_fidelity,
            "generations": generation,
            "history": history,
            "time": elapsed,
            "cma_result": es.result,
        }

    def plot_results(self, results: dict, save_path: str | None = None):
        """Create visualization plots with high precision display."""
        from .visualization import plot_optimization_results

        return plot_optimization_results(
            self.positions,
            results["final_statevector"],
            self.target,
            results,
            self.history,
            self.config.n_qubits,
            self.config.sigma,
            self.config.box_size,
            save_path,
        )

    def save_results(self, results: dict, filepath: str = None):
        """Save high-precision parameters to text file."""
        from .visualization import save_optimization_results

        config_dict = {
            "n_qubits": self.config.n_qubits,
            "sigma": self.config.sigma,
            "x0": self.config.x0,
            "box_size": self.config.box_size,
            "method": self.config.method,
            "high_precision": self.config.high_precision,
            "n_params": self.config.n_params,
            "n_states": self.config.n_states,
            "delta_x": self.config.delta_x,
            "tolerance": self.config.tolerance,
            "max_iter": self.config.max_iter,
            "max_fun": self.config.max_fun,
            "enable_refinement": self.config.enable_refinement,
            "verbose": self.config.verbose,
        }
        return save_optimization_results(results, config_dict, filepath)

    def get_optimized_circuit(
        self,
        params: np.ndarray | None = None,
        include_measurements: bool = False,
    ) -> QuantumCircuit:
        from .export import build_optimized_circuit

        return build_optimized_circuit(self, params, include_measurements)

    def export_qasm(
        self,
        params: np.ndarray | None = None,
        include_measurements: bool = False,
        version: int = 2,
    ) -> str:
        from .export import export_to_qasm, export_to_qasm3

        if version == 2:
            return export_to_qasm(self, params, include_measurements)
        elif version == 3:
            return export_to_qasm3(self, params, include_measurements)
        else:
            raise ValueError(f"OpenQASM version must be 2 or 3, got {version}")

    def save_circuit(
        self,
        filepath: str,
        params: np.ndarray | None = None,
        format: str = "qasm",
        include_measurements: bool = False,
        **kwargs,
    ) -> str:
        from .export import save_circuit

        return str(save_circuit(self, filepath, params, format, include_measurements, **kwargs))
