"""Main Gaussian state optimizer."""

import copy
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from scipy.optimize import basinhopping, differential_evolution, minimize

from .adam import AdamOptimizer, AdamWithRestarts
from .ansatz import DefaultAnsatz
from .compat import HAS_CUSTATEVEC
from .config import OptimizationPipeline, OptimizerConfig, TargetFunction
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
            self.circuit, basis_gates=["ry", "cx", "x"], optimization_level=1
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

    def _compute_target_wavefunction(self) -> "ComplexArray":
        """Compute normalized target wavefunction based on config."""
        x = self.positions
        dx = self.config.delta_x

        if self.config.target_function == TargetFunction.GAUSSIAN:
            psi = self._gaussian(x)
        elif self.config.target_function == TargetFunction.LORENTZIAN:
            psi = self._lorentzian(x)
        elif self.config.target_function == TargetFunction.SECH:
            psi = self._sech(x)  # ADD THIS CASE
        elif self.config.target_function == TargetFunction.CUSTOM:
            if self.config.custom_target_fn is None:
                raise ValueError("custom_target_fn required for CUSTOM target")
            psi = self.config.custom_target_fn(x)
        else:
            raise ValueError(f"Unknown target function: {self.config.target_function}")

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

    def get_statevector(self, params: ParameterArray, backend: str = "auto") -> ComplexArray:
        """
        Get statevector with automatic backend selection.

        Args:
            params: Circuit parameters
            backend: 'auto', 'custatevec', 'gpu', or 'cpu'

        Returns:
            Statevector as numpy array
        """
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
        params: Optional[ParameterArray] = None,
        psi: Optional[ComplexArray] = None,
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
        self, population: "NDArray[np.float64]", backend: str = "auto"
    ) -> "FloatArray":
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
        # Single dot product with pre-conjugated target
        overlap = np.dot(self._target_conj, psi_circuit)
        return overlap.real**2 + overlap.imag**2

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
            self.optimize_stage(
                self.best_params, f"Refinement (tol={tol:.0e})", max_iter=3000, tolerance=tol
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
            "infidelity": 1 - final_fidelity,
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

        # Build all shifted parameter sets at once
        # Shape: (2 * n_params, n_params)
        params_shifted = np.zeros((2 * n_params, n_params))

        for i in range(n_params):
            # Forward shift
            params_shifted[2 * i] = params.copy()
            params_shifted[2 * i, i] += shift

            # Backward shift
            params_shifted[2 * i + 1] = params.copy()
            params_shifted[2 * i + 1, i] -= shift

        # Single batched GPU call for all shifts
        fidelities = self._gpu_evaluator.compute_fidelities_batched(params_shifted)

        # Compute gradients from shift results
        gradient = np.zeros(n_params)
        for i in range(n_params):
            fid_plus = fidelities[2 * i]
            fid_minus = fidelities[2 * i + 1]
            gradient[i] = (fid_plus - fid_minus) / 2

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
                gradient = self._compute_gradient_sequential_impl(params)

            # Track best
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

    def optimize(self, initial_params: Optional[np.ndarray] = None) -> dict:
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

    def plot_results(self, results: dict, save_path: Optional[str] = None):
        """Create visualization plots with high precision display"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 11))

            x = self.positions
            psi_circuit = results["final_statevector"]
            psi_target = self.target

            # Plot 1: Probability densities (log scale option for high precision)
            ax = axes[0, 0]
            ax.plot(x, np.abs(psi_circuit) ** 2, "b-", label="Circuit", linewidth=2)
            ax.plot(
                x, np.abs(psi_target) ** 2, "r--", label="Target Gaussian", linewidth=2, alpha=0.8
            )
            ax.set_xlabel("Position x", fontsize=11)
            ax.set_ylabel("|ψ(x)|²", fontsize=11)
            ax.set_title(
                f"Probability Density (Fidelity = {results['fidelity']:.10f})", fontsize=12
            )
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

            # Plot 2: Real and imaginary parts
            ax = axes[0, 1]
            ax.plot(x, np.real(psi_circuit), "b-", label="Circuit (Real)", linewidth=1.5)
            ax.plot(
                x, np.imag(psi_circuit), "b--", label="Circuit (Imag)", linewidth=1.5, alpha=0.7
            )
            ax.plot(x, np.real(psi_target), "r-", label="Target (Real)", linewidth=1.5, alpha=0.8)
            ax.plot(x, np.imag(psi_target), "r--", label="Target (Imag)", linewidth=1.5, alpha=0.5)
            ax.set_xlabel("Position x", fontsize=11)
            ax.set_ylabel("Amplitude", fontsize=11)
            ax.set_title("Wavefunction Components", fontsize=12)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

            # Plot 3: Difference (log scale for high precision)
            ax = axes[1, 0]
            difference = np.abs(psi_circuit - psi_target) ** 2
            max_diff = np.max(difference)
            ax.plot(x, difference, "g-", linewidth=2)
            ax.set_xlabel("Position x", fontsize=11)
            ax.set_ylabel("|ψ_circuit - ψ_target|²", fontsize=11)
            ax.set_title(f"Squared Difference (max = {max_diff:.3e})", fontsize=12)
            ax.set_yscale("log")
            ax.grid(True, alpha=0.3, which="both")

            # Plot 4: Convergence with infidelity tracking
            ax = axes[1, 1]
            if len(self.history["fidelity"]) > 0:
                fidelities = np.array(self.history["fidelity"])
                infidelities = 1 - fidelities

                # Plot on log scale to see high precision improvement
                ax.semilogy(
                    self.history["iteration"], infidelities, "g-", linewidth=1.5, label="Infidelity"
                )
                ax.axhline(y=1e-3, color="r", linestyle="--", alpha=0.5, label="F=0.999")
                ax.axhline(y=1e-4, color="orange", linestyle="--", alpha=0.5, label="F=0.9999")
                ax.axhline(
                    y=results["infidelity"],
                    color="blue",
                    linestyle="-",
                    alpha=0.7,
                    label=f"Final: 1-F={results['infidelity']:.2e}",
                )
                ax.set_xlabel("Function Evaluation", fontsize=11)
                ax.set_ylabel("Infidelity (1 - F)", fontsize=11)
                ax.set_title("Optimization Progress (Log Scale)", fontsize=12)
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3, which="both")

            # Add high-precision statistics text
            stats_text = (
                f"High Precision Results:\n"
                f"Fidelity:    {results['fidelity']:.12f}\n"
                f"Infidelity:  {results['infidelity']:.3e}\n"
                f"Circuit: μ={results['circuit_mean']:.8f}, σ={results['circuit_std']:.8f}\n"
                f"Target:  μ={results['target_mean']:.8f}, σ={results['target_std']:.8f}\n"
                f"Errors:  Δμ={results['mean_error']:.3e}, Δσ={results['std_error']:.3e}\n"
                f"Rel. σ error: {results['relative_std_error'] * 100:.2f}%\n"
                f"Time: {results['time']:.1f}s, Evals: {results['n_evaluations']}"
            )
            fig.text(
                0.02,
                0.02,
                stats_text,
                fontsize=9,
                family="monospace",
                bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.6},
            )

            plt.suptitle(
                f"High Precision Gaussian State (n={self.config.n_qubits}, σ={self.config.sigma:.4f}, box=±{self.config.box_size:.2f})",
                fontsize=14,
                fontweight="bold",
            )
            plt.tight_layout()

            if save_path:
                try:
                    plt.savefig(save_path, dpi=200, bbox_inches="tight")
                    print(f"Plot saved to: {save_path}")
                except Exception as e:
                    print(f"Warning: Could not save plot: {e}")

            plt.show()

            return fig

        except Exception as e:
            print(f"Warning: Could not create plot: {e}")
            import traceback

            traceback.print_exc()
            return None

    def save_results(self, results: dict, filepath: str = None):
        """Save high-precision parameters to text file"""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"gaussian_highprec_q{self.config.n_qubits}_s{self.config.sigma:.4f}_{timestamp}.txt"

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("=" * 80 + "\n")
                f.write("HIGH PRECISION GAUSSIAN STATE PREPARATION\n")
                f.write("=" * 80 + "\n\n")

                f.write("CONFIGURATION\n")
                f.write("-" * 40 + "\n")
                f.write(f"Timestamp:          {datetime.now().isoformat()}\n")
                f.write(f"Number of qubits:   {self.config.n_qubits}\n")
                f.write(f"Number of params:   {self.config.n_params}\n")
                f.write(f"Target sigma:       {self.config.sigma:.10f}\n")
                f.write(f"Target x0:          {self.config.x0:.10f}\n")
                f.write(f"Box size:           +/-{self.config.box_size:.6f}\n")
                f.write(f"Grid points:        {self.config.n_states}\n")
                f.write(f"Grid spacing:       {self.config.delta_x:.10f}\n")
                f.write(f"Optimizer:          {self.config.method}\n")
                f.write(f"Max iterations:     {self.config.max_iter}\n")
                f.write(f"Max fun evals:      {self.config.max_fun}\n")
                f.write(f"Tolerance:          {self.config.tolerance:.2e}\n")
                f.write(f"High precision:     {self.config.high_precision}\n")
                f.write(f"Refinement enabled: {self.config.enable_refinement}\n\n")

                f.write("HIGH PRECISION RESULTS\n")
                f.write("-" * 40 + "\n")
                f.write(f"Fidelity:           {results['fidelity']:.15f}\n")
                f.write(f"Infidelity (1-F):   {results['infidelity']:.3e}\n")
                f.write(f"Circuit mean:       {results['circuit_mean']:.12f}\n")
                f.write(f"Circuit std:        {results['circuit_std']:.12f}\n")
                f.write(f"Target mean:        {results['target_mean']:.12f}\n")
                f.write(f"Target std:         {results['target_std']:.12f}\n")
                f.write(f"Error in mean:      {results['mean_error']:.3e}\n")
                f.write(f"Error in std:       {results['std_error']:.3e}\n")
                f.write(f"Relative std err:   {results['relative_std_error'] * 100:.4f}%\n")
                f.write(f"Optimization time:  {results['time']:.2f} seconds\n")
                f.write(f"Function evals:     {results['n_evaluations']}\n")
                f.write(f"Success:            {results['success']}\n")
                f.write(f"Message:            {results.get('optimizer_message', 'N/A')}\n\n")

                f.write("OPTIMAL PARAMETERS (15 decimal places)\n")
                f.write("-" * 40 + "\n")
                f.write("# Index    Value\n")
                params = results["optimal_params"]
                for i, param in enumerate(params):
                    f.write(f"{i:5d}    {param:+.15f}\n")

                f.write("\n" + "=" * 80 + "\n")
                f.write("# To load parameters:\n")
                f.write(
                    f"# params = np.loadtxt('{os.path.basename(filepath)}', skiprows=N, usecols=1)\n"
                )

            print(f"\nResults saved to: {filepath}")

            # Save numpy array with full precision
            np_file = filepath.replace(".txt", "_params.npy")
            np.save(np_file, results["optimal_params"])
            print(f"Parameters saved to: {np_file}")

            # Save JSON with results
            json_file = filepath.replace(".txt", "_results.json")
            json_data = {
                "fidelity": float(results["fidelity"]),
                "infidelity": float(results["infidelity"]),
                "circuit_mean": float(results["circuit_mean"]),
                "circuit_std": float(results["circuit_std"]),
                "target_mean": float(results["target_mean"]),
                "target_std": float(results["target_std"]),
                "mean_error": float(results["mean_error"]),
                "std_error": float(results["std_error"]),
                "time": float(results["time"]),
                "n_evaluations": int(results["n_evaluations"]),
                "config": {
                    "n_qubits": self.config.n_qubits,
                    "sigma": self.config.sigma,
                    "x0": self.config.x0,
                    "box_size": self.config.box_size,
                    "method": self.config.method,
                    "high_precision": self.config.high_precision,
                },
            }
            with open(json_file, "w") as f:
                json.dump(json_data, f, indent=2)
            print(f"JSON results saved to: {json_file}")

        except Exception as e:
            print(f"Error saving results: {e}")
            import traceback

            traceback.print_exc()

        return filepath

    def get_optimized_circuit(
        self,
        params: Optional[np.ndarray] = None,
        include_measurements: bool = False,
    ) -> QuantumCircuit:
        from .export import build_optimized_circuit

        return build_optimized_circuit(self, params, include_measurements)

    def export_qasm(
        self,
        params: Optional[np.ndarray] = None,
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
        params: Optional[np.ndarray] = None,
        format: str = "qasm",
        include_measurements: bool = False,
        **kwargs,
    ) -> str:
        from .export import save_circuit

        return str(save_circuit(self, filepath, params, format, include_measurements, **kwargs))
