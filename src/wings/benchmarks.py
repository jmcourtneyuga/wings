"""Benchmarking utilities for backend comparison."""

import time
from typing import Any, Optional

import numpy as np

from .compat import HAS_CUSTATEVEC
from .config import OptimizerConfig
from .optimizer import GaussianOptimizer


def _get_gpu_count():
    """Get number of available GPUs."""
    if not HAS_CUSTATEVEC:
        return 0
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount()
    except Exception:
        return 0


__all__ = [
    "benchmark_gpu",
    "benchmark_multi_gpu",
    "find_gpu_crossover",
    "benchmark_all_backends",
    "BenchmarkResult",
]


class BenchmarkResult:
    """Container for benchmark results."""

    def __init__(self):
        self.results: dict[str, dict[str, float]] = {}
        self.winner: Optional[str] = None
        self.recommendation: str = ""

    def add_result(self, backend: str, metric: str, value: float):
        if backend not in self.results:
            self.results[backend] = {}
        self.results[backend][metric] = value

    def __repr__(self) -> str:
        lines = ["BenchmarkResult:"]
        for backend, metrics in self.results.items():
            lines.append(f"  {backend}:")
            for metric, value in metrics.items():
                lines.append(f"    {metric}: {value}")
        if self.winner:
            lines.append(f"  Winner: {self.winner}")
        return "\n".join(lines)


def benchmark_gpu(
    n_qubits: int = 8,
    sigma: float = 0.5,
    n_trials: int = 10,
    verbose: bool = True,
) -> BenchmarkResult:
    """
    Benchmark GPU vs CPU performance for a specific configuration.

    Parameters
    ----------
    n_qubits : int
        Number of qubits to benchmark
    sigma : float
        Gaussian width parameter
    n_trials : int
        Number of trials for timing (more = more accurate)
    verbose : bool
        Print detailed results

    Returns
    -------
    BenchmarkResult
        Object containing timing results and recommendations

    Examples
    --------
    >>> result = benchmark_gpu(n_qubits=12, sigma=0.5)
    >>> print(f"GPU speedup: {result.results['gpu']['speedup']:.2f}x")
    """
    result = BenchmarkResult()

    if verbose:
        print(f"\n{'=' * 80}")
        print("GPU BENCHMARK")
        print(f"{'=' * 80}")
        print(f"  Qubits: {n_qubits}")
        print(f"  Parameters: {n_qubits * n_qubits}")

    # CPU configuration
    config_cpu = OptimizerConfig(
        n_qubits=n_qubits,
        sigma=sigma,
        box_size=4 * sigma,
        verbose=False,
        use_gpu=False,
        use_custatevec=False,
        parallel_gradients=False,
    )

    if verbose:
        print("\nInitializing CPU optimizer...")
    optimizer_cpu = GaussianOptimizer(config_cpu)
    test_params = optimizer_cpu.get_initial_params("smart")

    # ========================================
    # Benchmark 1: Single evaluation
    # ========================================
    if verbose:
        print("\n1. Single Statevector Evaluation")
        print("-" * 50)

    # CPU timing
    start = time.perf_counter()
    for _ in range(n_trials * 10):
        optimizer_cpu.get_statevector(test_params)
    cpu_single = (time.perf_counter() - start) / (n_trials * 10)
    result.add_result("cpu", "single_eval_ms", cpu_single * 1000)

    if verbose:
        print(f"  CPU:  {cpu_single * 1000:.2f} ms/eval")

    # GPU timing (Aer)
    config_gpu = OptimizerConfig(
        n_qubits=n_qubits,
        sigma=sigma,
        box_size=4 * sigma,
        verbose=False,
        use_gpu=True,
        use_custatevec=False,
        gpu_precision="double",
    )

    optimizer_gpu = GaussianOptimizer(config_gpu)

    if optimizer_gpu._gpu_evaluator and optimizer_gpu._gpu_evaluator.gpu_available:
        # Warm-up
        for _ in range(5):
            _ = optimizer_gpu.get_statevector(test_params)

        start = time.perf_counter()
        for _ in range(n_trials * 10):
            optimizer_gpu.get_statevector(test_params)
        gpu_single = (time.perf_counter() - start) / (n_trials * 10)

        result.add_result("gpu_aer", "single_eval_ms", gpu_single * 1000)
        result.add_result("gpu_aer", "speedup_vs_cpu", cpu_single / gpu_single)

        if verbose:
            print(f"  GPU (Aer):  {gpu_single * 1000:.2f} ms/eval")
            print(f"  Speedup: {cpu_single / gpu_single:.2f}x")
    elif verbose:
        print("  GPU (Aer): Not available")

    # cuStateVec timing
    if HAS_CUSTATEVEC:
        config_cusv = OptimizerConfig(
            n_qubits=n_qubits,
            sigma=sigma,
            box_size=4 * sigma,
            verbose=False,
            use_gpu=False,
            use_custatevec=True,
            gpu_precision="double",
        )

        optimizer_cusv = GaussianOptimizer(config_cusv)

        if optimizer_cusv._custatevec_evaluator is not None:
            # Warm-up
            for _ in range(5):
                _ = optimizer_cusv._custatevec_evaluator.compute_fidelity(test_params)

            start = time.perf_counter()
            for _ in range(n_trials * 10):
                _ = optimizer_cusv._custatevec_evaluator.compute_fidelity(test_params)
            cusv_single = (time.perf_counter() - start) / (n_trials * 10)

            result.add_result("custatevec", "single_eval_ms", cusv_single * 1000)
            result.add_result("custatevec", "speedup_vs_cpu", cpu_single / cusv_single)

            if verbose:
                print(f"  cuStateVec: {cusv_single * 1000:.2f} ms/eval")
                print(f"  Speedup: {cpu_single / cusv_single:.2f}x")

            # Cleanup
            optimizer_cusv._custatevec_evaluator.cleanup()
            if optimizer_cusv._custatevec_batch_evaluator:
                optimizer_cusv._custatevec_batch_evaluator.cleanup()


def benchmark_multi_gpu(
    n_qubits: int = 12,
    sigma: float = 0.5,
    batch_sizes: Optional[list[int]] = None,
    verbose: bool = True,
) -> BenchmarkResult:
    """
    Benchmark multi-GPU vs single-GPU performance.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (should be >= 12 for meaningful results)
    sigma : float
        Gaussian width
    batch_sizes : list of int, optional
        Batch sizes to test (default: [64, 128, 256])
    verbose : bool
        Print results

    Returns
    -------
    BenchmarkResult
        Timing comparisons for single vs multi-GPU
    """
    result = BenchmarkResult()

    n_gpus = _get_gpu_count()

    if n_gpus < 2:
        if verbose:
            print("Multi-GPU benchmark requires 2+ GPUs")
        result.recommendation = "Multi-GPU not available"
        return result

    if batch_sizes is None:
        batch_sizes = [64, 128, 256]

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"MULTI-GPU BENCHMARK ({n_gpus} GPUs, {n_qubits} qubits)")
        print(f"{'=' * 60}")

    # Single GPU config
    config_single = OptimizerConfig(
        n_qubits=n_qubits,
        sigma=sigma,
        verbose=False,
        use_custatevec=True,
        use_multi_gpu=False,
    )

    # Multi-GPU config
    config_multi = OptimizerConfig(
        n_qubits=n_qubits,
        sigma=sigma,
        verbose=False,
        use_custatevec=True,
        use_multi_gpu=True,
    )

    opt_single = GaussianOptimizer(config_single)
    opt_multi = GaussianOptimizer(config_multi)

    if opt_multi._multi_gpu_evaluator is None:
        if verbose:
            print("Multi-GPU evaluator not initialized")
        return result

    for batch_size in batch_sizes:
        population = np.random.randn(batch_size, config_single.n_params) * 0.1

        # Single GPU
        start = time.perf_counter()
        _ = opt_single.evaluate_population(population, backend="custatevec")
        single_time = time.perf_counter() - start

        # Multi GPU
        start = time.perf_counter()
        _ = opt_multi.evaluate_population(population, backend="multi_gpu")
        multi_time = time.perf_counter() - start

        speedup = single_time / multi_time

        result.add_result("single_gpu", f"batch_{batch_size}_ms", single_time * 1000)
        result.add_result("multi_gpu", f"batch_{batch_size}_ms", multi_time * 1000)
        result.add_result("multi_gpu", f"batch_{batch_size}_speedup", speedup)

        if verbose:
            print(
                f"  Batch {batch_size}: Single={single_time * 1000:.0f}ms, "
                f"Multi={multi_time * 1000:.0f}ms, Speedup={speedup:.2f}x"
            )

    # Cleanup
    if hasattr(opt_single, "cleanup"):
        opt_single.cleanup()
    if hasattr(opt_multi, "cleanup"):
        opt_multi.cleanup()

    avg_speedup = np.mean(
        [result.results["multi_gpu"].get(f"batch_{bs}_speedup", 1.0) for bs in batch_sizes]
    )

    result.winner = "multi_gpu" if avg_speedup > 1.2 else "single_gpu"
    result.recommendation = (
        f"Multi-GPU provides {avg_speedup:.1f}x speedup for {n_qubits} qubits"
        if avg_speedup > 1.2
        else f"Single GPU sufficient for {n_qubits} qubits"
    )

    if verbose:
        print(f"\nRecommendation: {result.recommendation}")

    return result


def benchmark_batched_evaluation(
    n_qubits: int = 10,
    sigma: float = 0.5,
    batch_sizes: Optional[list[int]] = None,
    verbose: bool = True,
) -> BenchmarkResult:
    """
    Benchmark batched evaluation (gradient-like workload) across backends.

    Parameters
    ----------
    n_qubits : int
        Number of qubits
    sigma : float
        Gaussian width
    batch_sizes : list of int, optional
        Batch sizes to test (default: [32, 64, 128])
    verbose : bool
        Print results

    Returns
    -------
    BenchmarkResult
        Timing comparisons for CPU vs GPU batched evaluation
    """
    result = BenchmarkResult()

    if batch_sizes is None:
        batch_sizes = [32, 64, 128]

    # CPU config
    config_cpu = OptimizerConfig(
        n_qubits=n_qubits,
        sigma=sigma,
        verbose=False,
        use_gpu=False,
        use_custatevec=False,
    )

    # GPU config
    config_gpu = OptimizerConfig(
        n_qubits=n_qubits,
        sigma=sigma,
        verbose=False,
        use_gpu=True,
        use_custatevec=False,
    )

    optimizer_cpu = GaussianOptimizer(config_cpu)
    optimizer_gpu = GaussianOptimizer(config_gpu)

    if verbose:
        print("\nBatched Evaluation (Gradient-like workload)")
        print("-" * 50)

    for batch_size in batch_sizes:
        population = np.random.randn(batch_size, config_cpu.n_params) * 0.1

        # CPU sequential
        start = time.perf_counter()
        np.array(
            [
                optimizer_cpu._compute_fidelity_fast(optimizer_cpu.get_statevector(p))
                for p in population
            ]
        )
        cpu_batch_time = time.perf_counter() - start

        result.add_result("cpu", f"batch_{batch_size}_ms", cpu_batch_time * 1000)

        if verbose:
            print(f"  Batch {batch_size}: CPU={cpu_batch_time * 1000:.0f}ms", end="")

        # GPU batched
        if optimizer_gpu._gpu_evaluator and optimizer_gpu._gpu_evaluator.gpu_available:
            start = time.perf_counter()
            optimizer_gpu.evaluate_population(population)
            gpu_batch_time = time.perf_counter() - start

            result.add_result("gpu_aer", f"batch_{batch_size}_ms", gpu_batch_time * 1000)

            if verbose:
                print(
                    f", GPU={gpu_batch_time * 1000:.0f}ms ({cpu_batch_time / gpu_batch_time:.1f}x)",
                    end="",
                )

        if verbose:
            print()

    # Determine winner
    backends = ["cpu"]
    if "gpu_aer" in result.results:
        backends.append("gpu_aer")
    if "custatevec" in result.results:
        backends.append("custatevec")

    best_time = float("inf")
    for backend in backends:
        if "single_eval_ms" in result.results.get(backend, {}):
            t = result.results[backend]["single_eval_ms"]
            if t < best_time:
                best_time = t
                result.winner = backend

    result.recommendation = f"Use {result.winner} for {n_qubits} qubits"

    if verbose:
        print(f"\n{'=' * 50}")
        print(f"RECOMMENDATION: {result.recommendation}")

    return result


def find_gpu_crossover(
    qubit_range: Optional[list[int]] = None,
    sigma: float = 0.5,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Find where GPU becomes faster than CPU for your hardware.

    This helps you determine the optimal backend for different problem sizes.

    Parameters
    ----------
    qubit_range : list of int, optional
        Qubit counts to test. Default: [6, 8, 10, 12, 14, 16, 18]
    sigma : float
        Gaussian width for test problems
    verbose : bool
        Print results table

    Returns
    -------
    dict
        Contains:
        - 'crossover_qubits': Qubit count where GPU becomes faster
        - 'results': Detailed timing for each qubit count
        - 'recommendation': Human-readable recommendation

    Examples
    --------
    >>> info = find_gpu_crossover()
    >>> print(f"GPU becomes faster at {info['crossover_qubits']} qubits")
    """
    if qubit_range is None:
        qubit_range = [6, 8, 10, 12, 14, 16, 18]

    results = []
    crossover = None

    if verbose:
        print(f"{'Qubits':<8} {'CPU (ms)':<12} {'cuSV (ms)':<12} {'Speedup':<10} {'Winner'}")
        print("-" * 55)

    for n_qubits in qubit_range:
        n_params = n_qubits * n_qubits

        # Skip if too large
        if n_qubits > 20:
            if verbose:
                print(f"{n_qubits:<8} Skipped (memory)")
            continue

        # CPU timing
        config_cpu = OptimizerConfig(
            n_qubits=n_qubits,
            sigma=sigma,
            box_size=4.0,
            verbose=False,
            use_gpu=False,
            use_custatevec=False,
        )

        opt_cpu = GaussianOptimizer(config_cpu)
        test_params = np.random.randn(n_params) * 0.1

        n_trials = 20 if n_qubits <= 14 else 5

        start = time.perf_counter()
        for _ in range(n_trials):
            psi = opt_cpu.get_statevector(test_params)
            _ = opt_cpu._compute_fidelity_fast(psi)
        cpu_time = (time.perf_counter() - start) / n_trials * 1000

        # cuStateVec timing
        cusv_time = float("inf")

        if HAS_CUSTATEVEC:
            config_cusv = OptimizerConfig(
                n_qubits=n_qubits,
                sigma=sigma,
                box_size=4.0,
                verbose=False,
                use_gpu=False,
                use_custatevec=True,
            )

            try:
                opt_cusv = GaussianOptimizer(config_cusv)

                if opt_cusv._custatevec_evaluator is not None:
                    # Warm-up
                    for _ in range(3):
                        _ = opt_cusv._custatevec_evaluator.compute_fidelity(test_params)

                    start = time.perf_counter()
                    for _ in range(n_trials):
                        _ = opt_cusv._custatevec_evaluator.compute_fidelity(test_params)
                    cusv_time = (time.perf_counter() - start) / n_trials * 1000

                    # Cleanup
                    opt_cusv._custatevec_evaluator.cleanup()
                    if opt_cusv._custatevec_batch_evaluator:
                        opt_cusv._custatevec_batch_evaluator.cleanup()
            except Exception:
                pass

        speedup = cpu_time / cusv_time if cusv_time > 0 and cusv_time != float("inf") else 0
        winner = "cuSV" if speedup > 1 else "CPU"

        # Track crossover point
        if crossover is None and speedup > 1:
            crossover = n_qubits

        results.append(
            {
                "n_qubits": n_qubits,
                "cpu_ms": cpu_time,
                "cusv_ms": cusv_time if cusv_time != float("inf") else None,
                "speedup": speedup,
                "winner": winner,
            }
        )

        if verbose:
            cusv_str = f"{cusv_time:.2f}" if cusv_time != float("inf") else "N/A"
            print(f"{n_qubits:<8} {cpu_time:<12.2f} {cusv_str:<12} {speedup:<10.2f} {winner}")

    if verbose:
        print("\n" + "=" * 55)
        print("RECOMMENDATION:")
        if crossover:
            print(f"  - Use CPU for qubits < {crossover}")
            print(f"  - Use cuStateVec for qubits >= {crossover}")
        else:
            print("  - CPU is faster for all tested sizes")
        print("  - For gradient computation, crossover is ~2 qubits lower")

    return {
        "crossover_qubits": crossover,
        "results": results,
        "recommendation": f"GPU crossover at {crossover} qubits"
        if crossover
        else "CPU faster for all sizes",
    }


def benchmark_all_backends(
    n_qubits: int = 10,
    sigma: float = 0.5,
) -> dict[str, BenchmarkResult]:
    """
    Comprehensive benchmark of all available backends.

    Parameters
    ----------
    n_qubits : int
        Number of qubits for benchmark
    sigma : float
        Gaussian width

    Returns
    -------
    dict
        Results for each benchmark type
    """
    print("=" * 80)
    print(f"COMPREHENSIVE BACKEND BENCHMARK ({n_qubits} qubits)")
    print("=" * 80)

    results = {}

    # Single evaluation benchmark
    results["single"] = benchmark_gpu(n_qubits, sigma, verbose=True)

    # Gradient benchmark would go here

    return results
