"""
Gaussian State Optimizer
========================

GPU-accelerated variational quantum state preparation for Gaussian wavefunctions.

This package provides high-performance tools for preparing Gaussian quantum states
using variational quantum algorithms with support for multiple acceleration backends:

- CPU (Qiskit Statevector)
- GPU (Qiskit Aer)
- cuStateVec (NVIDIA cuQuantum)

Quick Start
-----------
>>> from wings import GaussianOptimizer, OptimizerConfig
>>>
>>> # Configure optimization
>>> config = OptimizerConfig(
...     n_qubits=8,
...     sigma=0.5,
...     use_custatevec=True,  # Enable GPU acceleration
... )
>>>
>>> # Create optimizer and run
>>> optimizer = GaussianOptimizer(config)
>>> results = optimizer.optimize_ultra_precision(target_infidelity=1e-10)
>>> print(f"Fidelity: {results['fidelity']:.12f}")

For production campaigns with thousands of runs:

>>> from wings import run_production_campaign
>>>
>>> results = run_production_campaign(
...     n_qubits=8,
...     sigma=0.5,
...     total_runs=1000,
...     target_infidelity=1e-11,
... )
>>> results.print_summary()

Environment Variables
--------------------
GSO_BASE_DIR : str
    Base directory for all data storage (default: ~/.wings)
GSO_CACHE_DIR : str
    Directory for coefficient cache
GSO_CHECKPOINT_DIR : str
    Directory for optimization checkpoints
GSO_CAMPAIGN_DIR : str
    Directory for campaign results

See Also
--------
- Documentation: https://gaussian-state-optimizer.readthedocs.io
- GitHub: https://github.com/yourusername/gaussian-state-optimizer
"""

__version__ = "0.4.1"
__author__ = "Joshua M. Courtney"
__email__ = "joshuamcourtney@gmail.com"
__license__ = "MIT"

# Version tuple for programmatic comparison
VERSION = tuple(int(x) for x in __version__.split(".")[:3])


# Lazy imports to avoid loading heavy dependencies until needed
def __getattr__(name: str):
    """Lazy import of main classes to speed up package import."""

    _public_api = {
        # Core classes
        "GaussianOptimizer": ".optimizer",
        "OptimizerConfig": ".config",
        "OptimizationPipeline": ".config",
        # Composable pipeline
        "Pipeline": ".pipeline",
        "InitSearch": ".pipeline",
        "Adam": ".pipeline",
        "SPSA": ".pipeline",
        "NaturalGradient": ".pipeline",
        "LBFGSB": ".pipeline",
        "BasinHopping": ".pipeline",
        "Newton": ".pipeline",
        "GrowCircuit": ".pipeline",
        "CampaignConfig": ".config",
        "CampaignResults": ".campaign",
        "OptimizationManager": ".campaign",
        "TargetFunction": ".config",
        # Ansatz classes
        "DefaultAnsatz": ".ansatz",
        "CustomHardwareEfficientAnsatz": ".ansatz",
        "AnsatzProtocol": ".ansatz",
        # Ansatz library
        "EfficientSU2Ansatz": ".ansatz_library",
        "generate_entanglement_map": ".ansatz_library",
        # Evaluators
        "CuStateVecEvaluator": ".evaluators.custatevec",
        "BatchedCuStateVecEvaluator": ".evaluators.custatevec",
        "GPUCircuitEvaluator": ".evaluators.gpu",
        "ThreadSafeCircuitEvaluator": ".evaluators.cpu",
        "MultiGPUBatchEvaluator": ".evaluators.custatevec",
        # Convenience functions
        "run_production_campaign": ".campaign",
        "quick_optimization": ".campaign",
        "load_campaign_results": ".campaign",
        "list_campaigns": ".campaign",
        # Path configuration
        "PathConfig": ".paths",
        "get_path_config": ".paths",
        # Compatibility
        "CuQuantumCompat": ".compat",
        "get_cuda_dtype": ".compat",
        "get_compute_type": ".compat",
        "optimize_gaussian_state": ".convenience",
        "quick_optimize": ".convenience",
        # Benchmarking
        "benchmark_gpu": ".benchmarks",
        "find_gpu_crossover": ".benchmarks",
        "benchmark_all_backends": ".benchmarks",
        # Circuit export
        "build_optimized_circuit": ".export",
        "export_to_qasm": ".export",
        "export_to_qasm3": ".export",
        "save_circuit": ".export",
        # SPSA optimizer
        "SPSAOptimizer": ".spsa",
        # Barren plateau detection
        "BarrenPlateauDetector": ".barren_plateau",
        # Warm-start transfer learning
        "transfer_params": ".warm_start",
        # Natural gradient
        "compute_qfim_diagonal": ".natural_gradient",
        "compute_natural_gradient": ".natural_gradient",
        # Fidelity utilities
        "compute_fidelity_fast": ".fidelity",
        "compute_infidelity_direct": ".fidelity",
        # MPS initialization
        "mps_decompose": ".mps_init",
        "mps_to_statevector": ".mps_init",
        "mps_initial_params": ".mps_init",
        # Tracy-Widom distributions
        "tracy_widom_pdf": ".tracy_widom",
        "tracy_widom_wavefunction": ".tracy_widom",
        "list_tracy_widom_targets": ".tracy_widom",
        # Wavefunction library
        "harmonic_oscillator_eigenstate": ".wavefunctions",
        "superposition_of_gaussians": ".wavefunctions",
        "airy_wavefunction": ".wavefunctions",
        "morse_oscillator_eigenstate": ".wavefunctions",
        "squeezed_gaussian": ".wavefunctions",
        "plane_wave_packet": ".wavefunctions",
        "list_wavefunctions": ".wavefunctions",
        # Visualization / export
        "plot_optimization_results": ".visualization",
        "save_optimization_results": ".visualization",
        # Dashboard
        "OptimizationDashboard": ".dashboard",
        # JAX autodiff backend
        "compute_gradient_jax_default_ansatz": ".evaluators.jax_backend",
        # Multi-dimensional grids
        "NDGrid": ".nd_grid",
        "gaussian_nd": ".nd_grid",
        # Noise-aware optimization
        "NoiseConfig": ".evaluators.noisy",
        # Time evolution (v0.4.0 WI-4)
        "split_operator_step": ".time_evolution",
        "evolve_classical": ".time_evolution",
        "make_grid": ".time_evolution",
        "free_particle": ".time_evolution",
        "harmonic_potential": ".time_evolution",
        "morse_potential": ".time_evolution",
        "lennard_jones_potential": ".time_evolution",
        # Hardware-native execution (v0.4.0 WI-5)
        "transpile_for_hardware": ".hardware",
        "classical_state_fidelity": ".hardware",
        "counts_to_probabilities": ".hardware",
        "HardwareResult": ".hardware",
        # Result dataclass (v1.0.0)
        "OptimizationResult": ".results",
    }

    if name in _public_api:
        import importlib

        module_path = _public_api[name]
        module = importlib.import_module(module_path, package=__name__)
        return getattr(module, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Explicitly define __all__ for `from wings import *`
__all__ = [
    # Version info
    "__version__",
    "VERSION",
    # Core classes
    "GaussianOptimizer",
    "OptimizerConfig",
    "OptimizationPipeline",
    "TargetFunction",
    "CampaignConfig",
    "CampaignResults",
    "OptimizationManager",
    # Composable pipeline
    "Pipeline",
    "InitSearch",
    "Adam",
    "SPSA",
    "NaturalGradient",
    "LBFGSB",
    "BasinHopping",
    "Newton",
    "GrowCircuit",
    # Ansatz
    "DefaultAnsatz",
    "CustomHardwareEfficientAnsatz",
    "AnsatzProtocol",
    "EfficientSU2Ansatz",
    "generate_entanglement_map",
    # Evaluators
    "CuStateVecEvaluator",
    "BatchedCuStateVecEvaluator",
    "MultiGPUBatchEvaluator",
    "GPUCircuitEvaluator",
    "ThreadSafeCircuitEvaluator",
    # Functions
    "run_production_campaign",
    "quick_optimization",
    "load_campaign_results",
    "list_campaigns",
    # Configuration
    "PathConfig",
    "get_path_config",
    # Compatibility
    "CuQuantumCompat",
    "get_cuda_dtype",
    "get_compute_type",
    "optimize_gaussian_state",
    "quick_optimize",
    # Benchmarks
    "benchmark_gpu",
    "find_gpu_crossover",
    "benchmark_all_backends",
    "build_optimized_circuit",
    "export_to_qasm",
    "export_to_qasm3",
    "save_circuit",
    # SPSA optimizer
    "SPSAOptimizer",
    # Barren plateau detection
    "BarrenPlateauDetector",
    # Warm-start transfer learning
    "transfer_params",
    # Natural gradient
    "compute_qfim_diagonal",
    "compute_natural_gradient",
    # Fidelity utilities
    "compute_fidelity_fast",
    "compute_infidelity_direct",
    # MPS initialization
    "mps_decompose",
    "mps_to_statevector",
    "mps_initial_params",
    # Wavefunction library
    "harmonic_oscillator_eigenstate",
    "superposition_of_gaussians",
    "airy_wavefunction",
    "morse_oscillator_eigenstate",
    "squeezed_gaussian",
    "plane_wave_packet",
    "list_wavefunctions",
    # Tracy-Widom distributions
    "tracy_widom_pdf",
    "tracy_widom_wavefunction",
    "list_tracy_widom_targets",
    # Visualization / export
    "plot_optimization_results",
    "save_optimization_results",
    # Dashboard
    "OptimizationDashboard",
    # JAX autodiff backend
    "compute_gradient_jax_default_ansatz",
    # Multi-dimensional grids
    "NDGrid",
    "gaussian_nd",
    # Noise-aware optimization
    "NoiseConfig",
    # Time evolution (v0.4.0 WI-4)
    "split_operator_step",
    "evolve_classical",
    "make_grid",
    "free_particle",
    "harmonic_potential",
    "morse_potential",
    "lennard_jones_potential",
    # Hardware-native execution (v0.4.0 WI-5)
    "transpile_for_hardware",
    "classical_state_fidelity",
    "counts_to_probabilities",
    "HardwareResult",
    # Result dataclass (v1.0.0)
    "OptimizationResult",
]


def get_backend_info() -> dict:
    """
    Get information about available backends.

    Returns
    -------
    dict
        Dictionary containing:
        - 'cpu': Always True
        - 'gpu_aer': True if qiskit-aer is available
        - 'custatevec': True if cuQuantum is available
        - 'cuda_version': CUDA version string or None
        - 'gpu_name': GPU device name or None
    """
    from typing import Any

    info: dict[str, Any] = {
        "cpu": True,
        "gpu_aer": False,
        "custatevec": False,
        "cuda_version": None,
        "gpu_name": None,
    }

    # Check Qiskit Aer
    try:
        from qiskit_aer import AerSimulator

        backend = AerSimulator(method="statevector")
        info["gpu_aer"] = "GPU" in backend.available_devices()
    except ImportError:
        pass
    except Exception:
        pass

    # Check cuStateVec
    try:
        import cupy as cp
        from cuquantum.bindings import custatevec

        info["custatevec"] = True

        # Get CUDA info
        device = cp.cuda.Device(0)
        info["gpu_name"] = (
            device.name.decode() if hasattr(device.name, "decode") else str(device.name)
        )

        cuda_version = cp.cuda.runtime.runtimeGetVersion()
        major = cuda_version // 1000
        minor = (cuda_version % 1000) // 10
        info["cuda_version"] = f"{major}.{minor}"
    except ImportError:
        pass
    except Exception:
        pass

    return info


def print_backend_info() -> None:
    """Print information about available backends."""
    info = get_backend_info()

    print("Gaussian State Optimizer - Backend Information")
    print("=" * 50)
    print(f"  CPU (Qiskit Statevector): {'✓ Available' if info['cpu'] else '✗ Not available'}")
    print(f"  GPU (Qiskit Aer):         {'✓ Available' if info['gpu_aer'] else '✗ Not available'}")
    print(
        f"  cuStateVec:               {'✓ Available' if info['custatevec'] else '✗ Not available'}"
    )

    if info["cuda_version"]:
        print(f"  CUDA Version:             {info['cuda_version']}")
    if info["gpu_name"]:
        print(f"  GPU Device:               {info['gpu_name']}")

    print("=" * 50)
