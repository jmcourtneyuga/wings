# WINGS Usage Protocol

This document provides complete instructions for using WINGS across all three acceleration backends (CPU, Qiskit Aer GPU, and NVIDIA cuStateVec), covering installation, configuration, optimization workflows, and troubleshooting.

---

## Table of Contents

1. [Backend Overview](#1-backend-overview)
2. [Installation by Backend](#2-installation-by-backend)
3. [Backend Verification](#3-backend-verification)
4. [CPU Protocol](#4-cpu-protocol)
5. [Qiskit Aer GPU Protocol](#5-qiskit-aer-gpu-protocol)
6. [cuStateVec Protocol](#6-custatevec-protocol)
7. [Multi-GPU Protocol](#7-multi-gpu-protocol)
8. [Backend Selection Logic](#8-backend-selection-logic)
9. [Optimization Workflows](#9-optimization-workflows)
10. [Ansatz Selection Guide](#10-ansatz-selection-guide)
11. [Configuration Reference](#11-configuration-reference)
12. [Performance Tuning](#12-performance-tuning)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. Backend Overview

WINGS supports four acceleration tiers. The optimizer selects the best available backend automatically, but you can override this.

| Backend | Library | Precision | Typical Speedup | Best For |
|---------|---------|-----------|----------------|----------|
| **CPU** | Qiskit Statevector | float64 | 1x (baseline) | Development, small systems (n <= 8) |
| **Aer GPU** | qiskit-aer + CUDA | float64 | 2-5x | Medium systems, batched evaluation |
| **cuStateVec** | cuQuantum + CuPy | float64 | 5-20x | Production, single-GPU optimization |
| **Multi-GPU** | cuStateVec + threads | float64 | Linear scaling | Large systems (n >= 12), campaigns |

**Automatic priority**: Multi-GPU > cuStateVec > Aer GPU > CPU

---

## 2. Installation by Backend

### CPU Only (no GPU required)

```bash
pip install wings
```

Dependencies installed: `qiskit>=1.0`, `numpy>=1.20`, `scipy>=1.7`, `matplotlib>=3.5`

### Qiskit Aer GPU

Requires NVIDIA GPU with CUDA 11.0+ drivers installed.

```bash
pip install wings[gpu]
```

This installs `qiskit-aer>=0.13` with CUDA support. Verify CUDA is accessible:

```bash
nvidia-smi  # Should show your GPU
```

### cuStateVec (recommended for production)

Requires NVIDIA GPU with CUDA 11.0+ and cuQuantum SDK.

```bash
pip install wings[gpu]
pip install cuquantum-python>=23.0 cupy-cuda11x>=12.0
```

For CUDA 12.x systems, replace `cupy-cuda11x` with `cupy-cuda12x`.

### From Source (development)

```bash
git clone https://github.com/jmcourtneyuga/wings.git
cd wings
pip install -e ".[dev,gpu]"
pip install cuquantum-python cupy-cuda11x  # For cuStateVec
```

---

## 3. Backend Verification

After installation, verify which backends are available:

```python
from wings import print_backend_info
print_backend_info()
```

Expected output (full GPU setup):
```
WINGS Backend Information
========================
  CPU (Qiskit Statevector): Available
  GPU (Qiskit Aer):         Available (NVIDIA A100-SXM4-80GB)
  cuStateVec (cuQuantum):    Available (v24.03)
  CUDA version:              12.2
  CuPy version:              13.0
```

Programmatic check:

```python
from wings import get_backend_info
info = get_backend_info()
print(info["gpu_available"])        # True/False
print(info["custatevec_available"]) # True/False
```

---

## 4. CPU Protocol

Use CPU mode for development, testing, debugging, and systems with n <= 8 qubits.

### Basic Usage

```python
from wings import GaussianOptimizer, OptimizerConfig

config = OptimizerConfig(
    n_qubits=8,
    sigma=0.5,
    use_gpu=False,
    use_custatevec=False,
)

optimizer = GaussianOptimizer(config)
results = optimizer.run_optimization(target_fidelity=0.9999, max_total_time=120)

print(f"Fidelity: {results['fidelity']:.12f}")
print(f"Infidelity: {results['infidelity']:.3e}")
print(f"Time: {results['time']:.1f}s")
```

### CPU-Specific Configuration

```python
config = OptimizerConfig(
    n_qubits=8,
    sigma=0.5,

    # Disable all GPU paths
    use_gpu=False,
    use_custatevec=False,
    use_multi_gpu=False,

    # CPU parallelization for gradient computation
    parallel_gradients=True,
    n_workers=4,                  # Number of CPU threads (default: cpu_count - 1)
    parallel_backend="thread",    # "thread" (faster) or "process" (safer)
    gradient_chunk_size=8,        # Parameters per thread chunk

    # For faster iteration during development
    max_iter=2000,
    max_fun=10000,
)
```

### CPU with Stochastic Gradients (faster per step)

```python
config = OptimizerConfig(
    n_qubits=10,
    sigma=0.5,
    use_gpu=False,
    use_custatevec=False,
    gradient_sample_fraction=0.5,  # Only compute 50% of gradient components per step
)
```

### CPU with SPSA (constant cost per step)

```python
optimizer = GaussianOptimizer(config)
params = optimizer.get_initial_params("smart")

# SPSA: 2 evaluations per gradient regardless of n_params
result = optimizer.optimize_spsa(
    params,
    max_steps=5000,
    a=0.1,
    c=0.1,
    n_avg=3,          # Average 3 perturbations for lower variance
    max_time=300,
)
```

### When CPU is Sufficient

| Qubits | Parameters | CPU Time (F=0.9999) | Recommendation |
|--------|-----------|---------------------|----------------|
| 6 | 36 | ~30s | CPU is fine |
| 8 | 64 | ~3min | CPU is fine |
| 10 | 100 | ~15min | Consider GPU |
| 12 | 144 | ~2hr | Use GPU |

---

## 5. Qiskit Aer GPU Protocol

The Aer GPU backend provides moderate speedup (2-5x) with minimal setup. It runs the entire Qiskit circuit simulation on GPU via CUDA.

### Basic Usage

```python
config = OptimizerConfig(
    n_qubits=10,
    sigma=0.5,
    use_gpu=True,          # Enable Aer GPU
    use_custatevec=False,  # Don't use cuStateVec (use Aer instead)
    gpu_precision="double",
    gpu_batch_size=64,     # Circuits per GPU batch
)

optimizer = GaussianOptimizer(config)
results = optimizer.run_optimization(target_fidelity=0.999999, max_total_time=600)
```

### Aer GPU Configuration

```python
config = OptimizerConfig(
    n_qubits=12,
    sigma=0.5,

    # Aer GPU settings
    use_gpu=True,
    use_custatevec=False,
    gpu_precision="double",    # "double" (recommended) or "single" (2x faster, less accurate)
    gpu_batch_size=64,         # Number of circuits per batch
    gpu_blocking=True,         # Wait for GPU to finish before continuing
)
```

### Aer GPU Gradient Computation

With Aer GPU, all `2 * n_params` shifted circuits for the parameter-shift gradient are batched into a single GPU call:

```python
optimizer = GaussianOptimizer(config)
params = optimizer.get_initial_params("smart")

# This automatically uses batched GPU evaluation
gradient = optimizer.compute_gradient(params)  # method="auto" selects GPU
```

### When to Use Aer GPU vs cuStateVec

| Scenario | Use Aer GPU | Use cuStateVec |
|----------|-------------|----------------|
| No cuQuantum installed | Yes | N/A |
| n_qubits <= 10 | Either | Either |
| n_qubits >= 12 | No | Yes (5-10x faster) |
| Custom gate sets beyond RY/RZ/CX | Yes (Aer supports all Qiskit gates) | Only if gates are implemented |
| Need Qiskit noise models | Yes | No |

---

## 6. cuStateVec Protocol

cuStateVec is the fastest backend, providing 5-20x speedup over CPU by interfacing directly with NVIDIA's cuQuantum library. It bypasses Qiskit circuit simulation entirely, applying gates natively on GPU memory.

### Basic Usage

```python
config = OptimizerConfig(
    n_qubits=12,
    sigma=0.5,
    use_custatevec=True,
    gpu_precision="double",
)

optimizer = GaussianOptimizer(config)
results = optimizer.optimize_ultra_precision(
    target_infidelity=1e-10,
    max_total_time=3600,
)
```

### cuStateVec Configuration

```python
config = OptimizerConfig(
    n_qubits=14,
    sigma=0.5,

    # cuStateVec settings
    use_gpu=False,             # Disable Aer (not needed with cuStateVec)
    use_custatevec=True,
    gpu_precision="double",    # Always use double for high-fidelity work
    custatevec_batch_size=128, # Circuits per batch for gradient computation
)
```

### cuStateVec with Batched Gradient

cuStateVec uses multiple pre-allocated simulators for parallel gradient evaluation:

```python
# The optimizer automatically creates BatchedCuStateVecEvaluator
# with 4 simulators for round-robin evaluation
optimizer = GaussianOptimizer(config)

# Full pipeline: Adam -> Basin Hopping -> L-BFGS-B -> Fine Tuning
results = optimizer.run_optimization(
    mode="ultra",
    target_fidelity=1 - 1e-11,
    max_total_time=3600,
)
```

### cuStateVec with EfficientSU2 Ansatz

The EfficientSU2 ansatz uses RZ gates, which are natively supported by cuStateVec (v0.3.0):

```python
from wings import GaussianOptimizer, OptimizerConfig
from wings.ansatz_library import EfficientSU2Ansatz

ansatz = EfficientSU2Ansatz(
    n_qubits=12,
    layers=6,
    entanglement="circular",  # Also: "log_distance", "parity", "full"
)

config = OptimizerConfig(
    n_qubits=12,
    sigma=0.5,
    ansatz=ansatz,
    use_custatevec=True,
)

optimizer = GaussianOptimizer(config)
results = optimizer.run_optimization(target_fidelity=0.999999)
```

### cuStateVec Memory Requirements

| Qubits | Statevector Size | GPU Memory (double) | GPU Memory (single) |
|--------|-----------------|--------------------|--------------------|
| 10 | 1,024 states | ~16 KB | ~8 KB |
| 14 | 16,384 states | ~256 KB | ~128 KB |
| 18 | 262,144 states | ~4 MB | ~2 MB |
| 20 | 1,048,576 states | ~16 MB | ~8 MB |
| 24 | 16,777,216 states | ~256 MB | ~128 MB |
| 28 | 268,435,456 states | ~4 GB | ~2 GB |

The batch evaluator allocates `n_simulators * statevector_size` memory. With 4 simulators at 20 qubits: ~64 MB.

---

## 7. Multi-GPU Protocol

For large systems or production campaigns, distribute evaluation across multiple GPUs.

### Basic Usage

```python
config = OptimizerConfig(
    n_qubits=16,
    sigma=0.5,
    use_custatevec=True,
    use_multi_gpu=True,
    gpu_device_ids=[0, 1, 2, 3],  # None = auto-detect all
    simulators_per_gpu=2,
)

optimizer = GaussianOptimizer(config)
results = optimizer.run_optimization(target_fidelity=0.999999, max_total_time=7200)
```

### Multi-GPU with Campaigns

```python
from wings import run_production_campaign

results = run_production_campaign(
    n_qubits=14,
    sigma=0.5,
    total_runs=1000,
    target_infidelity=1e-11,
    use_multi_gpu=True,
    gpu_device_ids=[0, 1],
)

results.print_summary()
```

### Multi-GPU Gradient Distribution

The `MultiGPUBatchEvaluator` distributes the `2 * n_params` shifted circuits across GPUs, with each GPU processing its chunk in a separate thread:

```
GPU 0: circuits [0 .. n_params/2)
GPU 1: circuits [n_params/2 .. n_params)
```

This provides near-linear speedup for gradient computation.

---

## 8. Backend Selection Logic

The optimizer's `compute_gradient()` and `get_statevector()` methods select backends in this priority order:

```
1. Multi-GPU  (if use_multi_gpu=True and multiple GPUs detected)
2. cuStateVec (if use_custatevec=True and cuQuantum available)
3. Aer GPU    (if use_gpu=True and qiskit-aer with CUDA available)
4. CPU        (always available)
```

### Forcing a Specific Backend

```python
# Force CPU even when GPU is available
sv = optimizer.get_statevector(params, backend="cpu")
fid = optimizer.compute_fidelity(params=params, backend="cpu")

# Force cuStateVec
sv = optimizer.get_statevector(params, backend="custatevec")

# Force Aer GPU
gradient = optimizer.compute_gradient(params, method="gpu")
```

### Disabling Backends

```python
# CPU only
config = OptimizerConfig(use_gpu=False, use_custatevec=False, use_multi_gpu=False)

# Aer GPU only (no cuStateVec)
config = OptimizerConfig(use_gpu=True, use_custatevec=False)

# cuStateVec only (no Aer)
config = OptimizerConfig(use_gpu=False, use_custatevec=True)
```

---

## 9. Optimization Workflows

### Workflow A: Quick Development Run (CPU)

```python
from wings import quick_optimize

fidelity, results = quick_optimize(n_qubits=8, sigma=0.5)
```

### Workflow B: Standard Optimization (auto backend)

```python
from wings import optimize_gaussian_state

results, optimizer = optimize_gaussian_state(
    n_qubits=10,
    sigma=0.5,
    target_infidelity=1e-10,
    max_time=600,
)
```

### Workflow C: Full Pipeline Control

```python
from wings import GaussianOptimizer, OptimizerConfig, OptimizationPipeline

config = OptimizerConfig(n_qubits=10, sigma=0.5, use_custatevec=True)
optimizer = GaussianOptimizer(config)

pipeline = OptimizationPipeline(
    mode="ultra",
    target_fidelity=1 - 1e-11,
    max_total_time=3600,

    # Stage control
    use_init_search=True,
    init_strategies=["smart", "gaussian_product", "random", "random", "random"],

    use_adam_stage=True,
    adam_max_steps=2000,
    adam_lr=0.02,
    adam_time_fraction=0.3,

    use_basin_hopping=True,
    basin_hopping_threshold=0.9999,
    basin_hopping_iterations=30,

    use_lbfgs_refinement=True,
    lbfgs_tolerances=[1e-10, 1e-12, 1e-14],

    use_log_objective=True,          # v0.2.0: log(1-F) objective for refinement
    log_objective_threshold=0.999,

    use_fine_tuning=True,
    fine_tuning_threshold=0.9999,

    use_natural_gradient=False,      # v0.3.0: enable for curvature-aware updates
    use_adaptive_depth=False,        # v0.3.0: grow circuit as needed
)

results = optimizer.run_optimization(pipeline)
```

### Workflow D: SPSA for Large Systems

```python
from wings import GaussianOptimizer, OptimizerConfig

config = OptimizerConfig(n_qubits=14, sigma=0.5, use_custatevec=True)
optimizer = GaussianOptimizer(config)

# Phase 1: SPSA for broad exploration (cheap gradients)
params = optimizer.get_initial_params("smart")
r1 = optimizer.optimize_spsa(params, max_steps=3000, a=0.1, c=0.1, n_avg=3)

# Phase 2: Adam with parameter-shift for refinement
r2 = optimizer.optimize_adam(r1["params"], max_steps=1000, lr=0.01)

# Phase 3: L-BFGS-B for final polish
pipeline = OptimizationPipeline(
    use_init_search=False,
    use_adam_stage=False,
    use_lbfgs_refinement=True,
    use_log_objective=True,
    target_fidelity=1 - 1e-10,
)
results = optimizer.run_optimization(pipeline, initial_params=r2["params"])
```

### Workflow E: Warm-Start Across Qubit Counts

```python
from wings import GaussianOptimizer, OptimizerConfig
from wings.warm_start import transfer_params

# Start small: 8 qubits
config_8 = OptimizerConfig(n_qubits=8, sigma=0.5, use_custatevec=True)
opt_8 = GaussianOptimizer(config_8)
r8 = opt_8.run_optimization(target_fidelity=0.999999)

# Transfer to 10 qubits
params_10 = transfer_params(r8["optimal_params"], n_source=8, n_target=10)
config_10 = OptimizerConfig(n_qubits=10, sigma=0.5, use_custatevec=True)
opt_10 = GaussianOptimizer(config_10)
r10 = opt_10.run_optimization(initial_params=params_10, target_fidelity=0.999999)

# Transfer to 12 qubits
params_12 = transfer_params(r10["optimal_params"], n_source=10, n_target=12)
config_12 = OptimizerConfig(n_qubits=12, sigma=0.5, use_custatevec=True)
opt_12 = GaussianOptimizer(config_12)
r12 = opt_12.run_optimization(initial_params=params_12, target_fidelity=0.999999)
```

### Workflow F: Complex Wavepackets with Momentum

```python
from wings import GaussianOptimizer, OptimizerConfig
from wings.ansatz_library import EfficientSU2Ansatz

# Gaussian wavepacket with initial momentum k=2.0
# MUST use an ansatz with RZ gates for complex amplitudes
ansatz = EfficientSU2Ansatz(n_qubits=10, layers=5, entanglement="circular")

config = OptimizerConfig(
    n_qubits=10,
    sigma=0.5,
    momentum=2.0,       # Applies exp(ikx) phase
    ansatz=ansatz,
    use_custatevec=True,
)

optimizer = GaussianOptimizer(config)
results = optimizer.run_optimization(target_fidelity=0.9999)
```

### Workflow G: Custom Wavefunction

```python
import numpy as np
from wings import GaussianOptimizer, OptimizerConfig, TargetFunction

# Double-Gaussian superposition
def cat_state(x):
    return np.exp(-((x - 2)**2) / 0.5) + np.exp(-((x + 2)**2) / 0.5)

config = OptimizerConfig(
    n_qubits=10,
    target_function=TargetFunction.CUSTOM,
    custom_target_fn=cat_state,
    box_size=8.0,
    use_custatevec=True,
)

optimizer = GaussianOptimizer(config)
results = optimizer.run_optimization(target_fidelity=0.999)
```

### Workflow H: Using the Wavefunction Library

```python
from wings import GaussianOptimizer, OptimizerConfig, TargetFunction
from wings.wavefunctions import harmonic_oscillator_eigenstate, morse_oscillator_eigenstate

# Harmonic oscillator first excited state
config = OptimizerConfig(
    n_qubits=10,
    target_function=TargetFunction.CUSTOM,
    custom_target_fn=lambda x: harmonic_oscillator_eigenstate(x, n=1, sigma=1.0),
    box_size=8.0,
    use_custatevec=True,
)

optimizer = GaussianOptimizer(config)
results = optimizer.run_optimization(target_fidelity=0.999)
```

---

## 10. Ansatz Selection Guide

| Ansatz | Gates | Complex Phases? | Parameters | Best For |
|--------|-------|----------------|-----------|----------|
| `DefaultAnsatz` | RY + CNOT | No (real only) | `n * depth` | Standard Gaussians, sech, Lorentzian |
| `DefaultAnsatz(entanglement="log_distance")` | RY + CNOT (multi-scale) | No | `n * depth` | Wide or narrow Gaussians needing multi-scale correlations |
| `CustomHardwareEfficientAnsatz(rotation_gates=['ry', 'rz'])` | RY + RZ + CNOT | Yes | `n * 2 * layers` | Momentum wavepackets, complex targets |
| `EfficientSU2Ansatz` | RY + RZ + CNOT | Yes | `n * 2 * layers` | Momentum wavepackets, complex targets (cleaner API) |
| `EfficientSU2Ansatz(entanglement="circular")` | RY + RZ + circular CNOT | Yes | `n * 2 * layers` | Best general-purpose for complex targets |

**Rule of thumb**:
- Real target, centered Gaussian -> `DefaultAnsatz` (default)
- Real target, unusual shape -> `DefaultAnsatz(entanglement="log_distance")`
- Complex target (momentum != 0) -> `EfficientSU2Ansatz(entanglement="circular")`
- Hardware-specific gate set -> `CustomHardwareEfficientAnsatz` with matching gates

---

## 11. Configuration Reference

### OptimizerConfig Fields

```python
OptimizerConfig(
    # Problem
    n_qubits=9,              # Qubits (2^n grid points)
    sigma=1.0,               # Gaussian width
    x0=0.0,                  # Center position
    momentum=0.0,            # Wavepacket momentum (v0.2.0)
    box_size=None,           # Grid half-width (None = auto)
    auto_optimize_box=False, # Analytical optimal box size (v0.2.0)

    # Target
    target_function=TargetFunction.GAUSSIAN,
    gamma=None,              # Lorentzian width
    custom_target_fn=None,   # Callable for CUSTOM

    # Ansatz
    ansatz=None,             # None = DefaultAnsatz
    ansatz_depth=None,
    ansatz_kwargs=None,

    # Optimization
    method="L-BFGS-B",
    max_iter=10000,
    max_fun=50000,
    tolerance=1e-12,
    gtol=1e-12,
    use_analytic_gradients=True,
    gradient_sample_fraction=1.0,  # Stochastic gradient fraction (v0.2.0)
    high_precision=True,
    target_fidelity=0.9999999,

    # GPU
    use_gpu=True,
    use_custatevec=True,
    use_multi_gpu=False,
    gpu_device_ids=None,     # None = auto-detect
    gpu_precision="double",
    gpu_batch_size=64,
    custatevec_batch_size=128,
    simulators_per_gpu=2,

    # Parallelization
    n_workers=None,          # None = cpu_count - 1
    parallel_gradients=True,
    parallel_backend="thread",
    gradient_chunk_size=8,

    # Output
    verbose=True,
    plot_result=True,
    save_plot=True,
)
```

### OptimizationPipeline Fields

```python
OptimizationPipeline(
    mode="adaptive",                    # "adaptive", "ultra", "hybrid", "single_stage"
    target_fidelity=0.9999,
    max_total_time=3600,

    # Init search
    use_init_search=True,
    init_strategies=["smart", "gaussian_product", "random", "random", "random"],

    # Adam
    use_adam_stage=True,
    adam_max_steps=1000,
    adam_lr=0.01,
    adam_time_fraction=0.4,

    # Basin hopping
    use_basin_hopping=False,
    basin_hopping_threshold=0.9999,

    # L-BFGS-B
    use_lbfgs_refinement=True,
    lbfgs_tolerances=[1e-10, 1e-12, 1e-14],

    # Log objective (v0.2.0)
    use_log_objective=True,
    log_objective_threshold=0.999,

    # Fine tuning
    use_fine_tuning=True,
    fine_tuning_threshold=0.9999,

    # Natural gradient (v0.3.0)
    use_natural_gradient=False,
    natural_gradient_regularization=0.001,

    # Adaptive depth (v0.3.0)
    use_adaptive_depth=False,
    min_depth=2,
    max_depth=0,                        # 0 = use n_qubits
    depth_step=1,
    depth_fidelity_plateau=1e-6,
)
```

---

## 12. Performance Tuning

### CPU Optimization

1. **Increase thread count**: `n_workers=os.cpu_count()` (default leaves 1 core free)
2. **Use stochastic gradients**: `gradient_sample_fraction=0.5` halves gradient cost per step
3. **Use SPSA for n_qubits >= 10**: 2 evaluations per gradient vs `2 * n^2` for parameter-shift
4. **Reduce init strategies**: `init_strategies=["smart", "random"]` (fewer initial evaluations)

### Aer GPU Optimization

1. **Increase batch size**: `gpu_batch_size=128` or `256` (more circuits per GPU call)
2. **Use double precision**: `gpu_precision="double"` is essential for fidelity > 0.999999
3. **Enable blocking**: `gpu_blocking=True` (default) ensures synchronous evaluation

### cuStateVec Optimization

1. **Use batched evaluation**: Automatically enabled with `BatchedCuStateVecEvaluator`
2. **Increase batch simulators**: Default is 4 per GPU; 8 may help for large gradients
3. **Log-distance entanglement**: For Gaussian targets, `entanglement="log_distance"` often converges faster than linear
4. **CUDA stream pipelining** (v0.3.0): Automatically enabled -- overlaps gate operations

### Multi-GPU Optimization

1. **Balance GPUs**: `simulators_per_gpu=2` is default; increase for many-parameter systems
2. **Pin device IDs**: `gpu_device_ids=[0, 1]` to avoid unexpected GPU selection
3. **Campaign parallelism**: Combine multi-GPU with campaign's `n_parallel_runs`

### General Tips

- **Start with SPSA, finish with L-BFGS-B**: SPSA explores cheaply; L-BFGS-B refines precisely
- **Use warm-start**: Optimize at n=8, transfer to n=10, then n=12. Each step starts near the optimum.
- **Log objective for ultra-precision**: Automatically breaks through the F=0.999999 ceiling
- **Adaptive box size**: `auto_optimize_box=True` ensures the grid is neither too small (clipping) nor too large (wasting resolution)
- **Monitor for barren plateaus**: Enabled by default in Adam. If restarts happen frequently, try a shallower circuit or `EfficientSU2Ansatz`

---

## 13. Troubleshooting

### "GPU not available" but NVIDIA GPU is present

```bash
# Check CUDA
nvidia-smi
python -c "import qiskit_aer; print(qiskit_aer.AerSimulator().available_devices())"
```

If `GPU` not in available devices, reinstall qiskit-aer with CUDA:
```bash
pip install qiskit-aer --force-reinstall
```

### "cuStateVec not available"

```bash
python -c "from cuquantum.bindings import custatevec as cusv; h = cusv.create(); cusv.destroy(h); print('OK')"
```

If this fails, check:
1. CUDA toolkit installed: `nvcc --version`
2. cuQuantum installed: `pip install cuquantum-python`
3. CuPy matches CUDA version: `pip install cupy-cuda12x` (for CUDA 12.x)

### "RuntimeError: Simulation device GPU is not supported"

This means qiskit-aer was installed without CUDA support. On WSL2:
```bash
pip uninstall qiskit-aer
pip install qiskit-aer-gpu  # Separate GPU-enabled package
```

### "UserWarning: Complex target with DefaultAnsatz"

You're using `momentum != 0` with the default RY-only ansatz. Switch to:
```python
from wings.ansatz_library import EfficientSU2Ansatz
config = OptimizerConfig(ansatz=EfficientSU2Ansatz(n_qubits=8, layers=4))
```

### Optimization stalls at low fidelity (F < 0.99)

1. Try different initialization: `init_strategies=["smart", "gaussian_product", "random"] * 3`
2. Check box size: `auto_optimize_box=True`
3. Increase circuit depth: `DefaultAnsatz(n_qubits=8, depth=12)` (more than default n)
4. Use basin hopping: `use_basin_hopping=True, basin_hopping_iterations=50`
5. Try log-distance entanglement: `DefaultAnsatz(n_qubits=8, entanglement="log_distance")`

### Optimization converges but fidelity plateaus at F ~ 0.9999

1. Enable log-infidelity objective (default in v0.2.0+): `use_log_objective=True`
2. Increase L-BFGS-B tolerance progression: `lbfgs_tolerances=[1e-10, 1e-12, 1e-14, 1e-15]`
3. Try natural gradient for final refinement: `optimizer.optimize_natural_gradient(params, max_steps=200)`
4. Run a campaign: `run_production_campaign(total_runs=100)` to find the global minimum

### Out of GPU memory

1. Reduce `custatevec_batch_size` or `gpu_batch_size`
2. Reduce `simulators_per_gpu` from 2 to 1
3. Use single precision for exploration: `gpu_precision="single"` (switch to "double" for refinement)
4. Reduce qubit count if possible

### HPC / SLURM Cluster Usage

WINGS automatically detects common scratch directories. Set explicitly if needed:
```bash
export GSO_BASE_DIR=/scratch/$USER/wings
export GSO_CHECKPOINT_DIR=/scratch/$USER/wings/checkpoints
```

For multi-node GPU campaigns with SLURM:
```bash
srun --gpus-per-node=4 --ntasks=1 python my_campaign.py
```

Inside `my_campaign.py`:
```python
from wings import run_production_campaign
results = run_production_campaign(
    n_qubits=14, sigma=0.5, total_runs=1000,
    use_multi_gpu=True, gpu_device_ids=[0, 1, 2, 3],
)
```
