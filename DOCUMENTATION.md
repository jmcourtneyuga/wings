# WINGS Documentation

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Core Concepts](#core-concepts)
4. [API Reference](#api-reference)
5. [Configuration Guide](#configuration-guide)
6. [Optimization Strategies](#optimization-strategies)
7. [GPU Acceleration](#gpu-acceleration)
8. [Campaign Management](#campaign-management)
9. [Custom Ansatze](#custom-ansatze)
10. [Troubleshooting](#troubleshooting)
11. [Advanced Topics](#advanced-topics)

---

## Introduction

WINGS (Wavepacket Initialization on Neighboring Grid States) is a specialized toolkit for preparing continuous wavefunction states on discrete quantum registers using variational quantum algorithms. The primary use case is encoding Gaussian and other continuous probability distributions into quantum circuits with machine-precision fidelity.

### The Problem

Quantum algorithms for continuous-variable problems (molecular dynamics, quantum field theory simulations, solving PDEs) 
require mapping continuous wavefunctions ψ(x) onto a finite set of 2^n computational basis states. 
This discretization introduces errors, and finding the optimal circuit parameters to minimize these errors is computationally intensive.

### The Solution

WINGS provides:

1. **Parameterized quantum circuits** (ansatze) designed for wavefunction preparation
2. **High-performance gradient computation** using the parameter-shift rule
3. **GPU acceleration** to evaluate quantum circuits orders of magnitude faster
4. **Sophisticated optimization pipelines** combining Adam, L-BFGS-B, and basin hopping
5. **Production tooling** for running and managing thousands of optimization runs

---

## Installation

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.9 | 3.11+ |
| RAM | 8 GB | 32 GB |
| GPU | CUDA 11.0+ | NVIDIA A100/H100 |
| Storage | 1 GB | 10 GB (for campaigns) |

### Installation Options

#### Option 1: pip (Recommended)

```bash
# CPU only
pip install wings-quantum

# With GPU support
pip install wings-quantum[gpu]

# Development installation
pip install wings-quantum[dev,gpu]
```

#### Option 2: From Source

```bash
git clone https://github.com/jmcourtneyuga/wings.git
cd wings
pip install -e .
```

#### Option 3: Conda Environment

```bash
conda create -n wings python=3.11
conda activate wings
conda install -c conda-forge qiskit numpy scipy matplotlib
pip install cuquantum-python cupy-cuda11x  # For GPU support
pip install -e .
```

### Verifying Installation

```python
from wings import get_backend_info, print_backend_info

# Print available backends
print_backend_info()

# Programmatic check
info = get_backend_info()
print(f"cuStateVec available: {info['custatevec']}")
print(f"GPU name: {info['gpu_name']}")
```

---

## Core Concepts

### Wavefunction Discretization

A continuous wavefunction ψ(x) is mapped to a discrete quantum state |ψ⟩ on n qubits:

```
|ψ⟩ = Σᵢ ψ(xᵢ) |i⟩
```

where xᵢ are grid points in the interval [-L, L] and L is the box size. The number of grid points is 2^n.

### Variational Circuit

The ansatz circuit U(θ) prepares a parameterized state:

```
|ψ(θ)⟩ = U(θ)|0⟩
```

Optimization finds θ* that maximizes the fidelity:

```
F = |⟨ψ_target|ψ(θ)⟩|²
```

### Parameter-Shift Gradients

For circuits containing RY gates, exact gradients are computed via:

```
∂F/∂θᵢ = [F(θᵢ + π/2) - F(θᵢ - π/2)] / 2
```

This requires 2n circuit evaluations per gradient computation.

---

## API Reference

### GaussianOptimizer

The main optimization class.

```python
class GaussianOptimizer:
    """
    High-precision optimizer for quantum state preparation.
    
    Parameters
    ----------
    config : OptimizerConfig
        Configuration object specifying all optimization parameters.
    
    Attributes
    ----------
    config : OptimizerConfig
        The configuration used for this optimizer.
    target : np.ndarray
        The normalized target wavefunction.
    n_params : int
        Number of variational parameters.
    best_fidelity : float
        Best fidelity achieved during optimization.
    best_params : np.ndarray
        Parameters corresponding to best fidelity.
    history : dict
        Optimization history with fidelity and iteration data.
    """
```

#### Key Methods

##### `optimize_ultra_precision`

```python
def optimize_ultra_precision(
    self,
    target_infidelity: float = 1e-10,
    max_total_time: float = 3600,
    initial_params: np.ndarray = None,
) -> Dict[str, Any]:
    """
    Multi-stage optimization targeting extreme precision.
    
    Combines multiple optimization strategies:
    1. Smart initialization with multiple strategies
    2. Adam pre-training to escape shallow minima
    3. L-BFGS-B refinement with progressive tolerances
    4. Optional basin hopping for global search
    5. Final fine-tuning stage
    
    Parameters
    ----------
    target_infidelity : float
        Target value for 1-F (default: 1e-10)
    max_total_time : float
        Maximum total optimization time in seconds
    initial_params : np.ndarray, optional
        Starting parameters (None = auto-select)
    
    Returns
    -------
    dict
        Results dictionary containing:
        - 'fidelity': float - Final fidelity achieved
        - 'infidelity': float - 1 - fidelity
        - 'optimal_params': np.ndarray - Best parameters
        - 'time': float - Total optimization time
        - 'n_evaluations': int - Number of function evaluations
        - 'circuit_mean': float - Mean position of prepared state
        - 'circuit_std': float - Standard deviation of prepared state
        - 'target_mean': float - Mean of target wavefunction
        - 'target_std': float - Standard deviation of target
        - 'success': bool - Whether target was achieved
    """
```

##### `run_optimization`

```python
def run_optimization(
    self,
    pipeline: OptimizationPipeline = None,
    initial_params: np.ndarray = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Unified optimization entry point with configurable pipeline.
    
    Parameters
    ----------
    pipeline : OptimizationPipeline, optional
        Configuration for multi-stage optimization
    initial_params : np.ndarray, optional
        Starting parameters
    **kwargs
        Override pipeline settings
    
    Returns
    -------
    dict
        Optimization results
    """
```

##### `get_statevector`

```python
def get_statevector(
    self, 
    params: np.ndarray, 
    backend: str = 'auto'
) -> np.ndarray:
    """
    Compute the statevector for given parameters.
    
    Parameters
    ----------
    params : np.ndarray
        Circuit parameters of shape (n_params,)
    backend : str
        Backend selection: 'auto', 'custatevec', 'gpu', 'cpu'
    
    Returns
    -------
    np.ndarray
        Complex statevector of shape (2^n_qubits,)
    """
```

##### `compute_fidelity`

```python
def compute_fidelity(
    self, 
    params: np.ndarray = None, 
    psi: np.ndarray = None, 
    backend: str = 'auto'
) -> float:
    """
    Compute fidelity with target wavefunction.
    
    Parameters
    ----------
    params : np.ndarray, optional
        Circuit parameters (provide this OR psi)
    psi : np.ndarray, optional
        Pre-computed statevector (provide this OR params)
    backend : str
        Backend selection
    
    Returns
    -------
    float
        Fidelity value in [0, 1]
    """
```

##### `compute_gradient`

```python
def compute_gradient(
    self, 
    params: np.ndarray, 
    method: str = 'auto'
) -> np.ndarray:
    """
    Compute gradient using parameter-shift rule.
    
    Parameters
    ----------
    params : np.ndarray
        Current parameters
    method : str
        Method selection: 'auto', 'multi_gpu', 'custatevec', 
        'gpu', 'parallel', 'sequential'
    
    Returns
    -------
    np.ndarray
        Gradient of negative fidelity (for minimization)
    """
```

##### `get_initial_params`

```python
def get_initial_params(self, strategy: str = 'smart') -> np.ndarray:
    """
    Generate initial parameters using specified strategy.
    
    Parameters
    ----------
    strategy : str
        Initialization strategy:
        - 'smart': Heuristic based on target properties
        - 'gaussian_product': Product state approximation
        - 'random': Uniform random in [-π, π]
        - 'zero': All zeros
        - 'small_random': Small perturbations around zero
    
    Returns
    -------
    np.ndarray
        Initial parameter values
    """
```

---

## Configuration Guide

### OptimizerConfig

Complete configuration for single optimizations.

```python
from wings import OptimizerConfig, TargetFunction

config = OptimizerConfig(
    # Problem definition
    n_qubits=8,                              # Number of qubits (2^n grid points)
    sigma=0.5,                               # Gaussian width parameter
    x0=0.0,                                  # Wavefunction center position
    box_size=None,                           # Box size (auto if None)
    
    # Target wavefunction
    target_function=TargetFunction.GAUSSIAN, # GAUSSIAN, LORENTZIAN, SECH, CUSTOM
    gamma=None,                              # Width for Lorentzian
    custom_target_fn=None,                   # Custom function f(x) -> array
    
    # Optimizer settings
    method='L-BFGS-B',                       # Optimization method
    max_iter=10000,                          # Maximum iterations
    max_fun=50000,                           # Maximum function evaluations
    tolerance=1e-12,                         # Convergence tolerance
    gtol=1e-12,                              # Gradient tolerance
    use_analytic_gradients=True,             # Use parameter-shift gradients
    
    # GPU settings
    use_gpu=True,                            # Enable GPU (Aer)
    use_custatevec=True,                     # Enable cuStateVec
    use_multi_gpu=False,                     # Enable multi-GPU
    gpu_device_ids=None,                     # Specific GPU IDs (None = auto)
    gpu_precision='double',                  # 'double' or 'single'
    
    # Parallelization
    n_workers=None,                          # CPU workers (None = auto)
    parallel_gradients=True,                 # Parallelize gradient computation
    
    # Output
    verbose=True,                            # Print progress
)
```

### CampaignConfig

Configuration for large-scale optimization campaigns.

```python
from wings import CampaignConfig

config = CampaignConfig(
    # Problem
    n_qubits=8,
    sigma=0.5,
    
    # Scale
    total_runs=1000,
    runs_per_batch=50,
    
    # Targets
    target_infidelity=1e-10,
    acceptable_infidelity=1e-8,
    
    # Strategy distribution
    strategy_weights={
        'smart': 0.3,
        'gaussian_product': 0.2,
        'random': 0.4,
        'perturb_best': 0.1,
    },
    
    # Per-run settings
    max_iter_per_run=5000,
    use_ultra_precision=True,
    ultra_precision_time_limit=300,
    
    # Checkpointing
    checkpoint_interval=10,
    resume_from_checkpoint=True,
    
    # Output
    save_top_n_results=100,
    verbose=1,
)
```
### Circuit export
def get_optimized_circuit(
    self,
    params: np.ndarray = None,
    include_measurements: bool = False,
) -> QuantumCircuit:
    """
    Get the optimized quantum circuit with parameters bound.
    
    Parameters
    ----------
    params : np.ndarray, optional
        Parameter values. If None, uses self.best_params
    include_measurements : bool, default False
        Whether to add measurement gates
        
    Returns
    -------
    QuantumCircuit
        Qiskit circuit with concrete parameter values
    """

### Export_Qasm

def export_qasm(
    self,
    params: np.ndarray = None,
    include_measurements: bool = False,
    version: int = 2,
) -> str:
    """
    Export optimized circuit to OpenQASM string.
    
    Parameters
    ----------
    params : np.ndarray, optional
        Parameter values. If None, uses self.best_params
    include_measurements : bool, default False
        Whether to include measurement gates
    version : int, default 2
        OpenQASM version (2 or 3)
        
    Returns
    -------
    str
        OpenQASM format string
    """

### Save Circuit
def save_circuit(
    self,
    filepath: str,
    params: np.ndarray = None,
    format: str = 'qasm',
    include_measurements: bool = False,
) -> str:
    """
    Save optimized circuit to file.
    
    Parameters
    ----------
    filepath : str
        Output file path
    params : np.ndarray, optional
        Parameter values. If None, uses self.best_params
    format : str, default 'qasm'
        Output format: 'qasm', 'qasm3', 'qpy', 'png', 'svg', 'pdf'
    include_measurements : bool, default False
        Whether to include measurement gates
        
    Returns
    -------
    str
        Path to saved file
    """

#### Standalone Export Functions

For more control, you can use the standalone functions from `wings.export`:

```python
from wings.export import (
    build_optimized_circuit,  # Returns QuantumCircuit
    export_to_qasm,           # Returns QASM 2.0 string
    export_to_qasm3,          # Returns QASM 3.0 string
    save_circuit,             # Saves to file
)

# Build circuit with custom parameters
circuit = build_optimized_circuit(optimizer, my_params, include_measurements=True)

# Export with explicit parameters (not best_params)
qasm = export_to_qasm(optimizer, params=my_params)


### Integration with other Frameworks (e.g. Cirq)

import cirq
from cirq.contrib.qasm_import import circuit_from_qasm
qasm_str = optimizer.export_qasm()
cirq_circuit = circuit_from_qasm(qasm_str)

# Load QPY for transfer between Qiskit versions
from qiskit.qpy import load
with open("circuit.qpy", "rb") as f:
    circuits = load(f)
    circuit = circuits[0]



---

## Optimization Strategies

### Initialization Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| `smart` | Heuristic based on target shape | General use |
| `gaussian_product` | Product state approximation | Gaussian targets |
| `random` | Uniform random in [-π, π] | Exploration |
| `perturb_best` | Perturbation of best known | Refinement |
| `zero` | All parameters zero | Testing |

### Multi-Stage Pipeline

```python
from wings import OptimizationPipeline

pipeline = OptimizationPipeline(
    mode='ultra',
    target_fidelity=0.9999999999,
    max_total_time=3600,
    
    # Stage 1: Adam pre-training
    use_adam_stage=True,
    adam_max_steps=20000,
    adam_lr=0.01,
    adam_time_fraction=0.3,
    
    # Stage 2: Basin hopping
    use_basin_hopping=True,
    basin_hopping_threshold=0.9999,
    basin_hopping_iterations=30,
    
    # Stage 3: L-BFGS-B refinement
    use_lbfgs_refinement=True,
    lbfgs_tolerances=[1e-10, 1e-12, 1e-14],
    
    # Stage 4: Fine-tuning
    use_fine_tuning=True,
    fine_tuning_threshold=0.9999999,
)
```

---

## GPU Acceleration

### Backend Selection

WINGS automatically selects the best available backend:

1. **Multi-GPU** (if enabled and multiple GPUs available)
2. **cuStateVec** (if cuQuantum installed)
3. **Qiskit Aer GPU** (if CUDA available)
4. **CPU** (fallback)

### Manual Backend Selection

```python
# Force specific backend
statevector = optimizer.get_statevector(params, backend='custatevec')
gradient = optimizer.compute_gradient(params, method='gpu')
```

### cuStateVec Setup

```bash
# Install cuQuantum
pip install cuquantum-python

# Install CuPy (match your CUDA version)
pip install cupy-cuda11x  # CUDA 11.x
pip install cupy-cuda12x  # CUDA 12.x
```

### Multi-GPU Configuration

```python
from wings import OptimizerConfig

config = OptimizerConfig(
    n_qubits=14,
    use_multi_gpu=True,
    gpu_device_ids=[0, 1, 2, 3],  # Specific GPUs
    simulators_per_gpu=2,
)
```

---

## Campaign Management

### Running Campaigns

```python
from wings import CampaignConfig, OptimizationManager

config = CampaignConfig(
    n_qubits=8,
    sigma=0.5,
    total_runs=1000,
    target_infidelity=1e-10,
    
    # Strategy mix
    strategy_weights={
        'smart': 0.3,
        'gaussian_product': 0.2,
        'random': 0.4,
        'perturb_best': 0.1,
    },
    
    # Checkpointing
    checkpoint_interval=50,
    resume_from_checkpoint=True,
)

manager = OptimizationManager(config)
results = manager.run_campaign()
results.print_summary()
```

### Resuming Campaigns

```python
# Automatic resume
config = CampaignConfig(
    ...,
    resume_from_checkpoint=True,
)

# Manual resume
from wings import load_campaign_results
results = load_campaign_results("campaign_name")
```

### Analyzing Results

```python
# Get statistics
stats = results.get_statistics()
print(f"Success rate: {stats['success_rate']*100:.1f}%")
print(f"Best fidelity: {stats['best_fidelity']:.15f}")

# Get top results
top_10 = results.get_top_results(10)
for r in top_10:
    print(f"Run {r.run_id}: F={r.fidelity:.12f} ({r.strategy})")
```

---

## Custom Ansatze

### Implementing Custom Ansatz

```python
from qiskit import QuantumCircuit
import numpy as np

class MyCustomAnsatz:
    """Custom ansatz with alternating rotation layers."""
    
    def __init__(self, n_qubits: int, layers: int = 4):
        self._n_qubits = n_qubits
        self._layers = layers
        self._n_params = n_qubits * layers * 2  # RY and RZ per qubit per layer
    
    @property
    def n_params(self) -> int:
        return self._n_params
    
    def __call__(self, params, n_qubits=None, **kwargs):
        n = n_qubits or self._n_qubits
        qc = QuantumCircuit(n)
        
        idx = 0
        for layer in range(self._layers):
            # RY layer
            for q in range(n):
                qc.ry(params[idx], q)
                idx += 1
            
            # RZ layer
            for q in range(n):
                qc.rz(params[idx], q)
                idx += 1
            
            # Entanglement
            if layer < self._layers - 1:
                for q in range(n - 1):
                    qc.cx(q, q + 1)
        
        return qc

# Use custom ansatz
from wings import OptimizerConfig, GaussianOptimizer

ansatz = MyCustomAnsatz(n_qubits=8, layers=6)
config = OptimizerConfig(n_qubits=8, sigma=0.5, ansatz=ansatz)
optimizer = GaussianOptimizer(config)
```

---

## Troubleshooting

### Common Issues

#### "cuStateVec not available"

```bash
# Check CUDA installation
nvidia-smi

# Install cuQuantum
pip install cuquantum-python cupy-cuda11x
```

#### "Out of GPU memory"

```python
# Reduce batch size
config = OptimizerConfig(
    gpu_batch_size=32,  # Default: 64
    custatevec_batch_size=64,  # Default: 128
)
```

#### "Optimization not converging"

```python
# Try different initialization
for strategy in ['smart', 'gaussian_product', 'random']:
    params = optimizer.get_initial_params(strategy)
    results = optimizer.run_optimization(initial_params=params)
    if results['fidelity'] > 0.999:
        break

# Increase time budget
results = optimizer.optimize_ultra_precision(
    target_infidelity=1e-10,
    max_total_time=7200,  # 2 hours
)
```

#### "Poor fidelity for narrow Gaussians"

```python
# Increase grid resolution
config = OptimizerConfig(
    n_qubits=12,  # More grid points
    sigma=0.1,
    box_size=4.0,  # Smaller box for narrow Gaussian
)
```

---

## Advanced Topics

### Gradient Verification

```python
# Compare analytic vs numerical gradients
params = optimizer.get_initial_params('random')
analytic = optimizer.compute_gradient(params)
numerical = optimizer._compute_gradient_numerical(params, epsilon=1e-6)
error = np.max(np.abs(analytic - numerical))
print(f"Max gradient error: {error:.2e}")
```

### Custom Target Functions

```python
import numpy as np

# Superposition of Gaussians
def multi_gaussian(x, centers, widths, weights):
    psi = np.zeros_like(x)
    for c, w, a in zip(centers, widths, weights):
        psi += a * np.exp(-(x - c)**2 / (2 * w**2))
    return psi

# Wavepacket with momentum
def gaussian_with_momentum(x, sigma, k0):
    return np.exp(-x**2 / (2 * sigma**2)) * np.exp(1j * k0 * x)

# Use custom function
from wings import OptimizerConfig, TargetFunction

config = OptimizerConfig(
    n_qubits=10,
    target_function=TargetFunction.CUSTOM,
    custom_target_fn=lambda x: multi_gaussian(x, [-1, 1], [0.3, 0.3], [1, 1]),
)
```

### Exporting Circuits

```python
# Get optimized circuit
results = optimizer.optimize_ultra_precision()
optimal_params = results['optimal_params']

# Build final circuit
circuit = optimizer.ansatz(optimal_params, optimizer.config.n_qubits)

# Export to OpenQASM
qasm = circuit.qasm()

# Draw circuit
circuit.draw('mpl', filename='optimized_circuit.png')
```

### Integration with Other Frameworks

```python
# Using with PennyLane
import pennylane as qml

@qml.qnode(qml.device('default.qubit', wires=8))
def pennylane_circuit(params):
    # Recreate ansatz in PennyLane
    for i in range(8):
        qml.RY(params[i], wires=i)
    # ... rest of circuit
    return qml.state()
```

---

## Version History

### v0.1.0 (Initial Release)

- Core optimization functionality
- GPU acceleration via cuStateVec
- Multi-GPU support
- Campaign management
- CLI interface

---

## Support

- **Issues**: https://github.com/jmcourtneyuga/wings/issues
- **Discussions**: https://github.com/jmcourtneyuga/wings/discussions

---

*WINGS: Preparing quantum states, one wavepacket at a time.*
