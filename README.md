# WINGS: Wavepacket Initialization on Neighboring Grid States

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CUDA](https://img.shields.io/badge/CUDA-11.0+-green.svg)](https://developer.nvidia.com/cuda-toolkit)

**WINGS** is a high-performance, GPU-accelerated variational quantum state preparation toolkit for preparing continuous wavepacket states on discrete qubit grids. 
It enables researchers to achieve machine-precision fidelities (>0.999999999) when 
encoding Gaussian, Lorentzian, and other continuous wavefunctions into quantum circuits.

This degree of precision is not (!) a guarantee, seeing as the optimization landscape is full of barren plateaus and local minima. 
This is another reason why you can perform multiple jobs and batched jobs.

## Documentation

See [DOCUMENTATION.md] for the full API reference and guides.

## Overview

Quantum algorithms for simulating continuous systems—such as molecular dynamics, quantum field theories, and wave propagation—require encoding continuous wavefunctions onto discrete qubit registers. WINGS addresses this challenge by providing:

- **Variational quantum circuits** optimized to prepare target wavepackets with ultra-high fidelity
- **GPU acceleration** via NVIDIA cuStateVec (cuQuantum) for 5–20x speedups over CPU
- **Multi-GPU support** for scaling to larger qubit counts
- **Production-grade tooling** for running thousands of optimization campaigns with automatic checkpointing

## Features

### Target Wavefunctions

- **Gaussian**: Standard Gaussian wavepackets with configurable width (σ) and center (x₀)
- **Lorentzian**: Cauchy distributions for heavy-tailed profiles
- **Hyperbolic Secant**: Soliton-like profiles for nonlinear dynamics
- **Custom Functions**: Any user-defined wavefunction via callback

### Acceleration Backends

| Backend | Description | Speedup |
|---------|-------------|---------|
| CPU | Qiskit Statevector (baseline) | 1x |
| GPU (Aer) | Qiskit Aer with CUDA | 2–5x |
| cuStateVec | NVIDIA cuQuantum direct API | 5–20x |
| Multi-GPU | Parallel cuStateVec across GPUs | Linear scaling |

### Optimization Methods

- **L-BFGS-B**: Quasi-Newton method for high-precision convergence
- **Adam**: Momentum-based optimizer for escaping local minima
- **Basin Hopping**: Global optimization with local refinement
- **Hybrid Pipelines**: Adaptive multi-stage optimization

### Production Features

- Automatic checkpointing and resume for long campaigns
- Parallel campaign execution with configurable strategies
- Comprehensive result aggregation and statistical analysis
- Cross-platform path management for HPC clusters

## Installation

### Prerequisites

- Python 3.9 or higher
- Qiskit 1.0+
- NumPy, SciPy

### Basic Installation

```bash
pip install wings-quantum
```

### With GPU Support

```bash
# Install CUDA toolkit (11.0+) first, then:
pip install wings-quantum[gpu]

# For cuStateVec acceleration (recommended):
pip install cuquantum-python cupy-cuda11x
```

### From Source

```bash
git clone https://github.com/jmcourtneyuga/wings.git
cd wings
pip install -e ".[dev,gpu]"
```

### Verify Installation

```bash
# Check available backends
python -c "from wings import print_backend_info; print_backend_info()"

# Or use the CLI
gso info
```

## Quick Start

### Basic Optimization

```python
from wings import GaussianOptimizer, OptimizerConfig

# Configure the optimization
config = OptimizerConfig(
    n_qubits=8,           # 2^8 = 256 grid points
    sigma=0.5,            # Gaussian width
    use_custatevec=True,  # Enable GPU acceleration
)

# Create optimizer and run
optimizer = GaussianOptimizer(config)
results = optimizer.optimize_ultra_precision(target_infidelity=1e-10)

print(f"Fidelity achieved: {results['fidelity']:.12f}")
print(f"Infidelity (1-F):  {results['infidelity']:.3e}")
```

### High-Level Convenience API

```python
from wings import optimize_gaussian_state

results, optimizer = optimize_gaussian_state(
    n_qubits=10,
    sigma=0.5,
    target_infidelity=1e-11,
    max_time=3600,  # 1 hour
    plot=True,
    save=True,
)
```

### Production Campaign

```python
from wings import run_production_campaign

results = run_production_campaign(
    n_qubits=8,
    sigma=0.5,
    total_runs=1000,
    target_infidelity=1e-11,
)

results.print_summary()
```

### Custom Target Wavefunction

```python
import numpy as np
from wings import OptimizerConfig, GaussianOptimizer, TargetFunction

# Define a double-Gaussian wavepacket
def double_gaussian(x):
    return np.exp(-((x - 1)**2) / 0.5) + np.exp(-((x + 1)**2) / 0.5)

config = OptimizerConfig(
    n_qubits=10,
    target_function=TargetFunction.CUSTOM,
    custom_target_fn=double_gaussian,
    box_size=5.0,
)

optimizer = GaussianOptimizer(config)
results = optimizer.optimize_ultra_precision(target_infidelity=1e-9)
```

### Custom Ansatz

```python
from wings import CustomHardwareEfficientAnsatz, OptimizerConfig, GaussianOptimizer

# Create a custom hardware-efficient ansatz
ansatz = CustomHardwareEfficientAnsatz(
    n_qubits=8,
    layers=6,
    entanglement='circular',  # 'linear', 'circular', or 'full'
    rotation_gates=['ry', 'rz'],
)

config = OptimizerConfig(
    n_qubits=8,
    sigma=0.5,
    ansatz=ansatz,
    use_custatevec=True,
)

optimizer = GaussianOptimizer(config)
results = optimizer.run_optimization()
```

### Exporting Optimized Circuits:

from wings import GaussianOptimizer, OptimizerConfig

# Run optimization
config = OptimizerConfig(n_qubits=8, sigma=0.5)
optimizer = GaussianOptimizer(config)
results = optimizer.optimize_ultra_precision(target_infidelity=1e-10)

# Get the optimized circuit
circuit = optimizer.get_optimized_circuit()
print(circuit.draw())

# Export to OpenQASM 2.0
qasm_str = optimizer.export_qasm()
print(qasm_str)

# Export to OpenQASM 3.0
qasm3_str = optimizer.export_qasm(version=3)

# Save to file (various formats)
optimizer.save_circuit("gaussian_prep.qasm")          # OpenQASM 2.0
optimizer.save_circuit("gaussian_prep.qasm3", format='qasm3')  # OpenQASM 3.0
optimizer.save_circuit("gaussian_prep.qpy", format='qpy')      # Qiskit QPY
optimizer.save_circuit("gaussian_prep.png", format='png')      # Circuit diagram

## Command-Line Interface

WINGS includes a full CLI for common tasks:

```bash
# Run optimization
gso optimize --qubits 8 --sigma 0.5 --target-infidelity 1e-10

# Run production campaign
gso campaign --qubits 8 --sigma 0.5 --runs 1000 --resume

# Benchmark GPU performance
gso benchmark --qubits 12

# Find GPU crossover point
gso crossover --min-qubits 6 --max-qubits 18

# Show backend information
gso info

# List/load campaigns
gso campaigns list
gso campaigns load campaign_q8_s0.50_20250203_120000
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GSO_BASE_DIR` | Base directory for all data | `~/.wings` |
| `GSO_CACHE_DIR` | Coefficient cache directory | `$GSO_BASE_DIR/cache` |
| `GSO_OUTPUT_DIR` | Simulation outputs | `$GSO_BASE_DIR/output` |
| `GSO_CHECKPOINT_DIR` | Optimization checkpoints | `$GSO_BASE_DIR/checkpoints` |
| `GSO_CAMPAIGN_DIR` | Campaign results | `$GSO_BASE_DIR/campaigns` |

### HPC Cluster Support

WINGS automatically detects common HPC scratch directories:

- `/scratch/$USER` (SLURM)
- `/work/$USER` (PBS/Torque)
- `/gpfs/scratch/$USER` (GPFS)
- `/lustre/scratch/$USER` (Lustre)
- `$SCRATCH` and `$WORK` environment variables

## Project Structure

```
wings/
├── src/
│   └── wings/
│       ├── __init__.py          # Public API and lazy imports
│       ├── py.typed             # PEP 561 marker
│       ├── optimizer.py         # Core GaussianOptimizer class
│       ├── config.py            # Configuration dataclasses
│       ├── ansatz.py            # Quantum circuit ansatz implementations
│       ├── campaign.py          # Large-scale campaign management
│       ├── results.py           # Result tracking and analysis
│       ├── adam.py              # Adam optimizer implementation
│       ├── convenience.py       # High-level convenience functions
│       ├── cli.py               # Command-line interface
│       ├── benchmarks.py        # Performance benchmarking
│       ├── paths.py             # Cross-platform path management
│       ├── compat.py            # cuQuantum compatibility layer
│       ├── types.py             # Type aliases
│       └── evaluators/
│           ├── __init__.py
│           ├── cpu.py           # Thread-safe CPU evaluator
│           ├── gpu.py           # Qiskit Aer GPU evaluator
│           └── custatevec.py    # NVIDIA cuStateVec evaluator
├── tests/
│   ├── conftest.py              # Shared fixtures
│   ├── unit/                    # Unit tests
│   └── integration/             # Integration tests
├── docs/                        # Sphinx documentation
├── examples/                    # Example notebooks and scripts
├── pyproject.toml
├── Makefile
├── README.md
├── DOCUMENTATION.md
├── CHANGELOG.md
└── LICENSE
```

## Performance

### Typical Results

| Qubits | Grid Points | Parameters | GPU Time | Best Infidelity |
|--------|-------------|------------|----------|-----------------|
| 4 | 16 | 16 | ~1000s | 1e-10
| 5 | 32 | 25 | ~1500s | 1e-8 | 
| 8 | 256 | 64 | ~3000s | 1e-6 |

### Benchmarking

```python
from wings import benchmark_gpu, find_gpu_crossover

# Benchmark specific configuration
result = benchmark_gpu(n_qubits=12, sigma=0.5)
print(f"GPU speedup: {result.results['custatevec']['speedup_vs_cpu']:.1f}x")

# Find crossover point where GPU becomes faster
crossover = find_gpu_crossover(qubit_range=range(6, 18, 2))
```

## API Reference

### Core Classes

- **`GaussianOptimizer`**: Main optimizer class with all optimization methods
- **`OptimizerConfig`**: Configuration for single optimizations
- **`CampaignConfig`**: Configuration for large-scale campaigns
- **`CampaignResults`**: Aggregated results from campaigns
- **`OptimizationPipeline`**: Multi-stage optimization configuration

### Ansatz Classes

- **`DefaultAnsatz`**: Standard hardware-efficient ansatz with RY rotations and linear CNOT entanglement
- **`CustomHardwareEfficientAnsatz`**: Configurable ansatz with multiple entanglement patterns
- **`AnsatzProtocol`**: Protocol for implementing custom ansatze

### Evaluators

- **`ThreadSafeCircuitEvaluator`**: CPU-based thread-safe evaluator
- **`GPUCircuitEvaluator`**: Qiskit Aer GPU evaluator
- **`CuStateVecEvaluator`**: Single-GPU cuStateVec evaluator
- **`BatchedCuStateVecEvaluator`**: Batched GPU evaluation
- **`MultiGPUBatchEvaluator`**: Multi-GPU parallel evaluator

### Convenience Functions

- **`optimize_gaussian_state()`**: High-level optimization with sensible defaults
- **`quick_optimize()`**: Fast optimization for testing
- **`run_production_campaign()`**: Launch large-scale campaigns
- **`load_campaign_results()`**: Load saved campaign results
- **`list_campaigns()`**: List available campaigns

### Exporting

- **`build_optimized_circuit()`**: Build a QuantumCircuit with bound parameters
- **`export_to_qasm()`**: Export to OpenQASM 2.0 string
- **`export_to_qasm3()`**: Export to OpenQASM 3.0 string  
- **`save_circuit()`**: Save circuit to file (QASM, QPY, PNG, SVG, PDF)


## Development

### Setup

```bash
git clone https://github.com/jmcourtneyuga/wings.git
cd wings
pip install -e ".[dev]"
```

### Running Tests

```bash
# Fast unit tests
make test

# All tests including integration
make test-all

# With coverage
make coverage
```

### Code Quality

```bash
# Format and lint
make format
make lint

# Full check (format + lint + test)
make check
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

If you use WINGS in your research, please cite:

```bibtex
@software{wings2026,
  title={WINGS: Wavepacket Initialization on Neighboring Grid States},
  author={Joshua Courtney},
  year={2026},
  url={https://github.com/jmcourtneyuga/wings}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Qiskit](https://qiskit.org/) for the quantum circuit framework
- [NVIDIA cuQuantum](https://developer.nvidia.com/cuquantum-sdk) for GPU acceleration
- [CuPy](https://cupy.dev/) for GPU array operations

## See Also

- [Full Documentation](DOCUMENTATION.md) - Detailed API documentation and guides
- [Examples](examples/) - Jupyter notebooks and scripts
