# Changelog

All notable changes to WINGS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial public release

## [0.1.0] - 2026-02-03

### Added

#### Core Features
- `GaussianOptimizer` class for variational quantum state preparation
- Support for multiple target wavefunctions: Gaussian, Lorentzian, hyperbolic secant, and custom functions
- High-precision optimization achieving fidelities > 0.999999999
- Parameter-shift rule for exact analytic gradients

#### Ansatz Support
- `DefaultAnsatz` with hardware-efficient RY + CNOT structure
- `CustomHardwareEfficientAnsatz` with configurable entanglement patterns (linear, circular, full)
- `AnsatzProtocol` for implementing custom ansatze

#### GPU Acceleration
- NVIDIA cuStateVec integration via cuQuantum
- Qiskit Aer GPU backend support
- Multi-GPU parallelization for large-scale optimization
- Automatic backend selection with fallback

#### Optimization Methods
- L-BFGS-B quasi-Newton optimization
- Adam optimizer with warm restarts
- Basin hopping for global optimization
- Multi-stage adaptive pipelines

#### Campaign Management
- `CampaignConfig` for large-scale optimization campaigns
- `OptimizationManager` for running thousands of optimizations
- Automatic checkpointing and resume functionality
- Configurable strategy distribution

#### Utilities
- Cross-platform path configuration for HPC clusters
- Comprehensive benchmarking suite
- Command-line interface (`gso`)
- Result visualization and export

### Dependencies
- Qiskit >= 1.0
- NumPy >= 1.20
- SciPy >= 1.7
- Optional: cuQuantum, CuPy for GPU support

---

## Version Numbering

- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

[Unreleased]: https://github.com/yourusername/wings/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/yourusername/wings/releases/tag/v0.1.0
