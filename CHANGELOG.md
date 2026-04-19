# Changelog

All notable changes to WINGS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.1] - 2026-04-19

### Changed
- Version bump for PyPI re-publish. Includes accumulated fixes to `adam.py`, `campaign.py`, `cli.py`, `compat.py`, `evaluators/__init__.py`, `evaluators/cpu.py`, `evaluators/gpu.py`, `paths.py`, `results.py`, and `types.py` made on top of the 0.4.0 tag.

## [0.4.0] - 2026-04-01

### Added

#### Tracy-Widom Distribution Targets
- `TargetFunction.TRACY_WIDOM_GOE` (β=1, GOE), `TRACY_WIDOM_GUE` (β=2, GUE), `TRACY_WIDOM_GSE` (β=4, GSE) as first-class enum members with automatic box sizing (`box_size=8.0`) and optimizer dispatch via Painlevé II solver.
- `tracy_widom_pdf()`, `tracy_widom_wavefunction()`, `list_tracy_widom_targets()` exposed in public API.
- `solve_painleve_ii()` for the Hastings-McLeod solution to the Painlevé II transcendent (DOP853 integration, 1e-12 tolerance).
- Analytic + numerical expressibility proof confirming DefaultAnsatz (RY+CNOT) achieves F > 0.99 for all three TW distributions at depth L=2N.

#### Composable Pipeline System
- `Pipeline` class with composable optimization stages: `InitSearch`, `Adam`, `SPSA`, `LBFGS`, `NaturalGradient`, `BasinHopping`, `Newton`, `GrowCircuit`.

#### JAX Backend
- JAX autodiff backend (`backend="jax"`) with `compute_gradient_jax_default_ansatz()`.

#### Time Evolution
- Split-operator time evolution module: `split_operator_step()`, `evolve_classical()`, `make_grid()`, with built-in potentials (`free_particle`, `harmonic_potential`, `morse_potential`, `lennard_jones_potential`).

#### Hardware Execution
- Hardware-native transpilation and execution: `transpile_for_hardware()`, `classical_state_fidelity()`, `counts_to_probabilities()`, `HardwareResult`.

#### Other
- `OptimizationResult` dataclass for structured results.
- N-dimensional grid support (`NDGrid`, `gaussian_nd`).
- Noise-aware optimization (`NoiseConfig`).
- MPS initialization: `mps_decompose()`, `mps_to_statevector()`, `mps_initial_params()`.

### Fixed
- CI workflow moved to `.github/workflows/` (was incorrectly in `tests/.github/workflows/`).
- CI install extras corrected from `[test]` to `[dev]`.
- Coverage path corrected from `src/gaussian_state_optimizer` to `src/wings`.

---

## [0.3.0] - 2026-03-29

### Added

#### New Ansatz Designs
- `EfficientSU2Ansatz` with alternating RY-RZ rotations per qubit and configurable entanglement. Unlike `DefaultAnsatz`, this ansatz can produce complex amplitudes, making it suitable for momentum wavepackets and other phase-structured targets. Parameters per layer: `2 * n_qubits`.
- `generate_entanglement_map()` utility supporting six entanglement patterns: `linear`, `circular`, `reverse_linear`, `parity`, `log_distance`, and `full`. Log-distance is particularly effective for Gaussian state preparation because qubit `i` (representing position bit `2^i`) needs correlations at multiple spatial scales.
- Entanglement topology parameter added to `DefaultAnsatz`: `DefaultAnsatz(n_qubits=8, entanglement="log_distance")`. Backward compatible -- default remains `"linear"`.

#### Adaptive Circuit Depth
- `GaussianOptimizer.grow_circuit()` method for incremental layer-ramp optimization. Starts with a shallow circuit, optimizes to convergence, then adds layers initialized near zero to preserve achieved fidelity. Avoids barren plateaus that afflict deep random circuits.
- `OptimizationPipeline` fields: `use_adaptive_depth`, `min_depth`, `max_depth`, `depth_step`, `depth_fidelity_plateau`.

#### Barren Plateau Detection
- `BarrenPlateauDetector` class monitors gradient norms in real time during Adam optimization. Detects when gradients vanish at low fidelity (barren plateau) versus at high fidelity (convergence). Automatically triggers random restart mitigation with escalating strategies (`random_restart` -> `reduce_depth` -> `identity_init`).
- Integrated into `optimize_adam()` -- active by default with conservative thresholds.

#### Quantum Natural Gradient
- `compute_qfim_diagonal()` computes the diagonal of the Quantum Fisher Information Matrix via the parameter-shift rule. Cost: `2 * n_params` circuit evaluations (same as a standard gradient).
- `compute_natural_gradient()` rescales the Euclidean gradient by the inverse diagonal QFIM with Tikhonov regularization, accounting for the geometry of the quantum state manifold.
- `GaussianOptimizer.optimize_natural_gradient()` method combining QNG with Adam momentum for improved convergence near the optimum.
- `OptimizationPipeline` fields: `use_natural_gradient`, `natural_gradient_regularization`.

#### Warm-Start Transfer Learning
- `transfer_params()` function transfers optimized parameters from a smaller qubit count to a larger one. Preserves existing layer parameters and initializes new qubit/layer parameters near zero. Assumes `DefaultAnsatz` structure (`depth = n_qubits`, `n_params = n_qubits^2`).

#### CUDA Stream Pipelining
- Per-simulator CUDA streams in `CuStateVecSimulator` for overlapping gate operations across batched circuit evaluations.
- `apply_rz()` gate added to `CuStateVecSimulator`, enabling native cuStateVec acceleration for `EfficientSU2Ansatz` and other RZ-gate ansatze.
- `synchronize()` method for explicit stream synchronization.

### Changed
- Transpiler basis gates now include `rz` alongside `ry`, `cx`, `x` to support RZ-gate ansatze.

---

## [0.2.0] - 2026-03-29

### Added

#### Optimization Speed
- **Vectorized parameter-shift construction**: Replaced Python for-loops with NumPy vectorized operations in `_compute_gradient_gpu_impl()`, `compute_gradient_batched()`, and `compute_gradient_parallel()`. Eliminates O(n_params) Python loop overhead in gradient computation. Gradient extraction also vectorized via array slicing.
- **Stochastic parameter-shift gradients**: `compute_gradient_stochastic()` method samples a random subset of `k = n_params * fraction` coordinates per gradient evaluation. With Adam's momentum buffers smoothing missing components, `fraction=0.5` yields ~2x fewer evaluations per step with minimal convergence impact. Controlled by `OptimizerConfig.gradient_sample_fraction` (default `1.0` = full gradient, backward compatible).
- **SPSA optimizer**: New `SPSAOptimizer` class (`src/wings/spsa.py`) estimates the full gradient from only 2 function evaluations regardless of parameter count, using Rademacher perturbation vectors and Spall (1992/1998) gain sequences. `GaussianOptimizer.optimize_spsa()` provides the same interface as `optimize_adam()`. Supports multi-sample averaging via `n_avg` parameter.

#### Accuracy
- **Logarithmic fidelity objective**: `objective_log_infidelity()` and `objective_and_gradient_log_infidelity()` methods minimize `log(1-F)` instead of `-F` during L-BFGS-B refinement stages. The gradient amplification factor `1/(1-F)` keeps gradients informative as `F -> 1`, breaking through premature convergence at `F > 0.999999`. Activated automatically when `pipeline.use_log_objective=True` (default) and fidelity exceeds `log_objective_threshold`.
- **Direct infidelity computation**: `_compute_infidelity_direct()` uses the Pythagorean identity `1 - |<t|p>|^2 = ||p - <t|p>*t||^2` to compute infidelity without catastrophic cancellation. Accurate to ~1e-15 versus ~1e-12 for naive `1 - F` when `F` is near unity. Used in pipeline finalization and log-infidelity objective.
- **Adaptive box-size computation**: `optimal_box_size()` analytically minimizes the sum of truncation error (`erfc(L / (sqrt(2) * sigma))`) and discretization error (`(dx / sigma)^2`) for Gaussian, Lorentzian, and hyperbolic secant targets. Enabled via `OptimizerConfig(auto_optimize_box=True)`.

#### Wavefunction Library
- New module `src/wings/wavefunctions.py` with six analytically-defined wavefunctions:
  - `harmonic_oscillator_eigenstate(x, n, sigma, x0)` -- quantum harmonic oscillator `|n>` via Hermite polynomials
  - `superposition_of_gaussians(x, centers, sigmas, amplitudes)` -- coherent superposition with complex amplitude support
  - `airy_wavefunction(x, x0, scale)` -- Airy function `Ai(x)` for linear potentials
  - `morse_oscillator_eigenstate(x, n, D_e, a, x_e, mu)` -- anharmonic vibrational states via generalized Laguerre polynomials
  - `squeezed_gaussian(x, x0, sigma, squeeze_r)` -- minimum-uncertainty state with tunable position/momentum squeezing
  - `plane_wave_packet(x, k0, sigma, x0)` -- Gaussian wavepacket with initial momentum `exp(ik_0 x)`
- `list_wavefunctions()` returns a dictionary of available wavefunctions with descriptions.

#### Complex-Valued Wavepackets
- `OptimizerConfig.momentum` parameter applies `exp(ikx)` phase to any target wavefunction.
- `TargetFunction.GAUSSIAN_WAVEPACKET` enum for direct Gaussian-with-momentum construction.
- **Ansatz validation**: `UserWarning` issued when `momentum != 0` with `DefaultAnsatz`, which cannot encode complex phases (RY + CNOT produces only real amplitudes). Recommends `CustomHardwareEfficientAnsatz(rotation_gates=['ry', 'rz'])` or `EfficientSU2Ansatz`.
- `optimize_gaussian_state()` and `quick_optimize()` accept `momentum` parameter.

### Changed
- `_pipeline_finalize()` now uses `_compute_infidelity_direct()` for the reported infidelity, improving precision at extreme fidelities.

---

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

[Unreleased]: https://github.com/jmcourtneyuga/wings/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/jmcourtneyuga/wings/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/jmcourtneyuga/wings/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/jmcourtneyuga/wings/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/jmcourtneyuga/wings/releases/tag/v0.1.0
