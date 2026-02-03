"""Configuration classes for optimization."""

import json
import multiprocessing as mp
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np
from numpy.typing import NDArray

from .ansatz import AnsatzProtocol
from .paths import get_path_config

__all__ = [
    "TargetFunction",
    "OptimizerConfig",
    "OptimizationPipeline",
    "OptimizationStrategy",
    "CampaignConfig",
]


class TargetFunction(Enum):
    """Available target wavefunction types."""

    GAUSSIAN = "gaussian"
    LORENTZIAN = "lorentzian"
    SECH = "sech"  # Hyperbolic secant (soliton-like)
    CUSTOM = "custom"


@dataclass
class OptimizerConfig:
    """
    Configuration for the GaussianOptimizer.

    Supports multiple target wavefunctions, custom ansatze, and multi-GPU execution.

    Examples
    --------
    Basic Gaussian:

    >>> config = OptimizerConfig(n_qubits=8, sigma=0.5)

    Shifted Gaussian:

    >>> config = OptimizerConfig(n_qubits=8, sigma=0.5, x0=2.0)  # Center at x=2.0

    Lorentzian:

    >>> config = OptimizerConfig(
    ...     n_qubits=8,
    ...     target_function=TargetFunction.LORENTZIAN,
    ...     gamma=0.3,  # Width parameter
    ...     x0=1.5,     # Shift to x=1.5
    ... )

    Custom wavefunction:

    >>> def double_gaussian(x):
    ...     return np.exp(-((x-1)**2)/0.5) + np.exp(-((x+1)**2)/0.5)
    >>>
    >>> config = OptimizerConfig(
    ...     n_qubits=10,
    ...     target_function=TargetFunction.CUSTOM,
    ...     custom_target_fn=double_gaussian,
    ... )

    Multi-GPU:

    >>> config = OptimizerConfig(
    ...     n_qubits=14,
    ...     use_multi_gpu=True,
    ...     gpu_device_ids=[0, 1, 2, 3],
    ... )
    """

    # ========================================
    # Problem Parameters
    # ========================================
    n_qubits: int = 9
    sigma: float = 1.0
    x0: float = 0.0  # Center position (shift wavefunction left/right)
    box_size: Optional[float] = None

    # ========================================
    # Target Wavefunction
    # ========================================
    target_function: TargetFunction = TargetFunction.GAUSSIAN
    gamma: Optional[float] = None  # Width for Lorentzian (uses sigma if None)
    custom_target_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None

    # ========================================
    # Ansatz Configuration
    # ========================================
    ansatz: Optional[AnsatzProtocol] = None
    ansatz_depth: Optional[int] = None
    ansatz_kwargs: Optional[dict[str, Any]] = None

    # ========================================
    # Optimizer Settings
    # ========================================
    method: str = "L-BFGS-B"
    max_iter: int = 10000
    max_fun: int = 50000
    tolerance: float = 1e-12
    gtol: float = 1e-12
    use_analytic_gradients: bool = True

    # High precision mode
    high_precision: bool = True
    adaptive_tolerance: bool = True

    # For difficult cases
    use_multistart: bool = False
    n_restarts: int = 5

    # Refinement stage
    enable_refinement: bool = True
    target_fidelity: float = 0.9999999
    refinement_iter: int = 5000

    # ========================================
    # Visualization
    # ========================================
    plot_result: bool = True
    save_plot: bool = True

    # ========================================
    # GPU Configuration
    # ========================================
    use_gpu: bool = True
    gpu_precision: str = "double"  # 'double' or 'single'
    gpu_batch_size: int = 64
    gpu_blocking: bool = True

    # cuStateVec
    use_custatevec: bool = True
    custatevec_batch_size: int = 128

    # Multi-GPU support
    use_multi_gpu: bool = False
    gpu_device_ids: Optional[list[int]] = None  # None = auto-detect all GPUs
    simulators_per_gpu: int = 2  # Number of simulators per GPU for batching

    # ========================================
    # Parallelization
    # ========================================
    n_workers: Optional[int] = None  # None = auto-detect
    parallel_gradients: bool = True
    parallel_backend: str = "thread"  # 'thread' or 'process'
    gradient_chunk_size: int = 8

    # ========================================
    # Output
    # ========================================
    verbose: bool = True

    def __post_init__(self):
        # Auto-detect number of workers if not specified
        if self.n_workers is None:
            self.n_workers = max(1, mp.cpu_count() - 1)

        # Only auto-calculate box_size if not explicitly provided
        if self.box_size is None:
            if self.sigma < 0.5:
                self.box_size = max(4.0, 12 * self.sigma)
            else:
                self.box_size = 10.0

            # Ensure adequate coverage
            min_box = 8 * self.sigma
            if self.box_size < min_box:
                if self.verbose:
                    print(
                        f"Info: Auto-adjusting box size from {self.box_size:.2f} to {min_box:.2f} for sigma={self.sigma}"
                    )
                self.box_size = min_box
        else:
            if self.verbose:
                print(f"Using user-specified box size: +/-{self.box_size:.2f}")

        # Validate custom target function
        if self.target_function == TargetFunction.CUSTOM and self.custom_target_fn is None:
            raise ValueError("custom_target_fn must be provided when target_function is CUSTOM")

        if self.verbose:
            print(f"Target function: {self.target_function.value}")
            if self.x0 != 0.0:
                print(f"Wavefunction center: x0 = {self.x0}")
            print(f"Parallelization: {self.n_workers} workers, backend='{self.parallel_backend}'")
            if self.use_multi_gpu:
                gpu_str = f"devices {self.gpu_device_ids}" if self.gpu_device_ids else "auto-detect"
                print(f"Multi-GPU enabled: {gpu_str}")

    @property
    def n_params(self) -> int:
        """Number of variational parameters."""
        if self.ansatz is not None and hasattr(self.ansatz, 'n_params'):
            return self.ansatz.n_params
        return self.n_qubits * self.n_qubits

    @property
    def n_states(self) -> int:
        """Number of basis states (2^n_qubits)."""
        return 2**self.n_qubits

    @property
    def positions(self) -> NDArray[np.float64]:
        """Position grid for wavefunction."""
        return np.linspace(-self.box_size, self.box_size, self.n_states)

    @property
    def delta_x(self) -> float:
        """Grid spacing."""
        return 2 * self.box_size / (self.n_states - 1)


@dataclass
class OptimizationPipeline:
    """Configuration for optimization pipeline stages."""

    mode: str = "adaptive"  # 'adaptive', 'ultra', 'hybrid', 'single_stage'

    target_fidelity: float = 0.9999
    target_infidelity: Optional[float] = None

    max_total_time: float = 3600  # seconds

    use_init_search: bool = True
    init_strategies: Optional[list[str]] = None

    use_adam_stage: bool = True
    adam_max_steps: int = 1000
    adam_lr: float = 0.01
    adam_max_time: Optional[float] = None
    adam_time_fraction: float = 0.4

    use_basin_hopping: bool = False
    basin_hopping_threshold: float = 0.9999
    basin_hopping_iterations: int = 30

    use_lbfgs_refinement: bool = True
    lbfgs_tolerances: Optional[list[float]] = None
    lbfgs_time_fraction: float = 0.8

    use_fine_tuning: bool = True
    fine_tuning_threshold: float = 0.9999

    verbose: bool = True

    def __post_init__(self):
        if self.target_infidelity is None:
            self.target_infidelity = 1 - self.target_fidelity
        if self.init_strategies is None:
            self.init_strategies = ["smart", "gaussian_product", "random", "random", "random"]
        if self.lbfgs_tolerances is None:
            self.lbfgs_tolerances = [1e-10, 1e-12, 1e-14]


class OptimizationStrategy(Enum):
    """Available optimization strategies for campaign runs."""

    SMART = "smart"
    GAUSSIAN_PRODUCT = "gaussian_product"
    RANDOM = "random"
    PERTURB_BEST = "perturb_best"
    ZERO = "zero"


@dataclass
class CampaignConfig:
    """
    Configuration for large-scale optimization campaigns.

    Designed for running thousands of optimizations to guarantee
    finding the global minimum with machine precision.
    """

    # ========================================
    # Problem Parameters
    # ========================================
    n_qubits: int = 8
    sigma: float = 0.5
    x0: float = 0.0
    box_size: Optional[float] = None
    target_function: TargetFunction = TargetFunction.GAUSSIAN
    gamma: Optional[float] = None
    custom_target_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None

    # ========================================
    # Campaign Scale
    # ========================================
    total_runs: int = 1000
    runs_per_batch: int = 50

    # ========================================
    # Optimization Targets
    # ========================================
    target_infidelity: float = 1e-10
    target_fidelity: Optional[float] = None
    acceptable_infidelity: float = 1e-8

    # ========================================
    # Strategy Distribution
    # ========================================
    strategy_weights: dict[str, float] = field(
        default_factory=lambda: {
            "smart": 0.3,
            "gaussian_product": 0.2,
            "random": 0.4,
            "perturb_best": 0.1,
        }
    )

    # ========================================
    # Per-Run Settings
    # ========================================
    max_iter_per_run: int = 5000
    tolerance_per_run: float = 1e-12
    use_ultra_precision: bool = True
    ultra_precision_time_limit: float = 300

    # ========================================
    # Refinement Settings
    # ========================================
    refine_top_n: int = 10
    refinement_time_limit: float = 600

    # ========================================
    # Parallelization
    # ========================================
    n_parallel_runs: Optional[int] = None
    use_gpu: bool = True
    use_custatevec: bool = True
    use_multi_gpu: bool = False
    gpu_device_ids: Optional[list[int]] = None
    gpu_precision: str = "double"

    # ========================================
    # Checkpointing
    # ========================================
    checkpoint_interval: int = 10
    checkpoint_dir: Optional[str] = None
    resume_from_checkpoint: bool = True

    # ========================================
    # Output Settings
    # ========================================
    campaign_name: Optional[str] = None
    output_dir: Optional[str] = None
    save_all_results: bool = False
    save_top_n_results: int = 100
    verbose: int = 1

    # ========================================
    # Seed Management
    # ========================================
    base_seed: int = 42

    def __post_init__(self):
        """Initialize computed fields."""
        if self.target_fidelity is None:
            self.target_fidelity = 1 - self.target_infidelity

        if self.box_size is None:
            self.box_size = max(4 * self.sigma, 2.0)

        if self.n_parallel_runs is None:
            self.n_parallel_runs = max(1, mp.cpu_count() - 2)

        if self.campaign_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fn_suffix = (
                f"_{self.target_function.value}"
                if self.target_function != TargetFunction.GAUSSIAN
                else ""
            )
            self.campaign_name = (
                f"campaign_q{self.n_qubits}_s{self.sigma:.2f}{fn_suffix}_{timestamp}"
            )

        if self.checkpoint_dir is None:
            self.checkpoint_dir = str(
                get_path_config(verbose=False).checkpoint_dir / self.campaign_name
            )
        if self.output_dir is None:
            self.output_dir = str(get_path_config(verbose=False).campaign_dir / self.campaign_name)

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        total_weight = sum(self.strategy_weights.values())
        self.strategy_weights = {k: v / total_weight for k, v in self.strategy_weights.items()}

    @property
    def n_params(self) -> int:
        return self.n_qubits * self.n_qubits

    def get_strategy_for_run(self, run_id: int) -> str:
        """Deterministically assign strategy based on run_id."""
        np.random.seed(self.base_seed + run_id)
        strategies = list(self.strategy_weights.keys())
        weights = list(self.strategy_weights.values())
        return np.random.choice(strategies, p=weights)

    def get_seed_for_run(self, run_id: int) -> int:
        """Get reproducible seed for run."""
        return self.base_seed + run_id * 1000

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        d = asdict(self)
        d["target_function"] = self.target_function.value
        d["custom_target_fn"] = None  # Cannot serialize functions
        return d

    def save(self, filepath: Optional[str] = None) -> str:
        """Save configuration to JSON."""
        if filepath is None:
            filepath = os.path.join(self.output_dir, "campaign_config.json")

        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

        return filepath

    @classmethod
    def load(cls, filepath: str) -> "CampaignConfig":
        """Load configuration from JSON."""
        with open(filepath) as f:
            data = json.load(f)

        if isinstance(data.get("target_function"), str):
            data["target_function"] = TargetFunction(data["target_function"])

        if isinstance(data.get("strategy_weights"), str):
            data["strategy_weights"] = json.loads(data["strategy_weights"])

        return cls(**data)
