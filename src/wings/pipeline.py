"""
Composable optimization pipeline for WINGS.

Replaces the rigid boolean-flag pipeline with a flexible stage-list
architecture. Users define pipelines as ordered lists of stage objects,
each with its own hyperparameters.

Usage:

    from wings.pipeline import Pipeline, InitSearch, Adam, SPSA, LBFGSB, Newton

    pipeline = Pipeline(
        target_fidelity=1 - 1e-10,
        stages=[
            InitSearch(strategies=["mps", "smart", "random"]),
            SPSA(max_steps=2000, a=0.1, c=0.1),
            Adam(max_steps=1000, lr=0.02),
            LBFGSB(tolerances=[1e-10, 1e-12, 1e-14]),
            Newton(max_steps=50),
        ],
    )

    results = optimizer.run_pipeline(pipeline)

Presets:

    Pipeline.quick()        # Fast result for testing
    Pipeline.standard()     # Good default for most problems
    Pipeline.ultra()        # Maximum precision
    Pipeline.for_momentum() # Complex targets needing phase encoding
"""

from dataclasses import dataclass, field
from typing import Optional

__all__ = [
    "Stage",
    "InitSearch",
    "Adam",
    "SPSA",
    "NaturalGradient",
    "LBFGSB",
    "BasinHopping",
    "Newton",
    "GrowCircuit",
    "Pipeline",
]


# ============================================================================
# Base Stage
# ============================================================================


@dataclass
class Stage:
    """Base class for pipeline stages. All stages share a name and verbose flag."""

    name: str = ""
    verbose: bool = True

    def describe(self) -> str:
        """One-line description for pipeline summary."""
        return self.name or type(self).__name__


# ============================================================================
# Stage Definitions
# ============================================================================


@dataclass
class InitSearch(Stage):
    """
    Test multiple initialization strategies and pick the best.

    Strategies: "smart", "gaussian_product", "random", "mps", "zero", "perturb_best"
    """

    name: str = "Initialization Search"
    strategies: list[str] = field(
        default_factory=lambda: ["smart", "gaussian_product", "random", "random", "random"]
    )


@dataclass
class Adam(Stage):
    """
    Adam optimizer with warm restarts and optional stochastic gradients.

    Good for escaping shallow local minima via momentum. Use as the
    primary exploration stage before refinement.
    """

    name: str = "Adam"
    max_steps: int = 1000
    lr: float = 0.02
    max_time: Optional[float] = None
    convergence_window: int = 100
    convergence_threshold: float = 1e-8
    gradient_fraction: float = 1.0  # < 1.0 enables stochastic gradients


@dataclass
class SPSA(Stage):
    """
    SPSA optimizer -- 2 evaluations per gradient, any parameter count.

    Best for large circuits (n_params > 100) where parameter-shift is expensive.
    Use for broad exploration, then switch to Adam or L-BFGS-B for refinement.
    """

    name: str = "SPSA"
    max_steps: int = 3000
    a: float = 0.1
    c: float = 0.1
    A: Optional[float] = None  # None = 0.1 * max_steps
    n_avg: int = 1
    max_time: Optional[float] = None


@dataclass
class NaturalGradient(Stage):
    """
    Quantum Natural Gradient using diagonal QFIM.

    Accounts for the geometry of the quantum state manifold.
    Most effective near the optimum where curvature matters.
    """

    name: str = "Natural Gradient"
    max_steps: int = 200
    lr: float = 0.01
    regularization: float = 0.001
    max_time: Optional[float] = None


@dataclass
class LBFGSB(Stage):
    """
    L-BFGS-B quasi-Newton refinement with progressive tolerances.

    The workhorse for high-precision convergence. Automatically switches
    to log-infidelity objective when fidelity exceeds log_threshold.
    """

    name: str = "L-BFGS-B"
    tolerances: list[float] = field(default_factory=lambda: [1e-10, 1e-12, 1e-14])
    max_iter: int = 3000
    use_log_objective: bool = True
    log_threshold: float = 0.999


@dataclass
class BasinHopping(Stage):
    """
    Basin hopping for escaping deep local minima.

    Combines random jumps with local optimization. Use when Adam
    gets stuck at moderate fidelity (F ~ 0.99 but target is higher).
    """

    name: str = "Basin Hopping"
    n_iterations: int = 30
    temperature: float = 0.5
    step_size: float = 0.5


@dataclass
class Newton(Stage):
    """
    Diagonal-Hessian Newton refinement for final polish.

    Uses second-order parameter-shift (Mari et al. 2021) to compute
    exact diagonal Hessian, then takes preconditioned gradient steps.
    Best used when fidelity is already > 0.9999.
    """

    name: str = "Newton Refinement"
    max_steps: int = 50
    lr: float = 0.3
    epsilon: float = 1e-6


@dataclass
class GrowCircuit(Stage):
    """
    Add one layer to the circuit (adaptive depth / layer-ramp).

    New layer is initialized near zero so fidelity is approximately preserved.
    Use between optimization stages when convergence plateaus due to
    insufficient circuit expressibility.
    """

    name: str = "Grow Circuit"


# ============================================================================
# Pipeline
# ============================================================================


@dataclass
class Pipeline:
    """
    Composable optimization pipeline.

    A pipeline is an ordered list of stages. Each stage receives the
    best parameters from the previous stage. The pipeline stops early
    if target_fidelity is reached or max_total_time is exceeded.

    Args:
        target_fidelity: Stop when this fidelity is achieved
        max_total_time: Wall-clock time limit in seconds
        stages: Ordered list of Stage objects
        verbose: Print progress for each stage
    """

    target_fidelity: float = 0.9999
    max_total_time: float = 3600
    stages: list[Stage] = field(default_factory=list)
    verbose: bool = True

    @property
    def target_infidelity(self) -> float:
        return 1.0 - self.target_fidelity

    # ================================================================
    # Presets
    # ================================================================

    @classmethod
    def quick(cls, target_fidelity: float = 0.999, max_time: float = 60) -> "Pipeline":
        """
        Fast pipeline for development and testing.

        InitSearch -> Adam(200 steps)
        """
        return cls(
            target_fidelity=target_fidelity,
            max_total_time=max_time,
            stages=[
                InitSearch(strategies=["smart", "random"]),
                Adam(max_steps=200, lr=0.03),
            ],
        )

    @classmethod
    def standard(
        cls, target_fidelity: float = 0.999999, max_time: float = 600
    ) -> "Pipeline":
        """
        Standard pipeline for most problems.

        InitSearch -> Adam -> L-BFGS-B
        """
        return cls(
            target_fidelity=target_fidelity,
            max_total_time=max_time,
            stages=[
                InitSearch(strategies=["smart", "gaussian_product", "random", "random"]),
                Adam(max_steps=1000, lr=0.02),
                LBFGSB(tolerances=[1e-10, 1e-12]),
            ],
        )

    @classmethod
    def ultra(
        cls, target_fidelity: float = None, target_infidelity: float = 1e-11,
        max_time: float = 3600,
    ) -> "Pipeline":
        """
        Maximum precision pipeline.

        InitSearch(mps) -> SPSA -> Adam -> BasinHopping -> L-BFGS-B(log) -> Newton
        """
        if target_fidelity is None:
            target_fidelity = 1.0 - target_infidelity
        return cls(
            target_fidelity=target_fidelity,
            max_total_time=max_time,
            stages=[
                InitSearch(strategies=["mps", "smart", "gaussian_product", "random", "random"]),
                SPSA(max_steps=2000, a=0.1, c=0.1, n_avg=3),
                Adam(max_steps=1500, lr=0.02),
                BasinHopping(n_iterations=30),
                LBFGSB(tolerances=[1e-10, 1e-12, 1e-14], use_log_objective=True),
                Newton(max_steps=50, lr=0.3),
            ],
        )

    @classmethod
    def for_momentum(
        cls, target_fidelity: float = 0.9999, max_time: float = 1200
    ) -> "Pipeline":
        """
        Pipeline for complex targets (momentum wavepackets, phase-encoded states).

        Requires EfficientSU2Ansatz or CustomHardwareEfficientAnsatz with RZ gates.
        Uses Natural Gradient which respects the quantum state geometry --
        important when the ansatz must encode both amplitudes and phases.

        InitSearch(mps) -> Adam -> NaturalGradient -> L-BFGS-B
        """
        return cls(
            target_fidelity=target_fidelity,
            max_total_time=max_time,
            stages=[
                InitSearch(strategies=["mps", "smart", "random", "random"]),
                Adam(max_steps=1000, lr=0.02),
                NaturalGradient(max_steps=300, lr=0.01),
                LBFGSB(tolerances=[1e-10, 1e-12], use_log_objective=True),
            ],
        )

    @classmethod
    def layer_ramp(
        cls,
        target_fidelity: float = 0.99999,
        max_time: float = 1800,
        n_grows: int = 3,
    ) -> "Pipeline":
        """
        Adaptive depth pipeline -- grow circuit incrementally.

        Starts shallow, optimizes, grows, repeats. Avoids barren plateaus.

        InitSearch -> [Adam -> GrowCircuit] x n_grows -> L-BFGS-B -> Newton
        """
        stages = [InitSearch(strategies=["smart", "random"])]
        for i in range(n_grows):
            stages.append(Adam(max_steps=500, lr=0.02 / (i + 1)))
            stages.append(GrowCircuit())
        stages.append(LBFGSB(tolerances=[1e-10, 1e-12]))
        stages.append(Newton(max_steps=30))
        return cls(
            target_fidelity=target_fidelity,
            max_total_time=max_time,
            stages=stages,
        )

    @classmethod
    def exploration(
        cls, target_fidelity: float = 0.999, max_time: float = 300
    ) -> "Pipeline":
        """
        Cheap exploration for finding a good basin.

        Uses SPSA (2 evals per step) for maximum parameter-count scalability.
        Follow with Pipeline.standard() or Pipeline.ultra() for refinement.

        InitSearch -> SPSA(5000 steps)
        """
        return cls(
            target_fidelity=target_fidelity,
            max_total_time=max_time,
            stages=[
                InitSearch(strategies=["mps", "smart", "random"]),
                SPSA(max_steps=5000, a=0.1, c=0.1, n_avg=1),
            ],
        )

    # ================================================================
    # Utilities
    # ================================================================

    def summary(self) -> str:
        """Print a readable summary of the pipeline stages."""
        lines = [
            f"Pipeline: {len(self.stages)} stages, "
            f"target F={self.target_fidelity}, max_time={self.max_total_time}s",
            "",
        ]
        for i, stage in enumerate(self.stages):
            lines.append(f"  {i + 1}. {stage.describe()}")
            # Show key hyperparameters
            for k, v in stage.__dict__.items():
                if k in ("name", "verbose"):
                    continue
                lines.append(f"       {k}={v}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        stage_names = " -> ".join(s.describe() for s in self.stages)
        return f"Pipeline({stage_names})"
