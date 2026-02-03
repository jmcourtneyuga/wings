"""High-level convenience functions for interactive use."""

from typing import Any, Optional

from .config import OptimizerConfig, TargetFunction
from .optimizer import GaussianOptimizer

__all__ = [
    "optimize_gaussian_state",
    "quick_optimize",
]


def optimize_gaussian_state(
    n_qubits: int = 8,
    sigma: float = 1.0,
    x0: float = 0.0,
    box_size: Optional[float] = None,
    target_function: "TargetFunction" = None,
    gamma: Optional[float] = None,
    custom_target_fn=None,
    target_fidelity: float = 0.999999,
    target_infidelity: Optional[float] = None,
    high_precision: bool = True,
    max_time: float = 1800,
    use_gpu: bool = True,
    use_custatevec: bool = True,
    use_multi_gpu: bool = False,
    gpu_device_ids: Optional[list] = None,
    plot: bool = True,
    save: bool = True,
    verbose: bool = True,
) -> tuple[dict[str, Any], GaussianOptimizer]:
    """
    Main optimization function with high precision capabilities.

    This is the recommended entry point for interactive use.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (default: 8)
    sigma : float
        Gaussian width parameter (default: 1.0)
    box_size : float, optional
        Box size for position grid. If None, auto-calculated based on sigma.
    target_fidelity : float
        Target fidelity to achieve (default: 0.999999)
    target_infidelity : float, optional
        Alternative to target_fidelity: specify 1-F directly (e.g., 1e-10)
    high_precision : bool
        Enable high precision mode (default: True)
    max_time : float
        Maximum optimization time in seconds (default: 1800 = 30 min)
    use_gpu : bool
        Use GPU acceleration if available (default: True)
    use_custatevec : bool
        Use cuStateVec if available (default: True)
    plot : bool
        Generate result plots (default: True)
    save : bool
        Save results to files (default: True)
    verbose : bool
        Print progress information (default: True)

    Returns
    -------
    results : dict
        Dictionary containing optimization results:
        - 'fidelity': Final fidelity achieved
        - 'infidelity': 1 - fidelity
        - 'params': Optimal parameters
        - 'time': Total optimization time
        - 'n_evaluations': Number of function evaluations
        - 'circuit_mean', 'circuit_std': Statistics of prepared state
        - 'target_mean', 'target_std': Statistics of target Gaussian
    optimizer : GaussianOptimizer
        The optimizer instance (for further analysis)

    Examples
    --------
    Basic usage:

    >>> results, opt = optimize_gaussian_state(n_qubits=8, sigma=0.5)
    >>> print(f"Fidelity: {results['fidelity']:.12f}")

    High precision with specific target:

    >>> results, opt = optimize_gaussian_state(
    ...     n_qubits=10,
    ...     sigma=0.5,
    ...     target_infidelity=1e-11,
    ...     max_time=3600,
    ... )
    """
    if verbose:
        print("=" * 80)
        print("HIGH PRECISION GAUSSIAN STATE OPTIMIZER")
        print("=" * 80)

    # Determine target
    if target_infidelity is not None:
        effective_target_infidelity = target_infidelity
    else:
        effective_target_infidelity = 1 - target_fidelity

    # Auto-calculate box size if not provided
    if box_size is None:
        box_size = max(10.0, 8 * sigma)

    # Create configuration
    if target_function is None:
        target_function = TargetFunction.GAUSSIAN

    # Create configuration
    config = OptimizerConfig(
        n_qubits=n_qubits,
        sigma=sigma,
        x0=x0,
        box_size=box_size,
        # Target function
        target_function=target_function,
        gamma=gamma,
        custom_target_fn=custom_target_fn,
        # Optimizer settings
        method="L-BFGS-B",
        max_iter=10000,
        max_fun=200000,
        tolerance=1e-14,
        gtol=1e-14,
        high_precision=high_precision,
        use_analytic_gradients=True,
        use_gpu=use_gpu,
        use_custatevec=use_custatevec,
        use_multi_gpu=use_multi_gpu,
        gpu_device_ids=gpu_device_ids,
        gpu_precision="double",
        verbose=verbose,
        target_fidelity=1 - effective_target_infidelity,
    )

    if verbose:
        print("\nConfiguration:")
        print(f"  n_qubits:         {config.n_qubits}")
        print(f"  Target function:  {target_function.value}")
        print(f"  Target sigma:     {config.sigma:.6f}")
        if x0 != 0.0:
            print(f"  Center (x0):      {x0:.6f}")
        print(f"  Box size:         ±{config.box_size:.4f}")
        print(f"  Grid points:      {config.n_states}")
        print(f"  Grid spacing:     {config.delta_x:.8f}")
        print(f"  Target infidelity: {effective_target_infidelity:.2e}")
        print()

    # Create optimizer
    optimizer = GaussianOptimizer(config)

    # Run ultra-precision optimization
    results = optimizer.optimize_ultra_precision(
        target_infidelity=effective_target_infidelity,
        max_total_time=max_time,
    )

    # Display results
    if verbose:
        print("\n" + "=" * 80)
        print("OPTIMIZATION COMPLETE")
        print("=" * 80)
        print(f"Fidelity achieved:    {results['fidelity']:.15f}")
        print(f"Infidelity (1-F):     {results['infidelity']:.3e}")
        print(f"Time taken:           {results['time']:.2f} seconds")
        print(f"Function evaluations: {results['n_evaluations']}")
        print()
        print("State properties:")
        print(
            f"  Circuit mean: {results['circuit_mean']:.10f} (target: {results['target_mean']:.6f})"
        )
        print(
            f"  Circuit std:  {results['circuit_std']:.10f} (target: {results['target_std']:.6f})"
        )

        if results["fidelity"] >= (1 - effective_target_infidelity):
            print(f"\n✓ SUCCESS: Achieved target infidelity of {effective_target_infidelity:.2e}")
        else:
            print(f"\n⚠ Target not achieved. Current: 1-F = {results['infidelity']:.3e}")
            _print_suggestions(results["fidelity"])

    # Plot results
    if plot:
        plot_file = f"gaussian_q{n_qubits}_s{sigma:.4f}.png" if save else None
        optimizer.plot_results(results, save_path=plot_file)

    # Save results
    if save:
        optimizer.save_results(results)

    return results, optimizer


def _print_suggestions(fidelity: float) -> None:
    """Print optimization suggestions based on achieved fidelity."""
    if fidelity < 0.999:
        print("\nSuggestions to improve fidelity:")
        print("  1. Increase max_time")
        print("  2. Ensure high_precision=True")
        print("  3. Check box_size matches Gaussian extent (~8*sigma)")
        print("  4. Increase n_qubits for better resolution")
    else:
        print("\nFor higher precision:")
        print("  - Increase max_time (current optimization may need more iterations)")
        print("  - Try running multiple times (different initial conditions)")
        print("  - Use run_production_campaign() for systematic search")


def quick_optimize(
    n_qubits: int,
    sigma: float,
    x0: float = 0.0,
    target_function: "TargetFunction" = None,
    verbose: bool = True,
) -> tuple[float, dict[str, Any]]:
    """
    Quick optimization with sensible defaults.

    Useful for testing or when you just want a quick result.

    Parameters
    ----------
    n_qubits : int
        Number of qubits
    sigma : float
        Gaussian width
    verbose : bool
        Print progress (default: True)

    Returns
    -------
    fidelity : float
        Best fidelity achieved
    results : dict
        Full results dictionary

    Examples
    --------
    >>> fidelity, results = quick_optimize(8, 0.5)
    >>> print(f"Achieved fidelity: {fidelity:.10f}")
    """
    results, _ = optimize_gaussian_state(
        n_qubits=n_qubits,
        sigma=sigma,
        x0=x0,
        target_function=target_function,
        target_infidelity=1e-10,
        max_time=300,  # 5 minutes
        plot=False,
        save=False,
        verbose=verbose,
    )
    return results["fidelity"], results
