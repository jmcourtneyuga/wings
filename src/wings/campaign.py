"""Campaign management and convenience functions."""

import glob
import json
import logging
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from typing import Any, Optional

import numpy as np

from .config import CampaignConfig, OptimizationPipeline, OptimizerConfig, TargetFunction
from .optimizer import GaussianOptimizer
from .paths import get_path_config
from .results import CampaignResults, RunResult
from .types import ParameterArray

logger = logging.getLogger(__name__)

__all__ = [
    "OptimizationManager",
    "run_production_campaign",
    "quick_optimization",
    "load_campaign_results",
    "list_campaigns",
]


class OptimizationManager:
    """
    Production manager for large-scale optimization campaigns.

    Features:
    - Run thousands of optimizations
    - Automatic checkpointing and resume
    - Parallel execution with GPU support
    - Result aggregation and analysis
    - Progress tracking and logging

    Example:
        config = CampaignConfig(n_qubits=8, sigma=0.5, total_runs=1000)
        manager = OptimizationManager(config)
        results = manager.run_campaign()
        results.print_summary()
    """

    def __init__(self, config: CampaignConfig) -> None:
        self.config: CampaignConfig = config
        self.results: CampaignResults = CampaignResults(config)
        # Checkpoint tracking
        self._checkpoint_file: str = os.path.join(config.checkpoint_dir, "checkpoint.pkl")
        self._completed_runs: set[int] = set()

        # Logging
        self._log_file = os.path.join(config.output_dir, "campaign.log")

        # Resume from checkpoint if requested
        if config.resume_from_checkpoint:
            self._try_resume()

        # Save config
        config.save()

        self._log(f"OptimizationManager initialized for campaign: {config.campaign_name}")
        self._log(f"  Total runs: {config.total_runs}")
        self._log(f"  Target infidelity: {config.target_infidelity:.0e}")
        self._log(f"  Output directory: {config.output_dir}")

    def _log(self, message: str, level: int = 1) -> None:
        """Log message to file and optionally console"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"

        # Write to file
        with open(self._log_file, "a") as f:
            f.write(log_entry + "\n")

        # Print if verbose enough
        if self.config.verbose >= level:
            print(message)

    def _try_resume(self) -> None:
        """Attempt to resume from checkpoint"""
        if not os.path.exists(self._checkpoint_file):
            self._log("No checkpoint found, starting fresh")
            return

        try:
            with open(self._checkpoint_file, "rb") as f:
                checkpoint = pickle.load(f)

            self.results = checkpoint["results"]
            self._completed_runs = checkpoint["completed_runs"]

            self._log(f"Resumed from checkpoint: {len(self._completed_runs)} runs completed")
            self._log(
                f"  Best fidelity so far: {self.results.best_result.fidelity:.12f}"
                if self.results.best_result
                else "  No successful runs yet"
            )

        except (OSError, FileNotFoundError) as e:
            self._log(f"Could not read checkpoint file: {e}")
            self._completed_runs = set()
        except (pickle.UnpicklingError, KeyError, EOFError) as e:
            self._log(f"Checkpoint file corrupted: {e}")
            self._log("Starting fresh")
            self._completed_runs = set()
        except Exception as e:
            # Catch-all for truly unexpected errors, but log the type
            self._log(f"Unexpected error resuming checkpoint ({type(e).__name__}): {e}")
            self._completed_runs = set()

    def _save_checkpoint(self) -> None:
        """Save checkpoint to disk"""
        checkpoint = {
            "results": self.results,
            "completed_runs": self._completed_runs,
            "timestamp": datetime.now().isoformat(),
        }

        # Write to temp file first, then rename (atomic)
        temp_file = self._checkpoint_file + ".tmp"
        with open(temp_file, "wb") as f:
            pickle.dump(checkpoint, f)

        os.replace(temp_file, self._checkpoint_file)

        # Also save results
        self.results.save()

    def _create_optimizer_config(self) -> OptimizerConfig:
        """Create OptimizerConfig from CampaignConfig"""
        return OptimizerConfig(
            n_qubits=self.config.n_qubits,
            sigma=self.config.sigma,
            x0=self.config.x0,
            box_size=self.config.box_size,
            # Target function settings
            target_function=self.config.target_function,
            gamma=self.config.gamma,
            custom_target_fn=self.config.custom_target_fn,
            # Optimizer settings
            method="L-BFGS-B",
            max_iter=self.config.max_iter_per_run,
            max_fun=self.config.max_iter_per_run * 2,
            tolerance=self.config.tolerance_per_run,
            gtol=self.config.tolerance_per_run,
            high_precision=True,
            use_analytic_gradients=True,
            verbose=False,  # Quiet for mass runs
            # GPU settings
            use_gpu=self.config.use_gpu,
            use_custatevec=self.config.use_custatevec,
            gpu_precision=self.config.gpu_precision,
            # Multi-GPU settings
            use_multi_gpu=self.config.use_multi_gpu,
            gpu_device_ids=self.config.gpu_device_ids,
            # Other
            parallel_gradients=False,  # Avoid nested parallelism
            target_fidelity=self.config.target_fidelity,
        )

    def _run_single_optimization(self, run_id: int) -> RunResult:
        """
        Execute a single optimization run.

        Returns RunResult with success/failure status.
        """
        from .optimizer import GaussianOptimizer

        strategy = self.config.get_strategy_for_run(run_id)
        seed = self.config.get_seed_for_run(run_id)

        start_time = time.time()

        try:
            # Set random seed for reproducibility
            np.random.seed(seed)

            # Create optimizer
            opt_config = self._create_optimizer_config()
            optimizer = GaussianOptimizer(opt_config)

            # Get initial parameters based on strategy
            if strategy == "perturb_best" and self.results.best_result is not None:
                # Perturb from best known result
                initial_params = self.results.best_result.params + 0.1 * np.random.randn(
                    opt_config.n_params
                )
            else:
                initial_params = optimizer.get_initial_params(strategy)

            # Run optimization
            pipeline = OptimizationPipeline(
                mode="ultra" if self.config.use_ultra_precision else "adaptive",
                target_fidelity=self.config.target_fidelity,
                max_total_time=self.config.ultra_precision_time_limit,
                use_basin_hopping=self.config.use_ultra_precision,
                verbose=False,
            )
            optimizer.run_optimization(pipeline, initial_params)

            # Compute circuit statistics
            final_psi = optimizer.get_statevector(optimizer.best_params)
            circuit_stats = optimizer.compute_statistics(final_psi)

            elapsed = time.time() - start_time

            return RunResult(
                run_id=run_id,
                strategy=strategy,
                seed=seed,
                fidelity=optimizer.best_fidelity,
                infidelity=1 - optimizer.best_fidelity,
                params=optimizer.best_params.copy(),
                circuit_std=circuit_stats["std"],
                circuit_mean=circuit_stats["mean"],
                n_evaluations=optimizer.n_evals,
                time_seconds=elapsed,
                success=True,
            )

        except Exception as e:
            elapsed = time.time() - start_time
            error_msg = f"{type(e).__name__}: {str(e)}"
            self._log(f"Run {run_id} failed: {error_msg}", level=2)

            return RunResult(
                run_id=run_id,
                strategy=strategy,
                seed=seed,
                fidelity=0.0,
                infidelity=1.0,
                params=None,
                circuit_std=None,
                circuit_mean=None,
                n_evaluations=0,
                time_seconds=elapsed,
                success=False,
                error_message=error_msg,
            )

    def _run_batch_sequential(self, run_ids: list[int]) -> list[RunResult]:
        """Run a batch of optimizations sequentially (GPU-optimized)"""
        results = []
        for run_id in run_ids:
            if run_id in self._completed_runs:
                continue

            if self.config.verbose >= 1:
                print(f"  Starting run {run_id}/{self.config.total_runs}...", end=" ")

            result = self._run_single_optimization(run_id)
            results.append(result)

            if self.config.verbose >= 1:
                if result.success:
                    print(f"F={result.fidelity:.10f}, time={result.time_seconds:.1f}s")
                else:
                    print(f"FAILED: {result.error_message}")

        return results

    def _run_batch_parallel(self, run_ids: list[int]) -> list[RunResult]:
        """Run a batch of optimizations in parallel (CPU mode)"""
        # Filter out completed runs
        run_ids = [r for r in run_ids if r not in self._completed_runs]

        if not run_ids:
            return []

        results = []

        # Use ProcessPoolExecutor for true parallelism
        # Note: GPU/cuStateVec should be disabled for parallel runs
        with ProcessPoolExecutor(max_workers=self.config.n_parallel_runs) as executor:
            futures = {executor.submit(self._run_single_optimization, rid): rid for rid in run_ids}

            for future in futures:
                try:
                    result = future.result(timeout=self.config.ultra_precision_time_limit * 2)
                    results.append(result)

                    if self.config.verbose >= 1:
                        run_id = futures[future]
                        if result.success:
                            print(f"  Run {run_id}: F={result.fidelity:.10f}")
                        else:
                            print(f"  Run {run_id}: FAILED")

                except Exception as e:
                    run_id = futures[future]
                    self._log(f"Run {run_id} exception: {e}")

        return results

    def run_campaign(self) -> CampaignResults:
        """
        Execute the full optimization campaign.

        Returns:
            CampaignResults with all run data and statistics
        """
        self._log("=" * 80)
        self._log(f"STARTING OPTIMIZATION CAMPAIGN: {self.config.campaign_name}")
        self._log("=" * 80)

        self.results.start_time = time.time()

        # Determine which runs still need to be done
        all_run_ids = list(range(self.config.total_runs))
        remaining_runs = [r for r in all_run_ids if r not in self._completed_runs]

        self._log(f"Runs to complete: {len(remaining_runs)}/{self.config.total_runs}")

        # Process in batches
        batch_size = self.config.runs_per_batch
        n_batches = (len(remaining_runs) + batch_size - 1) // batch_size

        for batch_idx in range(n_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(remaining_runs))
            batch_run_ids = remaining_runs[batch_start:batch_end]

            self._log(
                f"\nBatch {batch_idx + 1}/{n_batches} (runs {batch_run_ids[0]}-{batch_run_ids[-1]})"
            )

            # Run batch (sequential for GPU, parallel for CPU)
            if self.config.use_gpu or self.config.use_custatevec:
                batch_results = self._run_batch_sequential(batch_run_ids)
            else:
                batch_results = self._run_batch_parallel(batch_run_ids)

            # Process results
            for result in batch_results:
                self.results.add_result(result)
                self._completed_runs.add(result.run_id)

            # Checkpoint
            if (batch_idx + 1) % max(1, self.config.checkpoint_interval // batch_size) == 0:
                self._log("Saving checkpoint...")
                self._save_checkpoint()

            # Check if target achieved
            if (
                self.results.best_result
                and self.results.best_result.fidelity >= self.config.target_fidelity
            ):
                self._log(f"\n TARGET ACHIEVED at run {self.results.best_result.run_id}!")
                self._log(f"  Fidelity: {self.results.best_result.fidelity:.15f}")
                # Continue running to find potentially better solutions

        self.results.end_time = time.time()

        # Final checkpoint and save
        self._save_checkpoint()

        # Refinement phase
        if self.config.refine_top_n > 0:
            self._run_refinement_phase()

        # Final save
        self.results.save()

        self._log("\nCampaign complete!")
        self.results.print_summary()

        return self.results

    def _run_refinement_phase(self):
        """Refine top N results with extended optimization"""
        self._log("\n" + "=" * 60)
        self._log("REFINEMENT PHASE")
        self._log("=" * 60)

        top_results = self.results.get_top_results(self.config.refine_top_n)

        if not top_results:
            self._log("No results to refine")
            return

        self._log(f"Refining top {len(top_results)} results")

        # Create optimizer for refinement
        opt_config = self._create_optimizer_config()
        opt_config.verbose = True
        optimizer = GaussianOptimizer(opt_config)

        best_refined_fidelity = 0
        best_refined_params = None

        for i, result in enumerate(top_results):
            self._log(
                f"\nRefining result {i + 1}/{len(top_results)} (original F={result.fidelity:.12f})"
            )

            # Reset optimizer state
            optimizer.best_fidelity = 0
            optimizer.best_params = None
            optimizer.n_evals = 0

            try:
                # Run ultra-precision refinement starting from this result
                refined = optimizer.optimize_ultra_precision(
                    target_infidelity=self.config.target_infidelity / 10,  # Even tighter
                    max_total_time=self.config.refinement_time_limit,
                    initial_params=result.params,
                )

                self._log(
                    f"  Refined: F={refined['fidelity']:.15f} "
                    f"(improvement: {refined['fidelity'] - result.fidelity:.3e})"
                )

                if refined["fidelity"] > best_refined_fidelity:
                    best_refined_fidelity = refined["fidelity"]
                    best_refined_params = optimizer.best_params.copy()

            except Exception as e:
                self._log(f"  Refinement failed: {e}")

        # Update best result if refinement improved it
        if best_refined_fidelity > self.results.best_result.fidelity:
            self._log("\n* Refinement improved best result")
            self._log(f"  Before: {self.results.best_result.fidelity:.15f}")
            self._log(f"  After:  {best_refined_fidelity:.15f}")

            # Create new best result
            final_psi = optimizer.get_statevector(best_refined_params)
            circuit_stats = optimizer.compute_statistics(final_psi)

            self.results.best_result = RunResult(
                run_id=-1,  # Refinement result
                strategy="refinement",
                seed=0,
                fidelity=best_refined_fidelity,
                infidelity=1 - best_refined_fidelity,
                params=best_refined_params,
                circuit_std=circuit_stats["std"],
                circuit_mean=circuit_stats["mean"],
                n_evaluations=optimizer.n_evals,
                time_seconds=0,
                success=True,
            )

    def get_best_parameters(self) -> np.ndarray:
        """Get the best parameters found across all runs"""
        if self.results.best_result is None:
            raise ValueError("No successful optimization runs yet")
        return self.results.best_result.params.copy()

    def get_best_fidelity(self) -> float:
        """Get the best fidelity achieved"""
        if self.results.best_result is None:
            return 0.0
        return self.results.best_result.fidelity

    def export_best_result(self, output_path: str = None) -> str:
        """Export best result to file"""
        if self.results.best_result is None:
            raise ValueError("No successful optimization runs yet")

        if output_path is None:
            output_path = os.path.join(
                self.config.output_dir,
                f"best_result_q{self.config.n_qubits}_s{self.config.sigma:.2f}.npz",
            )

        np.savez(
            output_path,
            params=self.results.best_result.params,
            fidelity=self.results.best_result.fidelity,
            infidelity=self.results.best_result.infidelity,
            n_qubits=self.config.n_qubits,
            sigma=self.config.sigma,
            box_size=self.config.box_size,
        )

        self._log(f"Best result exported to: {output_path}")
        return output_path


def run_production_campaign(
    n_qubits: int,
    sigma: float,
    total_runs: int = 1000,
    target_infidelity: float = 1e-10,
    box_size: Optional[float] = None,
    x0: float = 0.0,
    target_function: "TargetFunction" = None,
    gamma: Optional[float] = None,
    custom_target_fn=None,
    use_multi_gpu: bool = False,
    gpu_device_ids: Optional[list[int]] = None,
    campaign_name: str = None,
    resume: bool = True,
    **kwargs,
) -> CampaignResults:
    """
    Convenience function to run a production optimization campaign.

    Args:
        n_qubits: Number of qubits
        sigma: Gaussian width
        total_runs: Total number of optimization runs
        target_infidelity: Target infidelity (1 - fidelity)
        campaign_name: Name for campaign (auto-generated if None)
        resume: Whether to resume from checkpoint if available
        **kwargs: Additional arguments passed to CampaignConfig
        x0: Wavefunction center position (default: 0.0)
        target_function: Target wavefunction type (default: GAUSSIAN)
        gamma: Width parameter for Lorentzian
        custom_target_fn: Custom wavefunction function
        use_multi_gpu: Enable multi-GPU acceleration
        gpu_device_ids: Specific GPU device IDs to use

    Returns:
        CampaignResults with all results and statistics

    Example:
        results = run_production_campaign(
            n_qubits=8,
            sigma=0.5,
            total_runs=1000,
            target_infidelity=1e-11
        )
        print(f"Best fidelity: {results.best_result.fidelity}")
    """
    # Default to GAUSSIAN if not specified
    if target_function is None:
        target_function = TargetFunction.GAUSSIAN

    config = CampaignConfig(
        n_qubits=n_qubits,
        sigma=sigma,
        x0=x0,
        box_size=box_size,
        target_function=target_function,
        gamma=gamma,
        custom_target_fn=custom_target_fn,
        total_runs=total_runs,
        target_infidelity=target_infidelity,
        use_multi_gpu=use_multi_gpu,
        gpu_device_ids=gpu_device_ids,
        campaign_name=campaign_name,
        resume_from_checkpoint=resume,
        **kwargs,
    )

    manager = OptimizationManager(config)
    results = manager.run_campaign()

    return results


def quick_optimization(
    n_qubits: int,
    sigma: float,
    x0: float = 0.0,
    target_function: "TargetFunction" = None,
    n_runs: int = 50,
    target_infidelity: float = 1e-8,
    verbose: bool = True,
) -> tuple[ParameterArray, float, dict[str, Any]]:
    """
    Quick optimization for testing or small-scale problems.

    Args:
        n_qubits: Number of qubits
        sigma: Gaussian width
        n_runs: Number of optimization runs (default 50)
        target_infidelity: Target infidelity
        verbose: Print progress

    Returns:
        (best_params, best_fidelity, results_dict)
    """
    if target_function is None:
        target_function = TargetFunction.GAUSSIAN

    config = CampaignConfig(
        n_qubits=n_qubits,
        sigma=sigma,
        x0=x0,
        target_function=target_function,
        total_runs=n_runs,
        target_infidelity=target_infidelity,
        runs_per_batch=10,
        checkpoint_interval=100,  # Effectively no checkpointing
        verbose=1 if verbose else 0,
        use_ultra_precision=False,  # Faster
        max_iter_per_run=2000,
    )

    manager = OptimizationManager(config)
    results = manager.run_campaign()

    return (results.best_result.params, results.best_result.fidelity, results.get_statistics())


def load_campaign_results(campaign_name_or_path: str) -> CampaignResults:
    """
    Load results from a previous campaign.

    Args:
        campaign_name_or_path: Campaign name or full path to results file

    Returns:
        CampaignResults object
    """
    if os.path.exists(campaign_name_or_path):
        return CampaignResults.load(campaign_name_or_path)

    # Try to find in campaign directory
    campaign_dir = get_path_config(verbose=False).campaign_dir / campaign_name_or_path
    results_file = os.path.join(campaign_dir, "campaign_results.pkl")

    if os.path.exists(results_file):
        return CampaignResults.load(results_file)

    raise FileNotFoundError(f"Could not find campaign results for: {campaign_name_or_path}")


def list_campaigns() -> list[dict[str, Any]]:
    """
    List all available campaigns.

    Returns:
        List of dicts with campaign info
    """
    campaigns = []

    campaign_base = get_path_config(verbose=False).campaign_dir
    for campaign_dir in glob.glob(str(campaign_base / "campaign_*")):
        config_file = os.path.join(campaign_dir, "campaign_config.json")
        results_file = os.path.join(campaign_dir, "campaign_results_summary.json")

        info = {
            "name": os.path.basename(campaign_dir),
            "path": campaign_dir,
            "has_config": os.path.exists(config_file),
            "has_results": os.path.exists(results_file),
        }

        if os.path.exists(results_file):
            try:
                with open(results_file) as f:
                    summary = json.load(f)
                info["best_fidelity"] = summary.get("statistics", {}).get("best_fidelity")
                info["total_runs"] = summary.get("statistics", {}).get("total_runs")
            except (OSError, json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Could not parse results file {results_file}: {e}")

        campaigns.append(info)

    return sorted(campaigns, key=lambda x: x["name"], reverse=True)
