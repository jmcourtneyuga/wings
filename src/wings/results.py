"""Result classes for optimization campaigns."""

import json
import os
import pickle
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from .config import CampaignConfig

__all__ = [
    "RunResult",
    "CampaignResults",
]


@dataclass
class RunResult:
    run_id: int
    fidelity: float
    infidelity: float
    params: NDArray[np.float64]
    circuit_std: float
    circuit_mean: float
    time_seconds: float
    n_evaluations: int
    strategy: str
    seed: int
    success: bool
    error_message: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to serializable dictionary"""
        return {
            "run_id": self.run_id,
            "strategy": self.strategy,
            "seed": self.seed,
            "fidelity": float(self.fidelity),
            "infidelity": float(self.infidelity),
            "params": self.params.tolist() if self.params is not None else None,
            "circuit_std": float(self.circuit_std) if self.circuit_std else None,
            "circuit_mean": float(self.circuit_mean) if self.circuit_mean else None,
            "n_evaluations": self.n_evaluations,
            "time_seconds": float(self.time_seconds),
            "success": self.success,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RunResult":
        """Create from dictionary"""
        d = d.copy()
        if d.get("params") is not None:
            d["params"] = np.array(d["params"])
        return cls(**d)


class CampaignResults:
    """
    Aggregates and analyzes results from optimization campaign.

    Tracks:
    - All run results
    - Best results found
    - Statistics across runs
    - Convergence analysis
    """

    def __init__(self, config: CampaignConfig) -> None:
        self.config: CampaignConfig = config
        self.results: list[RunResult] = []
        self._top_results: list[RunResult] = []
        self.best_result: Optional[RunResult] = None
        self.completed_runs: int = 0
        self.failed_runs: int = 0
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

        # Track top N results
        self._top_n = config.save_top_n_results

    def add_result(self, result: RunResult) -> None:
        """Add a run result and update tracking"""
        self.results.append(result)
        self.completed_runs += 1

        if not result.success:
            self.failed_runs += 1
            return

        # Update best result
        if self.best_result is None or result.fidelity > self.best_result.fidelity:
            self.best_result = result
            if self.config.verbose >= 1:
                print(
                    f"   New best: Run {result.run_id}, F={result.fidelity:.12f}, "
                    f"1-F={result.infidelity:.3e}"
                )

        # Update top N
        self._update_top_results(result)

    def _update_top_results(self, result: RunResult):
        """Maintain sorted list of top N results"""
        self._top_results.append(result)
        self._top_results.sort(key=lambda x: x.fidelity, reverse=True)
        if len(self._top_results) > self._top_n:
            self._top_results = self._top_results[: self._top_n]

    def get_top_results(self, n: int = None) -> list[RunResult]:
        """Get top N results by fidelity"""
        if n is None:
            n = self._top_n
        return self._top_results[:n]

    def get_statistics(self) -> dict[str, Any]:
        """Compute statistics across all successful runs"""
        successful = [r for r in self.results if r.success]

        if not successful:
            return {"error": "No successful runs"}

        fidelities = np.array([r.fidelity for r in successful])
        infidelities = 1 - fidelities
        times = np.array([r.time_seconds for r in successful])
        evals = np.array([r.n_evaluations for r in successful])

        # Strategy breakdown
        strategy_stats = {}
        for strategy in self.config.strategy_weights:
            strat_results = [r for r in successful if r.strategy == strategy]
            if strat_results:
                strat_fids = [r.fidelity for r in strat_results]
                strategy_stats[strategy] = {
                    "count": len(strat_results),
                    "best_fidelity": max(strat_fids),
                    "mean_fidelity": np.mean(strat_fids),
                    "std_fidelity": np.std(strat_fids),
                }

        return {
            "total_runs": len(self.results),
            "successful_runs": len(successful),
            "failed_runs": self.failed_runs,
            "success_rate": len(successful) / len(self.results) if self.results else 0,
            "best_fidelity": float(np.max(fidelities)),
            "best_infidelity": float(np.min(infidelities)),
            "mean_fidelity": float(np.mean(fidelities)),
            "std_fidelity": float(np.std(fidelities)),
            "median_fidelity": float(np.median(fidelities)),
            "fidelity_percentiles": {
                "50": float(np.percentile(fidelities, 50)),
                "90": float(np.percentile(fidelities, 90)),
                "95": float(np.percentile(fidelities, 95)),
                "99": float(np.percentile(fidelities, 99)),
                "99.9": float(np.percentile(fidelities, 99.9)),
            },
            "total_time_seconds": float(np.sum(times)),
            "mean_time_per_run": float(np.mean(times)),
            "total_evaluations": int(np.sum(evals)),
            "mean_evaluations_per_run": float(np.mean(evals)),
            "strategy_breakdown": strategy_stats,
            "target_achieved": self.best_result.fidelity >= self.config.target_fidelity
            if self.best_result
            else False,
            "runs_above_target": int(np.sum(fidelities >= self.config.target_fidelity)),
            "runs_above_acceptable": int(np.sum(infidelities <= self.config.acceptable_infidelity)),
        }

    def print_summary(self):
        """Print human-readable summary"""
        stats = self.get_statistics()

        print("\n" + "=" * 80)
        print("CAMPAIGN RESULTS SUMMARY")
        print("=" * 80)

        print(
            f"\nRuns: {stats['successful_runs']}/{stats['total_runs']} successful "
            f"({stats['success_rate'] * 100:.1f}%)"
        )

        print("\nBest Result:")
        print(f"  Fidelity:     {stats['best_fidelity']:.15f}")
        print(f"  Infidelity:   {stats['best_infidelity']:.3e}")
        print(f"  Target:       {self.config.target_infidelity:.0e}")
        print(f"  Achieved:     {' YES' if stats['target_achieved'] else ' NO'}")

        print("\nFidelity Distribution:")
        print(f"  Mean:         {stats['mean_fidelity']:.10f}")
        print(f"  Std:          {stats['std_fidelity']:.3e}")
        print(f"  Median:       {stats['median_fidelity']:.10f}")

        print("\nFidelity Percentiles:")
        for pct, val in stats["fidelity_percentiles"].items():
            print(f"  {pct}%:".ljust(10) + f"{val:.12f}")

        print("\nRuns Meeting Targets:")
        print(
            f"  Above target ({self.config.target_infidelity:.0e}):     {stats['runs_above_target']}"
        )
        print(
            f"  Above acceptable ({self.config.acceptable_infidelity:.0e}): {stats['runs_above_acceptable']}"
        )

        print("\nComputation:")
        print(f"  Total time:        {stats['total_time_seconds'] / 3600:.2f} hours")
        print(f"  Mean time/run:     {stats['mean_time_per_run']:.1f}s")
        print(f"  Total evaluations: {stats['total_evaluations']:,}")

        print("\nStrategy Performance:")
        for strat, strat_stats in stats["strategy_breakdown"].items():
            print(f"  {strat}:")
            print(
                f"    Runs: {strat_stats['count']}, "
                f"Best: {strat_stats['best_fidelity']:.10f}, "
                f"Mean: {strat_stats['mean_fidelity']:.8f}"
            )

        print("=" * 80)

    def save(self, filepath: str = None):
        """Save results to disk"""
        if filepath is None:
            filepath = os.path.join(self.config.output_dir, "campaign_results.pkl")

        # Save full pickle
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

        # Save JSON summary (without params to save space)
        json_path = filepath.replace(".pkl", "_summary.json")
        summary = {
            "config": self.config.to_dict(),
            "statistics": self.get_statistics(),
            "best_result": self.best_result.to_dict() if self.best_result else None,
            "top_results": [r.to_dict() for r in self._top_results],
        }
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        # Save best params as numpy
        if self.best_result is not None:
            np.save(filepath.replace(".pkl", "_best_params.npy"), self.best_result.params)

        return filepath

    @classmethod
    def load(cls, filepath: str) -> "CampaignResults":
        """Load results from disk"""
        with open(filepath, "rb") as f:
            return pickle.load(f)
