"""Result visualization and file export for WINGS."""

import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

__all__ = ["plot_optimization_results", "save_optimization_results"]


def plot_optimization_results(
    positions,
    psi_circuit,
    psi_target,
    results,
    history,
    n_qubits,
    sigma,
    box_size,
    save_path=None,
):
    """Create visualization plots with high precision display.

    Parameters
    ----------
    positions : array-like
        Grid positions (x values).
    psi_circuit : array-like
        Circuit statevector amplitudes.
    psi_target : array-like
        Target wavefunction amplitudes.
    results : dict
        Optimization results dictionary.
    history : dict
        Optimization history with 'fidelity' and 'iteration' keys.
    n_qubits : int
        Number of qubits.
    sigma : float
        Target Gaussian width.
    box_size : float
        Half-width of the simulation box.
    save_path : str, optional
        Path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure or None
        The figure object, or None if plotting fails.
    """
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 11))

        x = positions

        # Plot 1: Probability densities (log scale option for high precision)
        ax = axes[0, 0]
        ax.plot(x, np.abs(psi_circuit) ** 2, "b-", label="Circuit", linewidth=2)
        ax.plot(x, np.abs(psi_target) ** 2, "r--", label="Target Gaussian", linewidth=2, alpha=0.8)
        ax.set_xlabel("Position x", fontsize=11)
        ax.set_ylabel("|ψ(x)|²", fontsize=11)
        ax.set_title(f"Probability Density (Fidelity = {results['fidelity']:.10f})", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Plot 2: Real and imaginary parts
        ax = axes[0, 1]
        ax.plot(x, np.real(psi_circuit), "b-", label="Circuit (Real)", linewidth=1.5)
        ax.plot(x, np.imag(psi_circuit), "b--", label="Circuit (Imag)", linewidth=1.5, alpha=0.7)
        ax.plot(x, np.real(psi_target), "r-", label="Target (Real)", linewidth=1.5, alpha=0.8)
        ax.plot(x, np.imag(psi_target), "r--", label="Target (Imag)", linewidth=1.5, alpha=0.5)
        ax.set_xlabel("Position x", fontsize=11)
        ax.set_ylabel("Amplitude", fontsize=11)
        ax.set_title("Wavefunction Components", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Plot 3: Difference (log scale for high precision)
        ax = axes[1, 0]
        difference = np.abs(psi_circuit - psi_target) ** 2
        max_diff = np.max(difference)
        ax.plot(x, difference, "g-", linewidth=2)
        ax.set_xlabel("Position x", fontsize=11)
        ax.set_ylabel("|ψ_circuit - ψ_target|²", fontsize=11)
        ax.set_title(f"Squared Difference (max = {max_diff:.3e})", fontsize=12)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, which="both")

        # Plot 4: Convergence with infidelity tracking
        ax = axes[1, 1]
        if len(history["fidelity"]) > 0:
            fidelities = np.array(history["fidelity"])
            infidelities = 1 - fidelities

            # Plot on log scale to see high precision improvement
            ax.semilogy(history["iteration"], infidelities, "g-", linewidth=1.5, label="Infidelity")
            ax.axhline(y=1e-3, color="r", linestyle="--", alpha=0.5, label="F=0.999")
            ax.axhline(y=1e-4, color="orange", linestyle="--", alpha=0.5, label="F=0.9999")
            ax.axhline(
                y=results["infidelity"],
                color="blue",
                linestyle="-",
                alpha=0.7,
                label=f"Final: 1-F={results['infidelity']:.2e}",
            )
            ax.set_xlabel("Function Evaluation", fontsize=11)
            ax.set_ylabel("Infidelity (1 - F)", fontsize=11)
            ax.set_title("Optimization Progress (Log Scale)", fontsize=12)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3, which="both")

        # Add high-precision statistics text
        stats_text = (
            f"High Precision Results:\n"
            f"Fidelity:    {results['fidelity']:.12f}\n"
            f"Infidelity:  {results['infidelity']:.3e}\n"
            f"Circuit: μ={results['circuit_mean']:.8f}, σ={results['circuit_std']:.8f}\n"
            f"Target:  μ={results['target_mean']:.8f}, σ={results['target_std']:.8f}\n"
            f"Errors:  Δμ={results['mean_error']:.3e}, Δσ={results['std_error']:.3e}\n"
            f"Rel. σ error: {results['relative_std_error'] * 100:.2f}%\n"
            f"Time: {results['time']:.1f}s, Evals: {results['n_evaluations']}"
        )
        fig.text(
            0.02,
            0.02,
            stats_text,
            fontsize=9,
            family="monospace",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.6},
        )

        plt.suptitle(
            f"High Precision Gaussian State (n={n_qubits}, σ={sigma:.4f}, box=±{box_size:.2f})",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()

        if save_path:
            try:
                plt.savefig(save_path, dpi=200, bbox_inches="tight")
                print(f"Plot saved to: {save_path}")
            except Exception as e:
                print(f"Warning: Could not save plot: {e}")

        plt.show()

        return fig

    except Exception as e:
        print(f"Warning: Could not create plot: {e}")
        import traceback

        traceback.print_exc()
        return None


def save_optimization_results(
    results,
    config_dict,
    filepath=None,
):
    """Save high-precision parameters to text file, numpy array, and JSON.

    Parameters
    ----------
    results : dict
        Optimization results dictionary.
    config_dict : dict
        Configuration parameters dictionary with keys: n_qubits, sigma, x0,
        box_size, n_states, delta_x, method, max_iter, max_fun, tolerance,
        high_precision, enable_refinement, n_params.
    filepath : str, optional
        Output text file path.  Auto-generated from config if not provided.

    Returns
    -------
    str
        The filepath used for the text output.
    """
    if filepath is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = (
            f"gaussian_highprec_q{config_dict['n_qubits']}"
            f"_s{config_dict['sigma']:.4f}_{timestamp}.txt"
        )

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("HIGH PRECISION GAUSSIAN STATE PREPARATION\n")
            f.write("=" * 80 + "\n\n")

            f.write("CONFIGURATION\n")
            f.write("-" * 40 + "\n")
            f.write(f"Timestamp:          {datetime.now().isoformat()}\n")
            f.write(f"Number of qubits:   {config_dict['n_qubits']}\n")
            f.write(f"Number of params:   {config_dict['n_params']}\n")
            f.write(f"Target sigma:       {config_dict['sigma']:.10f}\n")
            f.write(f"Target x0:          {config_dict['x0']:.10f}\n")
            f.write(f"Box size:           +/-{config_dict['box_size']:.6f}\n")
            f.write(f"Grid points:        {config_dict['n_states']}\n")
            f.write(f"Grid spacing:       {config_dict['delta_x']:.10f}\n")
            f.write(f"Optimizer:          {config_dict['method']}\n")
            f.write(f"Max iterations:     {config_dict['max_iter']}\n")
            f.write(f"Max fun evals:      {config_dict['max_fun']}\n")
            f.write(f"Tolerance:          {config_dict['tolerance']:.2e}\n")
            f.write(f"High precision:     {config_dict['high_precision']}\n")
            f.write(f"Refinement enabled: {config_dict['enable_refinement']}\n\n")

            f.write("HIGH PRECISION RESULTS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Fidelity:           {results['fidelity']:.15f}\n")
            f.write(f"Infidelity (1-F):   {results['infidelity']:.3e}\n")
            f.write(f"Circuit mean:       {results['circuit_mean']:.12f}\n")
            f.write(f"Circuit std:        {results['circuit_std']:.12f}\n")
            f.write(f"Target mean:        {results['target_mean']:.12f}\n")
            f.write(f"Target std:         {results['target_std']:.12f}\n")
            f.write(f"Error in mean:      {results['mean_error']:.3e}\n")
            f.write(f"Error in std:       {results['std_error']:.3e}\n")
            f.write(f"Relative std err:   {results['relative_std_error'] * 100:.4f}%\n")
            f.write(f"Optimization time:  {results['time']:.2f} seconds\n")
            f.write(f"Function evals:     {results['n_evaluations']}\n")
            f.write(f"Success:            {results['success']}\n")
            f.write(f"Message:            {results.get('optimizer_message', 'N/A')}\n\n")

            f.write("OPTIMAL PARAMETERS (15 decimal places)\n")
            f.write("-" * 40 + "\n")
            f.write("# Index    Value\n")
            params = results["optimal_params"]
            for i, param in enumerate(params):
                f.write(f"{i:5d}    {param:+.15f}\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write(
                f"# To load parameters:\n"
                f"# params = np.loadtxt('{os.path.basename(filepath)}', skiprows=N, usecols=1)\n"
            )

        print(f"\nResults saved to: {filepath}")

        # Save numpy array with full precision
        np_file = filepath.replace(".txt", "_params.npy")
        np.save(np_file, results["optimal_params"])
        print(f"Parameters saved to: {np_file}")

        # Save JSON with results
        json_file = filepath.replace(".txt", "_results.json")
        json_data = {
            "fidelity": float(results["fidelity"]),
            "infidelity": float(results["infidelity"]),
            "circuit_mean": float(results["circuit_mean"]),
            "circuit_std": float(results["circuit_std"]),
            "target_mean": float(results["target_mean"]),
            "target_std": float(results["target_std"]),
            "mean_error": float(results["mean_error"]),
            "std_error": float(results["std_error"]),
            "time": float(results["time"]),
            "n_evaluations": int(results["n_evaluations"]),
            "config": {
                "n_qubits": config_dict["n_qubits"],
                "sigma": config_dict["sigma"],
                "x0": config_dict["x0"],
                "box_size": config_dict["box_size"],
                "method": config_dict["method"],
                "high_precision": config_dict["high_precision"],
            },
        }
        with open(json_file, "w") as f:
            json.dump(json_data, f, indent=2)
        print(f"JSON results saved to: {json_file}")

    except Exception as e:
        print(f"Error saving results: {e}")
        import traceback

        traceback.print_exc()

    return filepath
