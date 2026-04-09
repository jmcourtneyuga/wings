"""
Noise-aware optimization for variational quantum circuits.

Provides noise configuration, noise-robust objective functions,
and circuit depth penalties for hardware-aware optimization.

Requires qiskit-aer for full noise model simulation (optional).
"""

from dataclasses import dataclass

__all__ = ["NoiseConfig"]


@dataclass
class NoiseConfig:
    """
    Configuration for noise-aware optimization.

    Noise parameters model common hardware imperfections:
    - depolarizing_rate: probability of depolarizing error per gate
    - amplitude_damping_rate: T1 decay rate
    - gate_error_1q: single-qubit gate error rate
    - gate_error_2q: two-qubit gate error rate (typically 5-10x gate_error_1q)
    - readout_error: measurement bit-flip probability

    Reference: Sharma et al., NJP 22, 043006 (2020)
    """

    depolarizing_rate: float = 0.0
    amplitude_damping_rate: float = 0.0
    gate_error_1q: float = 0.0
    gate_error_2q: float = 0.0
    readout_error: float = 0.0

    def has_noise(self) -> bool:
        """Check if any noise is configured."""
        return any(
            [
                self.depolarizing_rate > 0,
                self.amplitude_damping_rate > 0,
                self.gate_error_1q > 0,
                self.gate_error_2q > 0,
                self.readout_error > 0,
            ]
        )

    def noise_robust_objective(
        self,
        fidelity_ideal: float,
        fidelity_noisy: float,
        robustness_weight: float = 0.1,
    ) -> float:
        """
        Compute noise-robust objective function.

        L(theta) = (1 - F_ideal) + lambda * (F_ideal - F_noisy)

        The first term drives toward high ideal fidelity.
        The second term penalizes parameters that are sensitive to noise.

        Args:
            fidelity_ideal: Fidelity from ideal simulation
            fidelity_noisy: Fidelity from noisy simulation
            robustness_weight: Lambda parameter (0 = ignore noise, 1 = equal weight)

        Returns:
            Combined objective value (lower is better)
        """
        infidelity = 1.0 - fidelity_ideal
        noise_gap = fidelity_ideal - fidelity_noisy
        return infidelity + robustness_weight * max(0.0, noise_gap)

    def depth_penalty(self, n_cx_gates: int, weight: float = 0.001) -> float:
        """
        Circuit depth penalty proportional to two-qubit gate count.

        Deeper circuits accumulate more noise. This penalty encourages
        the optimizer to prefer shallower circuits when possible.

        Args:
            n_cx_gates: Number of CNOT (CX) gates in the circuit
            weight: Penalty weight per gate

        Returns:
            Penalty value to add to objective
        """
        return weight * n_cx_gates

    def build_qiskit_noise_model(self):
        """
        Build a Qiskit Aer noise model from this configuration.

        Requires qiskit-aer. Raises ImportError if not available.
        """
        try:
            from qiskit_aer.noise import (
                NoiseModel,
                ReadoutError,
                depolarizing_error,
            )
        except ImportError as err:
            raise ImportError(
                "qiskit-aer is required for noise model simulation. "
                "Install with: pip install qiskit-aer"
            ) from err

        noise_model = NoiseModel()

        if self.depolarizing_rate > 0:
            error_1q = depolarizing_error(self.depolarizing_rate)
            error_2q = depolarizing_error(self.depolarizing_rate)
            noise_model.add_all_qubit_quantum_error(error_1q, ["ry", "rz", "x"])
            noise_model.add_all_qubit_quantum_error(error_2q, ["cx"])

        if self.gate_error_1q > 0:
            error_1q = depolarizing_error(self.gate_error_1q)
            noise_model.add_all_qubit_quantum_error(error_1q, ["ry", "rz", "x"])

        if self.gate_error_2q > 0:
            error_2q = depolarizing_error(self.gate_error_2q)
            noise_model.add_all_qubit_quantum_error(error_2q, ["cx"])

        if self.readout_error > 0:
            p = self.readout_error
            ro_error = ReadoutError([[1 - p, p], [p, 1 - p]])
            noise_model.add_all_qubit_readout_error(ro_error)

        return noise_model
