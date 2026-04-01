"""Barren plateau detection for variational quantum optimization."""

import math
from collections import deque

import numpy as np

__all__ = ["BarrenPlateauDetector"]


class BarrenPlateauDetector:
    """Detect barren plateaus by monitoring gradient norms and fidelity.

    A barren plateau is flagged when average gradient norms are vanishingly
    small *and* fidelity is far from convergence, indicating the optimizer
    is stuck rather than successfully converged.

    Parameters
    ----------
    n_params : int
        Number of variational parameters.
    window : int
        Number of recent samples to consider for detection.
    threshold : float
        Base threshold for gradient norm (scaled by sqrt(n_params)).
    convergence_fidelity : float
        Fidelity above which small gradients are assumed to indicate
        convergence rather than a barren plateau.
    """

    def __init__(
        self,
        n_params: int,
        window: int = 50,
        threshold: float = 0.01,
        convergence_fidelity: float = 0.99,
    ) -> None:
        self.n_params = n_params
        self.window = window
        self.threshold = threshold
        self.convergence_fidelity = convergence_fidelity

        self._recent_norms: deque[float] = deque(maxlen=window)
        self._recent_fidelities: deque[float] = deque(maxlen=window)

        self.gradient_norm_history: list[float] = []
        self.fidelity_history: list[float] = []
        self._n_detections: int = 0

    def update(self, gradient: np.ndarray, fidelity: float) -> None:
        """Record a gradient and fidelity observation."""
        norm = float(np.linalg.norm(gradient))
        self._recent_norms.append(norm)
        self._recent_fidelities.append(fidelity)
        self.gradient_norm_history.append(norm)
        self.fidelity_history.append(fidelity)

    def is_barren_plateau(self) -> bool:
        """Return True if a barren plateau is detected.

        Detection requires at least ``window`` samples, an average gradient
        norm below ``threshold * sqrt(n_params)``, and an average fidelity
        below ``convergence_fidelity``.
        """
        if len(self._recent_norms) < self.window:
            return False

        avg_norm = sum(self._recent_norms) / len(self._recent_norms)
        avg_fidelity = sum(self._recent_fidelities) / len(self._recent_fidelities)

        scaled_threshold = self.threshold * math.sqrt(self.n_params)

        return avg_norm < scaled_threshold and avg_fidelity < self.convergence_fidelity

    def suggest_mitigation(self) -> str:
        """Suggest a mitigation strategy based on detection count.

        Returns
        -------
        str
            One of ``"random_restart"``, ``"reduce_depth"``, or
            ``"identity_init"``.
        """
        self._n_detections += 1
        if self._n_detections == 1:
            return "random_restart"
        elif self._n_detections <= 3:
            return "reduce_depth"
        else:
            return "identity_init"

    def reset(self) -> None:
        """Clear all recorded state."""
        self._recent_norms.clear()
        self._recent_fidelities.clear()
        self.gradient_norm_history.clear()
        self.fidelity_history.clear()
