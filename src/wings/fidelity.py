"""Fidelity and infidelity computation utilities."""

import numpy as np

from .types import ComplexArray

__all__ = ["compute_fidelity_fast", "compute_infidelity_direct"]


def compute_fidelity_fast(target_conj: ComplexArray, psi_circuit: ComplexArray) -> float:
    """Compute |<target|psi>|^2 using pre-conjugated target."""
    overlap = np.dot(target_conj, psi_circuit)
    return overlap.real**2 + overlap.imag**2


def compute_infidelity_direct(
    target_conj: ComplexArray, target: ComplexArray, psi_circuit: ComplexArray
) -> float:
    """Compute 1-F without catastrophic cancellation via ||psi - <t|p>*t||^2."""
    overlap = np.dot(target_conj, psi_circuit)
    residual = psi_circuit - overlap * target
    infidelity = np.real(np.vdot(residual, residual))
    return max(0.0, infidelity)
