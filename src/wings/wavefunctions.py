"""
Expanded wavefunction library for WINGS v0.2.0.

Provides analytically-defined wavefunctions beyond the built-in Gaussian,
Lorentzian, and hyperbolic secant targets. Each function takes a position
array and returns a (possibly unnormalized) complex wavefunction that will
be normalized during target construction.
"""

import math
from typing import Optional

import numpy as np

__all__ = [
    "harmonic_oscillator_eigenstate",
    "superposition_of_gaussians",
    "airy_wavefunction",
    "morse_oscillator_eigenstate",
    "squeezed_gaussian",
    "plane_wave_packet",
    "list_wavefunctions",
]


def harmonic_oscillator_eigenstate(
    x: np.ndarray,
    n: int = 0,
    sigma: float = 1.0,
    x0: float = 0.0,
) -> np.ndarray:
    """
    Quantum harmonic oscillator eigenstate |n>.

    psi_n(x) = (1/(2^n * n!))^(1/2) * (1/(pi*sigma^2))^(1/4) * H_n(xi) * exp(-xi^2/2)
    where xi = (x - x0) / sigma, H_n is the Hermite polynomial.

    Args:
        x: Position array
        n: Quantum number (n=0 is ground state)
        sigma: Width parameter (related to mass and frequency: sigma = sqrt(hbar/(m*omega)))
        x0: Center position

    Returns:
        Complex wavefunction array
    """
    from scipy.special import hermite

    xi = (x - x0) / sigma
    H_n = hermite(n)

    # Normalization: (2^n * n! * sqrt(pi) * sigma)^(-1/2)
    norm = (2**n * math.factorial(n) * np.sqrt(np.pi) * sigma) ** (-0.5)
    psi = norm * H_n(xi) * np.exp(-(xi**2) / 2)

    return psi.astype(np.complex128)


def superposition_of_gaussians(
    x: np.ndarray,
    centers: Optional[list] = None,
    sigmas: Optional[list] = None,
    amplitudes: Optional[list] = None,
) -> np.ndarray:
    """
    Superposition of Gaussian wavepackets.

    psi(x) = sum_i a_i * exp(-(x - x_i)^2 / (2 * sigma_i^2))

    Useful for modeling quantum interference patterns, cat states,
    and multi-modal distributions.

    Args:
        x: Position array
        centers: List of center positions (default: [-1, 1])
        sigmas: List of widths (default: [0.5, 0.5])
        amplitudes: List of complex amplitudes (default: [1, 1])

    Returns:
        Complex wavefunction array
    """
    if centers is None:
        centers = [-1.0, 1.0]
    if sigmas is None:
        sigmas = [0.5] * len(centers)
    if amplitudes is None:
        amplitudes = [1.0] * len(centers)

    psi = np.zeros_like(x, dtype=np.complex128)
    for x_i, s_i, a_i in zip(centers, sigmas, amplitudes):
        psi += a_i * np.exp(-((x - x_i) ** 2) / (2 * s_i**2))

    return psi


def airy_wavefunction(
    x: np.ndarray,
    x0: float = 0.0,
    scale: float = 1.0,
) -> np.ndarray:
    """
    Airy function wavefunction Ai((x - x0) / scale).

    The Airy function is the solution to the Schrodinger equation
    for a particle in a linear potential V(x) = F*x. It oscillates
    for x < 0 and decays exponentially for x > 0.

    Args:
        x: Position array
        x0: Turning point position
        scale: Length scale (related to force: scale = (hbar^2 / (2*m*F))^(1/3))

    Returns:
        Complex wavefunction array
    """
    from scipy.special import airy

    xi = (x - x0) / scale
    ai_vals, _, _, _ = airy(xi)  # airy returns (Ai, Ai', Bi, Bi')

    return ai_vals.astype(np.complex128)


def morse_oscillator_eigenstate(
    x: np.ndarray,
    n: int = 0,
    D_e: float = 10.0,
    a: float = 1.0,
    x_e: float = 0.0,
    mu: float = 1.0,
) -> np.ndarray:
    """
    Morse oscillator eigenstate.

    The Morse potential is V(x) = D_e * (1 - exp(-a*(x-x_e)))^2.
    Eigenstates are: psi_n(z) = N * z^s * exp(-z/2) * L_n^(2s-2n)(z)
    where z = 2*lambda*exp(-a*(x-x_e)), lambda = sqrt(2*mu*D_e)/a,
    s = lambda - n - 0.5, and L_n^alpha is the generalized Laguerre polynomial.

    Args:
        x: Position array
        n: Vibrational quantum number
        D_e: Dissociation energy
        a: Width parameter of the potential
        x_e: Equilibrium position
        mu: Reduced mass

    Returns:
        Complex wavefunction array

    Raises:
        ValueError: If n exceeds the number of bound states
    """
    from scipy.special import eval_genlaguerre

    lam = np.sqrt(2 * mu * D_e) / a
    n_max = int(np.floor(lam - 0.5))

    if n > n_max:
        raise ValueError(
            f"Quantum number n={n} exceeds maximum bound state n_max={n_max} for lambda={lam:.2f}"
        )

    s = lam - n - 0.5
    z = 2 * lam * np.exp(-a * (x - x_e))

    # Generalized Laguerre polynomial L_n^(2s-2n)(z) = L_n^(2*lam-2*n-1)(z)
    alpha = 2 * s  # = 2*lam - 2*n - 1
    L_n = eval_genlaguerre(n, alpha, z)

    # Wavefunction (unnormalized -- will be normalized by WINGS)
    psi = z**s * np.exp(-z / 2) * L_n

    # Handle potential overflow/underflow
    psi = np.where(np.isfinite(psi), psi, 0.0)

    return psi.astype(np.complex128)


def squeezed_gaussian(
    x: np.ndarray,
    x0: float = 0.0,
    sigma: float = 1.0,
    squeeze_r: float = 0.0,
) -> np.ndarray:
    """
    Squeezed Gaussian (minimum uncertainty state with asymmetric uncertainties).

    In position representation, squeezing parameter r modifies the width:
    psi(x) = exp(-(x-x0)^2 / (2 * sigma_eff^2))
    where sigma_eff = sigma * exp(-r).

    r > 0: position-squeezed (narrower in x, broader in p)
    r < 0: momentum-squeezed (broader in x, narrower in p)
    r = 0: coherent state (standard Gaussian)

    Args:
        x: Position array
        x0: Center position
        sigma: Base width (before squeezing)
        squeeze_r: Squeezing parameter

    Returns:
        Complex wavefunction array
    """
    sigma_eff = sigma * np.exp(-squeeze_r)
    psi = np.exp(-((x - x0) ** 2) / (2 * sigma_eff**2))

    return psi.astype(np.complex128)


def plane_wave_packet(
    x: np.ndarray,
    k0: float = 1.0,
    sigma: float = 1.0,
    x0: float = 0.0,
) -> np.ndarray:
    """
    Gaussian wavepacket with initial momentum k0.

    psi(x) = exp(i*k0*x) * exp(-(x-x0)^2 / (2*sigma^2))

    This is a minimum-uncertainty state with mean position x0
    and mean momentum k0.

    Note: This target has complex phases. The DefaultAnsatz (RY+CNOT only)
    CANNOT encode complex phases. Use CustomHardwareEfficientAnsatz with
    rotation_gates=['ry', 'rz'] for this target.

    Args:
        x: Position array
        k0: Initial momentum (wavenumber)
        sigma: Position-space width
        x0: Center position

    Returns:
        Complex wavefunction array (with nonzero imaginary part when k0 != 0)
    """
    envelope = np.exp(-((x - x0) ** 2) / (2 * sigma**2))
    phase = np.exp(1j * k0 * x)

    return (phase * envelope).astype(np.complex128)


def list_wavefunctions() -> dict:
    """
    List all available wavefunctions with descriptions.

    Returns:
        Dictionary mapping function names to one-line descriptions
    """
    return {
        "harmonic_oscillator_eigenstate": "Quantum harmonic oscillator |n> (Hermite-Gaussian)",
        "superposition_of_gaussians": "Coherent superposition of Gaussian wavepackets",
        "airy_wavefunction": "Airy function Ai(x) for linear potential",
        "morse_oscillator_eigenstate": "Morse oscillator bound state (anharmonic vibration)",
        "squeezed_gaussian": "Squeezed minimum-uncertainty state",
        "plane_wave_packet": "Gaussian wavepacket with momentum (requires RZ gates)",
    }
