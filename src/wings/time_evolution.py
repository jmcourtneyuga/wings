"""
Time-dependent wavepacket evolution via split-operator FFT.

Provides classical time evolution of wavefunctions under arbitrary
potentials, producing trajectories that can be optimized into
quantum circuits step-by-step.

Reference: Feit, Fleck & Steiger, JCP 47, 412 (1982)
"""

import numpy as np

__all__ = [
    "make_grid",
    "split_operator_step",
    "evolve_classical",
    "free_particle",
    "harmonic_potential",
    "morse_potential",
    "lennard_jones_potential",
]


def make_grid(n_points: int, L: float) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Create position and momentum grids for split-operator method.

    Args:
        n_points: Number of grid points
        L: Box half-width (grid spans [-L, L])

    Returns:
        (x, dx, k) where x is position grid, dx is spacing, k is momentum grid
    """
    x = np.linspace(-L, L, n_points)
    dx = x[1] - x[0]

    # Momentum grid (FFT convention)
    k = np.fft.fftfreq(n_points, d=dx) * 2 * np.pi

    return x, dx, k


def split_operator_step(
    psi: np.ndarray,
    V: np.ndarray,
    T_k: np.ndarray,
    dt: float,
) -> np.ndarray:
    """
    One split-operator time step (second-order Strang splitting).

    psi(t+dt) = exp(-iV*dt/2) * IFFT[ exp(-iT_k*dt) * FFT[ exp(-iV*dt/2) * psi(t) ] ]

    This is unconditionally stable and preserves unitarity.

    Args:
        psi: Wavefunction in position space
        V: Potential energy array V(x) in position space
        T_k: Kinetic energy array T(k) = k^2/(2m) in momentum space
        dt: Time step

    Returns:
        Evolved wavefunction
    """
    # Half-step potential
    psi = psi * np.exp(-0.5j * V * dt)

    # Full-step kinetic (in momentum space)
    psi_k = np.fft.fft(psi)
    psi_k = psi_k * np.exp(-1j * T_k * dt)
    psi = np.fft.ifft(psi_k)

    # Half-step potential
    psi = psi * np.exp(-0.5j * V * dt)

    return psi


def evolve_classical(
    psi_0: np.ndarray,
    V: np.ndarray,
    T_k: np.ndarray,
    dt: float,
    n_steps: int,
    save_every: int = 1,
) -> list[np.ndarray]:
    """
    Evolve wavefunction for n_steps and return trajectory.

    Args:
        psi_0: Initial wavefunction
        V: Potential energy array
        T_k: Kinetic energy array
        dt: Time step
        n_steps: Number of time steps
        save_every: Save wavefunction every N steps

    Returns:
        List of wavefunctions [psi(0), psi(dt*save_every), ...]
    """
    trajectory = [psi_0.copy()]
    psi = psi_0.copy()

    for step in range(1, n_steps + 1):
        psi = split_operator_step(psi, V, T_k, dt)
        if step % save_every == 0:
            trajectory.append(psi.copy())

    return trajectory


# ============================================================
# Built-in potentials
# ============================================================


def free_particle(x: np.ndarray) -> np.ndarray:
    """V(x) = 0 everywhere."""
    return np.zeros_like(x)


def harmonic_potential(
    x: np.ndarray,
    omega: float = 1.0,
    mass: float = 1.0,
    x0: float = 0.0,
) -> np.ndarray:
    """Harmonic oscillator: V(x) = 0.5 * mass * omega^2 * (x - x0)^2."""
    return 0.5 * mass * omega**2 * (x - x0) ** 2


def morse_potential(
    x: np.ndarray,
    D_e: float = 10.0,
    a: float = 1.0,
    x_e: float = 0.0,
) -> np.ndarray:
    """Morse potential: V(x) = D_e * (1 - exp(-a*(x-x_e)))^2."""
    return D_e * (1.0 - np.exp(-a * (x - x_e))) ** 2


def lennard_jones_potential(
    x: np.ndarray,
    epsilon: float = 1.0,
    sigma_lj: float = 1.0,
) -> np.ndarray:
    """Lennard-Jones: V(r) = 4*eps*((sigma/r)^12 - (sigma/r)^6). Clips at small r."""
    r = np.maximum(np.abs(x), 0.1 * sigma_lj)  # Prevent singularity
    return 4.0 * epsilon * ((sigma_lj / r) ** 12 - (sigma_lj / r) ** 6)
