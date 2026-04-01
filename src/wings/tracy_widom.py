"""
Tracy-Widom distribution wavefunctions for WINGS.

Provides discretized Tracy-Widom probability density functions (TW_1, TW_2, TW_4)
as target wavefunctions for variational quantum state preparation. The distributions
are computed via the Painlevé II transcendent (Hastings-McLeod solution).

Mathematical background
-----------------------
The Tracy-Widom distributions describe the fluctuations of the largest eigenvalue
of large random matrices drawn from the classical ensembles:

    TW_1 (β=1): Gaussian Orthogonal Ensemble  (GOE, real symmetric)
    TW_2 (β=2): Gaussian Unitary Ensemble     (GUE, Hermitian)
    TW_4 (β=4): Gaussian Symplectic Ensemble   (GSE, quaternion self-dual)

All three are expressed in terms of the Hastings-McLeod solution q(s) to the
Painlevé II equation:

    q''(s) = s q(s) + 2 q(s)^3,    q(s) ~ Ai(s)  as  s -> +∞.

Defining R(s) = ∫_s^∞ q(t)^2 dt, the CDFs are:

    F_2(s) = exp( -∫_s^∞ (t - s) q(t)^2 dt )
    F_1(s) = exp( -½ ∫_s^∞ q(t) dt ) · F_2(s)^(1/2)
    F_4(s) = cosh(½ ∫_s^∞ q(t) dt) · F_2(s)^(1/2)

The PDFs f_β(s) = F_β'(s) are used as (real, non-negative) target wavefunctions.
Since |ψ(x)|^2 ~ f_β(x), we set ψ_β(x) = √(f_β(x)) and normalize.

References
----------
[1] Tracy & Widom, Commun. Math. Phys. 159, 151-174 (1994).
[2] Tracy & Widom, Commun. Math. Phys. 177, 727-754 (1996).
[3] Bornemann, Math. Comp. 79, 871-915 (2010).  (Numerical methods)
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import airy
from typing import Optional, Tuple

__all__ = [
    "tracy_widom_pdf",
    "tracy_widom_wavefunction",
    "solve_painleve_ii",
    "TW_BETA_1",
    "TW_BETA_2",
    "TW_BETA_4",
]

TW_BETA_1 = 1
TW_BETA_2 = 2
TW_BETA_4 = 4

_VALID_BETAS = {TW_BETA_1, TW_BETA_2, TW_BETA_4}


# ---------------------------------------------------------------------------
# Core: solve Painlevé II for the Hastings-McLeod solution
# ---------------------------------------------------------------------------

def solve_painleve_ii(
    s_max: float = 8.0,
    s_min: float = -8.0,
    n_points: int = 4096,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve the Painlevé II ODE  q'' = s q + 2 q^3  backward from s_max.

    Uses the Airy function asymptotics q(s) ~ Ai(s) for the initial condition
    at large positive s, then integrates backward toward negative s.

    Parameters
    ----------
    s_max : float
        Right boundary (large positive s where q ~ Ai(s)).
    s_min : float
        Left boundary (large negative s; PDF is negligible here).
    n_points : int
        Number of grid points in the output.

    Returns
    -------
    s_grid : ndarray, shape (n_points,)
        Grid of s values from s_min to s_max.
    q_vals : ndarray, shape (n_points,)
        Hastings-McLeod solution q(s).
    qp_vals : ndarray, shape (n_points,)
        Derivative q'(s).
    """
    # Airy initial conditions at s_max
    ai, aip, _, _ = airy(s_max)  # Ai(s_max), Ai'(s_max)

    # ODE system: y = [q, q']
    # q'' = s*q + 2*q^3
    def painleve_rhs(s, y):
        q, qp = y
        return [qp, s * q + 2.0 * q**3]

    # Integrate backward: from s_max down to s_min
    s_eval = np.linspace(s_max, s_min, n_points)

    sol = solve_ivp(
        painleve_rhs,
        t_span=(s_max, s_min),
        y0=[ai, aip],
        t_eval=s_eval,
        method="DOP853",
        rtol=1e-12,
        atol=1e-14,
        max_step=0.01,
    )

    if not sol.success:
        raise RuntimeError(f"Painlevé II integration failed: {sol.message}")

    # Reverse to get ascending s order
    s_grid = sol.t[::-1]
    q_vals = sol.y[0, ::-1]
    qp_vals = sol.y[1, ::-1]

    return s_grid, q_vals, qp_vals


# ---------------------------------------------------------------------------
# Tracy-Widom CDFs and PDFs from Painlevé II solution
# ---------------------------------------------------------------------------

def _compute_tw_distributions(
    s_grid: np.ndarray,
    q_vals: np.ndarray,
) -> dict:
    """
    Compute F_1, F_2, F_4 CDFs and their PDFs from q(s).

    Uses the cumulative trapezoid rule for the integrals R(s) and U(s):
        R(s) = ∫_s^∞ q(t)^2 dt          (appears in F_2)
        U(s) = ∫_s^∞ q(t) dt            (appears in F_1, F_4)
        K(s) = ∫_s^∞ (t - s) q(t)^2 dt  (the log of F_2)

    Returns dict with keys 'F1','f1','F2','f2','F4','f4'.
    """
    ds = np.diff(s_grid)
    n = len(s_grid)
    q2 = q_vals**2

    # R(s) = ∫_s^∞ q(t)^2 dt  — integrate from right to left
    R = np.zeros(n)
    for i in range(n - 2, -1, -1):
        R[i] = R[i + 1] + 0.5 * (q2[i] + q2[i + 1]) * ds[i]

    # U(s) = ∫_s^∞ q(t) dt
    U = np.zeros(n)
    for i in range(n - 2, -1, -1):
        U[i] = U[i + 1] + 0.5 * (q_vals[i] + q_vals[i + 1]) * ds[i]

    # K(s) = ∫_s^∞ (t - s) q(t)^2 dt = ∫_s^∞ R(t) dt  (integration by parts)
    K = np.zeros(n)
    for i in range(n - 2, -1, -1):
        K[i] = K[i + 1] + 0.5 * (R[i] + R[i + 1]) * ds[i]

    # CDFs
    F2 = np.exp(-K)
    F1 = np.exp(-0.5 * U) * np.sqrt(np.maximum(F2, 0.0))
    F4_raw = np.cosh(0.5 * U) * np.sqrt(np.maximum(F2, 0.0))
    # Normalize F4 so F4(-∞)=0, F4(+∞)=1
    F4 = F4_raw / F4_raw[-1] if F4_raw[-1] > 0 else F4_raw

    # PDFs via numerical differentiation
    f1 = np.gradient(F1, s_grid)
    f2 = np.gradient(F2, s_grid)
    f4 = np.gradient(F4, s_grid)

    # Clamp to non-negative (numerical artifacts near tails)
    f1 = np.maximum(f1, 0.0)
    f2 = np.maximum(f2, 0.0)
    f4 = np.maximum(f4, 0.0)

    return {
        "F1": F1, "f1": f1,
        "F2": F2, "f2": f2,
        "F4": F4, "f4": f4,
    }


# ---------------------------------------------------------------------------
# Public API: Tracy-Widom wavefunctions for WINGS
# ---------------------------------------------------------------------------

def tracy_widom_pdf(
    s: np.ndarray,
    beta: int = 2,
    s_max: float = 8.0,
    s_min: float = -8.0,
    n_painleve: int = 8192,
) -> np.ndarray:
    """
    Evaluate the Tracy-Widom PDF f_β(s) on the given grid.

    Parameters
    ----------
    s : ndarray
        Points at which to evaluate the PDF.
    beta : int
        Ensemble index: 1 (GOE), 2 (GUE), or 4 (GSE).
    s_max, s_min : float
        Integration bounds for the Painlevé II solver.
    n_painleve : int
        Internal grid resolution for ODE integration.

    Returns
    -------
    pdf : ndarray
        Tracy-Widom PDF values f_β(s), same shape as input s.
    """
    if beta not in _VALID_BETAS:
        raise ValueError(f"beta must be one of {_VALID_BETAS}, got {beta}")

    s_grid, q_vals, _ = solve_painleve_ii(s_max, s_min, n_painleve)
    dists = _compute_tw_distributions(s_grid, q_vals)

    key = f"f{beta}"
    # Interpolate from internal grid onto user grid
    pdf = np.interp(s, s_grid, dists[key], left=0.0, right=0.0)

    return pdf


def tracy_widom_wavefunction(
    x: np.ndarray,
    beta: int = 2,
    s_max: float = 8.0,
    s_min: float = -8.0,
    n_painleve: int = 8192,
) -> np.ndarray:
    """
    Tracy-Widom target wavefunction for variational state preparation.

    Returns ψ_β(x) = √(f_β(x)) (real, non-negative), suitable for encoding
    the TW probability distribution into a quantum state via |ψ|^2 = f_β.

    The wavefunction is returned unnormalized; WINGS normalizes during
    target construction.

    Parameters
    ----------
    x : ndarray
        Position-space grid (mapped to the TW variable s).
    beta : int
        Ensemble index: 1 (GOE), 2 (GUE), or 4 (GSE).
    s_max, s_min : float
        Painlevé II integration bounds.
    n_painleve : int
        Internal ODE grid resolution.

    Returns
    -------
    psi : ndarray of complex128
        Wavefunction array ψ(x) = √(f_β(x)), as complex128 for WINGS
        compatibility.

    Notes
    -----
    The Tracy-Widom PDF is real and non-negative, so ψ_β(x) is purely real.
    This means the DefaultAnsatz (RY + CNOT only, no phase gates) is, in
    principle, not excluded on phase grounds. Whether the ansatz has sufficient
    *expressibility* to reach the TW shape is the subject of the numerical
    verification.
    """
    pdf = tracy_widom_pdf(x, beta=beta, s_max=s_max, s_min=s_min,
                          n_painleve=n_painleve)
    psi = np.sqrt(pdf)
    return psi.astype(np.complex128)


def list_tracy_widom_targets() -> dict:
    """List available Tracy-Widom targets with descriptions."""
    return {
        "tracy_widom_beta1": "TW_1 (GOE): largest eigenvalue of real symmetric random matrices",
        "tracy_widom_beta2": "TW_2 (GUE): largest eigenvalue of Hermitian random matrices",
        "tracy_widom_beta4": "TW_4 (GSE): largest eigenvalue of quaternion self-dual random matrices",
    }
