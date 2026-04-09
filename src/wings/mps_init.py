"""
MPS (Matrix Product State) initialization for variational quantum circuits.

Decomposes a target wavefunction into an MPS via sequential SVD,
then extracts approximate initial parameters for the DefaultAnsatz.

Reference: Ran, PRA 101, 032310 (2020)
"""

from typing import Optional

import numpy as np

__all__ = ["mps_decompose", "mps_to_statevector", "mps_initial_params"]


def mps_decompose(
    state: np.ndarray,
    n_qubits: int,
    bond_dim: Optional[int] = None,
) -> list[np.ndarray]:
    """
    Decompose a statevector into Matrix Product State tensors via sequential SVD.

    Args:
        state: Statevector of length 2^n_qubits
        n_qubits: Number of qubits
        bond_dim: Maximum bond dimension (None = no truncation)

    Returns:
        List of n_qubits MPS tensors. tensor[i] has shape (chi_left, 2, chi_right)
        where chi_left and chi_right are bond dimensions.
    """
    psi = state.copy().reshape(-1)
    tensors = []
    remaining = psi.copy()

    for i in range(n_qubits - 1):
        # Reshape: (chi_left * 2, remaining_dim)
        chi_left = remaining.shape[0] if remaining.ndim > 1 else 1
        if i == 0:
            M = remaining.reshape(2, -1)
        else:
            M = remaining.reshape(chi_left * 2, -1)

        U, S, Vh = np.linalg.svd(M, full_matrices=False)

        # Truncate to bond_dim
        if bond_dim is not None and len(S) > bond_dim:
            U = U[:, :bond_dim]
            S = S[:bond_dim]
            Vh = Vh[:bond_dim, :]

        chi_right = len(S)

        # Store tensor: reshape U to (chi_left, 2, chi_right)
        if i == 0:
            tensors.append(U.reshape(1, 2, chi_right))
        else:
            tensors.append(U.reshape(chi_left, 2, chi_right))

        # Remaining = S @ Vh
        remaining = np.diag(S) @ Vh

    # Last tensor: whatever remains
    chi_left = remaining.shape[0]
    tensors.append(remaining.reshape(chi_left, 2, 1))

    return tensors


def mps_to_statevector(tensors: list[np.ndarray]) -> np.ndarray:
    """
    Contract MPS tensors back into a statevector.

    Args:
        tensors: List of MPS tensors from mps_decompose

    Returns:
        Statevector of length 2^n_qubits
    """
    result = tensors[0].reshape(tensors[0].shape[1], tensors[0].shape[2])  # (2, chi)

    for i in range(1, len(tensors)):
        # result shape: (2^i, chi_left)
        # tensor[i] shape: (chi_left, 2, chi_right)
        t = tensors[i]
        # Contract: result @ tensor over the bond dimension
        new_result = np.einsum("...i,ijk->...jk", result, t)
        # Reshape: merge physical indices
        shape = new_result.shape
        result = new_result.reshape(-1, shape[-1])

    return result.reshape(-1)


def mps_initial_params(
    target: np.ndarray,
    n_qubits: int,
    bond_dim: Optional[int] = None,
) -> np.ndarray:
    """
    Generate initial parameters for DefaultAnsatz from MPS decomposition.

    Uses the MPS decomposition to reconstruct a high-fidelity approximation
    of the target state, then performs a fast local optimization to find
    circuit parameters that reproduce it.

    Args:
        target: Target statevector (length 2^n_qubits)
        n_qubits: Number of qubits
        bond_dim: MPS bond dimension (None = auto, uses min(8, 2^(n/2)))

    Returns:
        Parameter array of length n_qubits^2
    """
    if bond_dim is None:
        bond_dim = min(8, 2 ** (n_qubits // 2))

    n_qubits * n_qubits

    # Reconstruct target via MPS for a smoothed approximation
    tensors = mps_decompose(target, n_qubits, bond_dim=bond_dim)
    mps_target = mps_to_statevector(tensors)
    mps_target = mps_target / np.linalg.norm(mps_target)

    # Use the actual target for parameter fitting
    tgt = target / np.linalg.norm(target)

    # Quick parameter optimization using scipy
    from scipy.optimize import minimize

    def _neg_fidelity(params):
        """Compute negative fidelity using statevector simulation."""
        psi = _simulate_default_ansatz(params, n_qubits)
        return -(np.abs(np.vdot(tgt, psi)) ** 2)

    # Start from a physics-informed guess
    x0 = _extract_initial_guess(tgt, n_qubits)

    result = minimize(
        _neg_fidelity,
        x0,
        method="L-BFGS-B",
        options={"maxiter": 200, "ftol": 1e-12},
    )

    return result.x


def _extract_initial_guess(target: np.ndarray, n_qubits: int) -> np.ndarray:
    """Extract a physics-informed initial guess for the optimizer."""
    n_params = n_qubits * n_qubits
    params = np.zeros(n_params)

    # The DefaultAnsatz starts with |0...01> (X on last qubit)
    # First layer is RY rotations on each qubit
    # For qubit i, RY(theta)|0> = cos(t/2)|0> + sin(t/2)|1>
    # except last qubit starts in |1>: RY(theta)|1> = -sin(t/2)|0> + cos(t/2)|1>

    # Compute marginal probabilities for each qubit from the target
    n_states = 2**n_qubits
    probs = np.abs(target) ** 2

    for i in range(n_qubits):
        # Probability of qubit i being |1>
        mask = np.arange(n_states) & (1 << i)
        p1: float = np.sum(probs[mask > 0])
        p1 = np.clip(p1, 0.01, 0.99)

        if i == n_qubits - 1:
            # Last qubit starts in |1>, RY(theta)|1> gives P(|1>) = cos^2(t/2)
            params[i] = 2 * np.arccos(np.sqrt(p1))
        else:
            # Other qubits start in |0>, RY(theta)|0> gives P(|1>) = sin^2(t/2)
            params[i] = 2 * np.arcsin(np.sqrt(p1))

    # Small random for entangling layers to break symmetry
    remaining = n_params - n_qubits
    if remaining > 0:
        params[n_qubits:] = np.random.randn(remaining) * 0.1

    return params


def _simulate_default_ansatz(params: np.ndarray, n_qubits: int) -> np.ndarray:
    """
    Fast statevector simulation of DefaultAnsatz without Qiskit overhead.

    Simulates: X(last) -> [RY layer -> CX entangling layer] * depth
    """
    from .ansatz_library import generate_entanglement_map

    n = n_qubits
    depth = n  # DefaultAnsatz default depth = n_qubits
    dim = 2**n

    # Start with |0...0>
    psi = np.zeros(dim, dtype=np.complex128)
    psi[0] = 1.0

    # Apply X on last qubit: |0...0> -> |0...01> (qubit n-1 flipped)
    # In Qiskit convention, qubit n-1 is the most significant bit
    # Actually in the circuit: qc.x(n-1) flips qubit index n-1
    # Qiskit uses little-endian: qubit 0 is least significant bit
    # So X on qubit n-1 flips bit (n-1), changing state index by 2^(n-1)
    psi[0] = 0.0
    psi[1 << (n - 1)] = 1.0

    # Layer 0: RY on each qubit
    for i in range(n):
        psi = _apply_ry(psi, float(params[i]), i, n)

    # Layers 1..depth-1: CX + RY
    ent_map = generate_entanglement_map(n, "linear")
    for d in range(depth - 1):
        for ctrl, tgt in ent_map:
            psi = _apply_cx(psi, ctrl, tgt, n)
        for i in range(n):
            param_idx = n + n * d + i
            psi = _apply_ry(psi, float(params[param_idx]), i, n)

    return psi


def _apply_ry(psi: np.ndarray, theta: float, qubit: int, n_qubits: int) -> np.ndarray:
    """Apply RY(theta) gate to a specific qubit in the statevector (vectorized)."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)

    # Reshape to isolate the target qubit dimension
    # Shape: (2^(n-q-1), 2, 2^q) where q=qubit
    shape = (2 ** (n_qubits - qubit - 1), 2, 2**qubit)
    psi_r = psi.reshape(shape)

    result = np.empty_like(psi_r)
    # RY matrix: [[cos, -sin], [sin, cos]]
    result[:, 0, :] = c * psi_r[:, 0, :] - s * psi_r[:, 1, :]
    result[:, 1, :] = s * psi_r[:, 0, :] + c * psi_r[:, 1, :]

    return result.reshape(-1)


def _apply_cx(psi: np.ndarray, ctrl: int, tgt: int, n_qubits: int) -> np.ndarray:
    """Apply CX gate (ctrl -> tgt) to the statevector (vectorized)."""
    dim = 2**n_qubits
    indices = np.arange(dim)

    # Find indices where control qubit is 1 and target qubit is 0
    ctrl_mask = 1 << ctrl
    tgt_mask = 1 << tgt

    # Indices where ctrl=1, tgt=0 (these get swapped with ctrl=1, tgt=1)
    mask = ((indices & ctrl_mask) != 0) & ((indices & tgt_mask) == 0)
    idx_from = indices[mask]
    idx_to = idx_from ^ tgt_mask  # flip target bit

    result = psi.copy()
    result[idx_from] = psi[idx_to]
    result[idx_to] = psi[idx_from]

    return result
