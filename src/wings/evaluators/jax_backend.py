"""
JAX-based statevector simulator with automatic differentiation.

Provides O(1)-cost gradient computation via reverse-mode AD (jax.grad),
compared to O(n_params) for parameter-shift rules.

Requires: jax, jaxlib (optional dependencies)
"""

import numpy as np

__all__ = [
    "HAS_JAX",
    "make_zero_state",
    "apply_ry_jax",
    "apply_rz_jax",
    "apply_x_jax",
    "apply_cnot_jax",
    "compute_fidelity_jax",
    "compute_gradient_jax_default_ansatz",
]

try:
    import jax
    import jax.numpy as jnp
    from jax import grad as jax_grad

    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jnp = None
    jax = None


def make_zero_state(n_qubits: int):
    """Create |0...0> state as (2,)*n_qubits tensor."""
    if not HAS_JAX:
        raise ImportError("JAX is required. Install with: pip install jax jaxlib")
    sv = jnp.zeros([2] * n_qubits, dtype=jnp.complex128)
    idx = tuple([0] * n_qubits)
    sv = sv.at[idx].set(1.0 + 0j)
    return sv


def apply_ry_jax(sv, theta, target):
    """Apply RY(theta) to target qubit of tensor-shaped statevector."""
    c = jnp.cos(theta / 2)
    s = jnp.sin(theta / 2)
    gate = jnp.array([[c, -s], [s, c]], dtype=jnp.complex128)
    sv = jnp.tensordot(gate, sv, axes=([1], [target]))
    sv = jnp.moveaxis(sv, 0, target)
    return sv


def apply_rz_jax(sv, theta, target):
    """Apply RZ(theta) to target qubit."""
    phase_p = jnp.exp(-1j * theta / 2)
    phase_m = jnp.exp(1j * theta / 2)
    gate = jnp.array([[phase_p, 0], [0, phase_m]], dtype=jnp.complex128)
    sv = jnp.tensordot(gate, sv, axes=([1], [target]))
    sv = jnp.moveaxis(sv, 0, target)
    return sv


def apply_x_jax(sv, target):
    """Apply Pauli-X to target qubit."""
    gate = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex128)
    sv = jnp.tensordot(gate, sv, axes=([1], [target]))
    sv = jnp.moveaxis(sv, 0, target)
    return sv


def apply_cnot_jax(sv, control, target):
    """Apply CNOT gate. Flips target when control is |1>."""
    n_qubits = sv.ndim
    # Split along control qubit
    # When control = |0>, do nothing; when control = |1>, apply X to target
    slices_0 = [slice(None)] * n_qubits
    slices_1 = [slice(None)] * n_qubits
    slices_0[control] = 0
    slices_1[control] = 1

    part_0 = sv[tuple(slices_0)]  # control=0 subspace (unchanged)
    part_1 = sv[tuple(slices_1)]  # control=1 subspace (flip target)

    # Flip target qubit in part_1
    part_1 = jnp.flip(part_1, axis=target if target < control else target - 1)

    # Reconstruct
    sv = sv.at[tuple(slices_0)].set(part_0)
    sv = sv.at[tuple(slices_1)].set(part_1)
    return sv


def compute_fidelity_jax(sv_tensor, target_flat):
    """Compute |<target|psi>|^2. sv is tensor-shaped, target is flat."""
    sv_flat = sv_tensor.reshape(-1)
    overlap = jnp.vdot(target_flat, sv_flat)
    return (overlap * jnp.conj(overlap)).real


def _default_ansatz_jax(params, n_qubits, target):
    """
    Apply DefaultAnsatz and compute negative fidelity (for minimization).
    params: 1D array of length n_qubits^2
    """
    depth = n_qubits
    sv = make_zero_state(n_qubits)

    # Initial X on last qubit
    sv = apply_x_jax(sv, n_qubits - 1)

    # First layer: RY on each qubit
    for i in range(n_qubits):
        sv = apply_ry_jax(sv, params[i], i)

    # Subsequent layers: CNOT + RY
    for d in range(depth - 1):
        for i in range(n_qubits - 1):
            sv = apply_cnot_jax(sv, i, i + 1)
        for i in range(n_qubits):
            param_idx = n_qubits + n_qubits * d + i
            sv = apply_ry_jax(sv, params[param_idx], i)

    fid = compute_fidelity_jax(sv, target)
    return -fid  # Negative for minimization


def compute_gradient_jax_default_ansatz(
    params: np.ndarray,
    target: np.ndarray,
    n_qubits: int,
) -> np.ndarray:
    """
    Compute gradient using JAX automatic differentiation.

    Cost: ~3x forward pass, regardless of parameter count.

    Args:
        params: Parameter array (numpy)
        target: Target statevector (numpy, flat)
        n_qubits: Number of qubits

    Returns:
        Gradient as numpy array
    """
    if not HAS_JAX:
        raise ImportError("JAX is required. Install with: pip install jax jaxlib")

    params_jax = jnp.array(params, dtype=jnp.float64)
    target_jax = jnp.array(target, dtype=jnp.complex128)

    grad_fn = jax_grad(lambda p: _default_ansatz_jax(p, n_qubits, target_jax))
    grad = grad_fn(params_jax)

    return np.array(grad, dtype=np.float64)
