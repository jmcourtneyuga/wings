"""Unit tests for JAX autodiff backend (v0.4.0 WI-2)."""

import numpy as np
import pytest

# Check if JAX is available
try:
    import jax
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

skip_no_jax = pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")


@pytest.mark.unit
class TestJAXBackend:

    def test_import_without_jax(self):
        """Module should import without crashing even if JAX is missing."""
        from wings.evaluators import jax_backend
        assert hasattr(jax_backend, "HAS_JAX")

    @skip_no_jax
    def test_apply_ry_produces_valid_state(self):
        from wings.evaluators.jax_backend import apply_ry_jax, make_zero_state
        import jax.numpy as jnp
        sv = make_zero_state(4)
        sv = apply_ry_jax(sv, 0.5, 0)
        norm = float(jnp.sum(jnp.abs(sv)**2))
        assert abs(norm - 1.0) < 1e-10

    @skip_no_jax
    def test_apply_rz_produces_phases(self):
        from wings.evaluators.jax_backend import apply_ry_jax, apply_rz_jax, make_zero_state
        import jax.numpy as jnp
        sv = make_zero_state(4)
        sv = apply_ry_jax(sv, 1.0, 0)  # Create superposition
        sv = apply_rz_jax(sv, 1.0, 0)  # Apply phase
        flat = sv.reshape(-1)
        assert jnp.any(jnp.abs(flat.imag) > 0.01)

    @skip_no_jax
    def test_fidelity_self_is_one(self):
        from wings.evaluators.jax_backend import compute_fidelity_jax, make_zero_state
        import jax.numpy as jnp
        sv = make_zero_state(4)
        target = sv.reshape(-1).copy()
        fid = compute_fidelity_jax(sv, target)
        assert abs(float(fid) - 1.0) < 1e-10

    @skip_no_jax
    def test_gradient_shape(self):
        from wings.evaluators.jax_backend import compute_gradient_jax_default_ansatz
        import jax.numpy as jnp
        n_qubits = 4
        n_params = n_qubits * n_qubits
        target = np.zeros(2**n_qubits, dtype=np.complex128)
        target[0] = 1.0
        params = np.random.randn(n_params) * 0.1
        grad = compute_gradient_jax_default_ansatz(params, target, n_qubits)
        assert grad.shape == (n_params,)

    @skip_no_jax
    def test_gradient_finite(self):
        from wings.evaluators.jax_backend import compute_gradient_jax_default_ansatz
        n_qubits = 4
        n_params = n_qubits * n_qubits
        target = np.zeros(2**n_qubits, dtype=np.complex128)
        target[0] = 1.0
        params = np.random.randn(n_params) * 0.1
        grad = compute_gradient_jax_default_ansatz(params, target, n_qubits)
        assert np.all(np.isfinite(grad))

    @skip_no_jax
    def test_gradient_nonzero(self):
        from wings.evaluators.jax_backend import compute_gradient_jax_default_ansatz
        n_qubits = 4
        target = np.zeros(2**n_qubits, dtype=np.complex128)
        target[0] = 1.0
        params = np.random.randn(n_qubits**2) * 0.5
        grad = compute_gradient_jax_default_ansatz(params, target, n_qubits)
        assert np.any(np.abs(grad) > 1e-8)
