"""Unit tests for MPS initialization (v0.4.0 WI-1)."""

import numpy as np
import pytest


@pytest.fixture
def gaussian_state_8q():
    """Normalized Gaussian on 8-qubit grid."""
    n = 8
    x = np.linspace(-4, 4, 2**n)
    psi = np.exp(-x**2 / (2 * 0.5**2))
    psi = psi.astype(np.complex128)
    psi /= np.linalg.norm(psi)
    return psi


@pytest.fixture
def product_state_4q():
    """Simple product state |0000>."""
    psi = np.zeros(16, dtype=np.complex128)
    psi[0] = 1.0
    return psi


@pytest.mark.unit
class TestMPSDecomposition:

    def test_mps_decompose_shape(self, gaussian_state_8q):
        from wings.mps_init import mps_decompose
        tensors = mps_decompose(gaussian_state_8q, n_qubits=8)
        assert len(tensors) == 8

    def test_mps_decompose_product_state_exact(self, product_state_4q):
        from wings.mps_init import mps_decompose, mps_to_statevector
        tensors = mps_decompose(product_state_4q, n_qubits=4)
        reconstructed = mps_to_statevector(tensors)
        np.testing.assert_allclose(np.abs(reconstructed), np.abs(product_state_4q), atol=1e-10)

    def test_mps_decompose_gaussian_high_fidelity(self, gaussian_state_8q):
        from wings.mps_init import mps_decompose, mps_to_statevector
        tensors = mps_decompose(gaussian_state_8q, n_qubits=8, bond_dim=8)
        reconstructed = mps_to_statevector(tensors)
        reconstructed /= np.linalg.norm(reconstructed)
        fidelity = np.abs(np.vdot(gaussian_state_8q, reconstructed))**2
        assert fidelity > 0.99

    def test_mps_decompose_truncation_degrades(self, gaussian_state_8q):
        from wings.mps_init import mps_decompose, mps_to_statevector
        f_high = _fidelity_from_mps(gaussian_state_8q, 8, bond_dim=8)
        f_low = _fidelity_from_mps(gaussian_state_8q, 8, bond_dim=2)
        assert f_high >= f_low

    def test_bond_dim_none_uses_full(self, product_state_4q):
        from wings.mps_init import mps_decompose, mps_to_statevector
        tensors = mps_decompose(product_state_4q, n_qubits=4, bond_dim=None)
        reconstructed = mps_to_statevector(tensors)
        fidelity = np.abs(np.vdot(product_state_4q, reconstructed))**2
        assert fidelity > 0.9999


def _fidelity_from_mps(state, n_qubits, bond_dim):
    from wings.mps_init import mps_decompose, mps_to_statevector
    tensors = mps_decompose(state, n_qubits, bond_dim=bond_dim)
    recon = mps_to_statevector(tensors)
    recon /= np.linalg.norm(recon)
    return np.abs(np.vdot(state, recon))**2


@pytest.mark.unit
class TestMPSInitialParams:

    def test_mps_initial_params_shape(self, gaussian_state_8q):
        from wings.mps_init import mps_initial_params
        params = mps_initial_params(gaussian_state_8q, n_qubits=8)
        assert params.shape == (64,)  # 8*8 for DefaultAnsatz

    def test_mps_initial_params_finite(self, gaussian_state_8q):
        from wings.mps_init import mps_initial_params
        params = mps_initial_params(gaussian_state_8q, n_qubits=8)
        assert np.all(np.isfinite(params))

    def test_mps_strategy_in_optimizer(self):
        from wings import GaussianOptimizer, OptimizerConfig
        config = OptimizerConfig(n_qubits=6, sigma=0.5, verbose=False, use_gpu=False, use_custatevec=False)
        opt = GaussianOptimizer(config)
        params = opt.get_initial_params("mps")
        assert params.shape == (opt.n_params,)
        fid = opt.compute_fidelity(params=params)
        assert fid > 0.5  # MPS init should be decent

    def test_mps_beats_random(self):
        from wings import GaussianOptimizer, OptimizerConfig
        config = OptimizerConfig(n_qubits=6, sigma=0.5, verbose=False, use_gpu=False, use_custatevec=False)
        opt = GaussianOptimizer(config)
        np.random.seed(42)
        fid_mps = opt.compute_fidelity(params=opt.get_initial_params("mps"))
        fid_rand = opt.compute_fidelity(params=opt.get_initial_params("random"))
        assert fid_mps > fid_rand
