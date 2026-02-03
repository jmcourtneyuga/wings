"""
Pytest configuration and shared fixtures for WINGS.

This file provides:
- Hardware detection for GPU/cuStateVec tests
- Skip markers for unavailable hardware
- Reusable fixtures for configs, optimizers, and test data
"""

import numpy as np
import pytest

# ============================================================================
# SKIP MARKERS FOR HARDWARE-DEPENDENT TESTS
# ============================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests (fast, no GPU)")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "gpu: Tests requiring GPU")
    config.addinivalue_line("markers", "custatevec: Tests requiring cuStateVec")
    config.addinivalue_line("markers", "multi_gpu: Tests requiring multiple GPUs")
    config.addinivalue_line("markers", "slow: Slow tests")


def _has_gpu() -> bool:
    """Check if GPU is available via Qiskit Aer."""
    try:
        from qiskit_aer import AerSimulator

        backend = AerSimulator(method="statevector")
        return "GPU" in backend.available_devices()
    except Exception:
        return False


def _has_custatevec() -> bool:
    """Check if cuStateVec is available."""
    try:
        from cuquantum.bindings import custatevec as cusv

        # Try to create a handle to verify it works
        handle = cusv.create()
        cusv.destroy(handle)
        return True
    except Exception:
        return False


def _has_multi_gpu() -> bool:
    """Check if multiple GPUs are available."""
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() >= 2
    except Exception:
        return False


# Cache the results
_GPU_AVAILABLE = None
_CUSTATEVEC_AVAILABLE = None
_MULTI_GPU_AVAILABLE = None


def gpu_available() -> bool:
    """Check if GPU is available (cached)."""
    global _GPU_AVAILABLE
    if _GPU_AVAILABLE is None:
        _GPU_AVAILABLE = _has_gpu()
    return _GPU_AVAILABLE


def custatevec_available() -> bool:
    """Check if cuStateVec is available (cached)."""
    global _CUSTATEVEC_AVAILABLE
    if _CUSTATEVEC_AVAILABLE is None:
        _CUSTATEVEC_AVAILABLE = _has_custatevec()
    return _CUSTATEVEC_AVAILABLE


def multi_gpu_available() -> bool:
    """Check if multiple GPUs are available (cached)."""
    global _MULTI_GPU_AVAILABLE
    if _MULTI_GPU_AVAILABLE is None:
        _MULTI_GPU_AVAILABLE = _has_multi_gpu()
    return _MULTI_GPU_AVAILABLE


def hardware_available(hardware_type: str) -> bool:
    """Check if specific hardware is available."""
    checks = {
        "gpu": gpu_available,
        "custatevec": custatevec_available,
        "multi_gpu": multi_gpu_available,
    }
    return checks.get(hardware_type, lambda: False)()


# Skip decorators for use in tests
skip_if_no_gpu = pytest.mark.skipif(not gpu_available(), reason="GPU not available")

skip_if_no_custatevec = pytest.mark.skipif(
    not custatevec_available(), reason="cuStateVec not available"
)

skip_if_no_multi_gpu = pytest.mark.skipif(
    not multi_gpu_available(), reason="Multiple GPUs not available"
)


def pytest_collection_modifyitems(config, items):
    """Automatically skip tests based on markers and hardware availability."""
    for item in items:
        # Skip GPU tests if no GPU
        if "gpu" in item.keywords and not gpu_available():
            item.add_marker(pytest.mark.skip(reason="GPU not available"))

        # Skip cuStateVec tests if not available
        if "custatevec" in item.keywords and not custatevec_available():
            item.add_marker(pytest.mark.skip(reason="cuStateVec not available"))

        # Skip multi-GPU tests if not enough GPUs
        if "multi_gpu" in item.keywords and not multi_gpu_available():
            item.add_marker(pytest.mark.skip(reason="Multiple GPUs not available"))


# ============================================================================
# FIXTURES: Configuration
# ============================================================================


@pytest.fixture
def small_config():
    """Small configuration for fast tests (6 qubits)."""
    from wings import OptimizerConfig

    return OptimizerConfig(
        n_qubits=6,
        sigma=0.5,
        box_size=4.0,
        verbose=False,
        use_gpu=False,
        use_custatevec=False,
        max_iter=100,
        max_fun=200,
    )


@pytest.fixture
def medium_config():
    """Medium configuration (8 qubits)."""
    from wings import OptimizerConfig

    return OptimizerConfig(
        n_qubits=8,
        sigma=0.5,
        box_size=4.0,
        verbose=False,
        use_gpu=False,
        use_custatevec=False,
        max_iter=500,
        max_fun=1000,
    )


@pytest.fixture
def gpu_config():
    """Configuration with GPU enabled."""
    from wings import OptimizerConfig

    return OptimizerConfig(
        n_qubits=8,
        sigma=0.5,
        box_size=4.0,
        verbose=False,
        use_gpu=True,
        use_custatevec=False,
        max_iter=100,
    )


@pytest.fixture
def custatevec_config():
    """Configuration with cuStateVec enabled."""
    from wings import OptimizerConfig

    return OptimizerConfig(
        n_qubits=8,
        sigma=0.5,
        box_size=4.0,
        verbose=False,
        use_gpu=False,
        use_custatevec=True,
        max_iter=100,
    )


@pytest.fixture
def multi_gpu_config():
    """Configuration for multi-GPU tests."""
    from wings import OptimizerConfig

    return OptimizerConfig(
        n_qubits=10,
        sigma=0.5,
        box_size=4.0,
        verbose=False,
        use_gpu=False,
        use_custatevec=True,
        use_multi_gpu=True,
        gpu_device_ids=None,  # Auto-detect
    )


@pytest.fixture
def lorentzian_config():
    """Configuration with Lorentzian target."""
    from wings import OptimizerConfig, TargetFunction

    return OptimizerConfig(
        n_qubits=6,
        sigma=0.5,
        target_function=TargetFunction.LORENTZIAN,
        gamma=0.3,
        box_size=4.0,
        verbose=False,
        use_gpu=False,
        use_custatevec=False,
    )


@pytest.fixture
def shifted_config():
    """Configuration with shifted wavefunction."""
    from wings import OptimizerConfig

    return OptimizerConfig(
        n_qubits=6,
        sigma=0.5,
        x0=1.5,
        box_size=6.0,
        verbose=False,
        use_gpu=False,
        use_custatevec=False,
    )


# ============================================================================
# FIXTURES: Optimizers
# ============================================================================


@pytest.fixture
def small_optimizer(small_config):
    """Small optimizer for fast tests."""
    from wings import GaussianOptimizer

    return GaussianOptimizer(small_config)


@pytest.fixture
def medium_optimizer(medium_config):
    """Medium optimizer."""
    from wings import GaussianOptimizer

    return GaussianOptimizer(medium_config)


# ============================================================================
# FIXTURES: Test Data
# ============================================================================


@pytest.fixture
def random_params_6q():
    """Random parameters for 6-qubit circuit."""
    np.random.seed(42)
    return np.random.randn(36) * 0.1  # 6*6 = 36 params


@pytest.fixture
def random_params_8q():
    """Random parameters for 8-qubit circuit."""
    np.random.seed(42)
    return np.random.randn(64) * 0.1  # 8*8 = 64 params


@pytest.fixture
def zero_params_6q():
    """Zero parameters for 6-qubit circuit."""
    return np.zeros(36)


@pytest.fixture
def population_small():
    """Small population for batch tests."""
    np.random.seed(42)
    return np.random.randn(10, 36) * 0.1


@pytest.fixture
def population_medium():
    """Medium population for batch tests."""
    np.random.seed(42)
    return np.random.randn(32, 64) * 0.1


# ============================================================================
# FIXTURES: Target Wavefunctions
# ============================================================================


@pytest.fixture
def gaussian_target():
    """Normalized Gaussian wavefunction factory."""

    def _gaussian(n_states: int, sigma: float = 0.5, x0: float = 0.0, box_size: float = 4.0):
        x = np.linspace(-box_size, box_size, n_states)
        psi = np.exp(-((x - x0) ** 2) / (2 * sigma**2))
        psi = psi.astype(np.complex128)
        psi /= np.linalg.norm(psi)
        return psi

    return _gaussian


@pytest.fixture
def make_gaussian():
    """Factory fixture for creating Gaussian targets."""

    def _make(n_states: int, sigma: float = 0.5, x0: float = 0.0, box_size: float = 4.0):
        x = np.linspace(-box_size, box_size, n_states)
        psi = np.exp(-((x - x0) ** 2) / (2 * sigma**2))
        psi = psi.astype(np.complex128)
        psi /= np.linalg.norm(psi)
        return psi

    return _make


@pytest.fixture
def lorentzian_target():
    """Normalized Lorentzian wavefunction factory."""

    def _lorentzian(n_states: int, gamma: float = 0.3, x0: float = 0.0, box_size: float = 4.0):
        x = np.linspace(-box_size, box_size, n_states)
        psi = gamma / ((x - x0) ** 2 + gamma**2)
        psi = psi.astype(np.complex128)
        psi /= np.linalg.norm(psi)
        return psi

    return _lorentzian


# ============================================================================
# HELPER FIXTURES
# ============================================================================


@pytest.fixture
def assert_valid_statevector():
    """Helper to validate statevector properties."""

    def _assert(sv: np.ndarray, n_qubits: int):
        assert sv.shape == (2**n_qubits,), f"Wrong shape: {sv.shape}"
        assert sv.dtype == np.complex128, f"Wrong dtype: {sv.dtype}"
        norm = np.linalg.norm(sv)
        assert np.isclose(norm, 1.0, atol=1e-10), f"Not normalized: {norm}"

    return _assert


@pytest.fixture
def assert_valid_fidelity():
    """Helper to validate fidelity values."""

    def _assert(fidelity: float):
        assert isinstance(fidelity, (float, np.floating)), f"Wrong type: {type(fidelity)}"
        assert 0.0 <= fidelity <= 1.0 + 1e-10, f"Out of range: {fidelity}"

    return _assert


@pytest.fixture(autouse=True)
def reset_random_seed():
    """Reset random seed before each test for reproducibility."""
    np.random.seed(42)
    yield


# ============================================================================
# CLEANUP FIXTURES
# ============================================================================


@pytest.fixture
def cleanup_gpu():
    """Cleanup GPU resources after test."""
    yield
    # Force garbage collection to release GPU memory
    import gc

    gc.collect()

    if custatevec_available():
        try:
            import cupy as cp

            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass


@pytest.fixture
def gpu_cleanup(cleanup_gpu):
    """Alias for cleanup_gpu fixture."""
    yield


# ============================================================================
# TEMPORARY DIRECTORY FIXTURES
# ============================================================================


@pytest.fixture
def temp_output_dir(tmp_path):
    """Temporary directory for test outputs."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def temp_checkpoint_dir(tmp_path):
    """Temporary directory for checkpoints."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    return checkpoint_dir
