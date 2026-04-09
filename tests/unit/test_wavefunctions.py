"""Unit tests for the expanded wavefunction library (P7)."""

import numpy as np
import pytest


@pytest.fixture
def grid():
    """Standard test grid."""
    return np.linspace(-10, 10, 256)


@pytest.fixture
def fine_grid():
    """Fine grid for precision tests."""
    return np.linspace(-15, 15, 1024)


@pytest.mark.unit
class TestHarmonicOscillator:
    """Tests for harmonic oscillator eigenstates."""

    def test_ground_state_shape(self, grid):
        from wings.wavefunctions import harmonic_oscillator_eigenstate

        psi = harmonic_oscillator_eigenstate(grid, n=0)
        assert psi.shape == grid.shape
        assert psi.dtype == np.complex128

    def test_ground_state_peak_at_center(self, grid):
        from wings.wavefunctions import harmonic_oscillator_eigenstate

        psi = harmonic_oscillator_eigenstate(grid, n=0, x0=2.0)
        peak_idx = np.argmax(np.abs(psi))
        assert abs(grid[peak_idx] - 2.0) < 0.1

    def test_ground_state_normalization(self, fine_grid):
        from wings.wavefunctions import harmonic_oscillator_eigenstate

        psi = harmonic_oscillator_eigenstate(fine_grid, n=0, sigma=1.0)
        dx = fine_grid[1] - fine_grid[0]
        norm = np.sum(np.abs(psi) ** 2) * dx
        assert abs(norm - 1.0) < 0.01

    def test_first_excited_has_node(self, grid):
        from wings.wavefunctions import harmonic_oscillator_eigenstate

        psi = harmonic_oscillator_eigenstate(grid, n=1)
        # n=1 should cross zero at x=0
        real_psi = psi.real
        center = len(grid) // 2
        # Check sign change near center
        assert real_psi[center - 5] * real_psi[center + 5] < 0

    def test_orthogonality(self, fine_grid):
        """Eigenstates should be approximately orthogonal on the grid."""
        from wings.wavefunctions import harmonic_oscillator_eigenstate

        dx = fine_grid[1] - fine_grid[0]
        psi_0 = harmonic_oscillator_eigenstate(fine_grid, n=0)
        psi_1 = harmonic_oscillator_eigenstate(fine_grid, n=1)
        overlap = np.abs(np.sum(np.conj(psi_0) * psi_1) * dx)
        assert overlap < 0.01

    def test_higher_states(self, grid):
        from wings.wavefunctions import harmonic_oscillator_eigenstate

        for n in range(5):
            psi = harmonic_oscillator_eigenstate(grid, n=n)
            assert np.all(np.isfinite(psi))


@pytest.mark.unit
class TestSuperpositionOfGaussians:
    def test_default_two_gaussians(self, grid):
        from wings.wavefunctions import superposition_of_gaussians

        psi = superposition_of_gaussians(grid)
        assert psi.shape == grid.shape
        assert psi.dtype == np.complex128

    def test_two_peaks(self, grid):
        from wings.wavefunctions import superposition_of_gaussians

        psi = superposition_of_gaussians(grid, centers=[-3, 3], sigmas=[0.5, 0.5])
        abs_psi = np.abs(psi)
        # Should have two local maxima
        left_peak = np.argmax(abs_psi[: len(grid) // 2])
        right_peak = len(grid) // 2 + np.argmax(abs_psi[len(grid) // 2 :])
        assert abs(grid[left_peak] - (-3)) < 0.5
        assert abs(grid[right_peak] - 3) < 0.5

    def test_single_gaussian_equivalent(self, grid):
        from wings.wavefunctions import superposition_of_gaussians

        psi = superposition_of_gaussians(grid, centers=[0], sigmas=[1.0], amplitudes=[1.0])
        expected = np.exp(-(grid**2) / 2)
        np.testing.assert_allclose(np.abs(psi), expected, atol=1e-10)

    def test_complex_amplitudes(self, grid):
        from wings.wavefunctions import superposition_of_gaussians

        psi = superposition_of_gaussians(grid, centers=[0], sigmas=[1.0], amplitudes=[1j])
        assert np.any(np.abs(psi.imag) > 0.1)


@pytest.mark.unit
class TestAiryWavefunction:
    def test_shape_and_dtype(self, grid):
        from wings.wavefunctions import airy_wavefunction

        psi = airy_wavefunction(grid)
        assert psi.shape == grid.shape
        assert psi.dtype == np.complex128

    def test_decay_for_positive_x(self, grid):
        from wings.wavefunctions import airy_wavefunction

        psi = airy_wavefunction(grid, x0=0, scale=1.0)
        # Ai(x) decays exponentially for x >> 0
        right_half = np.abs(psi[grid > 5])
        assert np.max(right_half) < 0.01

    def test_finite_values(self, grid):
        from wings.wavefunctions import airy_wavefunction

        psi = airy_wavefunction(grid)
        assert np.all(np.isfinite(psi))


@pytest.mark.unit
class TestMorseOscillator:
    def test_ground_state(self, grid):
        from wings.wavefunctions import morse_oscillator_eigenstate

        psi = morse_oscillator_eigenstate(grid, n=0)
        assert psi.shape == grid.shape
        assert np.all(np.isfinite(psi))

    def test_invalid_quantum_number(self, grid):
        from wings.wavefunctions import morse_oscillator_eigenstate

        with pytest.raises(ValueError, match="exceeds maximum"):
            morse_oscillator_eigenstate(grid, n=100, D_e=1.0)

    def test_higher_states_finite(self, grid):
        from wings.wavefunctions import morse_oscillator_eigenstate

        for n in range(3):
            psi = morse_oscillator_eigenstate(grid, n=n, D_e=20.0)
            finite_frac = np.mean(np.isfinite(psi))
            assert finite_frac > 0.95


@pytest.mark.unit
class TestSqueezedGaussian:
    def test_unsqueezed_is_gaussian(self, grid):
        from wings.wavefunctions import squeezed_gaussian

        psi = squeezed_gaussian(grid, sigma=1.0, squeeze_r=0.0)
        expected = np.exp(-(grid**2) / 2)
        np.testing.assert_allclose(np.abs(psi), expected, atol=1e-10)

    def test_positive_r_narrows(self):
        from wings.wavefunctions import squeezed_gaussian

        # Use a grid that includes x=0 exactly to avoid sampling artifacts
        g = np.linspace(-10, 10, 257)
        psi_normal = squeezed_gaussian(g, sigma=1.0, squeeze_r=0.0)
        psi_squeezed = squeezed_gaussian(g, sigma=1.0, squeeze_r=1.0)
        # Squeezed should be narrower -> higher peak
        assert np.max(np.abs(psi_squeezed)) >= np.max(np.abs(psi_normal))
        # Also check that the squeezed state has less total spread
        dx = g[1] - g[0]
        var_normal = np.sum(g**2 * np.abs(psi_normal) ** 2 * dx) / np.sum(
            np.abs(psi_normal) ** 2 * dx
        )
        var_squeezed = np.sum(g**2 * np.abs(psi_squeezed) ** 2 * dx) / np.sum(
            np.abs(psi_squeezed) ** 2 * dx
        )
        assert var_squeezed < var_normal

    def test_negative_r_broadens(self):
        from wings.wavefunctions import squeezed_gaussian

        g = np.linspace(-10, 10, 257)
        psi_normal = squeezed_gaussian(g, sigma=1.0, squeeze_r=0.0)
        psi_anti = squeezed_gaussian(g, sigma=1.0, squeeze_r=-1.0)
        # Anti-squeezed should be broader -> wider variance
        dx = g[1] - g[0]
        var_normal = np.sum(g**2 * np.abs(psi_normal) ** 2 * dx) / np.sum(
            np.abs(psi_normal) ** 2 * dx
        )
        var_anti = np.sum(g**2 * np.abs(psi_anti) ** 2 * dx) / np.sum(np.abs(psi_anti) ** 2 * dx)
        assert var_anti > var_normal


@pytest.mark.unit
class TestPlaneWavePacket:
    def test_zero_momentum_is_real(self, grid):
        from wings.wavefunctions import plane_wave_packet

        psi = plane_wave_packet(grid, k0=0.0)
        assert np.allclose(psi.imag, 0.0, atol=1e-15)

    def test_nonzero_momentum_is_complex(self, grid):
        from wings.wavefunctions import plane_wave_packet

        psi = plane_wave_packet(grid, k0=2.0)
        assert np.any(np.abs(psi.imag) > 0.1)

    def test_shape_and_dtype(self, grid):
        from wings.wavefunctions import plane_wave_packet

        psi = plane_wave_packet(grid, k0=1.0)
        assert psi.shape == grid.shape
        assert psi.dtype == np.complex128

    def test_envelope_is_gaussian(self, grid):
        from wings.wavefunctions import plane_wave_packet

        psi = plane_wave_packet(grid, k0=3.0, sigma=1.0)
        # |psi|^2 should be Gaussian regardless of momentum
        abs_psi = np.abs(psi)
        expected_envelope = np.exp(-(grid**2) / 2)
        np.testing.assert_allclose(abs_psi, expected_envelope, atol=1e-10)


@pytest.mark.unit
class TestListWavefunctions:
    def test_returns_dict(self):
        from wings.wavefunctions import list_wavefunctions

        result = list_wavefunctions()
        assert isinstance(result, dict)
        assert len(result) == 6

    def test_all_functions_importable(self):
        import wings.wavefunctions as wf
        from wings.wavefunctions import list_wavefunctions

        for name in list_wavefunctions():
            assert hasattr(wf, name)
