"""Unit tests for time-dependent wavepackets (v0.4.0 WI-4)."""

import numpy as np
import pytest


@pytest.mark.unit
class TestSplitOperator:
    def test_free_particle_spreads(self):
        """Free particle Gaussian should spread over time."""
        from wings.time_evolution import make_grid, split_operator_step

        n = 256
        x, dx, k = make_grid(n, L=10.0)
        sigma = 0.5
        psi = np.exp(-(x**2) / (2 * sigma**2)).astype(np.complex128)
        psi /= np.linalg.norm(psi)

        width_before = np.sqrt(np.sum(x**2 * np.abs(psi) ** 2) * dx)

        V = np.zeros_like(x)
        T_k = k**2 / 2.0  # mass=1
        for _ in range(100):
            psi = split_operator_step(psi, V, T_k, dt=0.01)

        width_after = np.sqrt(np.sum(x**2 * np.abs(psi) ** 2) * dx)
        assert width_after > width_before * 1.1

    def test_norm_preserved(self):
        """Split-operator should preserve norm."""
        from wings.time_evolution import make_grid, split_operator_step

        n = 128
        x, dx, k = make_grid(n, L=8.0)
        psi = np.exp(-(x**2) / 2).astype(np.complex128)
        psi /= np.linalg.norm(psi)

        V = 0.5 * x**2  # Harmonic
        T_k = k**2 / 2.0
        for _ in range(50):
            psi = split_operator_step(psi, V, T_k, dt=0.01)

        norm = np.linalg.norm(psi)
        assert abs(norm - 1.0) < 1e-10

    def test_harmonic_oscillator_revival(self):
        """Coherent state in harmonic potential should oscillate."""
        from wings.time_evolution import make_grid, split_operator_step

        n = 256
        x, dx, k = make_grid(n, L=10.0)
        x0 = 2.0
        psi_0 = np.exp(-((x - x0) ** 2) / 2).astype(np.complex128)
        psi_0 /= np.linalg.norm(psi_0)

        V = 0.5 * x**2
        T_k = k**2 / 2.0

        psi = psi_0.copy()
        # Evolve for one full period (T = 2*pi for omega=1)
        n_steps = 628  # 2*pi / 0.01
        for _ in range(n_steps):
            psi = split_operator_step(psi, V, T_k, dt=0.01)

        # Should return close to initial state
        fid = np.abs(np.vdot(psi_0, psi)) ** 2
        assert fid > 0.9

    def test_make_grid(self):
        from wings.time_evolution import make_grid

        x, dx, k = make_grid(64, L=5.0)
        assert len(x) == 64
        assert len(k) == 64
        assert abs(dx - 10.0 / 63) < 0.01


@pytest.mark.unit
class TestBuiltInPotentials:
    def test_free_particle(self):
        from wings.time_evolution import free_particle

        x = np.linspace(-5, 5, 64)
        V = free_particle(x)
        np.testing.assert_array_equal(V, np.zeros(64))

    def test_harmonic_oscillator(self):
        from wings.time_evolution import harmonic_potential

        x = np.linspace(-5, 5, 64)
        V = harmonic_potential(x, omega=1.0, mass=1.0)
        assert V[32] < V[0]  # Minimum at center

    def test_morse_potential(self):
        from wings.time_evolution import morse_potential

        x = np.linspace(-2, 5, 64)
        V = morse_potential(x, D_e=10.0, a=1.0, x_e=0.0)
        assert V[0] > 0  # Repulsive wall
        # Minimum at x_e
        min_idx = np.argmin(V)
        assert abs(x[min_idx] - 0.0) < 0.5

    def test_lennard_jones(self):
        from wings.time_evolution import lennard_jones_potential

        x = np.linspace(0.5, 5, 64)
        V = lennard_jones_potential(x, epsilon=1.0, sigma_lj=1.0)
        assert np.all(np.isfinite(V))


@pytest.mark.unit
class TestEvolveClassical:
    def test_evolve_returns_trajectory(self):
        from wings.time_evolution import evolve_classical, make_grid

        n = 64
        x, dx, k = make_grid(n, L=5.0)
        psi_0 = np.exp(-(x**2) / 2).astype(np.complex128)
        psi_0 /= np.linalg.norm(psi_0)

        V = np.zeros_like(x)
        T_k = k**2 / 2.0

        trajectory = evolve_classical(psi_0, V, T_k, dt=0.01, n_steps=10)
        assert len(trajectory) == 11  # Initial + 10 steps
        assert trajectory[0].shape == (n,)
