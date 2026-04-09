"""Unit tests for noise-aware optimization (v0.4.0 WI-6)."""

import pytest


@pytest.mark.unit
class TestNoiseConfig:
    def test_default_initialization(self):
        from wings.evaluators.noisy import NoiseConfig

        nc = NoiseConfig()
        assert nc.depolarizing_rate == 0.0
        assert nc.gate_error_1q == 0.0
        assert nc.gate_error_2q == 0.0
        assert nc.readout_error == 0.0

    def test_custom_values(self):
        from wings.evaluators.noisy import NoiseConfig

        nc = NoiseConfig(depolarizing_rate=0.01, gate_error_2q=0.005)
        assert nc.depolarizing_rate == 0.01
        assert nc.gate_error_2q == 0.005

    def test_has_noise(self):
        from wings.evaluators.noisy import NoiseConfig

        nc_clean = NoiseConfig()
        nc_noisy = NoiseConfig(depolarizing_rate=0.01)
        assert not nc_clean.has_noise()
        assert nc_noisy.has_noise()

    def test_noise_robust_objective_weight(self):
        from wings.evaluators.noisy import NoiseConfig

        nc = NoiseConfig(depolarizing_rate=0.01)
        # Noise-robust objective: (1-F_ideal) + lambda * (F_ideal - F_noisy)
        # With F_ideal=0.99, F_noisy=0.95, lambda=0.1:
        ideal = 0.99
        noisy = 0.95
        obj = nc.noise_robust_objective(ideal, noisy, robustness_weight=0.1)
        expected = (1 - ideal) + 0.1 * (ideal - noisy)
        assert abs(obj - expected) < 1e-10

    def test_depth_penalty(self):
        from wings.evaluators.noisy import NoiseConfig

        nc = NoiseConfig()
        # depth_penalty(n_cx_gates, weight)
        penalty = nc.depth_penalty(n_cx_gates=20, weight=0.001)
        assert abs(penalty - 0.02) < 1e-10

    def test_depth_penalty_zero_weight(self):
        from wings.evaluators.noisy import NoiseConfig

        nc = NoiseConfig()
        penalty = nc.depth_penalty(n_cx_gates=100, weight=0.0)
        assert penalty == 0.0


@pytest.mark.unit
class TestNoiseRobustObjective:
    def test_clean_objective_matches_infidelity(self):
        """With no noise, noise-robust objective = infidelity."""
        from wings.evaluators.noisy import NoiseConfig

        nc = NoiseConfig()
        obj = nc.noise_robust_objective(0.99, 0.99, robustness_weight=0.1)
        assert abs(obj - 0.01) < 1e-10

    def test_noisy_increases_objective(self):
        """More noise gap should increase the objective."""
        from wings.evaluators.noisy import NoiseConfig

        nc = NoiseConfig(depolarizing_rate=0.01)
        obj_small_gap = nc.noise_robust_objective(0.99, 0.98, 0.1)
        obj_large_gap = nc.noise_robust_objective(0.99, 0.90, 0.1)
        assert obj_large_gap > obj_small_gap
