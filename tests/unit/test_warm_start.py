"""Unit tests for warm-start transfer learning module."""

import numpy as np
import pytest


@pytest.mark.unit
class TestWarmStart:
    """Tests for transfer_params function."""

    def test_transfer_increases_param_count(self):
        """Transfer from 6q (36 params) to 8q (64 params)."""
        from wings.warm_start import transfer_params

        source = np.random.randn(36)
        result = transfer_params(source, n_source=6, n_target=8)
        assert result.shape == (64,)

    def test_transfer_preserves_existing_values(self):
        """First 6 params of layer 0 are preserved from source."""
        from wings.warm_start import transfer_params

        source = np.random.randn(36)
        result = transfer_params(source, n_source=6, n_target=8)
        # Layer 0: first n_source params should match source layer 0
        np.testing.assert_array_equal(result[:6], source[:6])

    def test_new_qubit_params_near_zero(self):
        """Positions 6,7 in each layer of 8q result are < 0.1."""
        from wings.warm_start import transfer_params

        np.random.seed(42)
        source = np.ones(36)  # All ones so new params stand out
        result = transfer_params(source, n_source=6, n_target=8, init_scale=0.01)
        # Check new qubit positions (indices 6,7) in each of the first 6 layers
        n_target = 8
        for layer in range(6):
            offset = layer * n_target
            assert abs(result[offset + 6]) < 0.1
            assert abs(result[offset + 7]) < 0.1

    def test_same_qubit_count_identity(self):
        """transfer_params(params, 6, 6) returns a copy of params."""
        from wings.warm_start import transfer_params

        source = np.random.randn(36)
        result = transfer_params(source, n_source=6, n_target=6)
        np.testing.assert_array_equal(result, source)
        # Must be a copy, not the same object
        assert result is not source

    def test_invalid_source_larger(self):
        """ValueError when n_source > n_target."""
        from wings.warm_start import transfer_params

        source = np.random.randn(64)
        with pytest.raises(ValueError):
            transfer_params(source, n_source=8, n_target=6)

    def test_warm_start_improves_initial_fidelity(self):
        """3q Adam(30 steps) -> transfer to 4q -> fidelity > 0."""
        from wings.adam import AdamOptimizer
        from wings.ansatz import DefaultAnsatz
        from wings.config import OptimizerConfig
        from wings.evaluators.cpu import ThreadSafeCircuitEvaluator
        from wings.warm_start import transfer_params

        # Build 3-qubit optimized params (9 params = fast gradient)
        n_source = 3
        config_3 = OptimizerConfig(n_qubits=n_source, sigma=1.0)
        ansatz_3 = DefaultAnsatz(n_source)

        # Create a simple Gaussian target
        x = np.linspace(-np.pi, np.pi, 2**n_source)
        target = np.exp(-(x**2) / 2)
        target = target / np.linalg.norm(target)

        evaluator_3 = ThreadSafeCircuitEvaluator(config_3, target)

        params = np.random.randn(ansatz_3.n_params) * 0.1
        adam = AdamOptimizer(n_params=ansatz_3.n_params, learning_rate=0.05)

        # Run 30 Adam steps with parameter-shift gradient
        for _ in range(30):
            grad = np.zeros_like(params)
            for i in range(len(params)):
                p_plus = params.copy()
                p_minus = params.copy()
                p_plus[i] += np.pi / 2
                p_minus[i] -= np.pi / 2
                grad[i] = (
                    -(evaluator_3.compute_fidelity(p_plus) - evaluator_3.compute_fidelity(p_minus))
                    / 2
                )
            params = adam.step(params, grad)

        # Transfer to 4 qubits
        n_target = 4
        transferred = transfer_params(params, n_source=n_source, n_target=n_target)

        # Build 4q target and evaluator
        config_4 = OptimizerConfig(n_qubits=n_target, sigma=1.0)
        x4 = np.linspace(-np.pi, np.pi, 2**n_target)
        target4 = np.exp(-(x4**2) / 2)
        target4 = target4 / np.linalg.norm(target4)

        evaluator_4 = ThreadSafeCircuitEvaluator(config_4, target4)
        fidelity = evaluator_4.compute_fidelity(transferred)
        assert fidelity > 0
