"""Unit tests for convenience functions."""

import pytest


@pytest.mark.unit
class TestConvenienceFunctions:
    def test_optimize_gaussian_state_import(self):
        from wings.convenience import optimize_gaussian_state

        assert callable(optimize_gaussian_state)

    def test_quick_optimize_import(self):
        from wings.convenience import quick_optimize

        assert callable(quick_optimize)

    @pytest.mark.gpu
    def test_quick_optimize_runs(self):
        from wings.convenience import quick_optimize

        fidelity, results = quick_optimize(
            n_qubits=6,
            sigma=0.5,
            verbose=False,
        )
        assert fidelity > 0
        assert "fidelity" in results
