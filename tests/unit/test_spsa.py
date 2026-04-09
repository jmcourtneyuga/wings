"""Unit tests for SPSA optimizer module."""

import numpy as np
import pytest


@pytest.mark.unit
class TestSPSAOptimizer:
    """Tests for SPSAOptimizer class."""

    def test_initialization(self):
        """Test SPSA optimizer initialization."""
        from wings.spsa import SPSAOptimizer

        spsa = SPSAOptimizer(n_params=10, a=0.1, c=0.1)
        assert spsa.n_params == 10
        assert spsa.k == 0

    def test_gain_sequences_decay(self):
        """Test that gain sequences a_k, c_k decay over iterations."""
        from wings.spsa import SPSAOptimizer

        spsa = SPSAOptimizer(n_params=10, a=1.0, c=1.0, A=10.0)

        a_values = []
        c_values = []
        for k in range(100):
            spsa.k = k
            a_values.append(spsa.get_a_k())
            c_values.append(spsa.get_c_k())

        # Both should be monotonically decreasing
        for i in range(1, len(a_values)):
            assert a_values[i] < a_values[i - 1]
            assert c_values[i] < c_values[i - 1]

    def test_perturbation_is_rademacher(self):
        """Test perturbation vector contains only +1 and -1."""
        from wings.spsa import SPSAOptimizer

        spsa = SPSAOptimizer(n_params=50)

        for _ in range(10):
            delta = spsa._generate_perturbation()
            assert delta.shape == (50,)
            assert set(np.unique(delta)).issubset({-1.0, 1.0})

    def test_gradient_estimate_shape(self):
        """Test gradient estimate has correct shape."""
        from wings.spsa import SPSAOptimizer

        spsa = SPSAOptimizer(n_params=10)

        def quadratic(x):
            return np.sum(x**2)

        params = np.ones(10)
        g_hat, n_evals = spsa.estimate_gradient(params, quadratic)

        assert g_hat.shape == (10,)
        assert n_evals == 2

    def test_gradient_estimate_averaging(self):
        """Test that n_avg > 1 uses more evaluations."""
        from wings.spsa import SPSAOptimizer

        spsa = SPSAOptimizer(n_params=10, n_avg=5)

        def quadratic(x):
            return np.sum(x**2)

        params = np.ones(10)
        g_hat, n_evals = spsa.estimate_gradient(params, quadratic)

        assert n_evals == 10  # 2 * n_avg

    def test_gradient_estimate_unbiased_on_quadratic(self):
        """Averaged over many samples, SPSA gradient should approximate true gradient."""
        from wings.spsa import SPSAOptimizer

        np.random.seed(42)
        n = 10
        params = np.random.randn(n)

        # Quadratic: f(x) = x^T x, true gradient = 2x
        def quadratic(x):
            return np.sum(x**2)

        true_grad = 2 * params

        # Average many SPSA estimates
        n_samples = 2000
        g_sum = np.zeros(n)
        for _ in range(n_samples):
            spsa = SPSAOptimizer(n_params=n, c=0.01)
            g_hat, _ = spsa.estimate_gradient(params, quadratic)
            g_sum += g_hat

        g_avg = g_sum / n_samples

        # SPSA is a high-variance estimator; even with 2000 samples,
        # individual components can deviate. Use generous tolerance.
        np.testing.assert_allclose(g_avg, true_grad, rtol=0.3, atol=0.15)

    def test_spsa_converges_on_quadratic(self):
        """SPSA should find minimum of a simple quadratic."""
        from wings.spsa import SPSAOptimizer

        np.random.seed(42)
        n = 5

        def quadratic(x):
            return np.sum((x - 1.0) ** 2)  # minimum at x = [1, 1, ..., 1]

        params = np.zeros(n)
        spsa = SPSAOptimizer(n_params=n, a=0.5, c=0.1, A=50.0)

        for _ in range(1000):
            params, _, _ = spsa.step(params, quadratic)

        # Should be near the minimum
        np.testing.assert_allclose(params, np.ones(n), atol=0.3)

    def test_step_updates_iteration_counter(self):
        """Test that step() increments k."""
        from wings.spsa import SPSAOptimizer

        spsa = SPSAOptimizer(n_params=5)

        assert spsa.k == 0
        spsa.step(np.zeros(5), lambda x: np.sum(x**2))
        assert spsa.k == 1
        spsa.step(np.zeros(5), lambda x: np.sum(x**2))
        assert spsa.k == 2

    def test_reset(self):
        """Test reset clears iteration counter."""
        from wings.spsa import SPSAOptimizer

        spsa = SPSAOptimizer(n_params=5)
        spsa.k = 100
        spsa.reset()
        assert spsa.k == 0

    def test_spsa_only_two_evals_per_step(self):
        """Verify SPSA uses exactly 2 evals per step (n_avg=1)."""
        from wings.spsa import SPSAOptimizer

        eval_count = 0

        def counting_fn(x):
            nonlocal eval_count
            eval_count += 1
            return np.sum(x**2)

        spsa = SPSAOptimizer(n_params=50, n_avg=1)
        params = np.zeros(50)
        _, _, n_evals = spsa.step(params, counting_fn)

        assert n_evals == 2
        assert eval_count == 2


@pytest.mark.unit
class TestOptimizeSPSA:
    """Tests for GaussianOptimizer.optimize_spsa()."""

    def test_optimize_spsa_basic(self, tiny_optimizer, random_params_3q):
        """Test basic SPSA optimization returns correct structure."""
        result = tiny_optimizer.optimize_spsa(
            random_params_3q,
            max_steps=50,
            a=0.05,
            c=0.1,
        )

        assert "params" in result
        assert "fidelity" in result
        assert "history" in result
        assert "steps" in result
        assert "total_evals" in result
        assert "time" in result
        assert result["fidelity"] > 0

    def test_optimize_spsa_improves_fidelity(self, tiny_optimizer, random_params_3q):
        """Test SPSA improves fidelity from initial params."""
        initial_fid = tiny_optimizer.compute_fidelity(params=random_params_3q)

        result = tiny_optimizer.optimize_spsa(
            random_params_3q,
            max_steps=100,
            a=0.05,
            c=0.1,
        )

        assert result["fidelity"] >= initial_fid

    def test_optimize_spsa_max_time(self, tiny_optimizer, random_params_3q):
        """Test SPSA respects max_time parameter."""
        import time

        start = time.time()

        result = tiny_optimizer.optimize_spsa(
            random_params_3q,
            max_steps=100000,
            max_time=2.0,
        )

        elapsed = time.time() - start
        assert elapsed < 5.0
        assert "fidelity" in result

    def test_optimize_spsa_eval_efficiency(self, tiny_optimizer):
        """Test SPSA uses far fewer evals than parameter-shift per step."""
        params = tiny_optimizer.get_initial_params("smart")

        result = tiny_optimizer.optimize_spsa(
            params,
            max_steps=10,
            n_avg=1,
        )

        # With n_avg=1: 2 evals for SPSA gradient + 1 for fidelity check = 3 per step
        # 10 steps should use ~30 evals (plus some from objective calls inside SPSA)
        # Parameter-shift would use 2*n_params evals per gradient
        assert result["total_evals"] < 10 * 2 * tiny_optimizer.n_params
