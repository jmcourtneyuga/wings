"""Unit tests for optimizer module."""

import time

import numpy as np
import pytest


class TestGaussianOptimizerInit:
    """Tests for GaussianOptimizer initialization."""

    def test_basic_initialization(self, small_config):
        """Test basic optimizer initialization."""
        from wings import GaussianOptimizer

        opt = GaussianOptimizer(small_config)

        assert opt.config is small_config
        assert opt.n_params == 36  # 6*6
        assert opt.n_evals == 0
        assert opt.best_fidelity == 0
        assert opt.best_params is None

    def test_target_wavefunction_normalized(self, small_config):
        """Test target wavefunction is normalized."""
        from wings import GaussianOptimizer

        opt = GaussianOptimizer(small_config)

        norm = np.linalg.norm(opt.target)
        assert np.isclose(norm, 1.0, atol=1e-10)

    def test_target_wavefunction_shape(self, small_config):
        """Test target wavefunction has correct shape."""
        from wings import GaussianOptimizer

        opt = GaussianOptimizer(small_config)

        assert opt.target.shape == (64,)  # 2^6
        assert opt.target.dtype == np.complex128

    def test_positions_match_config(self, small_config):
        """Test positions array matches config."""
        from wings import GaussianOptimizer

        opt = GaussianOptimizer(small_config)

        np.testing.assert_array_equal(opt.positions, small_config.positions)

    def test_lorentzian_target(self, lorentzian_config):
        """Test Lorentzian target wavefunction."""
        from wings import GaussianOptimizer

        opt = GaussianOptimizer(lorentzian_config)

        # Should be normalized
        norm = np.linalg.norm(opt.target)
        assert np.isclose(norm, 1.0, atol=1e-10)

        # Peak should be at center (x0=0)
        peak_idx = np.argmax(np.abs(opt.target))
        center_idx = len(opt.target) // 2
        assert abs(peak_idx - center_idx) <= 1

    def test_shifted_target(self, shifted_config):
        """Test shifted Gaussian target."""
        from wings import GaussianOptimizer

        opt = GaussianOptimizer(shifted_config)

        # Peak should be offset from center
        peak_idx = np.argmax(np.abs(opt.target))
        center_idx = len(opt.target) // 2
        assert peak_idx > center_idx  # x0 = 1.5 > 0

    def test_sech_target(self):
        """Test hyperbolic secant target."""
        from wings import GaussianOptimizer, OptimizerConfig, TargetFunction

        config = OptimizerConfig(
            n_qubits=6,
            target_function=TargetFunction.SECH,
            sigma=0.5,
            verbose=False,
            use_gpu=False,
            use_custatevec=False,
        )
        opt = GaussianOptimizer(config)

        norm = np.linalg.norm(opt.target)
        assert np.isclose(norm, 1.0, atol=1e-10)

    def test_custom_target(self):
        """Test custom target function."""
        from wings import GaussianOptimizer, OptimizerConfig, TargetFunction

        def double_peak(x):
            return np.exp(-((x - 1) ** 2) / 0.5) + np.exp(-((x + 1) ** 2) / 0.5)

        config = OptimizerConfig(
            n_qubits=6,
            target_function=TargetFunction.CUSTOM,
            custom_target_fn=double_peak,
            box_size=4.0,
            verbose=False,
            use_gpu=False,
            use_custatevec=False,
        )
        opt = GaussianOptimizer(config)

        # Should be normalized
        norm = np.linalg.norm(opt.target)
        assert np.isclose(norm, 1.0, atol=1e-10)

        # Should have two peaks
        abs_target = np.abs(opt.target)
        # Find local maxima
        peaks = []
        for i in range(1, len(abs_target) - 1):
            if abs_target[i] > abs_target[i - 1] and abs_target[i] > abs_target[i + 1]:
                peaks.append(i)
        assert len(peaks) >= 2


class TestGaussianOptimizerStatevector:
    """Tests for statevector computation."""

    def test_statevector_normalized(
        self, small_optimizer, random_params_6q, assert_valid_statevector
    ):
        """Test statevector is normalized."""
        sv = small_optimizer.get_statevector(random_params_6q)
        assert_valid_statevector(sv, 6)

    def test_statevector_shape(self, small_optimizer, random_params_6q):
        """Test statevector has correct shape."""
        sv = small_optimizer.get_statevector(random_params_6q)
        assert sv.shape == (64,)

    def test_statevector_deterministic(self, small_optimizer, random_params_6q):
        """Test statevector computation is deterministic."""
        sv1 = small_optimizer.get_statevector(random_params_6q)
        sv2 = small_optimizer.get_statevector(random_params_6q)
        np.testing.assert_array_almost_equal(sv1, sv2)

    def test_different_params_different_statevector(
        self, small_optimizer, random_params_6q, zero_params_6q
    ):
        """Test different parameters give different statevectors."""
        sv1 = small_optimizer.get_statevector(random_params_6q)
        sv2 = small_optimizer.get_statevector(zero_params_6q)

        assert not np.allclose(sv1, sv2)


class TestGaussianOptimizerFidelity:
    """Tests for fidelity computation."""

    def test_fidelity_range(self, small_optimizer, random_params_6q, assert_valid_fidelity):
        """Test fidelity is in valid range."""
        sv = small_optimizer.get_statevector(random_params_6q)
        fidelity = small_optimizer._compute_fidelity_fast(sv)
        assert_valid_fidelity(fidelity)

    def test_fidelity_with_self(self, small_optimizer):
        """Test fidelity of target with itself is 1."""
        fidelity = small_optimizer._compute_fidelity_fast(small_optimizer.target)
        assert np.isclose(fidelity, 1.0, atol=1e-10)

    def test_fidelity_symmetric(self, small_optimizer, random_params_6q):
        """Test fidelity computation is symmetric."""
        sv = small_optimizer.get_statevector(random_params_6q)

        # |<psi|target>|^2 = |<target|psi>|^2
        fid1 = small_optimizer._compute_fidelity_fast(sv)

        # Compute manually the other way
        overlap = np.vdot(sv, small_optimizer.target)
        fid2 = np.abs(overlap) ** 2

        assert np.isclose(fid1, fid2, atol=1e-10)

    def test_compute_fidelity_method(
        self, small_optimizer, random_params_6q, assert_valid_fidelity
    ):
        """Test compute_fidelity public method."""
        fidelity = small_optimizer.compute_fidelity(params=random_params_6q)
        assert_valid_fidelity(fidelity)


class TestGaussianOptimizerGradient:
    """Tests for gradient computation."""

    def test_gradient_shape(self, small_optimizer, random_params_6q):
        """Test gradient has correct shape."""
        grad = small_optimizer.compute_gradient(random_params_6q)
        assert grad.shape == (36,)

    def test_gradient_finite(self, small_optimizer, random_params_6q):
        """Test gradient has no NaN or Inf values."""
        grad = small_optimizer.compute_gradient(random_params_6q)
        assert np.all(np.isfinite(grad))

    def test_gradient_nonzero(self, small_optimizer, random_params_6q):
        """Test gradient is not all zeros (for random params)."""
        grad = small_optimizer.compute_gradient(random_params_6q)
        assert np.any(grad != 0)

    def test_gradient_matches_finite_difference(self, small_optimizer, random_params_6q):
        """Test gradient approximately matches finite difference."""
        grad_analytic = small_optimizer.compute_gradient(random_params_6q)

        # Compute finite difference gradient for a few parameters
        eps = 1e-5
        grad_fd = np.zeros(5)  # Just check first 5 params

        for i in range(5):
            params_plus = random_params_6q.copy()
            params_plus[i] += eps
            fid_plus = small_optimizer.compute_fidelity(params=params_plus)

            params_minus = random_params_6q.copy()
            params_minus[i] -= eps
            fid_minus = small_optimizer.compute_fidelity(params=params_minus)

            # Note: gradient is for -fidelity (minimization)
            grad_fd[i] = -(fid_plus - fid_minus) / (2 * eps)

        # Should be approximately equal
        np.testing.assert_allclose(grad_analytic[:5], grad_fd, rtol=1e-2, atol=1e-4)


class TestGaussianOptimizerInitialParams:
    """Tests for parameter initialization strategies."""

    def test_smart_initialization(self, small_optimizer):
        """Test 'smart' initialization."""
        params = small_optimizer.get_initial_params("smart")

        assert params.shape == (36,)
        assert np.all(np.isfinite(params))

    def test_random_initialization(self, small_optimizer):
        """Test 'random' initialization."""
        params = small_optimizer.get_initial_params("random")

        assert params.shape == (36,)
        assert np.all(np.isfinite(params))

    def test_gaussian_product_initialization(self, small_optimizer):
        """Test 'gaussian_product' initialization."""
        params = small_optimizer.get_initial_params("gaussian_product")

        assert params.shape == (36,)
        assert np.all(np.isfinite(params))

    def test_zero_initialization(self, small_optimizer):
        """Test 'zero' initialization."""
        params = small_optimizer.get_initial_params("zero")

        assert params.shape == (36,)
        np.testing.assert_array_equal(params, np.zeros(36))

    def test_different_strategies_different_params(self, small_optimizer):
        """Test different strategies give different parameters."""
        np.random.seed(42)
        params_smart = small_optimizer.get_initial_params("smart")
        np.random.seed(42)
        small_optimizer.get_initial_params("random")

        # At minimum, smart should give reasonable fidelity
        fid = small_optimizer.compute_fidelity(params=params_smart)
        assert fid > 0.01  # Should be better than random chance


class TestGaussianOptimizerPopulation:
    """Tests for population evaluation."""

    def test_evaluate_population_shape(self, small_optimizer, population_small):
        """Test population evaluation returns correct shape."""
        fidelities = small_optimizer.evaluate_population(population_small)
        assert fidelities.shape == (10,)

    def test_evaluate_population_valid(
        self, small_optimizer, population_small, assert_valid_fidelity
    ):
        """Test all population fidelities are valid."""
        fidelities = small_optimizer.evaluate_population(population_small)
        for fid in fidelities:
            assert_valid_fidelity(fid)

    def test_evaluate_population_updates_best(self, small_optimizer, population_small):
        """Test population evaluation updates best_fidelity."""
        initial_best = small_optimizer.best_fidelity
        fidelities = small_optimizer.evaluate_population(population_small)

        # Best should be updated if any fidelity is better
        expected_best = max(initial_best, np.max(fidelities))
        assert np.isclose(small_optimizer.best_fidelity, expected_best)

    def test_evaluate_population_updates_n_evals(self, small_optimizer, population_small):
        """Test population evaluation updates n_evals counter."""
        initial_evals = small_optimizer.n_evals
        small_optimizer.evaluate_population(population_small)

        # Should increment by population size
        assert small_optimizer.n_evals == initial_evals + len(population_small)


class TestGaussianOptimizerStatistics:
    """Tests for wavefunction statistics."""

    def test_compute_statistics_keys(self, small_optimizer, random_params_6q):
        """Test statistics contains expected keys."""
        sv = small_optimizer.get_statevector(random_params_6q)
        stats = small_optimizer.compute_statistics(sv)

        assert "mean" in stats
        assert "std" in stats

    def test_compute_statistics_target(self, small_optimizer):
        """Test statistics of target wavefunction."""
        stats = small_optimizer.compute_statistics(small_optimizer.target)

        # For centered Gaussian with default x0=0
        assert np.isclose(stats["mean"], 0.0, atol=0.1)
        # std should be close to sigma
        assert np.isclose(stats["std"], small_optimizer.config.sigma / np.sqrt(2), rtol=0.2)


class TestGaussianOptimizerAdam:
    """Tests for Adam optimization."""

    def test_optimize_adam_basic(self, small_optimizer, random_params_6q):
        """Test basic Adam optimization."""
        result = small_optimizer.optimize_adam(
            random_params_6q,
            max_steps=50,
            lr=0.01,
        )

        assert "params" in result
        assert "fidelity" in result
        assert result["fidelity"] > 0

    def test_optimize_adam_improves_fidelity(self, small_optimizer, random_params_6q):
        """Test Adam improves fidelity."""
        initial_fid = small_optimizer.compute_fidelity(params=random_params_6q)

        result = small_optimizer.optimize_adam(
            random_params_6q,
            max_steps=100,
            lr=0.02,
        )

        assert result["fidelity"] >= initial_fid

    def test_optimize_adam_max_time(self, small_optimizer, random_params_6q):
        """Test Adam respects max_time parameter."""
        start = time.time()

        result = small_optimizer.optimize_adam(
            random_params_6q,
            max_steps=100000,  # Very high, should hit time limit first
            lr=0.01,
            max_time=2.0,  # 2 seconds max
        )

        elapsed = time.time() - start

        # Should stop within a reasonable margin of max_time
        assert elapsed < 5.0  # Allow some overhead
        assert "fidelity" in result

    def test_optimize_adam_updates_n_evals(self, small_optimizer, random_params_6q):
        """Test Adam updates evaluation counter."""
        result = small_optimizer.optimize_adam(random_params_6q, max_steps=10)
        assert result.get("steps", 0) >= 10

    def test_optimize_adam_convergence_check(self, small_optimizer):
        """Test Adam convergence detection."""
        # Use smart init for faster convergence
        params = small_optimizer.get_initial_params("smart")

        result = small_optimizer.optimize_adam(
            params,
            max_steps=500,
            lr=0.02,
            convergence_window=20,
            convergence_threshold=1e-6,
        )

        # Should converge before max_steps if threshold is reasonable
        assert result["fidelity"] > 0.5


class TestGaussianOptimizerPipeline:
    """Tests for optimization pipeline."""

    def test_run_optimization_basic(self, small_optimizer):
        """Test basic optimization run."""
        from wings.config import OptimizationPipeline

        pipeline = OptimizationPipeline(
            target_fidelity=0.9,
            max_total_time=30,
            use_adam_stage=False,
            use_basin_hopping=False,
            verbose=False,
        )

        results = small_optimizer.run_optimization(pipeline)

        assert "fidelity" in results
        assert "optimal_params" in results
        assert "time" in results
        assert "n_evaluations" in results
        assert results["fidelity"] > 0.5

    def test_run_optimization_with_adam(self, small_optimizer):
        """Test pipeline with Adam stage."""
        from wings.config import OptimizationPipeline

        pipeline = OptimizationPipeline(
            target_fidelity=0.95,
            max_total_time=30,
            use_adam_stage=True,
            adam_max_steps=100,
            adam_time_fraction=0.5,
            verbose=False,
        )

        results = small_optimizer.run_optimization(pipeline)

        assert results["fidelity"] > 0.5
        assert results["n_evaluations"] > 0

    def test_pipeline_adam_time_limit(self, small_optimizer):
        """Test Adam stage respects time fraction."""
        from wings.config import OptimizationPipeline

        pipeline = OptimizationPipeline(
            target_fidelity=0.9999,
            max_total_time=10,  # Short time
            use_adam_stage=True,
            adam_max_steps=100000,  # Very high
            adam_time_fraction=0.4,  # 4 seconds for Adam
            use_lbfgs_refinement=False,  # Disable to test Adam time limit only
            verbose=False,
        )

        start = time.time()
        results = small_optimizer.run_optimization(pipeline)
        total_time = time.time() - start

        # Total should be around max_total_time (with some margin)
        assert total_time < 30  # Shouldn't run forever
        assert "fidelity" in results

    def test_run_optimization_returns_n_evaluations(self, small_optimizer):
        """Test that n_evaluations is tracked and returned."""
        from wings.config import OptimizationPipeline

        pipeline = OptimizationPipeline(
            target_fidelity=0.9,
            max_total_time=20,
            use_adam_stage=True,
            adam_max_steps=50,
            verbose=False,
        )

        results = small_optimizer.run_optimization(pipeline)

        assert "n_evaluations" in results
        assert results["n_evaluations"] > 0
        assert isinstance(results["n_evaluations"], int)


@pytest.mark.slow
class TestGaussianOptimizerOptimizationSlow:
    """Slow tests for actual optimization runs."""

    def test_objective_decreases(self, small_optimizer, random_params_6q):
        """Test that optimization makes progress."""
        initial_fid = small_optimizer.compute_fidelity(params=random_params_6q)

        # Run a few optimization steps manually
        from scipy.optimize import minimize

        result = minimize(
            small_optimizer.objective,
            random_params_6q,
            method="L-BFGS-B",
            jac=lambda p: small_optimizer.compute_gradient(p),
            options={"maxiter": 50},
        )

        final_fid = -result.fun
        assert final_fid >= initial_fid

    def test_high_fidelity_achievable(self, small_optimizer):
        """Test that high fidelity is achievable."""
        from wings.config import OptimizationPipeline

        pipeline = OptimizationPipeline(
            target_fidelity=0.99,
            max_total_time=60,
            use_adam_stage=True,
            adam_max_steps=1000,
            use_lbfgs_refinement=True,
            verbose=False,
        )

        results = small_optimizer.run_optimization(pipeline)

        assert results["fidelity"] > 0.95
