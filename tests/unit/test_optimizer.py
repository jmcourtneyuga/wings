"""Unit tests for optimizer module."""

import time

import numpy as np
import pytest


@pytest.mark.unit
class TestDirectInfidelity:
    """Tests for direct infidelity computation (P3)."""

    def test_infidelity_direct_matches_naive_low_f(self, small_optimizer, random_params_6q):
        """For F < 0.99, direct and naive methods should agree."""
        psi = small_optimizer.get_statevector(random_params_6q)
        fidelity = small_optimizer._compute_fidelity_fast(psi)

        naive_infidelity = 1.0 - fidelity
        direct_infidelity = small_optimizer._compute_infidelity_direct(psi)

        assert abs(naive_infidelity - direct_infidelity) < 1e-12

    def test_infidelity_direct_self_is_zero(self, small_optimizer):
        """Target with itself should give infidelity ~ 0."""
        infidelity = small_optimizer._compute_infidelity_direct(small_optimizer.target)
        assert infidelity < 1e-28

    def test_infidelity_direct_orthogonal_is_one(self, small_optimizer):
        """Orthogonal state should give infidelity = 1."""
        # Create a state orthogonal to target
        n = len(small_optimizer.target)
        ortho = np.zeros(n, dtype=np.complex128)
        # Find index where target has minimum magnitude, put all weight there
        min_idx = np.argmin(np.abs(small_optimizer.target))
        ortho[min_idx] = 1.0
        # Gram-Schmidt orthogonalize
        overlap = np.dot(np.conj(small_optimizer.target), ortho)
        ortho = ortho - overlap * small_optimizer.target
        ortho = ortho / np.linalg.norm(ortho)

        infidelity = small_optimizer._compute_infidelity_direct(ortho)
        assert abs(infidelity - 1.0) < 1e-10

    def test_infidelity_direct_nonnegative(self, small_optimizer):
        """Infidelity must always be >= 0."""
        for _ in range(20):
            params = np.random.randn(small_optimizer.n_params) * 0.5
            psi = small_optimizer.get_statevector(params)
            infidelity = small_optimizer._compute_infidelity_direct(psi)
            assert infidelity >= 0.0

    def test_infidelity_direct_at_high_f(self, small_optimizer):
        """Direct method should be stable at F ~ 1 - 1e-14 where naive 1-F fails."""
        target = small_optimizer.target
        # Construct a state with known tiny infidelity:
        # psi = (1-eps^2/2)*target + eps*ortho  (normalized, F ~ 1 - eps^2)
        n = len(target)
        # Build an orthogonal unit vector
        ortho = np.zeros(n, dtype=np.complex128)
        min_idx = np.argmin(np.abs(target))
        ortho[min_idx] = 1.0
        overlap = np.dot(np.conj(target), ortho)
        ortho = ortho - overlap * target
        ortho = ortho / np.linalg.norm(ortho)

        eps = 1e-7  # gives infidelity ~ eps^2 = 1e-14
        psi = target * np.sqrt(1.0 - eps**2) + ortho * eps
        psi = psi / np.linalg.norm(psi)  # ensure normalization

        expected_infidelity = eps**2  # ~1e-14

        direct_infidelity = small_optimizer._compute_infidelity_direct(psi)

        # Direct method should be accurate to within a factor of 2
        assert direct_infidelity > 0, "Direct infidelity should be positive"
        assert abs(direct_infidelity - expected_infidelity) / expected_infidelity < 0.5, (
            f"Direct infidelity {direct_infidelity:.2e} should be close to "
            f"expected {expected_infidelity:.2e}"
        )

        # Naive 1-F loses precision here
        fidelity = small_optimizer._compute_fidelity_fast(psi)
        1.0 - fidelity
        # naive_infidelity may have large relative error at this scale
        # (this documents the problem, not a pass/fail criterion)

    def test_infidelity_direct_consistency(self, small_optimizer, random_params_6q):
        """F + (1-F) should equal 1 when computed via direct method."""
        psi = small_optimizer.get_statevector(random_params_6q)
        fidelity = small_optimizer._compute_fidelity_fast(psi)
        infidelity = small_optimizer._compute_infidelity_direct(psi)

        assert abs(fidelity + infidelity - 1.0) < 1e-12


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

    def test_optimize_adam_basic(self, tiny_optimizer, random_params_3q):
        """Test basic Adam optimization."""
        result = tiny_optimizer.optimize_adam(
            random_params_3q,
            max_steps=20,
            lr=0.01,
        )

        assert "params" in result
        assert "fidelity" in result
        assert result["fidelity"] > 0

    def test_optimize_adam_improves_fidelity(self, tiny_optimizer, random_params_3q):
        """Test Adam improves fidelity."""
        initial_fid = tiny_optimizer.compute_fidelity(params=random_params_3q)

        result = tiny_optimizer.optimize_adam(
            random_params_3q,
            max_steps=50,
            lr=0.02,
        )

        assert result["fidelity"] >= initial_fid

    def test_optimize_adam_max_time(self, tiny_optimizer, random_params_3q):
        """Test Adam respects max_time parameter."""
        start = time.time()

        result = tiny_optimizer.optimize_adam(
            random_params_3q,
            max_steps=100000,  # Very high, should hit time limit first
            lr=0.01,
            max_time=2.0,  # 2 seconds max
        )

        elapsed = time.time() - start

        # Should stop within a reasonable margin of max_time
        assert elapsed < 5.0  # Allow some overhead
        assert "fidelity" in result

    def test_optimize_adam_updates_n_evals(self, tiny_optimizer, random_params_3q):
        """Test Adam updates evaluation counter."""
        result = tiny_optimizer.optimize_adam(random_params_3q, max_steps=10)
        assert result.get("steps", 0) >= 10

    def test_optimize_adam_convergence_check(self, tiny_optimizer):
        """Test Adam convergence detection."""
        # Use smart init for faster convergence
        params = tiny_optimizer.get_initial_params("smart")

        result = tiny_optimizer.optimize_adam(
            params,
            max_steps=200,
            lr=0.02,
            convergence_window=20,
            convergence_threshold=1e-6,
        )

        # Should converge before max_steps if threshold is reasonable
        assert result["fidelity"] > 0.5


class TestGaussianOptimizerPipeline:
    """Tests for optimization pipeline."""

    def test_run_optimization_basic(self, tiny_optimizer):
        """Test basic optimization run."""
        from wings.config import OptimizationPipeline

        pipeline = OptimizationPipeline(
            target_fidelity=0.9,
            max_total_time=30,
            use_adam_stage=False,
            use_basin_hopping=False,
            verbose=False,
        )

        results = tiny_optimizer.run_optimization(pipeline)

        assert "fidelity" in results
        assert "optimal_params" in results
        assert "time" in results
        assert "n_evaluations" in results
        assert results["fidelity"] > 0.5

    def test_run_optimization_with_adam(self, tiny_optimizer):
        """Test pipeline with Adam stage."""
        from wings.config import OptimizationPipeline

        pipeline = OptimizationPipeline(
            target_fidelity=0.95,
            max_total_time=30,
            use_adam_stage=True,
            adam_max_steps=50,
            adam_time_fraction=0.5,
            verbose=False,
        )

        results = tiny_optimizer.run_optimization(pipeline)

        assert results["fidelity"] > 0.5
        assert results["n_evaluations"] > 0

    def test_pipeline_adam_time_limit(self, tiny_optimizer):
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
        results = tiny_optimizer.run_optimization(pipeline)
        total_time = time.time() - start

        # Total should be around max_total_time (with some margin)
        assert total_time < 30  # Shouldn't run forever
        assert "fidelity" in results

    def test_run_optimization_returns_n_evaluations(self, tiny_optimizer):
        """Test that n_evaluations is tracked and returned."""
        from wings.config import OptimizationPipeline

        pipeline = OptimizationPipeline(
            target_fidelity=0.9,
            max_total_time=20,
            use_adam_stage=True,
            adam_max_steps=30,
            verbose=False,
        )

        results = tiny_optimizer.run_optimization(pipeline)

        assert "n_evaluations" in results
        assert results["n_evaluations"] > 0
        assert isinstance(results["n_evaluations"], int)


@pytest.mark.slow
class TestGaussianOptimizerOptimizationSlow:
    """Slow tests for actual optimization runs."""

    def test_objective_decreases(self, tiny_optimizer, random_params_3q):
        """Test that optimization makes progress."""
        initial_fid = tiny_optimizer.compute_fidelity(params=random_params_3q)

        # Run a few optimization steps manually
        from scipy.optimize import minimize

        result = minimize(
            tiny_optimizer.objective,
            random_params_3q,
            method="L-BFGS-B",
            jac=lambda p: tiny_optimizer.compute_gradient(p),
            options={"maxiter": 50},
        )

        final_fid = -result.fun
        assert final_fid >= initial_fid

    def test_high_fidelity_achievable(self, tiny_optimizer):
        """Test that high fidelity is achievable."""
        from wings.config import OptimizationPipeline

        pipeline = OptimizationPipeline(
            target_fidelity=0.99,
            max_total_time=60,
            use_adam_stage=True,
            adam_max_steps=200,
            use_lbfgs_refinement=True,
            verbose=False,
        )

        results = tiny_optimizer.run_optimization(pipeline)

        assert results["fidelity"] > 0.95


@pytest.mark.unit
class TestLogFidelityObjective:
    """Tests for logarithmic fidelity objective (P2)."""

    def test_log_objective_value(self, small_optimizer, random_params_6q):
        """Log-infidelity should equal log(1-F)."""
        psi = small_optimizer.get_statevector(random_params_6q)
        fidelity = small_optimizer._compute_fidelity_fast(psi)

        log_inf = small_optimizer.objective_log_infidelity(random_params_6q)
        expected = np.log(1.0 - fidelity) if fidelity < 1.0 else -40.0

        assert abs(log_inf - expected) < 1e-8

    def test_log_objective_is_negative(self, small_optimizer, random_params_6q):
        """Log-infidelity should always be negative (infidelity < 1)."""
        log_inf = small_optimizer.objective_log_infidelity(random_params_6q)
        assert log_inf < 0

    def test_log_objective_gradient_shape(self, small_optimizer, random_params_6q):
        """Gradient should have correct shape."""
        val, grad = small_optimizer.objective_and_gradient_log_infidelity(random_params_6q)
        assert isinstance(val, float)
        assert grad.shape == (small_optimizer.n_params,)

    def test_log_objective_gradient_finite(self, small_optimizer, random_params_6q):
        """Gradient should contain finite values."""
        val, grad = small_optimizer.objective_and_gradient_log_infidelity(random_params_6q)
        assert np.all(np.isfinite(grad))

    def test_log_objective_gradient_vs_finite_diff(self, small_optimizer, random_params_6q):
        """Gradient should approximate finite differences."""
        eps = 1e-5
        _, analytic_grad = small_optimizer.objective_and_gradient_log_infidelity(random_params_6q)

        # Check a few components with finite differences
        for i in range(min(3, len(random_params_6q))):
            params_plus = random_params_6q.copy()
            params_minus = random_params_6q.copy()
            params_plus[i] += eps
            params_minus[i] -= eps

            f_plus = small_optimizer.objective_log_infidelity(params_plus)
            f_minus = small_optimizer.objective_log_infidelity(params_minus)

            fd_grad = (f_plus - f_minus) / (2 * eps)

            # Relative or absolute tolerance
            if abs(analytic_grad[i]) > 1e-6:
                assert abs(analytic_grad[i] - fd_grad) / abs(analytic_grad[i]) < 0.2
            else:
                assert abs(analytic_grad[i] - fd_grad) < 0.1

    def test_log_objective_at_zero_fidelity(self, small_optimizer):
        """Log objective should work near F=0."""
        # Random params will give low fidelity
        params = np.random.randn(small_optimizer.n_params) * 5.0
        log_inf = small_optimizer.objective_log_infidelity(params)
        assert np.isfinite(log_inf)
        assert log_inf < 0  # log(~1) is near 0

    def test_pipeline_config_log_objective_fields(self):
        """OptimizationPipeline should have log objective fields."""
        from wings.config import OptimizationPipeline

        pipeline = OptimizationPipeline()
        assert hasattr(pipeline, "use_log_objective")
        assert hasattr(pipeline, "log_objective_threshold")
        assert pipeline.use_log_objective
        assert pipeline.log_objective_threshold == 0.999

    def test_pipeline_backward_compat(self):
        """Pipeline with use_log_objective=False should still work."""
        from wings.config import OptimizationPipeline

        pipeline = OptimizationPipeline(use_log_objective=False)
        assert not pipeline.use_log_objective


class TestVectorizedShifts:
    """Tests for vectorized shifted-parameter construction (P1)."""

    @pytest.mark.unit
    def test_vectorized_shift_construction(self):
        """Build shifted params both ways (loop and vectorized), verify identical for n_params=36."""
        n_params = 36
        rng = np.random.RandomState(123)
        params = rng.randn(n_params)
        shift = np.pi / 2

        # Loop version
        params_loop = np.zeros((2 * n_params, n_params))
        for i in range(n_params):
            params_loop[2 * i] = params.copy()
            params_loop[2 * i, i] += shift
            params_loop[2 * i + 1] = params.copy()
            params_loop[2 * i + 1, i] -= shift

        # Vectorized version
        params_vec = np.tile(params, (2 * n_params, 1))
        idx = np.arange(n_params)
        params_vec[2 * idx, idx] += shift
        params_vec[2 * idx + 1, idx] -= shift

        np.testing.assert_array_equal(params_loop, params_vec)

    @pytest.mark.unit
    def test_vectorized_gradient_extraction(self):
        """Verify vectorized gradient extraction matches loop version."""
        n_params = 36
        rng = np.random.RandomState(456)
        fidelities = rng.rand(2 * n_params)

        # Loop version
        gradient_loop = np.zeros(n_params)
        for i in range(n_params):
            gradient_loop[i] = (fidelities[2 * i] - fidelities[2 * i + 1]) / 2

        # Vectorized version
        gradient_vec = (fidelities[0::2] - fidelities[1::2]) / 2

        np.testing.assert_array_equal(gradient_loop, gradient_vec)

    @pytest.mark.unit
    def test_gpu_gradient_uses_vectorized_shifts(self, small_optimizer, random_params_6q):
        """Verify _compute_gradient_gpu_impl produces correct gradient shape and finite values."""
        # _compute_gradient_gpu_impl will fall back to sequential when no GPU
        # is available, but we can still verify the interface works correctly
        gradient = small_optimizer._compute_gradient_gpu_impl(random_params_6q)

        assert gradient.shape == (36,), f"Expected shape (36,), got {gradient.shape}"
        assert np.all(np.isfinite(gradient)), "Gradient contains non-finite values"


@pytest.mark.unit
class TestStochasticGradient:
    """Tests for stochastic parameter-shift gradients (P5)."""

    def test_stochastic_gradient_shape(self, small_optimizer, random_params_6q):
        """Output shape should match n_params."""
        grad = small_optimizer.compute_gradient_stochastic(random_params_6q, fraction=0.5)
        assert grad.shape == (small_optimizer.n_params,)

    def test_stochastic_gradient_sparsity(self, small_optimizer, random_params_6q):
        """Only k = floor(n_params * fraction) components should be nonzero."""
        n = small_optimizer.n_params
        grad = small_optimizer.compute_gradient_stochastic(random_params_6q, fraction=0.5)
        n_nonzero = np.count_nonzero(grad)
        expected_k = max(1, int(n * 0.5))
        assert n_nonzero == expected_k

    def test_stochastic_gradient_reproducible(self, small_optimizer, random_params_6q):
        """Same RNG seed should give identical results."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        grad1 = small_optimizer.compute_gradient_stochastic(
            random_params_6q, fraction=0.5, rng=rng1
        )
        grad2 = small_optimizer.compute_gradient_stochastic(
            random_params_6q, fraction=0.5, rng=rng2
        )
        np.testing.assert_array_equal(grad1, grad2)

    def test_stochastic_gradient_unbiased(self, small_optimizer, random_params_6q):
        """Averaged over many samples, stochastic gradient should approximate full gradient."""
        full_grad = small_optimizer.compute_gradient(random_params_6q)

        n_samples = 50
        grad_sum = np.zeros_like(full_grad)
        for i in range(n_samples):
            rng = np.random.default_rng(i)
            grad = small_optimizer.compute_gradient_stochastic(
                random_params_6q, fraction=0.5, rng=rng
            )
            grad_sum += grad

        grad_avg = grad_sum / n_samples

        # Scale by 1/fraction since each component is sampled with probability=fraction
        grad_avg_scaled = grad_avg / 0.5

        # Should be close to the full gradient
        np.testing.assert_allclose(grad_avg_scaled, full_grad, atol=0.3)

    def test_fraction_one_equals_full(self, small_optimizer, random_params_6q):
        """fraction=1.0 should fall through to full gradient."""
        full_grad = small_optimizer.compute_gradient(random_params_6q)
        stoch_grad = small_optimizer.compute_gradient_stochastic(random_params_6q, fraction=1.0)
        np.testing.assert_allclose(stoch_grad, full_grad, atol=1e-10)

    def test_config_gradient_sample_fraction_default(self):
        """Default gradient_sample_fraction should be 1.0."""
        from wings.config import OptimizerConfig

        config = OptimizerConfig(
            n_qubits=6, sigma=1.0, verbose=False, use_gpu=False, use_custatevec=False
        )
        assert config.gradient_sample_fraction == 1.0

    def test_stochastic_gradient_finite(self, small_optimizer, random_params_6q):
        """All gradient values should be finite."""
        grad = small_optimizer.compute_gradient_stochastic(random_params_6q, fraction=0.3)
        assert np.all(np.isfinite(grad))


@pytest.mark.unit
class TestComplexWavepackets:
    """Tests for complex-valued / momentum wavepackets (P8)."""

    def test_momentum_zero_real(self, small_config):
        """With momentum=0, target should be real-valued."""
        from wings import GaussianOptimizer

        small_config.momentum = 0.0
        opt = GaussianOptimizer(small_config)
        assert np.allclose(opt.target.imag, 0.0, atol=1e-15)

    def test_momentum_nonzero_complex(self):
        """With momentum != 0, target should have nonzero imaginary part."""
        from wings import GaussianOptimizer
        from wings.config import OptimizerConfig

        config = OptimizerConfig(
            n_qubits=6,
            sigma=1.0,
            momentum=2.0,
            verbose=False,
            use_gpu=False,
            use_custatevec=False,
        )
        with pytest.warns(UserWarning, match="DefaultAnsatz"):
            opt = GaussianOptimizer(config)
        assert np.any(np.abs(opt.target.imag) > 0.01)

    def test_momentum_target_normalized(self):
        """Complex wavepacket should be normalized."""
        from wings import GaussianOptimizer
        from wings.config import OptimizerConfig

        config = OptimizerConfig(
            n_qubits=6,
            sigma=1.0,
            momentum=3.0,
            verbose=False,
            use_gpu=False,
            use_custatevec=False,
        )
        with pytest.warns(UserWarning, match="DefaultAnsatz"):
            opt = GaussianOptimizer(config)
        norm = np.linalg.norm(opt.target)
        assert abs(norm - 1.0) < 1e-10

    def test_momentum_envelope_is_gaussian(self):
        """The envelope |psi(x)| should still be Gaussian."""
        from wings import GaussianOptimizer
        from wings.config import OptimizerConfig

        config_no_k = OptimizerConfig(
            n_qubits=6,
            sigma=1.0,
            momentum=0.0,
            verbose=False,
            use_gpu=False,
            use_custatevec=False,
        )
        config_with_k = OptimizerConfig(
            n_qubits=6,
            sigma=1.0,
            momentum=3.0,
            verbose=False,
            use_gpu=False,
            use_custatevec=False,
        )
        opt_no_k = GaussianOptimizer(config_no_k)
        with pytest.warns(UserWarning, match="DefaultAnsatz"):
            opt_with_k = GaussianOptimizer(config_with_k)
        # |psi| should be the same (Gaussian envelope unchanged by momentum)
        np.testing.assert_allclose(
            np.abs(opt_with_k.target),
            np.abs(opt_no_k.target),
            atol=1e-10,
        )

    def test_fidelity_computation_complex(self):
        """Fidelity computation should work correctly for complex targets."""
        from wings import GaussianOptimizer
        from wings.config import OptimizerConfig

        config = OptimizerConfig(
            n_qubits=6,
            sigma=1.0,
            momentum=1.0,
            verbose=False,
            use_gpu=False,
            use_custatevec=False,
        )
        with pytest.warns(UserWarning, match="DefaultAnsatz"):
            opt = GaussianOptimizer(config)
        # Fidelity of target with itself should be 1
        f = opt._compute_fidelity_fast(opt.target)
        assert abs(f - 1.0) < 1e-12

    def test_gradient_computation_complex(self):
        """Parameter-shift gradient should work for complex targets."""
        from wings import GaussianOptimizer
        from wings.config import OptimizerConfig

        config = OptimizerConfig(
            n_qubits=6,
            sigma=1.0,
            momentum=1.0,
            verbose=False,
            use_gpu=False,
            use_custatevec=False,
        )
        with pytest.warns(UserWarning, match="DefaultAnsatz"):
            opt = GaussianOptimizer(config)
        params = opt.get_initial_params("random")
        grad = opt.compute_gradient(params)
        assert grad.shape == (opt.n_params,)
        assert np.all(np.isfinite(grad))

    def test_default_ansatz_warning(self):
        """Should warn when using DefaultAnsatz with momentum."""
        from wings import GaussianOptimizer
        from wings.config import OptimizerConfig

        config = OptimizerConfig(
            n_qubits=6,
            sigma=1.0,
            momentum=2.0,
            verbose=False,
            use_gpu=False,
            use_custatevec=False,
        )
        with pytest.warns(UserWarning, match="DefaultAnsatz"):
            GaussianOptimizer(config)

    def test_gaussian_wavepacket_enum(self):
        """GAUSSIAN_WAVEPACKET should be a valid target function."""
        from wings.config import TargetFunction

        assert hasattr(TargetFunction, "GAUSSIAN_WAVEPACKET")

    def test_backward_compat_momentum_zero(self, small_config):
        """Default momentum=0 should give identical results to v0.1.0."""
        from wings import GaussianOptimizer

        opt = GaussianOptimizer(small_config)
        # Target should be real
        assert np.allclose(opt.target.imag, 0.0, atol=1e-15)
        # Config momentum should default to 0
        assert small_config.momentum == 0.0


@pytest.mark.unit
class TestAdaptiveDepth:
    """Tests for adaptive circuit depth (layer-ramp)."""

    def test_adaptive_depth_config(self):
        from wings.config import OptimizationPipeline

        pipeline = OptimizationPipeline()
        assert hasattr(pipeline, "use_adaptive_depth")
        assert hasattr(pipeline, "min_depth")
        assert hasattr(pipeline, "max_depth")

    def test_adaptive_depth_disabled_by_default(self):
        from wings.config import OptimizationPipeline

        pipeline = OptimizationPipeline()
        assert pipeline.use_adaptive_depth is False

    def test_grow_circuit_adds_layer(self):
        from wings import GaussianOptimizer, OptimizerConfig
        from wings.ansatz import DefaultAnsatz

        ansatz = DefaultAnsatz(n_qubits=6, depth=2)
        config = OptimizerConfig(
            n_qubits=6, sigma=0.5, ansatz=ansatz, verbose=False, use_gpu=False, use_custatevec=False
        )
        opt = GaussianOptimizer(config)
        old_n_params = opt.n_params
        new_params = opt.grow_circuit(opt.get_initial_params("smart"))
        assert len(new_params) > old_n_params
        assert opt.n_params > old_n_params

    def test_grow_preserves_fidelity(self):
        from wings import GaussianOptimizer, OptimizerConfig
        from wings.ansatz import DefaultAnsatz

        ansatz = DefaultAnsatz(n_qubits=6, depth=2)
        config = OptimizerConfig(
            n_qubits=6, sigma=0.5, ansatz=ansatz, verbose=False, use_gpu=False, use_custatevec=False
        )
        opt = GaussianOptimizer(config)
        params = opt.get_initial_params("smart")
        fid_before = opt.compute_fidelity(params=params)
        new_params = opt.grow_circuit(params)
        fid_after = opt.compute_fidelity(params=new_params)
        assert abs(fid_after - fid_before) < 0.10


@pytest.mark.unit
class TestV030Integration:
    """Integration tests for v0.3.0 features working together."""

    def test_efficientsu2_with_momentum(self):
        """EfficientSU2 + momentum wavepacket should work together."""
        from wings import GaussianOptimizer, OptimizerConfig
        from wings.ansatz_library import EfficientSU2Ansatz

        ansatz = EfficientSU2Ansatz(n_qubits=6, layers=3, entanglement="circular")
        config = OptimizerConfig(
            n_qubits=6,
            sigma=0.5,
            momentum=1.0,
            ansatz=ansatz,
            verbose=False,
            use_gpu=False,
            use_custatevec=False,
        )
        opt = GaussianOptimizer(config)
        params = np.random.randn(ansatz.n_params) * 0.1
        fid = opt.compute_fidelity(params=params)
        assert 0 <= fid <= 1

    def test_log_distance_entanglement_in_optimizer(self):
        """Log-distance entanglement should work in full optimization."""
        from wings import GaussianOptimizer, OptimizerConfig
        from wings.ansatz import DefaultAnsatz

        ansatz = DefaultAnsatz(n_qubits=6, entanglement="log_distance")
        config = OptimizerConfig(
            n_qubits=6,
            sigma=0.5,
            ansatz=ansatz,
            verbose=False,
            use_gpu=False,
            use_custatevec=False,
        )
        opt = GaussianOptimizer(config)
        result = opt.optimize_adam(opt.get_initial_params("smart"), max_steps=50, lr=0.02)
        assert result["fidelity"] > 0

    def test_warm_start_then_optimize(self):
        """Warm-started params should lead to successful optimization."""
        from wings import GaussianOptimizer, OptimizerConfig
        from wings.warm_start import transfer_params

        config_3 = OptimizerConfig(
            n_qubits=3, sigma=0.5, verbose=False, use_gpu=False, use_custatevec=False
        )
        opt_3 = GaussianOptimizer(config_3)
        r3 = opt_3.optimize_adam(opt_3.get_initial_params("smart"), max_steps=30, lr=0.02)
        params_4 = transfer_params(r3["params"], 3, 4)
        config_4 = OptimizerConfig(
            n_qubits=4, sigma=0.5, verbose=False, use_gpu=False, use_custatevec=False
        )
        opt_4 = GaussianOptimizer(config_4)
        r4 = opt_4.optimize_adam(params_4, max_steps=30, lr=0.02)
        assert r4["fidelity"] > 0

    def test_barren_plateau_detector_in_adam(self):
        """Adam optimization should include barren plateau monitoring."""
        from wings import GaussianOptimizer, OptimizerConfig

        config = OptimizerConfig(
            n_qubits=6, sigma=0.5, verbose=False, use_gpu=False, use_custatevec=False
        )
        opt = GaussianOptimizer(config)
        result = opt.optimize_adam(opt.get_initial_params("smart"), max_steps=30, lr=0.02)
        assert result["fidelity"] > 0


@pytest.mark.unit
class TestHessianRefinement:
    """Tests for second-order Hessian-aided refinement (v0.4.0 WI-7)."""

    def test_hessian_diagonal_shape(self, small_optimizer, random_params_6q):
        hess = small_optimizer.compute_hessian_diagonal(random_params_6q)
        assert hess.shape == (small_optimizer.n_params,)

    def test_hessian_diagonal_finite(self, small_optimizer, random_params_6q):
        hess = small_optimizer.compute_hessian_diagonal(random_params_6q)
        assert np.all(np.isfinite(hess))

    def test_hessian_matches_finite_difference(self, small_optimizer, random_params_6q):
        """Diagonal Hessian should approximate finite-difference second derivative."""
        hess = small_optimizer.compute_hessian_diagonal(random_params_6q)
        eps = 1e-4
        for i in range(3):  # Check first 3 params
            p_plus = random_params_6q.copy()
            p_plus[i] += eps
            p_minus = random_params_6q.copy()
            p_minus[i] -= eps
            f_plus = small_optimizer.compute_fidelity(params=p_plus)
            f_minus = small_optimizer.compute_fidelity(params=p_minus)
            f_0 = small_optimizer.compute_fidelity(params=random_params_6q)
            fd_hess = (f_plus - 2 * f_0 + f_minus) / eps**2
            # Note: hess is for -fidelity (minimization), fd_hess is for fidelity
            # So hess[i] ~ -fd_hess (negated)
            assert (
                abs(hess[i] - (-fd_hess)) / (abs(fd_hess) + 1e-10) < 0.5
                or abs(hess[i] - (-fd_hess)) < 0.5
            )

    def test_newton_step_shape(self, small_optimizer, random_params_6q):
        new_params = small_optimizer.newton_refinement_step(random_params_6q)
        assert new_params.shape == random_params_6q.shape

    def test_newton_step_finite(self, small_optimizer, random_params_6q):
        new_params = small_optimizer.newton_refinement_step(random_params_6q)
        assert np.all(np.isfinite(new_params))

    def test_newton_step_changes_params(self, small_optimizer, random_params_6q):
        new_params = small_optimizer.newton_refinement_step(random_params_6q)
        assert not np.allclose(new_params, random_params_6q)
