"""Integration tests for wings.

These tests verify that multiple components work together correctly.
"""

import os

import numpy as np
import pytest


@pytest.mark.integration
class TestEndToEndOptimization:
    """End-to-end optimization tests."""

    def test_simple_optimization_cpu(self, small_config):
        """Test complete optimization on CPU."""
        from wings import GaussianOptimizer
        from wings.config import OptimizationPipeline

        opt = GaussianOptimizer(small_config)

        pipeline = OptimizationPipeline(
            target_fidelity=0.95,
            max_total_time=60,
            use_adam_stage=True,
            adam_max_steps=500,
            use_lbfgs_refinement=True,
            verbose=False,
        )

        results = opt.run_optimization(pipeline)

        assert "fidelity" in results
        assert "optimal_params" in results
        assert "time" in results
        assert results["fidelity"] > 0.8  # Should achieve decent fidelity
        assert results["optimal_params"].shape == (36,)

    @pytest.mark.slow
    def test_high_fidelity_optimization(self, medium_config):
        """Test optimization targeting high fidelity."""
        from wings import GaussianOptimizer
        from wings.config import OptimizationPipeline

        opt = GaussianOptimizer(medium_config)

        pipeline = OptimizationPipeline(
            target_fidelity=0.999,
            max_total_time=120,
            use_adam_stage=True,
            adam_max_steps=2000,
            use_lbfgs_refinement=True,
            verbose=False,
        )

        results = opt.run_optimization(pipeline)

        assert results["fidelity"] > 0.99

    def test_lorentzian_optimization(self, lorentzian_config):
        """Test optimization with Lorentzian target."""
        from wings import GaussianOptimizer
        from wings.config import OptimizationPipeline

        opt = GaussianOptimizer(lorentzian_config)

        pipeline = OptimizationPipeline(
            target_fidelity=0.9,
            max_total_time=60,
            verbose=False,
        )

        results = opt.run_optimization(pipeline)

        assert results["fidelity"] > 0.7

    def test_shifted_gaussian_optimization(self, shifted_config):
        """Test optimization with shifted wavefunction."""
        from wings import GaussianOptimizer
        from wings.config import OptimizationPipeline

        opt = GaussianOptimizer(shifted_config)

        pipeline = OptimizationPipeline(
            target_fidelity=0.9,
            max_total_time=60,
            verbose=False,
        )

        results = opt.run_optimization(pipeline)

        assert results["fidelity"] > 0.7
        # Circuit mean should be close to x0
        assert abs(results["circuit_mean"] - 1.5) < 1.0


@pytest.mark.integration
@pytest.mark.gpu
class TestGPUIntegration:
    """Integration tests with GPU backends."""

    def test_gpu_optimization(self, gpu_config):
        """Test optimization with GPU backend."""
        from wings import GaussianOptimizer
        from wings.config import OptimizationPipeline

        opt = GaussianOptimizer(gpu_config)

        if opt._gpu_evaluator and opt._gpu_evaluator.gpu_available:
            pipeline = OptimizationPipeline(
                target_fidelity=0.95,
                max_total_time=60,
                verbose=False,
            )

            results = opt.run_optimization(pipeline)

            assert results["fidelity"] > 0.8
        else:
            pytest.skip("GPU not available")


@pytest.mark.integration
@pytest.mark.custatevec
class TestCuStateVecIntegration:
    """Integration tests with cuStateVec backend."""

    def test_custatevec_optimization(self, custatevec_config, gpu_cleanup):
        """Test optimization with cuStateVec backend."""
        from wings import GaussianOptimizer
        from wings.config import OptimizationPipeline

        opt = GaussianOptimizer(custatevec_config)

        if opt._custatevec_evaluator is not None:
            pipeline = OptimizationPipeline(
                target_fidelity=0.95,
                max_total_time=60,
                verbose=False,
            )

            results = opt.run_optimization(pipeline)

            assert results["fidelity"] > 0.8

            opt.cleanup()
        else:
            pytest.skip("cuStateVec not available")

    def test_custatevec_matches_cpu(self, gpu_cleanup):
        """Test cuStateVec gives same results as CPU."""
        from wings import GaussianOptimizer, OptimizerConfig

        np.random.seed(42)
        params = np.random.randn(36) * 0.1

        # CPU config
        cpu_config = OptimizerConfig(
            n_qubits=6,
            sigma=0.5,
            verbose=False,
            use_gpu=False,
            use_custatevec=False,
        )

        # cuStateVec config
        cusv_config = OptimizerConfig(
            n_qubits=6,
            sigma=0.5,
            verbose=False,
            use_gpu=False,
            use_custatevec=True,
        )

        cpu_opt = GaussianOptimizer(cpu_config)
        cusv_opt = GaussianOptimizer(cusv_config)

        if cusv_opt._custatevec_evaluator is not None:
            fid_cpu = cpu_opt.compute_fidelity(params=params)
            fid_cusv = cusv_opt.compute_fidelity(params=params)

            assert np.isclose(fid_cpu, fid_cusv, rtol=1e-5)

            cusv_opt.cleanup()
        else:
            pytest.skip("cuStateVec not available")


@pytest.mark.integration
class TestConvenienceFunctions:
    """Tests for high-level convenience functions."""

    @pytest.mark.slow
    def test_optimize_gaussian_state(self):
        """Test optimize_gaussian_state convenience function."""
        from wings import optimize_gaussian_state

        results, opt = optimize_gaussian_state(
            n_qubits=6,
            sigma=0.5,
            target_fidelity=0.95,
            max_time=60,
            use_gpu=False,
            use_custatevec=False,
            plot=False,
            save=False,
            verbose=False,
        )

        assert results["fidelity"] > 0.8
        assert "optimal_params" in results

    def test_quick_optimize(self):
        """Test quick_optimize convenience function."""
        from wings import quick_optimize

        fidelity, results = quick_optimize(
            n_qubits=6,
            sigma=0.5,
            verbose=False,
        )

        assert fidelity > 0.5
        assert isinstance(results, dict)


@pytest.mark.integration
class TestCampaignIntegration:
    """Tests for campaign management."""

    def test_small_campaign(self, temp_output_dir, temp_checkpoint_dir):
        """Test running a small campaign."""
        from wings.campaign import OptimizationManager
        from wings.config import CampaignConfig

        config = CampaignConfig(
            n_qubits=6,
            sigma=0.5,
            total_runs=5,
            runs_per_batch=5,
            max_iter_per_run=100,
            use_ultra_precision=False,
            verbose=0,
            output_dir=str(temp_output_dir),
            checkpoint_dir=str(temp_checkpoint_dir),
        )

        manager = OptimizationManager(config)
        results = manager.run_campaign()

        assert results.best_result is not None
        assert results.best_result.fidelity > 0
        assert len(results.all_results) == 5

    def test_campaign_checkpointing(self, temp_output_dir, temp_checkpoint_dir):
        """Test campaign checkpointing and resume."""
        from wings.campaign import OptimizationManager
        from wings.config import CampaignConfig

        config = CampaignConfig(
            n_qubits=6,
            sigma=0.5,
            total_runs=10,
            runs_per_batch=5,
            checkpoint_interval=5,
            max_iter_per_run=50,
            use_ultra_precision=False,
            verbose=0,
            output_dir=str(temp_output_dir),
            checkpoint_dir=str(temp_checkpoint_dir),
        )

        # Run first batch
        manager = OptimizationManager(config)

        # Simulate partial run
        for run_id in range(5):
            result = manager._run_single_optimization(run_id)
            manager.results.add_result(result)
            manager._completed_runs.add(run_id)

        manager._save_checkpoint()

        # Create new manager with resume
        config2 = CampaignConfig(
            n_qubits=6,
            sigma=0.5,
            total_runs=10,
            runs_per_batch=5,
            checkpoint_interval=5,
            max_iter_per_run=50,
            use_ultra_precision=False,
            verbose=0,
            output_dir=str(temp_output_dir),
            checkpoint_dir=str(temp_checkpoint_dir),
            resume_from_checkpoint=True,
        )

        manager2 = OptimizationManager(config2)

        # Should have resumed with 5 completed runs
        assert len(manager2._completed_runs) == 5


@pytest.mark.integration
class TestResultsSaving:
    """Tests for saving and loading results."""

    def test_save_and_load_results(self, small_config, temp_output_dir):
        """Test saving and loading optimization results."""
        from wings import GaussianOptimizer
        from wings.config import OptimizationPipeline

        opt = GaussianOptimizer(small_config)

        pipeline = OptimizationPipeline(
            target_fidelity=0.9,
            max_total_time=30,
            verbose=False,
        )

        results = opt.run_optimization(pipeline)

        # Save results
        filepath = str(temp_output_dir / "test_results.txt")
        opt.save_results(results, filepath)

        assert os.path.exists(filepath)

        # Check associated files
        params_file = filepath.replace(".txt", "_params.npy")
        assert os.path.exists(params_file)

        # Load params
        loaded_params = np.load(params_file)
        np.testing.assert_array_almost_equal(loaded_params, results["optimal_params"])


@pytest.mark.integration
class TestCustomAnsatzIntegration:
    """Tests for custom ansatz integration."""

    def test_custom_ansatz_optimization(self):
        """Test optimization with custom ansatz."""
        from wings import GaussianOptimizer, OptimizerConfig
        from wings.ansatz import CustomHardwareEfficientAnsatz
        from wings.config import OptimizationPipeline

        ansatz = CustomHardwareEfficientAnsatz(
            n_qubits=6,
            layers=4,
            entanglement="circular",
        )

        config = OptimizerConfig(
            n_qubits=6,
            sigma=0.5,
            ansatz=ansatz,
            verbose=False,
            use_gpu=False,
            use_custatevec=False,
        )

        opt = GaussianOptimizer(config)

        pipeline = OptimizationPipeline(
            target_fidelity=0.9,
            max_total_time=60,
            verbose=False,
        )

        results = opt.run_optimization(pipeline)

        assert results["fidelity"] > 0.5


@pytest.mark.integration
class TestTargetFunctionIntegration:
    """Tests for different target functions."""

    @pytest.mark.parametrize(
        "target_fn,expected_min_fid",
        [
            ("gaussian", 0.7),
            ("lorentzian", 0.6),
            ("sech", 0.6),
        ],
    )
    def test_target_functions(self, target_fn, expected_min_fid):
        """Test optimization with different target functions."""
        from wings import GaussianOptimizer, OptimizerConfig, TargetFunction
        from wings.config import OptimizationPipeline

        tf_map = {
            "gaussian": TargetFunction.GAUSSIAN,
            "lorentzian": TargetFunction.LORENTZIAN,
            "sech": TargetFunction.SECH,
        }

        config = OptimizerConfig(
            n_qubits=6,
            sigma=0.5,
            target_function=tf_map[target_fn],
            gamma=0.3 if target_fn == "lorentzian" else None,
            verbose=False,
            use_gpu=False,
            use_custatevec=False,
        )

        opt = GaussianOptimizer(config)

        pipeline = OptimizationPipeline(
            target_fidelity=0.9,
            max_total_time=60,
            verbose=False,
        )

        results = opt.run_optimization(pipeline)

        assert results["fidelity"] > expected_min_fid
