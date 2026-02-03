"""Unit tests for campaign module."""

import os

import numpy as np
import pytest


class TestCampaignResults:
    """Tests for CampaignResults class."""

    def test_initialization(self):
        """Test CampaignResults initialization."""
        from wings.config import CampaignConfig
        from wings.results import CampaignResults

        config = CampaignConfig(n_qubits=6, sigma=0.5, verbose=0)
        results = CampaignResults(config)

        assert results.config is config
        assert results.best_result is None
        assert len(results.results) == 0

    def test_add_result(self):
        """Test adding results."""
        from wings.config import CampaignConfig
        from wings.results import CampaignResults, RunResult

        config = CampaignConfig(n_qubits=6, sigma=0.5, verbose=0)
        results = CampaignResults(config)

        run_result = RunResult(
            run_id=0,
            strategy="smart",
            seed=42,
            fidelity=0.95,
            infidelity=0.05,
            params=np.zeros(36),
            circuit_std=0.5,
            circuit_mean=0.0,
            n_evaluations=100,
            time_seconds=10.0,
            success=True,
        )

        results.add_result(run_result)

        assert len(results.results) == 1
        assert results.best_result is not None
        assert results.best_result.fidelity == 0.95

    def test_best_result_updated(self):
        """Test best result is updated correctly."""
        from wings.config import CampaignConfig
        from wings.results import CampaignResults, RunResult

        config = CampaignConfig(n_qubits=6, sigma=0.5, verbose=0)
        results = CampaignResults(config)

        # Add first result
        results.add_result(
            RunResult(
                run_id=0,
                strategy="smart",
                seed=42,
                fidelity=0.9,
                infidelity=0.1,
                params=np.zeros(36),
                circuit_std=0.5,
                circuit_mean=0.0,
                n_evaluations=100,
                time_seconds=10.0,
                success=True,
            )
        )

        assert results.best_result.fidelity == 0.9

        # Add better result
        results.add_result(
            RunResult(
                run_id=1,
                strategy="random",
                seed=43,
                fidelity=0.95,
                infidelity=0.05,
                params=np.ones(36),
                circuit_std=0.5,
                circuit_mean=0.0,
                n_evaluations=100,
                time_seconds=10.0,
                success=True,
            )
        )

        assert results.best_result.fidelity == 0.95
        assert results.best_result.run_id == 1

        # Add worse result
        results.add_result(
            RunResult(
                run_id=2,
                strategy="random",
                seed=44,
                fidelity=0.85,
                infidelity=0.15,
                params=np.zeros(36),
                circuit_std=0.5,
                circuit_mean=0.0,
                n_evaluations=100,
                time_seconds=10.0,
                success=True,
            )
        )

        # Best should still be run 1
        assert results.best_result.fidelity == 0.95
        assert results.best_result.run_id == 1

    def test_get_statistics(self):
        """Test getting statistics."""
        from wings.config import CampaignConfig
        from wings.results import CampaignResults, RunResult

        config = CampaignConfig(n_qubits=6, sigma=0.5, verbose=0)
        results = CampaignResults(config)

        for i in range(5):
            results.add_result(
                RunResult(
                    run_id=i,
                    strategy="random",
                    seed=42 + i,
                    fidelity=0.8 + 0.04 * i,
                    infidelity=0.2 - 0.04 * i,
                    params=np.zeros(36),
                    circuit_std=0.5,
                    circuit_mean=0.0,
                    n_evaluations=100,
                    time_seconds=10.0,
                    success=True,
                )
            )

        stats = results.get_statistics()

        assert "best_fidelity" in stats
        assert "mean_fidelity" in stats
        assert "total_runs" in stats
        assert stats["total_runs"] == 5
        assert stats["best_fidelity"] == pytest.approx(0.96)

    def test_save_and_load(self, temp_output_dir):
        """Test saving and loading results."""
        from wings.config import CampaignConfig
        from wings.results import CampaignResults, RunResult

        config = CampaignConfig(
            n_qubits=6,
            sigma=0.5,
            verbose=0,
            output_dir=str(temp_output_dir),
            checkpoint_dir=str(temp_output_dir),
        )
        results = CampaignResults(config)

        results.add_result(
            RunResult(
                run_id=0,
                strategy="smart",
                seed=42,
                fidelity=0.95,
                infidelity=0.05,
                params=np.zeros(36),
                circuit_std=0.5,
                circuit_mean=0.0,
                n_evaluations=100,
                time_seconds=10.0,
                success=True,
            )
        )

        filepath = results.save()
        assert os.path.exists(filepath)

        loaded = CampaignResults.load(filepath)
        assert loaded.best_result.fidelity == 0.95


class TestRunResult:
    """Tests for RunResult class."""

    def test_initialization(self):
        """Test RunResult initialization."""
        from wings.results import RunResult

        result = RunResult(
            run_id=5,
            strategy="smart",
            seed=12345,
            fidelity=0.999,
            infidelity=0.001,
            params=np.array([1.0, 2.0, 3.0]),
            circuit_std=0.5,
            circuit_mean=0.0,
            n_evaluations=1000,
            time_seconds=30.5,
            success=True,
        )

        assert result.run_id == 5
        assert result.fidelity == 0.999
        assert result.strategy == "smart"
        assert result.success

    def test_failed_result(self):
        """Test failed run result."""
        from wings.results import RunResult

        result = RunResult(
            run_id=0,
            strategy="random",
            seed=42,
            fidelity=0.0,
            infidelity=1.0,
            params=None,
            circuit_std=0.0,
            circuit_mean=0.0,
            n_evaluations=0,
            time_seconds=0.0,
            success=False,
            error_message="Test error",
        )

        assert not result.success
        assert result.error_message == "Test error"


class TestOptimizationManager:
    """Tests for OptimizationManager class."""

    def test_initialization(self, temp_output_dir, temp_checkpoint_dir):
        """Test manager initialization."""
        from wings.campaign import OptimizationManager
        from wings.config import CampaignConfig

        config = CampaignConfig(
            n_qubits=6,
            sigma=0.5,
            total_runs=10,
            verbose=0,
            output_dir=str(temp_output_dir),
            checkpoint_dir=str(temp_checkpoint_dir),
        )

        manager = OptimizationManager(config)

        assert manager.config is config
        assert manager.results is not None
        assert len(manager._completed_runs) == 0

    def test_run_single_optimization(self, temp_output_dir, temp_checkpoint_dir):
        """Test running a single optimization."""
        from wings.campaign import OptimizationManager
        from wings.config import CampaignConfig

        config = CampaignConfig(
            n_qubits=6,
            sigma=0.5,
            total_runs=1,
            max_iter_per_run=50,
            use_ultra_precision=False,
            verbose=0,
            output_dir=str(temp_output_dir),
            checkpoint_dir=str(temp_checkpoint_dir),
        )

        manager = OptimizationManager(config)
        result = manager._run_single_optimization(0)

        assert result.run_id == 0
        assert result.success
        assert result.fidelity > 0

    def test_create_optimizer_config(self, temp_output_dir, temp_checkpoint_dir):
        """Test optimizer config creation."""
        from wings.campaign import OptimizationManager
        from wings.config import CampaignConfig

        config = CampaignConfig(
            n_qubits=8,
            sigma=0.7,
            x0=1.0,
            verbose=0,
            output_dir=str(temp_output_dir),
            checkpoint_dir=str(temp_checkpoint_dir),
        )

        manager = OptimizationManager(config)
        opt_config = manager._create_optimizer_config()

        assert opt_config.n_qubits == 8
        assert opt_config.sigma == 0.7
        assert opt_config.x0 == 1.0

    def test_checkpoint_save_load(self, temp_output_dir, temp_checkpoint_dir):
        """Test checkpoint saving and loading."""
        from wings.campaign import OptimizationManager
        from wings.config import CampaignConfig

        config = CampaignConfig(
            n_qubits=6,
            sigma=0.5,
            total_runs=5,
            max_iter_per_run=30,
            verbose=0,
            output_dir=str(temp_output_dir),
            checkpoint_dir=str(temp_checkpoint_dir),
        )

        # Run a few optimizations
        manager = OptimizationManager(config)
        for i in range(3):
            result = manager._run_single_optimization(i)
            manager.results.add_result(result)
            manager._completed_runs.add(i)

        manager._save_checkpoint()

        # Create new manager with resume
        config2 = CampaignConfig(
            n_qubits=6,
            sigma=0.5,
            total_runs=5,
            max_iter_per_run=30,
            verbose=0,
            output_dir=str(temp_output_dir),
            checkpoint_dir=str(temp_checkpoint_dir),
            resume_from_checkpoint=True,
        )

        manager2 = OptimizationManager(config2)

        assert len(manager2._completed_runs) == 3


class TestConvenienceFunctions:
    """Tests for convenience module functions."""

    def test_run_production_campaign_signature(self):
        """Test run_production_campaign accepts new parameters."""
        import inspect

        from wings.campaign import run_production_campaign

        sig = inspect.signature(run_production_campaign)
        params = list(sig.parameters.keys())

        assert "n_qubits" in params
        assert "sigma" in params
        assert "total_runs" in params

    def test_quick_optimization_signature(self):
        """Test quick_optimization accepts new parameters."""
        import inspect

        from wings.campaign import quick_optimization

        sig = inspect.signature(quick_optimization)
        params = list(sig.parameters.keys())

        assert "n_qubits" in params
        assert "sigma" in params


class TestListCampaigns:
    """Tests for campaign listing."""

    def test_list_campaigns_empty(self, temp_output_dir, monkeypatch):
        """Test listing campaigns when none exist."""

        # Create empty campaign directory
        campaign_dir = temp_output_dir / "campaigns"
        campaign_dir.mkdir()

        # This test would need path mocking to work properly
        # Skipping actual test for now
        pass

    def test_load_campaign_results_not_found(self):
        """Test loading non-existent campaign raises error."""
        from wings.campaign import load_campaign_results

        with pytest.raises(FileNotFoundError):
            load_campaign_results("nonexistent_campaign_12345")
