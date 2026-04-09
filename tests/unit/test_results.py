"""Unit tests for results module."""

import numpy as np
import pytest


@pytest.mark.unit
class TestRunResult:
    def test_run_result_creation(self):
        from wings.results import RunResult

        r = RunResult(
            run_id=0,
            fidelity=0.99,
            infidelity=0.01,
            params=np.zeros(10),
            circuit_std=0.1,
            circuit_mean=0.5,
            time_seconds=1.0,
            n_evaluations=100,
            strategy="smart",
            seed=42,
            success=True,
        )
        assert r.fidelity == 0.99
        assert r.run_id == 0
        assert r.success is True

    def test_run_result_failed(self):
        from wings.results import RunResult

        r = RunResult(
            run_id=1,
            fidelity=0.0,
            infidelity=1.0,
            params=None,
            circuit_std=0.0,
            circuit_mean=0.0,
            time_seconds=0.1,
            n_evaluations=0,
            strategy="random",
            seed=43,
            success=False,
            error_message="test error",
        )
        assert r.success is False
        assert r.error_message == "test error"


@pytest.mark.unit
class TestCampaignResults:
    def test_campaign_results_init(self):
        from wings.config import CampaignConfig
        from wings.results import CampaignResults

        cfg = CampaignConfig(n_qubits=8, sigma=0.5, campaign_name="test")
        cr = CampaignResults(config=cfg)
        assert len(cr.results) == 0

    def test_add_result(self):
        from wings.config import CampaignConfig
        from wings.results import CampaignResults, RunResult

        cfg = CampaignConfig(n_qubits=8, sigma=0.5, campaign_name="test")
        cr = CampaignResults(config=cfg)
        r = RunResult(
            run_id=0,
            fidelity=0.99,
            infidelity=0.01,
            params=np.zeros(10),
            circuit_std=0.1,
            circuit_mean=0.5,
            time_seconds=1.0,
            n_evaluations=100,
            strategy="smart",
            seed=42,
            success=True,
        )
        cr.add_result(r)
        assert len(cr.results) == 1


@pytest.mark.unit
class TestOptimizationResult:
    def test_creation(self):
        from wings.results import OptimizationResult

        r = OptimizationResult(
            optimal_params=np.zeros(10),
            fidelity=0.999,
            infidelity=0.001,
            time=5.0,
            n_evaluations=1000,
            success=True,
        )
        assert r.fidelity == 0.999

    def test_dict_access(self):
        from wings.results import OptimizationResult

        r = OptimizationResult(
            optimal_params=np.zeros(10),
            fidelity=0.999,
            infidelity=0.001,
            time=5.0,
            n_evaluations=1000,
            success=True,
        )
        assert r["fidelity"] == 0.999
        assert "fidelity" in r

    def test_to_dict(self):
        from wings.results import OptimizationResult

        r = OptimizationResult(
            optimal_params=np.zeros(10),
            fidelity=0.999,
            infidelity=0.001,
            time=5.0,
            n_evaluations=1000,
            success=True,
        )
        d = r.to_dict()
        assert isinstance(d, dict)
        assert d["fidelity"] == 0.999
