"""Unit tests for composable optimization pipeline."""

import numpy as np
import pytest


@pytest.mark.unit
class TestPipelineStages:
    """Tests for individual pipeline stage configuration."""

    def test_init_search_defaults(self):
        from wings.pipeline import InitSearch
        stage = InitSearch()
        assert "smart" in stage.strategies
        assert stage.describe() == "Initialization Search"

    def test_adam_defaults(self):
        from wings.pipeline import Adam
        stage = Adam()
        assert stage.max_steps == 1000
        assert stage.lr == 0.02

    def test_spsa_defaults(self):
        from wings.pipeline import SPSA
        stage = SPSA()
        assert stage.max_steps == 3000
        assert stage.n_avg == 1

    def test_lbfgsb_defaults(self):
        from wings.pipeline import LBFGSB
        stage = LBFGSB()
        assert stage.use_log_objective is True
        assert len(stage.tolerances) == 3

    def test_newton_defaults(self):
        from wings.pipeline import Newton
        stage = Newton()
        assert stage.max_steps == 50

    def test_grow_circuit_stage(self):
        from wings.pipeline import GrowCircuit
        stage = GrowCircuit()
        assert stage.describe() == "Grow Circuit"

    def test_natural_gradient_stage(self):
        from wings.pipeline import NaturalGradient
        stage = NaturalGradient(max_steps=100, lr=0.005)
        assert stage.max_steps == 100
        assert stage.regularization == 0.001


@pytest.mark.unit
class TestPipelinePresets:
    """Tests for pipeline preset constructors."""

    def test_quick_preset(self):
        from wings.pipeline import Pipeline
        p = Pipeline.quick()
        assert len(p.stages) == 2
        assert p.target_fidelity == 0.999

    def test_standard_preset(self):
        from wings.pipeline import Pipeline
        p = Pipeline.standard()
        assert len(p.stages) == 3
        assert p.target_fidelity == 0.999999

    def test_ultra_preset(self):
        from wings.pipeline import Pipeline
        p = Pipeline.ultra(target_infidelity=1e-10)
        assert len(p.stages) == 6
        assert abs(p.target_fidelity - (1 - 1e-10)) < 1e-15

    def test_for_momentum_preset(self):
        from wings.pipeline import Pipeline
        p = Pipeline.for_momentum()
        assert len(p.stages) == 4
        # Should include NaturalGradient stage
        from wings.pipeline import NaturalGradient
        assert any(isinstance(s, NaturalGradient) for s in p.stages)

    def test_layer_ramp_preset(self):
        from wings.pipeline import Pipeline, GrowCircuit
        p = Pipeline.layer_ramp(n_grows=3)
        grow_count = sum(1 for s in p.stages if isinstance(s, GrowCircuit))
        assert grow_count == 3

    def test_exploration_preset(self):
        from wings.pipeline import Pipeline, SPSA
        p = Pipeline.exploration()
        assert any(isinstance(s, SPSA) for s in p.stages)


@pytest.mark.unit
class TestPipelineComposition:
    """Tests for custom pipeline composition."""

    def test_custom_pipeline(self):
        from wings.pipeline import Pipeline, InitSearch, Adam, LBFGSB
        p = Pipeline(
            target_fidelity=0.999,
            stages=[InitSearch(), Adam(max_steps=50), LBFGSB()],
        )
        assert len(p.stages) == 3

    def test_empty_pipeline(self):
        from wings.pipeline import Pipeline
        p = Pipeline(stages=[])
        assert len(p.stages) == 0

    def test_repeated_stages(self):
        from wings.pipeline import Pipeline, Adam
        p = Pipeline(stages=[Adam(max_steps=100, lr=0.05), Adam(max_steps=50, lr=0.01)])
        assert len(p.stages) == 2

    def test_summary_string(self):
        from wings.pipeline import Pipeline
        p = Pipeline.standard()
        s = p.summary()
        assert "Pipeline" in s
        assert "Adam" in s
        assert "L-BFGS-B" in s

    def test_repr(self):
        from wings.pipeline import Pipeline
        p = Pipeline.quick()
        r = repr(p)
        assert "Pipeline" in r
        assert "Adam" in r

    def test_target_infidelity_property(self):
        from wings.pipeline import Pipeline
        p = Pipeline(target_fidelity=0.999)
        assert abs(p.target_infidelity - 0.001) < 1e-15


@pytest.mark.unit
class TestRunPipeline:
    """Tests for optimizer.run_pipeline() execution."""

    def test_quick_pipeline_runs(self, small_optimizer):
        from wings.pipeline import Pipeline
        p = Pipeline.quick(target_fidelity=0.9, max_time=30)
        results = small_optimizer.run_pipeline(p)
        assert "fidelity" in results
        assert "optimal_params" in results
        assert "time" in results
        assert "n_stages_completed" in results
        assert results["fidelity"] > 0

    def test_default_pipeline_runs(self, small_optimizer):
        """run_pipeline() with no args should use Pipeline.standard()."""
        results = small_optimizer.run_pipeline()
        assert results["fidelity"] > 0

    def test_custom_two_stage_pipeline(self, small_optimizer):
        from wings.pipeline import Pipeline, InitSearch, Adam
        p = Pipeline(
            target_fidelity=0.9,
            max_total_time=30,
            stages=[
                InitSearch(strategies=["smart", "random"]),
                Adam(max_steps=100, lr=0.02),
            ],
            verbose=False,
        )
        results = small_optimizer.run_pipeline(p)
        assert results["fidelity"] > 0.3
        assert results["n_stages_completed"] >= 1

    def test_early_stopping_on_target(self, small_optimizer):
        """Pipeline should stop early if target is already met."""
        from wings.pipeline import Pipeline, InitSearch, Adam
        # Set very low target that init search might already satisfy
        p = Pipeline(
            target_fidelity=0.001,  # Trivially achievable
            max_total_time=60,
            stages=[
                InitSearch(strategies=["smart"]),
                Adam(max_steps=1000),  # Should be skipped
            ],
            verbose=False,
        )
        results = small_optimizer.run_pipeline(p)
        assert results["success"]

    def test_spsa_stage_in_pipeline(self, small_optimizer):
        from wings.pipeline import Pipeline, SPSA
        p = Pipeline(
            target_fidelity=0.9,
            max_total_time=30,
            stages=[SPSA(max_steps=50, a=0.05, c=0.1)],
            verbose=False,
        )
        results = small_optimizer.run_pipeline(p)
        assert results["fidelity"] > 0

    def test_newton_stage_in_pipeline(self, small_optimizer):
        from wings.pipeline import Pipeline, InitSearch, Adam, Newton
        p = Pipeline(
            target_fidelity=0.99,
            max_total_time=60,
            stages=[
                InitSearch(strategies=["smart"]),
                Adam(max_steps=100, lr=0.02),
                Newton(max_steps=5, lr=0.3),
            ],
            verbose=False,
        )
        results = small_optimizer.run_pipeline(p)
        assert results["fidelity"] > 0

    def test_results_dict_complete(self, small_optimizer):
        from wings.pipeline import Pipeline
        p = Pipeline.quick(target_fidelity=0.9, max_time=20)
        p.verbose = False
        results = small_optimizer.run_pipeline(p)
        expected_keys = [
            "optimal_params", "fidelity", "infidelity", "time",
            "n_evaluations", "success", "n_stages_completed",
        ]
        for key in expected_keys:
            assert key in results, f"Missing key: {key}"
