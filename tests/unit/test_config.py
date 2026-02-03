"""Unit tests for config module."""

import os

import numpy as np
import pytest


class TestTargetFunction:
    """Tests for TargetFunction enum."""

    def test_enum_values(self):
        """Test expected enum values exist."""
        from wings import TargetFunction

        assert TargetFunction.GAUSSIAN.value == "gaussian"
        assert TargetFunction.LORENTZIAN.value == "lorentzian"
        assert TargetFunction.SECH.value == "sech"
        assert TargetFunction.CUSTOM.value == "custom"

    def test_enum_from_string(self):
        """Test creating enum from string."""
        from wings import TargetFunction

        assert TargetFunction("gaussian") == TargetFunction.GAUSSIAN
        assert TargetFunction("lorentzian") == TargetFunction.LORENTZIAN

    def test_invalid_value_raises(self):
        """Test invalid values raise ValueError."""
        from wings import TargetFunction

        with pytest.raises(ValueError):
            TargetFunction("invalid")


class TestOptimizerConfig:
    """Tests for OptimizerConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        from wings import OptimizerConfig

        config = OptimizerConfig(verbose=False)

        assert config.n_qubits == 9
        assert config.sigma == 1.0
        assert config.x0 == 0.0
        assert config.method == "L-BFGS-B"

    def test_custom_values(self):
        """Test setting custom values."""
        from wings import OptimizerConfig

        config = OptimizerConfig(
            n_qubits=10,
            sigma=0.5,
            x0=2.0,
            box_size=8.0,
            verbose=False,
        )

        assert config.n_qubits == 10
        assert config.sigma == 0.5
        assert config.x0 == 2.0
        assert config.box_size == 8.0

    def test_n_params_property(self):
        """Test n_params computed property."""
        from wings import OptimizerConfig

        config = OptimizerConfig(n_qubits=8, verbose=False)
        assert config.n_params == 64  # 8 * 8

    def test_n_states_property(self):
        """Test n_states computed property."""
        from wings import OptimizerConfig

        config = OptimizerConfig(n_qubits=8, verbose=False)
        assert config.n_states == 256  # 2^8

    def test_positions_property(self):
        """Test positions grid property."""
        from wings import OptimizerConfig

        config = OptimizerConfig(n_qubits=6, box_size=4.0, verbose=False)
        positions = config.positions

        assert len(positions) == 64
        assert positions[0] == -4.0
        assert positions[-1] == 4.0

    def test_delta_x_property(self):
        """Test grid spacing property."""
        from wings import OptimizerConfig

        config = OptimizerConfig(n_qubits=6, box_size=4.0, verbose=False)
        expected = 8.0 / 63
        assert np.isclose(config.delta_x, expected)

    def test_auto_box_size(self):
        """Test automatic box size calculation."""
        from wings import OptimizerConfig

        config = OptimizerConfig(sigma=0.3, verbose=False)
        assert config.box_size >= 8 * 0.3

    def test_explicit_box_size_preserved(self):
        """Test explicit box_size is preserved."""
        from wings import OptimizerConfig

        config = OptimizerConfig(sigma=0.5, box_size=20.0, verbose=False)
        assert config.box_size == 20.0

    def test_target_function_lorentzian(self):
        """Test Lorentzian target function config."""
        from wings import OptimizerConfig, TargetFunction

        config = OptimizerConfig(
            target_function=TargetFunction.LORENTZIAN,
            gamma=0.3,
            verbose=False,
        )
        assert config.target_function == TargetFunction.LORENTZIAN
        assert config.gamma == 0.3

    def test_custom_target_requires_function(self):
        """Test CUSTOM target requires custom_target_fn."""
        from wings import OptimizerConfig, TargetFunction

        with pytest.raises(ValueError, match="custom_target_fn"):
            OptimizerConfig(
                target_function=TargetFunction.CUSTOM,
                verbose=False,
            )

    def test_custom_target_with_function(self):
        """Test CUSTOM target with provided function."""
        from wings import OptimizerConfig, TargetFunction

        def my_func(x):
            return np.exp(-(x**4))

        config = OptimizerConfig(
            target_function=TargetFunction.CUSTOM,
            custom_target_fn=my_func,
            verbose=False,
        )
        assert config.custom_target_fn is my_func

    def test_multi_gpu_settings(self):
        """Test multi-GPU configuration."""
        from wings import OptimizerConfig

        config = OptimizerConfig(
            use_multi_gpu=True,
            gpu_device_ids=[0, 1],
            simulators_per_gpu=4,
            verbose=False,
        )

        assert config.use_multi_gpu
        assert config.gpu_device_ids == [0, 1]
        assert config.simulators_per_gpu == 4


class TestOptimizationPipeline:
    """Tests for OptimizationPipeline configuration."""

    def test_default_values(self):
        """Test default pipeline values."""
        from wings.config import OptimizationPipeline

        pipeline = OptimizationPipeline()

        assert pipeline.mode == "adaptive"
        assert pipeline.target_fidelity == 0.9999
        assert pipeline.use_adam_stage

    def test_target_infidelity_computed(self):
        """Test target_infidelity is computed."""
        from wings.config import OptimizationPipeline

        pipeline = OptimizationPipeline(target_fidelity=0.99999)
        assert np.isclose(pipeline.target_infidelity, 1e-5)

    def test_adam_time_fraction(self):
        """Test adam_time_fraction is respected."""
        from wings.config import OptimizationPipeline

        pipeline = OptimizationPipeline(
            adam_time_fraction=0.3,
            max_total_time=100,
        )

        assert pipeline.adam_time_fraction == 0.3
        # Adam should get 30 seconds
        assert pipeline.max_total_time * pipeline.adam_time_fraction == 30.0

    def test_basin_hopping_settings(self):
        """Test basin hopping configuration."""
        from wings.config import OptimizationPipeline

        pipeline = OptimizationPipeline(
            use_basin_hopping=True,
            basin_hopping_threshold=0.999,
            basin_hopping_iterations=50,
        )

        assert pipeline.use_basin_hopping
        assert pipeline.basin_hopping_threshold == 0.999
        assert pipeline.basin_hopping_iterations == 50


class TestCampaignConfig:
    """Tests for CampaignConfig dataclass."""

    def test_default_values(self):
        """Test default campaign configuration."""
        from wings.config import CampaignConfig

        config = CampaignConfig(verbose=0)

        assert config.n_qubits == 8
        assert config.sigma == 0.5
        assert config.total_runs == 1000

    def test_target_fidelity_computed(self):
        """Test target_fidelity computed from target_infidelity."""
        from wings.config import CampaignConfig

        config = CampaignConfig(target_infidelity=1e-8, verbose=0)
        assert np.isclose(config.target_fidelity, 1 - 1e-8)

    def test_save_and_load(self, tmp_path):
        """Test saving and loading configuration."""
        from wings.config import CampaignConfig

        config = CampaignConfig(
            n_qubits=12,
            sigma=0.7,
            verbose=0,
            output_dir=str(tmp_path),
            checkpoint_dir=str(tmp_path),
        )

        filepath = config.save(str(tmp_path / "test_config.json"))
        assert os.path.exists(filepath)

        loaded = CampaignConfig.load(filepath)
        assert loaded.n_qubits == 12
        assert loaded.sigma == 0.7

    def test_strategy_weights_normalized(self):
        """Test strategy weights are normalized."""
        from wings.config import CampaignConfig

        config = CampaignConfig(
            strategy_weights={"smart": 1.0, "random": 1.0},
            verbose=0,
        )

        total = sum(config.strategy_weights.values())
        assert np.isclose(total, 1.0)

    def test_get_strategy_deterministic(self):
        """Test strategy selection is deterministic."""
        from wings.config import CampaignConfig

        config = CampaignConfig(base_seed=42, verbose=0)

        s1 = config.get_strategy_for_run(5)
        s2 = config.get_strategy_for_run(5)
        assert s1 == s2
