"""Unit tests for path configuration."""

import pytest


@pytest.mark.unit
class TestPathConfig:
    def test_get_path_config(self):
        from wings.paths import get_path_config

        config = get_path_config(verbose=False)
        assert config is not None
        assert hasattr(config, "cache_dir")
        assert hasattr(config, "output_dir")
        assert hasattr(config, "checkpoint_dir")
        assert hasattr(config, "campaign_dir")

    def test_path_config_dirs_are_paths(self):
        from pathlib import Path

        from wings.paths import get_path_config

        config = get_path_config(verbose=False)
        assert isinstance(config.cache_dir, Path)

    def test_path_config_singleton(self):
        from wings.paths import get_path_config

        c1 = get_path_config(verbose=False)
        c2 = get_path_config(verbose=False)
        # Should return same or equivalent config
        assert str(c1.cache_dir) == str(c2.cache_dir)
