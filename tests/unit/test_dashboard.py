"""Unit tests for interactive dashboard (v0.4.0 WI-8)."""

import numpy as np
import pytest


@pytest.mark.unit
class TestOptimizationDashboard:

    def test_initialization(self):
        from wings.dashboard import OptimizationDashboard
        dash = OptimizationDashboard()
        assert dash.steps == []
        assert dash.fidelities == []

    def test_update_accumulates(self):
        from wings.dashboard import OptimizationDashboard
        dash = OptimizationDashboard()
        dash.update(step=0, fidelity=0.5, gradient_norm=1.0)
        dash.update(step=1, fidelity=0.6, gradient_norm=0.8)
        assert len(dash.steps) == 2
        assert len(dash.fidelities) == 2
        assert dash.fidelities[-1] == 0.6

    def test_update_with_params(self):
        from wings.dashboard import OptimizationDashboard
        dash = OptimizationDashboard()
        params = np.array([0.1, 0.2, 0.3])
        dash.update(step=0, fidelity=0.5, gradient_norm=1.0, params=params)
        assert len(dash.param_history) == 1

    def test_get_summary(self):
        from wings.dashboard import OptimizationDashboard
        dash = OptimizationDashboard()
        for i in range(10):
            dash.update(step=i, fidelity=0.5 + i*0.05, gradient_norm=1.0/(i+1))
        summary = dash.get_summary()
        assert "best_fidelity" in summary
        assert "total_steps" in summary
        assert summary["total_steps"] == 10
        assert summary["best_fidelity"] == pytest.approx(0.95)

    def test_save_html_creates_file(self, tmp_path):
        """save_html should create a file even without plotly (falls back to text)."""
        from wings.dashboard import OptimizationDashboard
        dash = OptimizationDashboard()
        for i in range(5):
            dash.update(step=i, fidelity=0.5 + i*0.1, gradient_norm=1.0)
        filepath = str(tmp_path / "test_dashboard.html")
        dash.save_html(filepath)
        import os
        assert os.path.exists(filepath)

    def test_import_does_not_require_plotly(self):
        """Importing dashboard should not fail even if plotly is missing."""
        from wings.dashboard import OptimizationDashboard
        dash = OptimizationDashboard()
        assert dash is not None
