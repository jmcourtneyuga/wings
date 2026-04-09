"""
Interactive visualization dashboard for WINGS optimization.

Provides real-time and post-hoc visualization of optimization convergence,
gradient norms, and parameter landscapes.

Requires plotly for interactive plots (optional dependency).
Falls back to text/HTML summary when plotly is not available.
"""

from typing import Optional

import numpy as np

__all__ = ["OptimizationDashboard"]

# Check for plotly availability without requiring it at import time
_HAS_PLOTLY = False
try:
    import plotly  # noqa: F401 — used only to detect availability

    _HAS_PLOTLY = True
except ImportError:
    pass


class OptimizationDashboard:
    """
    Tracks and visualizes optimization progress.

    Usage:
        dash = OptimizationDashboard()
        for step in optimization_loop:
            dash.update(step, fidelity, gradient_norm, params)
        dash.save_html("convergence.html")
    """

    def __init__(self) -> None:
        self.steps: list[int] = []
        self.fidelities: list[float] = []
        self.infidelities: list[float] = []
        self.gradient_norms: list[float] = []
        self.param_history: list[np.ndarray] = []
        self.timestamps: list[float] = []

    def update(
        self,
        step: int,
        fidelity: float,
        gradient_norm: float = 0.0,
        params: Optional[np.ndarray] = None,
    ) -> None:
        """Record one optimization step."""
        import time

        self.steps.append(step)
        self.fidelities.append(fidelity)
        self.infidelities.append(max(1.0 - fidelity, 1e-16))
        self.gradient_norms.append(gradient_norm)
        self.timestamps.append(time.time())
        if params is not None:
            self.param_history.append(params.copy())

    def get_summary(self) -> dict:
        """Return summary statistics."""
        if not self.fidelities:
            return {"total_steps": 0, "best_fidelity": 0.0}
        return {
            "total_steps": len(self.steps),
            "best_fidelity": max(self.fidelities),
            "best_infidelity": min(self.infidelities),
            "final_fidelity": self.fidelities[-1],
            "final_gradient_norm": self.gradient_norms[-1] if self.gradient_norms else 0.0,
        }

    def plot_convergence(self):
        """Plot infidelity convergence (requires plotly)."""
        if not _HAS_PLOTLY:
            raise ImportError(
                "plotly is required for interactive plots. Install with: pip install plotly"
            )
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=self.steps,
                y=self.infidelities,
                mode="lines",
                name="Infidelity (1-F)",
            )
        )
        fig.update_layout(
            title="Optimization Convergence",
            xaxis_title="Step",
            yaxis_title="Infidelity (1-F)",
            yaxis_type="log",
        )
        return fig

    def plot_gradient_norms(self):
        """Plot gradient norm history (requires plotly)."""
        if not _HAS_PLOTLY:
            raise ImportError("plotly is required. Install with: pip install plotly")
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=self.steps,
                y=self.gradient_norms,
                mode="lines",
                name="Gradient Norm",
            )
        )
        fig.update_layout(
            title="Gradient Norm History",
            xaxis_title="Step",
            yaxis_title="||grad||",
            yaxis_type="log",
        )
        return fig

    def save_html(self, filepath: str) -> None:
        """
        Save dashboard to HTML file.

        Uses plotly if available, otherwise generates a plain HTML summary.
        """
        if _HAS_PLOTLY:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            fig = make_subplots(
                rows=2, cols=1, subplot_titles=("Infidelity Convergence", "Gradient Norm")
            )

            fig.add_trace(
                go.Scatter(
                    x=self.steps,
                    y=self.infidelities,
                    mode="lines",
                    name="Infidelity",
                ),
                row=1,
                col=1,
            )

            fig.add_trace(
                go.Scatter(
                    x=self.steps,
                    y=self.gradient_norms,
                    mode="lines",
                    name="Gradient Norm",
                ),
                row=2,
                col=1,
            )

            fig.update_yaxes(type="log", row=1, col=1)
            fig.update_yaxes(type="log", row=2, col=1)
            fig.update_layout(title="WINGS Optimization Dashboard", height=800)
            fig.write_html(filepath)
        else:
            # Fallback: plain HTML summary
            summary = self.get_summary()
            html = "<html><body>"
            html += "<h1>WINGS Optimization Dashboard</h1>"
            html += "<h2>Summary</h2><ul>"
            for k, v in summary.items():
                html += f"<li><b>{k}</b>: {v}</li>"
            html += "</ul>"
            if self.fidelities:
                html += f"<p>Steps: {len(self.steps)}, "
                html += f"Best F: {max(self.fidelities):.12f}, "
                html += f"Final F: {self.fidelities[-1]:.12f}</p>"
            html += "</body></html>"
            with open(filepath, "w") as f:
                f.write(html)
