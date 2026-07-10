"""Plotly overlay of experimental vs simulated patterns + difference curve."""

from __future__ import annotations

from numpy.typing import NDArray


def overlay_figure(
    x: NDArray, exp: NDArray, calc: NDArray, *,
    x_label: str = "log₁₀(d / Å)", title: str = "experimental vs simulated",
    metrics: dict | None = None,
):
    """Return a Plotly figure: experimental, simulated, and (exp − calc) difference."""
    import plotly.graph_objects as go

    sub = title
    if metrics:
        sub += (f"   Rwp={metrics.get('rwp', float('nan')):.1f}%  "
                f"cos={metrics.get('cosine', float('nan')):.3f}")

    fig = go.Figure()
    fig.add_scatter(x=x, y=exp, mode="lines", name="experimental",
                    line=dict(color="#1f77b4", width=1))
    fig.add_scatter(x=x, y=calc, mode="lines", name="simulated",
                    line=dict(color="#d62728", width=1))
    diff = calc - exp
    fig.add_scatter(x=x, y=diff - 0.25, mode="lines", name="difference",
                    line=dict(color="#7f7f7f", width=1))
    fig.update_layout(title=sub, xaxis_title=x_label, yaxis_title="normalized intensity",
                      height=460, legend=dict(orientation="h", y=1.05),
                      margin=dict(l=50, r=20, t=50, b=45))
    return fig
