"""Dash app to browse the experimental PXRD store (DATA_ROADMAP.md §3).

Loads the unified index (`exp_data/index.csv`) and lets you:

* filter/sort/search the index table (by source, audit verdict, quality, or any
  column via the table's native filters) and pick a pattern;
* view that pattern **side by side in 2θ and in log-d** — the two coordinates
  have different x-ranges and orientation (high 2θ ⇒ low d), so a shared x-axis
  would be misleading; the log-d panel previews the encoder/production coordinate
  (`DESIGN_DECISIONS.md` §1);
* toggle raw vs background-subtracted vs both, overlay peak positions where the
  source ships them, and override the wavelength (useful for the
  `missing_wavelength` patterns — see how they read against an assumed Kα line);
* read the essential metadata (labels, cell, provenance) for the selection.

The 2θ→d conversion is Bragg's law, `d = λ / (2 sin θ)` with `θ = 2θ/2`; log-d is
`log₁₀(d/Å)`. Pure helpers (`filter_records`, `make_figure`, `metadata_rows`) are
separated from the Dash wiring so they can be tested without launching a server.

Run (from the package directory)::

    uv run python -m crystalai_data.visualization.pattern_browser
    uv run python -m crystalai_data.visualization.pattern_browser --port 8051
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..xrddata import XRDDatabase

# Columns surfaced in the index table (order matters).
_TABLE_COLS = [
    "id", "source", "source_id", "formula", "space_group", "crystal_system",
    "wavelength_A", "quality_flag", "audit", "n_phases", "cif_id",
]
_FALLBACK_WAVELENGTH_A = 1.5406  # assumed Cu Kα when a pattern has none and no override
_VIEW_STYLE = {
    "raw": ("#1f77b4", "raw"),
    "bgsub": ("#d62728", "bgsub"),
}


# --------------------------------------------------------------------------- #
# Pure helpers (no Dash) — unit-testable
# --------------------------------------------------------------------------- #
def two_theta_to_log_d(two_theta, wavelength_A: float) -> np.ndarray:
    """Bragg 2θ (deg) → log₁₀(d/Å). Non-physical points (θ≤0) become NaN."""
    tt = np.asarray(two_theta, dtype=float)
    theta = np.radians(tt / 2.0)
    sin_t = np.sin(theta)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = wavelength_A / (2.0 * sin_t)
        d = np.where((sin_t > 1e-9) & (d > 0), d, np.nan)
        return np.log10(d)


def resolve_wavelength(row: pd.Series, override) -> tuple[float, bool]:
    """Pick the wavelength for the log-d conversion → (value, is_assumed)."""
    if override not in (None, ""):
        return float(override), False
    if pd.notna(row.get("wavelength_A")):
        return float(row["wavelength_A"]), False
    return _FALLBACK_WAVELENGTH_A, True


def filter_records(index: pd.DataFrame, source, audit, quality) -> list[dict]:
    """Index rows (table columns only) after the dropdown filters, as records."""
    df = index
    if source:
        df = df[df["source"] == source]
    if audit:
        df = df[df["audit"] == audit]
    if quality:
        df = df[df["quality_flag"] == quality]
    out = df[_TABLE_COLS].copy()
    out["wavelength_A"] = out["wavelength_A"].round(5)
    return out.to_dict("records")


def _add_profile(fig, xdb, pid, row, which, wavelength, x_native_is_2theta):
    """Add one profile (raw/bgsub) to both panels; return its (x, y, ymax)."""
    color, label = _VIEW_STYLE[which]
    x, y = xdb.load_pattern(int(pid), which)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    logd = two_theta_to_log_d(x, wavelength) if x_native_is_2theta else np.log10(x)
    fig.add_trace(
        go.Scatter(x=x, y=y, mode="lines", name=label, line=dict(color=color, width=1)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=logd, y=y, mode="lines", name=label, line=dict(color=color, width=1),
                   showlegend=False),
        row=1, col=2,
    )
    return y


def _add_peak_stems(fig, xdb, pid, wavelength, ymax, x_native_is_2theta):
    """Overlay peak positions as vertical stems in both panels."""
    px, _ = xdb.load_pattern(int(pid), "peaks")
    px = np.asarray(px, float)
    plogd = two_theta_to_log_d(px, wavelength) if x_native_is_2theta else np.log10(px)

    def stems(xs):
        X: list = []
        Y: list = []
        for xv in xs:
            X += [xv, xv, None]
            Y += [0, ymax, None]
        return X, Y

    for col, xs in ((1, px), (2, plogd)):
        sx, sy = stems(xs)
        fig.add_trace(
            go.Scatter(x=sx, y=sy, mode="lines", name="peaks",
                       line=dict(color="#2ca02c", width=1, dash="dot"),
                       showlegend=(col == 1)),
            row=1, col=col,
        )


def make_figure(xdb: XRDDatabase, row: pd.Series, view: str, show_peaks: bool,
                wavelength_override) -> go.Figure:
    """Build the side-by-side (2θ | log-d) figure for one selected pattern."""
    pid = int(row["id"])
    wavelength, assumed = resolve_wavelength(row, wavelength_override)
    x_native_is_2theta = str(row.get("x_coord")) != "d"

    left_title = "Intensity vs 2θ (deg)" if x_native_is_2theta else "Intensity vs d (Å)"
    wl_note = f"λ={wavelength:.5g} Å" + (" (assumed)" if assumed else "")
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(left_title, f"Intensity vs log₁₀(d/Å)   [{wl_note}]"),
        horizontal_spacing=0.08,
    )

    whichs = ["raw", "bgsub"] if view == "both" else [view]
    ymax = 1.0
    for which in whichs:
        rel = row.get(f"{which}_path")
        if rel is None or (isinstance(rel, float) and pd.isna(rel)):
            continue
        y = _add_profile(fig, xdb, pid, row, which, wavelength, x_native_is_2theta)
        if y.size:
            ymax = max(ymax, float(np.nanmax(y)))

    peaks_rel = row.get("peaks_path")
    if show_peaks and not (peaks_rel is None or (isinstance(peaks_rel, float) and pd.isna(peaks_rel))):
        _add_peak_stems(fig, xdb, pid, wavelength, ymax, x_native_is_2theta)

    fig.update_xaxes(title_text=left_title.split(" vs ")[1], row=1, col=1)
    fig.update_xaxes(title_text="log₁₀(d / Å)", row=1, col=2)
    fig.update_yaxes(title_text="Intensity", row=1, col=1)
    fig.update_layout(
        height=460, margin=dict(l=50, r=20, t=50, b=45),
        legend=dict(orientation="h", yanchor="bottom", y=1.08, x=0),
        title=f"{row['source']} · {row['source_id']}",
    )
    return fig


def metadata_rows(row: pd.Series) -> list[tuple[str, str]]:
    """(label, value) pairs for the metadata panel, skipping empty fields."""
    def val(k, fmt=str):
        v = row.get(k)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        try:
            return fmt(v)
        except Exception:
            return str(v)

    cell = None
    if pd.notna(row.get("a")):
        cell = (f"a={val('a', lambda x: f'{x:.4f}')}  b={val('b', lambda x: f'{x:.4f}')}  "
                f"c={val('c', lambda x: f'{x:.4f}')}  α={val('alpha', lambda x: f'{x:g}')}  "
                f"β={val('beta', lambda x: f'{x:g}')}  γ={val('gamma', lambda x: f'{x:g}')}")

    fields = [
        ("id", val("id")),
        ("source / id", f"{row.get('source')} / {row.get('source_id')}"),
        ("formula", val("formula")),
        ("space group", val("space_group", lambda x: str(int(x)))),
        ("crystal system", val("crystal_system", lambda x: str(int(x)))),
        ("cell", cell),
        ("wavelength (Å)", val("wavelength_A", lambda x: f"{x:.5g}")),
        ("x coordinate", val("x_coord")),
        ("n phases", val("n_phases", lambda x: str(int(x)))),
        ("quality", val("quality_flag")),
        ("audit", val("audit")),
        ("cif_id (crystals.sqlite)", val("cif_id", lambda x: str(int(x)))),
        ("cif_path", val("cif_path")),
        ("bgsub", ("auto" if row.get("autobg") == 1 else "native") if pd.notna(row.get("autobg")) else None),
        ("notes", val("notes")),
    ]
    return [(k, v) for k, v in fields if v not in (None, "", "nan")]


# --------------------------------------------------------------------------- #
# Dash app
# --------------------------------------------------------------------------- #
def build_app(store: str | Path | None = None):
    from dash import Dash, Input, Output, dash_table, dcc, html

    xdb = XRDDatabase(store)
    index = xdb.index.copy()
    if "audit" not in index.columns:
        index["audit"] = pd.NA

    def opts(col):
        vals = sorted(v for v in index[col].dropna().unique())
        return [{"label": str(v), "value": v} for v in vals]

    app = Dash(__name__, title="CrystalAI · experimental patterns")

    controls = html.Div(
        style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "alignItems": "center"},
        children=[
            dcc.Dropdown(id="f-source", options=opts("source"), placeholder="source",
                         style={"width": "140px"}),
            dcc.Dropdown(id="f-audit", options=opts("audit"), placeholder="audit",
                         style={"width": "190px"}),
            dcc.Dropdown(id="f-quality", options=opts("quality_flag"), placeholder="quality",
                         style={"width": "150px"}),
            dcc.RadioItems(id="view", options=["raw", "bgsub", "both"], value="raw",
                           inline=True, style={"marginLeft": "8px"}),
            dcc.Checklist(id="peaks", options=[{"label": " peaks", "value": "peaks"}], value=[]),
            dcc.Input(id="wl", type="number", placeholder="override λ (Å)",
                      debounce=True, style={"width": "130px"}),
        ],
    )

    app.layout = html.Div(
        style={"font-family": "system-ui, sans-serif", "margin": "16px", "maxWidth": "1250px"},
        children=[
            html.H3("CrystalAI — experimental PXRD browser"),
            html.Div(f"{len(index):,} patterns  ·  "
                     + "  ".join(f"{s}={int((index['source'] == s).sum())}"
                                 for s in sorted(index['source'].unique())),
                     style={"color": "#666", "marginBottom": "8px"}),
            controls,
            dash_table.DataTable(
                id="table",
                columns=[{"name": c, "id": c} for c in _TABLE_COLS],
                data=filter_records(index, None, None, None),
                row_selectable="single",
                filter_action="native", sort_action="native",
                page_size=12,
                style_table={"overflowX": "auto", "marginTop": "10px"},
                style_cell={"fontSize": "12px", "padding": "4px", "fontFamily": "monospace"},
                style_header={"fontWeight": "bold", "backgroundColor": "#f2f2f2"},
            ),
            dcc.Graph(id="graph"),
            html.Div(id="meta", style={"marginTop": "6px", "fontSize": "13px"}),
        ],
    )

    @app.callback(Output("table", "data"),
                  Input("f-source", "value"), Input("f-audit", "value"),
                  Input("f-quality", "value"))
    def _update_table(source, audit, quality):
        return filter_records(index, source, audit, quality)

    @app.callback(Output("graph", "figure"), Output("meta", "children"),
                  Input("table", "selected_row_ids"), Input("view", "value"),
                  Input("peaks", "value"), Input("wl", "value"))
    def _render(selected_ids, view, peaks_val, wl_override):
        if not selected_ids:
            return go.Figure(layout={"height": 460}), "Select a pattern (radio button on the left of a row)."
        row = index[index["id"] == selected_ids[0]].iloc[0]
        fig = make_figure(xdb, row, view, "peaks" in (peaks_val or []), wl_override)
        meta = html.Table(
            [html.Tr([html.Td(k, style={"fontWeight": "bold", "paddingRight": "12px",
                                         "verticalAlign": "top", "color": "#444"}),
                      html.Td(v)]) for k, v in metadata_rows(row)],
            style={"borderCollapse": "collapse"},
        )
        return fig, meta

    return app


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Dash browser for the experimental PXRD store")
    p.add_argument("--store", default=None, type=str, help="store root (default: exp_data)")
    p.add_argument("--host", default="127.0.0.1", type=str)
    p.add_argument("--port", default=8050, type=int)
    p.add_argument("--debug", action="store_true")
    args = p.parse_args(argv)

    app = build_app(args.store)
    print(f"[pattern_browser] serving on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
