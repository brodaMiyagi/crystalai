"""Interactive PXRD simulation dashboard (SIMXRD_ROADMAP §7, Phase 4).

Four tabs:
  1. Single Pattern Simulator  — one structure, tunable effects, 2θ or log-d view.
  2. Effect Decomposition       — pristine profile vs each physical effect added in turn.
  3. Augmentation Preview       — N stochastic draws from a training preset, overlaid.
  4. Sim vs Experimental        — refine effects against an ``exp_subset`` pattern until
                                  Rwp drops (criterion #6). Uses the opXRD subset patterns
                                  that ship an inline structure, so it needs no crystals DB.

Launch: ``uv run python scripts/launch_dashboard.py``.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import gradio as gr
import numpy as np
import plotly.graph_objects as go
from pymatgen.core import Structure

from crystalai_simxrd.comparison.compare import compare_structure_to_raw
from crystalai_simxrd.core.crystal import load_cif
from crystalai_simxrd.core.domain import Domain
from crystalai_simxrd.augmentations.presets import PRESETS, get_preset
from crystalai_simxrd.augmentations.profile_augmentations import ProfileAugmentor
from crystalai_simxrd.simulation.simulator import EffectConfig, simulate

warnings.filterwarnings("ignore")

_PKG = Path(__file__).resolve().parents[1]
_CIF_DIR = _PKG / "data" / "example_cifs"
EXAMPLES = sorted(p.stem for p in _CIF_DIR.glob("*.cif"))


def _structure(name: str) -> Structure:
    return load_cif(_CIF_DIR / f"{name}.cif")


# ---------------------------------------------------------------- experimental subset
def _load_exp_subset():
    """Return {label: (structure, two_theta, raw_intensity, wavelength)} for exp_subset pairs.

    RAW counts (the criterion-#6 comparison fits the background). opXRD pairs use the inline
    structure.cif; lab pairs resolve the ICSD structure from crystals.sqlite via cif_id if the
    DB is present (else skipped, keeping the dashboard self-contained)."""
    try:
        from crystalai_data.xrddata.database import XRDDatabase
    except Exception:
        return {}
    fixture = _PKG.parent / "crystalai-data" / "tests" / "fixtures" / "exp_subset"
    if not fixture.exists():
        return {}
    db = XRDDatabase(fixture)
    cdb = None
    crystals_db = _PKG.parent / "crystalai-data" / "crystals.sqlite"
    if crystals_db.exists():
        try:
            from crystalai_data.crystals.api import CrystalDatabase
            cdb = CrystalDatabase(crystals_db)
        except Exception:
            cdb = None

    out = {}
    for _, r in db.to_frame().iterrows():
        wl = r["wavelength_A"]
        if wl is None or (isinstance(wl, float) and np.isnan(wl)):
            continue
        st = None
        try:
            if isinstance(r.get("cif_path"), str):                 # opXRD inline CIF
                st = Structure.from_str(db.cif_text(int(r["id"])), fmt="cif")
            elif cdb is not None and np.isfinite(r.get("cif_id", np.nan)):  # lab → ICSD
                st = cdb.get(int(r["cif_id"]))
            if st is None:
                continue
            x, y = db.load_pattern(int(r["id"]), "raw")            # RAW counts
        except Exception:
            continue
        sg = "" if np.isnan(r["space_group"]) else f" SG{int(r['space_group'])}"
        out[f"{r['source_id']} (λ={float(wl):.4f}{sg})"] = (
            st, np.asarray(x, float), np.asarray(y, float), float(wl),
        )
    return out


EXP = _load_exp_subset()


# --------------------------------------------------------------------- tab 1: simulator
def _simulate_single(name, wavelength, size_nm, b_iso, microstrain, march_r,
                     zero_shift, background, axial, view):
    st = _structure(name)
    eff = EffectConfig(
        b_iso=b_iso,
        crystallite_size_nm=(size_nm if size_nm < 500 else np.inf),
        microstrain=microstrain,
        march_r=march_r,
        zero_shift_deg=zero_shift,
        axial_divergence=axial,
        background=(None if background == "none" else background),
    )
    dom = Domain.TWO_THETA if view == "2θ" else Domain.LOG_D
    p = simulate(st, wavelength, domain=dom, effects=eff)
    fig = go.Figure()
    fig.add_scatter(x=p.x_axis, y=p.intensity / max(p.intensity.max(), 1e-12),
                    mode="lines", line=dict(color="#d62728", width=1), name=name)
    fig.update_layout(
        title=f"{name}  ({p.metadata['n_peaks']} reflections, λ={wavelength:.4f} Å)",
        xaxis_title=("2θ (deg)" if view == "2θ" else "log₁₀(d / Å)"),
        yaxis_title="normalized intensity", height=460,
        margin=dict(l=50, r=20, t=50, b=45))
    return fig


# ------------------------------------------------------------- tab 2: effect decomposition
def _decompose(name, wavelength):
    st = _structure(name)
    steps = [
        ("pristine", EffectConfig()),
        ("+ Debye-Waller (B=0.8)", EffectConfig(b_iso=0.8)),
        ("+ size broadening (40 nm)", EffectConfig(b_iso=0.8, crystallite_size_nm=40)),
        ("+ pref. orient. (r=0.75)", EffectConfig(b_iso=0.8, crystallite_size_nm=40, march_r=0.75)),
        ("+ residual background", EffectConfig(b_iso=0.8, crystallite_size_nm=40, march_r=0.75,
                                               background="residual")),
    ]
    fig = go.Figure()
    for i, (label, eff) in enumerate(steps):
        p = simulate(st, wavelength, domain=Domain.TWO_THETA, effects=eff)
        y = p.intensity / max(p.intensity.max(), 1e-12)
        fig.add_scatter(x=p.x_axis, y=y + i * 1.05, mode="lines", name=label,
                        line=dict(width=1))
    fig.update_layout(title=f"effect decomposition — {name} (offset for clarity)",
                      xaxis_title="2θ (deg)", yaxis_title="normalized intensity (offset)",
                      height=560, legend=dict(orientation="h", y=1.06),
                      margin=dict(l=50, r=20, t=60, b=45))
    return fig


# --------------------------------------------------------------- tab 3: augmentation preview
def _augment_preview(name, preset, n, seed):
    st = _structure(name)
    aug_cfg, _ = get_preset(preset)
    augmentor = ProfileAugmentor(aug_cfg)
    rng = np.random.default_rng(int(seed))
    fig = go.Figure()
    for i in range(int(n)):
        a = augmentor(st, rng)
        fig.add_scatter(x=a.x_axis, y=a.intensity, mode="lines",
                        line=dict(width=1), opacity=0.75,
                        name=f"λ={a.wavelength:.3f} r={a.metadata['march_r']:.2f}")
    fig.update_layout(
        title=f"{n} augmented draws — {name} · preset '{preset}'",
        xaxis_title="log₁₀(d / Å)", yaxis_title="normalized intensity", height=460,
        legend=dict(orientation="h", y=-0.2, font=dict(size=9)),
        margin=dict(l=50, r=20, t=50, b=70))
    return fig


# ----------------------------------------------------------- tab 4: sim vs experimental
def _compare_exp(label, size_nm, b_iso, march_r, zero_shift):
    """Criterion-#6 comparison: forward sim vs RAW counts in 2θ, background+scale fitted."""
    if label not in EXP:
        return go.Figure(), "no experimental patterns available"
    st, tt, raw, wl = EXP[label]
    eff = EffectConfig(
        b_iso=b_iso,
        crystallite_size_nm=(size_nm if size_nm < 500 else np.inf),
        march_r=march_r,
        zero_shift_deg=zero_shift,
    )
    res = compare_structure_to_raw(st, tt, raw, wl, effects=eff)
    m = res["mask"]
    x, obs, calc, bg = res["two_theta"][m], res["obs"][m], res["calc"][m], res["background"][m]
    fig = go.Figure()
    fig.add_scatter(x=x, y=obs, mode="lines", name="experimental (raw)",
                    line=dict(color="#1f77b4", width=1))
    fig.add_scatter(x=x, y=calc, mode="lines", name="simulated + bg",
                    line=dict(color="#d62728", width=1))
    fig.add_scatter(x=x, y=bg, mode="lines", name="fitted background",
                    line=dict(color="#2ca02c", width=1, dash="dot"))
    fig.add_scatter(x=x, y=obs - calc - 0.15 * float(obs.max()), mode="lines",
                    name="difference", line=dict(color="#7f7f7f", width=1))
    gof = res["gof"]
    fig.update_layout(
        title=f"{label}   Rwp={res['rwp']:.1f}%  GoF={gof:.2f}  R_Bragg={res['r_bragg']:.1f}%",
        xaxis_title="2θ (deg)", yaxis_title="counts", height=460,
        legend=dict(orientation="h", y=1.06), margin=dict(l=60, r=20, t=50, b=45))
    gate = "✅ PASS" if gof < 4.0 else "above gate"
    txt = (f"**GoF = Rwp / noise-floor = {gof:.2f}**  ({gate}; criterion #6 gate GoF < 4)\n\n"
           f"Rwp = {res['rwp']:.1f}%  ·  noise-floor Rwp = {res['rwp_noise_floor']:.1f}%  ·  "
           f"cosine = {res['cosine']:.3f}\n\n"
           f"**R_Bragg = {res['r_bragg']:.1f}%** (reported bug-detector, not gated — a forward "
           f"simulator of a fixed structure floors at ~20–60% from texture/absorption; "
           f"≤5% needs a Rietveld refinement engine). Tune size / B / March-r / zero-shift.")
    return fig, txt


# --------------------------------------------------------------------------- build UI
def build() -> gr.Blocks:
    with gr.Blocks(title="CrystalAI · SimXRD") as demo:
        gr.Markdown("# CrystalAI — PXRD Simulation Dashboard")

        with gr.Tab("Single Pattern"):
            with gr.Row():
                with gr.Column(scale=1):
                    s_name = gr.Dropdown(EXAMPLES, value=EXAMPLES[0], label="structure")
                    s_wl = gr.Slider(0.3, 2.5, 1.5406, step=0.0001, label="wavelength (Å)")
                    s_size = gr.Slider(5, 500, 500, step=5, label="crystallite size (nm; 500=∞)")
                    s_b = gr.Slider(0, 3, 0.0, step=0.1, label="B_iso (Å²)")
                    s_strain = gr.Slider(0, 0.01, 0.0, step=0.0005, label="microstrain")
                    s_march = gr.Slider(0.5, 1.5, 1.0, step=0.01, label="March-Dollase r")
                    s_zero = gr.Slider(-0.1, 0.1, 0.0, step=0.005, label="zero-shift (deg)")
                    s_bg = gr.Radio(["none", "residual", "physical"], value="none", label="background")
                    s_axial = gr.Checkbox(False, label="axial divergence")
                    s_view = gr.Radio(["2θ", "log-d"], value="2θ", label="domain")
                s_plot = gr.Plot(scale=2)
            inputs = [s_name, s_wl, s_size, s_b, s_strain, s_march, s_zero, s_bg, s_axial, s_view]
            for c in inputs:
                c.change(_simulate_single, inputs, s_plot)
            demo.load(_simulate_single, inputs, s_plot)

        with gr.Tab("Effect Decomposition"):
            with gr.Row():
                d_name = gr.Dropdown(EXAMPLES, value=EXAMPLES[0], label="structure")
                d_wl = gr.Slider(0.3, 2.5, 1.5406, step=0.0001, label="wavelength (Å)")
            d_plot = gr.Plot()
            for c in (d_name, d_wl):
                c.change(_decompose, [d_name, d_wl], d_plot)
            demo.load(_decompose, [d_name, d_wl], d_plot)

        with gr.Tab("Augmentation Preview"):
            with gr.Row():
                a_name = gr.Dropdown(EXAMPLES, value=EXAMPLES[0], label="structure")
                a_preset = gr.Dropdown(sorted(PRESETS), value="PRODUCTION", label="preset")
                a_n = gr.Slider(1, 12, 6, step=1, label="draws")
                a_seed = gr.Number(0, label="seed", precision=0)
            a_plot = gr.Plot()
            a_btn = gr.Button("sample", variant="primary")
            a_btn.click(_augment_preview, [a_name, a_preset, a_n, a_seed], a_plot)
            demo.load(_augment_preview, [a_name, a_preset, a_n, a_seed], a_plot)

        with gr.Tab("Sim vs Experimental"):
            if not EXP:
                gr.Markdown("_No experimental patterns with inline structures found "
                            "(expected `crystalai-data/tests/fixtures/exp_subset`)._")
            else:
                gr.Markdown("Forward sim vs **raw** counts in 2θ; background + scale are "
                            "fitted (Rietveld convention). Gate: **GoF < 4**.")
                with gr.Row():
                    with gr.Column(scale=1):
                        e_label = gr.Dropdown(sorted(EXP), value=sorted(EXP)[0],
                                              label="experimental pattern (lab + opXRD)")
                        e_size = gr.Slider(5, 500, 60, step=5, label="crystallite size (nm)")
                        e_b = gr.Slider(0, 3, 0.5, step=0.1, label="B_iso (Å²)")
                        e_march = gr.Slider(0.5, 1.5, 1.0, step=0.01, label="March-Dollase r")
                        e_zero = gr.Slider(-0.2, 0.2, 0.0, step=0.005, label="zero-shift (deg)")
                        e_txt = gr.Markdown()
                    e_plot = gr.Plot(scale=2)
                e_in = [e_label, e_size, e_b, e_march, e_zero]
                for c in e_in:
                    c.change(_compare_exp, e_in, [e_plot, e_txt])
                demo.load(_compare_exp, e_in, [e_plot, e_txt])
    return demo


if __name__ == "__main__":
    build().launch()
