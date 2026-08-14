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

import tempfile
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
from crystalai_simxrd.io.xy_writer import write_xy
from crystalai_simxrd.profiles.caglioti import (
    GENERIC_BRAGG_BRENTANO,
    SIMXRD_DEFAULT,
    STADI_P,
    InstrumentParameters,
)
from crystalai_simxrd.simulation.simulator import EffectConfig, simulate

warnings.filterwarnings("ignore")

_PKG = Path(__file__).resolve().parents[1]
_CIF_DIR = _PKG / "data" / "example_cifs"
# Top-level examples (flat) + the lab_practicum_26 set (own subfolder, kept distinct
# from tests/conftest.py's "disordered/" fixture folder, which is not for the
# dashboard's example picker).
_LAB_SUBDIR = "lab_practicum_26"
EXAMPLES = sorted(p.stem for p in _CIF_DIR.glob("*.cif")) + sorted(
    f"{_LAB_SUBDIR}/{p.stem}" for p in (_CIF_DIR / _LAB_SUBDIR).glob("*.cif")
)

# Caglioti instrument presets selectable in the dashboard, keyed by display name.
_CAGLIOTI_PRESETS: dict[str, InstrumentParameters] = {
    p.name: p for p in (SIMXRD_DEFAULT, GENERIC_BRAGG_BRENTANO, STADI_P)
}
_MARCH_AXES = ["(0,0,1)", "(1,0,0)", "(0,1,0)", "(1,1,0)", "(1,0,1)", "(0,1,1)", "(1,1,1)"]


def _structure(name: str) -> Structure:
    return load_cif(_CIF_DIR / f"{name}.cif")


def _parse_axis(label: str) -> tuple[int, int, int]:
    h, k, index_l = (int(v) for v in label.strip("()").split(","))
    return (h, k, index_l)


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
# Single-pattern control order shared by the live plot and the .xy export button.
_SINGLE_ARG_NAMES = (
    "name", "wavelength", "view",
    "cag_preset", "cag_U", "cag_V", "cag_W", "cag_X", "cag_Y",
    "dw_on", "dw_b",
    "sz_on", "sz_size",
    "ms_on", "ms_val",
    "po_on", "po_r", "po_axis",
    "zs_on", "zs_val",
    "ax_on", "ax_L", "ax_H", "ax_S",
    "sl_on", "sl_width",
    "bg_kind", "bg_amp",
    "pn_on", "pn_lambda",
    "gn_on", "gn_mean", "gn_std",
    "seed",
)


def _build_pattern(*args):
    """Shared build: dashboard control values -> (SimulatedPattern, structure_name)."""
    a = dict(zip(_SINGLE_ARG_NAMES, args, strict=True))
    st = _structure(a["name"])

    instrument = InstrumentParameters(
        U=a["cag_U"], V=a["cag_V"], W=a["cag_W"], X=a["cag_X"], Y=a["cag_Y"],
        name=a["cag_preset"],
    )
    eff = EffectConfig(
        b_iso=(a["dw_b"] if a["dw_on"] else 0.0),
        crystallite_size_nm=(a["sz_size"] if a["sz_on"] else np.inf),
        microstrain=(a["ms_val"] if a["ms_on"] else 0.0),
        march_r=(a["po_r"] if a["po_on"] else 1.0),
        march_axis=(_parse_axis(a["po_axis"]) if a["po_on"] else (0, 0, 1)),
        zero_shift_deg=(a["zs_val"] if a["zs_on"] else 0.0),
        axial_divergence=bool(a["ax_on"]),
        axial_L=a["ax_L"], axial_H=a["ax_H"], axial_S=a["ax_S"],
        slit_width_deg=(a["sl_width"] if a["sl_on"] else 0.0),
        background=(None if a["bg_kind"] == "none" else a["bg_kind"]),
        background_rel_amplitude=a["bg_amp"],
        poisson_lambda_max=(a["pn_lambda"] if a["pn_on"] else None),
        gaussian_noise_mean=(a["gn_mean"] if a["gn_on"] else 0.0),
        gaussian_noise_std=(a["gn_std"] if a["gn_on"] else 0.0),
        rng_seed=(int(a["seed"]) if a["seed"] is not None else None),
    )
    dom = Domain.TWO_THETA if a["view"] == "2θ" else Domain.LOG_D
    p = simulate(st, a["wavelength"], domain=dom, instrument=instrument, effects=eff)
    return p, a["name"]


def _simulate_single(*args):
    p, name = _build_pattern(*args)
    view = args[_SINGLE_ARG_NAMES.index("view")]
    wavelength = args[_SINGLE_ARG_NAMES.index("wavelength")]
    fig = go.Figure()
    fig.add_scatter(x=p.x_axis, y=p.intensity / max(p.intensity.max(), 1e-12),
                    mode="lines", line=dict(color="#d62728", width=1), name=name)
    fig.update_layout(
        title=f"{name}  ({p.metadata['n_peaks']} reflections, λ={wavelength:.4f} Å)",
        xaxis_title=("2θ (deg)" if view == "2θ" else "log₁₀(d / Å)"),
        yaxis_title="normalized intensity", height=460,
        margin=dict(l=50, r=20, t=50, b=45))
    return fig


def _caglioti_preset_values(preset_name: str):
    """Preset dropdown -> (U, V, W, X, Y) to fill the editable Caglioti boxes."""
    p = _CAGLIOTI_PRESETS.get(preset_name, SIMXRD_DEFAULT)
    return p.U, p.V, p.W, p.X, p.Y


def _save_xy(*args):
    """Build + crop the current pattern and write it to a headered .xy file."""
    crop_lo, crop_hi, scale_100 = args[-3], args[-2], args[-1]
    p, name = _build_pattern(*args[:-3])
    crop = None
    if crop_lo is not None and crop_hi is not None and float(crop_lo) < float(crop_hi):
        crop = (float(crop_lo), float(crop_hi))

    a = dict(zip(_SINGLE_ARG_NAMES, args[:-3], strict=True))
    instrument = InstrumentParameters(
        U=a["cag_U"], V=a["cag_V"], W=a["cag_W"], X=a["cag_X"], Y=a["cag_Y"],
        name=a["cag_preset"],
    )
    safe_name = name.replace("/", "_")
    suffix = "_scaled100" if scale_100 else ""
    out_dir = Path(tempfile.mkdtemp(prefix="crystalai_simxrd_"))
    out_path = out_dir / f"{safe_name}_{p.wavelength:.4f}A_{p.domain.value}{suffix}.xy"
    write_xy(
        out_path,
        x=p.x_axis, y=p.intensity, domain=p.domain, wavelength=p.wavelength,
        structure_name=name, n_peaks=p.metadata["n_peaks"],
        effects=p.metadata["effects"], instrument=instrument, crop=crop,
        scale_to_100=bool(scale_100),
    )
    return str(out_path)


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
            gr.Markdown(
                "Each physical effect below has an **on/off toggle** (the pristine "
                "Bragg profile when off) and its own parameter widgets — continuous "
                "parameters as number/slider inputs, discrete choices as dropdowns."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    s_name = gr.Dropdown(EXAMPLES, value=EXAMPLES[0], label="structure")
                    s_wl = gr.Slider(0.3, 2.5, 1.5406, step=0.0001, label="wavelength (Å)")
                    s_view = gr.Radio(["2θ", "log-d"], value="2θ", label="domain")

                    with gr.Accordion("Instrument profile (Caglioti)", open=True):
                        c_preset = gr.Dropdown(
                            list(_CAGLIOTI_PRESETS), value=SIMXRD_DEFAULT.name,
                            label="preset (fills U/V/W/X/Y below; still editable)",
                        )
                        with gr.Row():
                            c_U = gr.Number(SIMXRD_DEFAULT.U, label="U (Gaussian, deg²)")
                            c_V = gr.Number(SIMXRD_DEFAULT.V, label="V (Gaussian, deg²)")
                            c_W = gr.Number(SIMXRD_DEFAULT.W, label="W (Gaussian, deg²)")
                        with gr.Row():
                            c_X = gr.Number(SIMXRD_DEFAULT.X, label="X (Lorentzian, deg)")
                            c_Y = gr.Number(SIMXRD_DEFAULT.Y, label="Y (Lorentzian, deg)")

                    with gr.Accordion("Debye-Waller (thermal)", open=False):
                        dw_on = gr.Checkbox(False, label="enable")
                        dw_b = gr.Slider(0, 3, 0.8, step=0.1, label="B_iso (Å²)")

                    with gr.Accordion("Crystallite-size (Scherrer) broadening", open=False):
                        sz_on = gr.Checkbox(False, label="enable")
                        sz_size = gr.Slider(5, 500, 40, step=5, label="crystallite size (nm)")

                    with gr.Accordion("Microstrain (Williamson-Hall) broadening", open=False):
                        ms_on = gr.Checkbox(False, label="enable")
                        ms_val = gr.Slider(0, 0.01, 0.002, step=0.0005, label="microstrain")

                    with gr.Accordion("Preferred orientation (March-Dollase)", open=False):
                        po_on = gr.Checkbox(False, label="enable")
                        po_r = gr.Slider(0.5, 1.5, 0.75, step=0.01, label="March-Dollase r")
                        po_axis = gr.Dropdown(_MARCH_AXES, value="(0,0,1)", label="texture axis (hkl)")

                    with gr.Accordion("Zero-shift", open=False):
                        zs_on = gr.Checkbox(False, label="enable")
                        zs_val = gr.Slider(-0.1, 0.1, 0.02, step=0.005, label="zero-shift (deg)")

                    with gr.Accordion("Axial divergence", open=False):
                        ax_on = gr.Checkbox(False, label="enable")
                        with gr.Row():
                            ax_L = gr.Number(200.0, label="detector distance L (mm)")
                            ax_H = gr.Number(2.0, label="source/sample half-height H (mm)")
                            ax_S = gr.Number(2.0, label="receiving-slit half-height S (mm)")

                    with gr.Accordion("Receiving-slit broadening", open=False):
                        sl_on = gr.Checkbox(False, label="enable")
                        sl_width = gr.Slider(0, 0.5, 0.1, step=0.01, label="slit width (deg)")

                    with gr.Accordion("Background", open=False):
                        bg_kind = gr.Radio(["none", "residual", "physical"], value="none",
                                           label="background model")
                        bg_amp = gr.Slider(0, 0.2, 0.03, step=0.005,
                                          label="residual background rel. amplitude "
                                                "(used when model = residual)")

                    with gr.Accordion("Counting (Poisson) noise", open=False):
                        pn_on = gr.Checkbox(False, label="enable")
                        pn_lambda = gr.Slider(1, 10000, 1000, step=1, label="λmax (peak counts)")

                    with gr.Accordion("Full-profile Gaussian noise", open=False):
                        gr.Markdown(
                            "Imitates the ripple left in a pattern by *manual* background "
                            "subtraction at inference: the pattern is normalized to [0,1] "
                            "(in 2θ, even for d/log-d output — converted afterward), "
                            "N(mean, std) is added, then it is renormalized to [0,1]."
                        )
                        gn_on = gr.Checkbox(False, label="enable")
                        gn_mean = gr.Slider(-0.2, 0.2, 0.0, step=0.005, label="mean")
                        gn_std = gr.Slider(0, 0.1, 0.01, step=0.001, label="standard deviation")

                    seed = gr.Number(0, precision=0,
                                     label="rng seed (residual bg / counting / Gaussian noise)")

                s_plot = gr.Plot(scale=2)

            with gr.Accordion("Crop + save as .xy", open=False):
                gr.Markdown(
                    "Leave crop bounds blank to export the full simulated range. "
                    "The saved file header lists every effect enabled above with its "
                    "parameter values."
                )
                with gr.Row():
                    crop_lo = gr.Number(value=None, label="crop min (x-axis units)")
                    crop_hi = gr.Number(value=None, label="crop max (x-axis units)")
                scale_100 = gr.Checkbox(
                    False,
                    label="scale intensity to 0–100 (strongest line in the saved range = 100; "
                          "standard PXRD relative-intensity convention)",
                )
                save_btn = gr.Button("save cropped pattern as .xy", variant="primary")
                save_file = gr.File(label="download")

            inputs = [
                s_name, s_wl, s_view,
                c_preset, c_U, c_V, c_W, c_X, c_Y,
                dw_on, dw_b,
                sz_on, sz_size,
                ms_on, ms_val,
                po_on, po_r, po_axis,
                zs_on, zs_val,
                ax_on, ax_L, ax_H, ax_S,
                sl_on, sl_width,
                bg_kind, bg_amp,
                pn_on, pn_lambda,
                gn_on, gn_mean, gn_std,
                seed,
            ]
            # order must match _SINGLE_ARG_NAMES exactly — consumed positionally by _build_pattern
            for c in inputs:
                c.change(_simulate_single, inputs, s_plot)
            demo.load(_simulate_single, inputs, s_plot)

            c_preset.change(_caglioti_preset_values, c_preset, [c_U, c_V, c_W, c_X, c_Y])

            save_btn.click(_save_xy, inputs + [crop_lo, crop_hi, scale_100], save_file)

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
