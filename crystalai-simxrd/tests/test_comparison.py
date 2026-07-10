"""Phase 4: comparison metrics + sim-vs-experimental machinery."""

import numpy as np
import pytest

from crystalai_simxrd.comparison import metrics
from crystalai_simxrd.comparison.compare import (
    compare_structure_to_experimental,
    compare_structure_to_raw,
)
from crystalai_simxrd.comparison.overlay import overlay_figure
from crystalai_simxrd.core.crystal import load_cif
from crystalai_simxrd.core.domain import Domain
from crystalai_simxrd.simulation.simulator import EffectConfig, simulate


def test_metrics_basic():
    a = np.array([0.0, 1.0, 0.5, 0.0, 0.2])
    assert metrics.rwp(a, a) < 1e-9
    assert metrics.rp(a, a) < 1e-9
    assert abs(metrics.cosine_similarity(a, a) - 1.0) < 1e-9
    assert abs(metrics.cosine_similarity(np.array([1.0, 0]), np.array([0, 1.0]))) < 1e-9
    assert abs(metrics.best_scale(2 * a, a) - 2.0) < 1e-9


def test_peak_position_rmsd():
    obs = np.array([10.0, 20.0, 30.0])
    calc = np.array([10.05, 19.9, 30.1])
    r = metrics.peak_position_rmsd(obs, calc)
    assert 0.05 < r < 0.15


@pytest.mark.filterwarnings("ignore")
def test_self_consistency_is_perfect(cif_dir):
    """A simulated pattern compared to itself → Rwp≈0, cos≈1 (machinery correctness)."""
    st = load_cif(cif_dir / "Si.cif")
    eff = EffectConfig(crystallite_size_nm=80)
    sim2t = simulate(st, 1.5406, domain=Domain.TWO_THETA, effects=eff)
    res = compare_structure_to_experimental(st, sim2t.x_axis, sim2t.intensity, 1.5406, effects=eff)
    assert res["rwp"] < 0.5
    assert res["cosine"] > 0.999
    assert res["shift"] == 0


@pytest.mark.filterwarnings("ignore")
def test_compare_to_raw_recovers_background(cif_dir):
    """A synthetic raw = scale·sim + smooth background must be fit to Rwp≈0 (bg recovery)."""
    st = load_cif(cif_dir / "CeO2.cif")
    wl, eff = 1.5406, EffectConfig(crystallite_size_nm=60)
    sim2t = simulate(st, wl, domain=Domain.TWO_THETA, effects=eff)
    tt, calc = sim2t.x_axis, sim2t.intensity
    # synthetic raw = strong peaks + a smooth background (linear in 2θ, exactly Chebyshev-fit)
    bg = 300.0 + 4.0 * (tt - tt.mean())
    raw = 5000.0 * calc / calc.max() + bg
    res = compare_structure_to_raw(st, tt, raw, wl, effects=eff, n_bg=8)
    # 2θ-domain fit: an exact model must recover Rwp≈0 and R_bragg≈0 (no Jacobian artifact)
    assert res["rwp"] < 0.5
    assert res["r_bragg"] < 0.5
    assert res["cosine"] > 0.9999
    assert res["scale"] > 0


@pytest.mark.filterwarnings("ignore")
def test_compare_to_raw_gof_and_bragg_keys(cif_dir):
    """compare_structure_to_raw reports GoF = Rwp/floor and a valid Rietveld R_Bragg."""
    st = load_cif(cif_dir / "Si.cif")
    wl, eff = 1.5406, EffectConfig(crystallite_size_nm=80)
    sim2t = simulate(st, wl, domain=Domain.TWO_THETA, effects=eff)
    raw = 4000.0 * sim2t.intensity / sim2t.intensity.max() + 250.0
    res = compare_structure_to_raw(st, sim2t.x_axis, raw, wl, effects=eff)
    for k in ("gof", "rwp_noise_floor", "r_bragg", "two_theta", "background"):
        assert k in res
    assert res["rwp_noise_floor"] > 0
    assert abs(res["gof"] - res["rwp"] / res["rwp_noise_floor"]) < 1e-6
    assert res["r_bragg"] < 2.0                      # exact model → tiny integrated-I error


def test_rietveld_r_bragg_texture_detected(cif_dir):
    """A wrong reflection intensity (texture) raises R_Bragg; the exact list gives ~0."""
    from crystalai_simxrd.comparison.metrics import rietveld_r_bragg

    st = load_cif(cif_dir / "CeO2.cif")
    sim = simulate(st, 1.5406, domain=Domain.TWO_THETA,
                   effects=EffectConfig(crystallite_size_nm=60))
    tt, ints = sim.x_axis, np.asarray(sim.peak_intensities, float)
    # obs profile is built from the true intensities; the exact I list must score ~0
    base = rietveld_r_bragg(tt, sim.intensity, sim.peak_two_theta, ints,
                            sim.peak_fwhm_g, sim.peak_fwhm_l)
    # now pass a perturbed I list (strongest reflection tripled) → integrated-I mismatch
    perturbed = ints.copy()
    perturbed[int(np.argmax(ints))] *= 3.0
    r_texture = rietveld_r_bragg(tt, sim.intensity, sim.peak_two_theta, perturbed,
                                 sim.peak_fwhm_g, sim.peak_fwhm_l)
    assert base < 2.0
    assert r_texture > base + 5.0      # a genuine |F|² change is clearly flagged


def test_fit_scale_and_background_linear():
    from crystalai_simxrd.comparison.compare import fit_scale_and_background

    x = np.linspace(-1, 1, 200)
    calc = np.exp(-((x - 0.1) ** 2) / 0.001)      # a peak
    truth_bg = 10.0 + 3.0 * x
    obs = 5.0 * calc + truth_bg
    scale, bg, y = fit_scale_and_background(obs, calc, n_bg=4)
    assert abs(scale - 5.0) < 0.1
    assert np.allclose(bg, truth_bg, atol=0.5)


@pytest.mark.filterwarnings("ignore")
def test_overlay_figure_builds(cif_dir):
    st = load_cif(cif_dir / "CeO2.cif")
    sim2t = simulate(st, 1.5406, domain=Domain.TWO_THETA)
    res = compare_structure_to_experimental(st, sim2t.x_axis, sim2t.intensity, 1.5406)
    fig = overlay_figure(res["log_d"], res["exp"], res["calc"], metrics=res)
    assert len(fig.data) == 3  # exp, calc, difference
