"""Simulated-vs-experimental comparison (SIMXRD_ROADMAP §7 criterion 6).

Rebins an experimental 2θ pattern onto the production log-d grid, simulates the known
structure at the same wavelength, aligns (zero-shift) and scales, and reports
agreement metrics. This is what the sim-vs-experimental dashboard tab and the
validation-against-`exp_subset` both call.

Two comparison modes:
  * :func:`compare_structure_to_experimental` — against a *background-subtracted* pattern,
    max-normalized. Good for shape agreement (cosine); its Rwp is a harsh diagnostic.
  * :func:`compare_structure_to_raw` — Rietveld convention (criterion #6 gate): against the
    *raw* counts, jointly fitting a smooth Chebyshev background **and** the scale by weighted
    least squares. The background counts inflate the Rwp denominator, so ``Rwp < 15`` is
    meaningful. This is the mode the "keep Rwp<15%, fit background on RAW" decision selected.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pymatgen.core import Structure

from ..core.bragg import BraggPeaks
from ..core.domain import Domain, two_theta_to_d
from ..profiles.convolver import resample_two_theta_to_log_d
from ..simulation.simulator import EffectConfig, simulate
from .alignment import apply_shift, best_shift
from .metrics import (
    best_scale,
    cosine_similarity,
    rietveld_r_bragg,
    rp,
    rwp,
    rwp_counts,
    rwp_noise_floor,
)


def rebin_experimental_to_log_d(
    two_theta: NDArray, intensity: NDArray, wavelength: float,
    d_window: tuple[float, float] = (0.7, 18.0), n_bins: int = 12000,
) -> tuple[NDArray, NDArray]:
    """Resample an experimental (2θ, I) pattern onto the fixed log₁₀(d) grid.

    Uses the **same Jacobian-corrected 2θ→log-d resample the simulator uses**, so the
    rebinned experimental pattern is directly comparable to a simulated one (a plain
    interpolation would differ by the Jacobian across the range).
    """
    tt = np.asarray(two_theta, float)
    y = np.asarray(intensity, float)
    order = np.argsort(tt)
    grid = np.linspace(np.log10(d_window[0]), np.log10(d_window[1]), n_bins)
    resampled = resample_two_theta_to_log_d(tt[order], y[order], grid, wavelength)
    return grid, resampled


def compare_structure_to_experimental(
    source: Structure | BraggPeaks,
    exp_two_theta: NDArray,
    exp_intensity: NDArray,
    wavelength: float,
    *,
    effects: EffectConfig | None = None,
    d_window: tuple[float, float] = (0.7, 18.0),
    n_bins: int = 12000,
    align: bool = True,
) -> dict:
    """Return agreement metrics over the experimentally-measured d-range.

    Keys: ``{log_d, exp, calc, mask, shift, scale, rwp, rp, cosine, rwp_counts}``.
    The plotted ``exp``/``calc`` are max-normalized over the full grid, but **all metrics
    are computed only over ``mask``** — the log-d bins the instrument actually scanned
    (from the experimental 2θ min/max). Scoring the whole [0.7,18] Å window would penalize
    every simulated reflection outside the measurement range against zero. ``rwp_counts`` is
    the Rietveld count-scale Rwp on the raw (un-normalized) rebinned experimental counts.
    """
    grid, exp_y = rebin_experimental_to_log_d(exp_two_theta, exp_intensity, wavelength, d_window, n_bins)
    sim = simulate(source, wavelength, domain=Domain.LOG_D, effects=effects,
                   d_window=d_window, n_log_d_bins=n_bins)

    # measured d-range from the experimental 2θ coverage (low 2θ → high d)
    tt = np.asarray(exp_two_theta, float)
    d_hi = two_theta_to_d(float(tt.min()), wavelength)
    d_lo = two_theta_to_d(float(tt.max()), wavelength)
    mask = ((grid >= np.log10(max(d_lo, d_window[0]))) &
            (grid <= np.log10(min(d_hi, d_window[1]))) & (exp_y > 0))

    exp_n = exp_y / max(float(exp_y.max()), 1e-12)
    calc = sim.intensity / max(float(sim.intensity.max()), 1e-12)
    k = best_shift(exp_n, calc) if align else 0
    calc = apply_shift(calc, k)
    s = best_scale(np.clip(exp_n[mask], 0, None), calc[mask])
    calc = s * calc

    rwp_c, _ = rwp_counts(exp_y[mask], sim.intensity[mask])
    return {
        "log_d": grid, "exp": exp_n, "calc": calc, "mask": mask,
        "shift": int(k), "scale": float(s),
        "rwp": rwp(np.clip(exp_n[mask], 1e-9, None), np.clip(calc[mask], 0, None)),
        "rp": rp(exp_n[mask], calc[mask]),
        "cosine": cosine_similarity(np.clip(exp_n[mask], 0, None), calc[mask]),
        "rwp_counts": rwp_c,
    }


def _chebyshev_basis(m: int, n_terms: int) -> NDArray:
    """``(n_terms, m)`` Chebyshev-T rows T₀…T_{n-1} on ``x ∈ [-1, 1]`` — a smooth
    background basis (the standard Rietveld background parameterization)."""
    x = np.linspace(-1.0, 1.0, m)
    t = np.empty((max(n_terms, 1), m))
    t[0] = 1.0
    if n_terms > 1:
        t[1] = x
    for j in range(2, n_terms):
        t[j] = 2.0 * x * t[j - 1] - t[j - 2]
    return t[:n_terms]


def fit_scale_and_background(
    obs: NDArray, calc: NDArray, n_bg: int = 8, weights: NDArray | None = None
) -> tuple[float, NDArray, NDArray]:
    """Weighted-LS fit of ``obs ≈ scale·calc + Σⱼ bⱼ·Tⱼ`` (scale + Chebyshev background).

    Poisson weights ``w = 1/max(obs, 1)`` by default. Returns ``(scale, background, y_calc)``
    where ``y_calc = scale·calc + background`` is the full model on the count scale.
    """
    o = np.clip(np.asarray(obs, float), 0, None)
    c = np.asarray(calc, float)
    w = 1.0 / np.clip(o, 1.0, None) if weights is None else np.asarray(weights, float)
    basis = _chebyshev_basis(o.size, n_bg)
    design = np.column_stack([c, *basis])           # columns: calc, T₀…T_{n_bg-1}
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(design * sw[:, None], o * sw, rcond=None)
    scale = float(coef[0])
    background = basis.T @ coef[1:]
    return scale, background, scale * c + background


def compare_structure_to_raw(
    source: Structure | BraggPeaks,
    raw_two_theta: NDArray,
    raw_intensity: NDArray,
    wavelength: float,
    *,
    effects: EffectConfig | None = None,
    n_bg: int = 8,
    two_theta_range: tuple[float, float] = (5.0, 150.0),
    internal_step: float = 0.01,
    align: bool = True,
) -> dict:
    """Rietveld-convention comparison against the **raw** pattern (criterion #6 gate).

    Done **in 2θ** — the domain where the background is physically smooth (a Chebyshev fit
    is exact there) and Rwp is conventionally defined. Fitting the background in log-d instead
    fights the resample Jacobian (a flat 2θ background becomes ``bg·J(log-d)``, unrepresentable
    by a log-d polynomial), which injects a spurious ~6% self-Rwp floor; 2θ removes that.

    Simulates the structure in 2θ, interpolates onto the experimental grid, then jointly fits
    scale + smooth Chebyshev background to the raw counts by weighted LS
    (``y_calc = scale·sim + background``). Rwp is the Poisson-weighted profile R with the
    background included — the convention under which ``Rwp < 15`` is meaningful.

    Keys: ``{two_theta, obs, calc, background, mask, shift, scale, rwp, rwp_noise_floor, gof,
    r_bragg, rp, cosine}`` (counts). ``r_bragg`` is the noise-insensitive integrated-intensity
    factor (bug detector: high ⇒ real structure-factor/position bug); ``rwp_noise_floor`` is the
    Rwp a perfect model would show under this pattern's counting noise, and
    ``gof = rwp / rwp_noise_floor`` (≈ √χ², → 1 when the fit is noise-limited).
    """
    tt_obs = np.asarray(raw_two_theta, float)
    y_obs = np.asarray(raw_intensity, float)
    order = np.argsort(tt_obs)
    tt_obs, y_obs = tt_obs[order], y_obs[order]

    sim = simulate(source, wavelength, domain=Domain.TWO_THETA, effects=effects,
                   two_theta_range=two_theta_range, internal_step=internal_step)
    calc = np.interp(tt_obs, sim.x_axis, sim.intensity, left=0.0, right=0.0)
    # coarse integer-bin pre-alignment (precise zero-shift is EffectConfig.zero_shift_deg)
    if align:
        k = best_shift(y_obs / max(float(y_obs.max()), 1e-12),
                       calc / max(float(calc.max()), 1e-12))
        calc = apply_shift(calc, k)
    else:
        k = 0
    # 2θ offset of that coarse shift, so the reflection centres used by R_Bragg stay aligned
    align_deg = k * float(np.median(np.diff(tt_obs))) if tt_obs.size > 1 else 0.0

    mask = (tt_obs >= sim.x_axis[0]) & (tt_obs <= sim.x_axis[-1]) & (y_obs > 0)
    om, cm = y_obs[mask], calc[mask]
    w = 1.0 / np.clip(om, 1.0, None)
    scale, bg, y_calc = fit_scale_and_background(om, cm, n_bg, weights=w)

    background = np.zeros_like(y_obs)
    background[mask] = bg
    calc_full = np.zeros_like(y_obs)
    calc_full[mask] = y_calc

    rwp_v = rwp(om, y_calc, weights=w)
    floor = rwp_noise_floor(y_calc, w, rng=np.random.default_rng(0))
    # Rietveld-partition R_Bragg over the measured range: apportion (obs − bg) among the
    # reflections by their calc shapes; I_calc,k on the count scale is scale·sim intensity.
    tt_m = tt_obs[mask]
    r_bragg_v = rietveld_r_bragg(
        tt_m, om - bg, np.asarray(sim.peak_two_theta) + align_deg,
        scale * np.asarray(sim.peak_intensities),
        np.asarray(sim.peak_fwhm_g), np.asarray(sim.peak_fwhm_l),
    )
    return {
        "two_theta": tt_obs, "obs": y_obs, "calc": calc_full, "background": background,
        "mask": mask, "shift": int(k), "scale": float(scale),
        "rwp": rwp_v,
        "rwp_noise_floor": floor,
        "gof": float(rwp_v / floor) if floor > 0 else float("inf"),
        "r_bragg": r_bragg_v,
        "rp": rp(om, y_calc),
        "cosine": cosine_similarity(om, y_calc),
    }
