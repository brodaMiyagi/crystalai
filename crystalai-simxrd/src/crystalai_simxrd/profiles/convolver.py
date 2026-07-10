"""Profile convolution: assemble the pattern in 2θ, then resample to log-d.

Production path (DESIGN_DECISIONS §1a, SIMXRD_ROADMAP §5): each reflection is a
TCH pseudo-Voigt on a fine 2θ grid (Caglioti Gaussian + size/strain Lorentzian);
the full 2θ pattern is assembled there (so 2θ-domain effects — background, counting
noise, later — act on it), then converted **once** to the uniform log-d grid with the
Jacobian intensity correction. Peak *positions* map exactly; peak *areas* are
preserved; widths become ≈ uniform in log-d.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..core.domain import jacobian_logd_per_two_theta
from .peak_shapes import pseudo_voigt, tch_mix


def convolve_two_theta(
    centers: NDArray,
    intensities: NDArray,
    fwhm_g: NDArray,
    fwhm_l: NDArray,
    two_theta_grid: NDArray,
    *,
    n_fwhm: float = 12.0,
) -> NDArray:
    """Assemble the 2θ pattern: each peak a TCH pseudo-Voigt, summed onto the grid.

    ``fwhm_g``/``fwhm_l`` are the per-peak Gaussian/Lorentzian FWHM (deg). Each profile
    is evaluated only within ±``n_fwhm`` FWHM of its centre for speed.
    """
    x = np.asarray(two_theta_grid, dtype=np.float64)
    pattern = np.zeros_like(x)
    if len(centers) == 0:
        return pattern

    lo, hi, n = x[0], x[-1], len(x)
    inv_span = (n - 1) / (hi - lo) if hi > lo else 0.0
    for c, inten, fg, fl in zip(centers, intensities, fwhm_g, fwhm_l):
        fwhm, eta = tch_mix(max(float(fg), 1e-6), float(fl))
        if fwhm < 1e-6:
            fwhm, eta = max(float(fg), 1e-6), 0.0
        half = n_fwhm * fwhm
        i0 = max(0, int(np.floor((c - half - lo) * inv_span)))
        i1 = min(n, int(np.ceil((c + half - lo) * inv_span)) + 1)
        if i1 <= i0:
            continue
        seg = x[i0:i1]
        pattern[i0:i1] += inten * pseudo_voigt(seg, float(c), fwhm, eta)
    return pattern


def resample_two_theta_to_log_d(
    two_theta_grid: NDArray,
    pattern_two_theta: NDArray,
    log_d_values: NDArray,
    wavelength: float,
) -> NDArray:
    """Resample a 2θ pattern (per-degree density) onto a log₁₀(d) grid, area-preserving.

    ``I_logd = I_2θ · |d(2θ)/d(log d)|`` — the Jacobian keeps each peak's integrated
    intensity invariant across the coordinate change. Log-d bins whose d is
    inaccessible at this wavelength (sinθ > 1) or fall outside the 2θ grid are zero.
    """
    log_d_values = np.asarray(log_d_values, dtype=np.float64)
    d = np.power(10.0, log_d_values)
    sin_theta = wavelength / (2.0 * d)
    tt = np.full_like(log_d_values, np.nan)
    accessible = sin_theta <= 1.0
    tt[accessible] = np.rad2deg(2.0 * np.arcsin(sin_theta[accessible]))

    tg = np.asarray(two_theta_grid, dtype=np.float64)
    inrange = accessible & (tt >= tg[0]) & (tt <= tg[-1])

    out = np.zeros_like(log_d_values)
    if not np.any(inrange):
        return out
    interp = np.interp(tt[inrange], tg, pattern_two_theta)
    jac_logd = jacobian_logd_per_two_theta(tt[inrange], wavelength)  # |d log d / d 2θ|
    out[inrange] = interp / np.where(jac_logd > 1e-30, jac_logd, 1e-30)
    return out


def resample_two_theta_to_d(
    two_theta_grid: NDArray,
    pattern_two_theta: NDArray,
    d_values: NDArray,
    wavelength: float,
) -> NDArray:
    """Resample a 2θ pattern onto a linear-d grid (validation helper), area-preserving."""
    log_d = np.log10(np.asarray(d_values, dtype=np.float64))
    # |d(2θ)/dd| = |d(2θ)/d log d| · |d log d / dd| = (1/jac_logd) · 1/(d ln10)
    out = resample_two_theta_to_log_d(two_theta_grid, pattern_two_theta, log_d, wavelength)
    return out / (np.asarray(d_values, dtype=np.float64) * np.log(10.0))
