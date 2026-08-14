"""Remap an experimental PXRD pattern onto PXRDGen's input grid.

PXRDGen's XRD encoder takes **7500 intensities equally spaced in 2θ over
[5°, 80°) at a step of 0.01°** (the training arrays hold 7501 points on
``arange(5, 80.01, 0.01)`` and the loader drops the last, leaving 5.00…79.99;
the shipped ``*_XRD.dat`` samples are already the 7500-long form). Patterns are
normalized by the maximum peak intensity (max → 1). The encoder has **no
wavelength input**: MP-20 training patterns were simulated with Cu Kα radiation,
so a pattern collected at a *different* wavelength must first be moved onto the
Cu Kα1 2θ scale before it is comparable.

We do this per data point via Bragg's law (wavelength-invariant d-spacing):

    d = λ_src / (2 sin θ_src)              # our reflection's d
    sin θ_cu = λ_cu / (2 d) = (λ_cu / λ_src) sin θ_src

then linearly resample the intensity onto the fixed 2θ grid. For genuine Cu Kα1
data this is the identity. NumPy only (no torch in the remap math). This mirrors
``baselines/alphadiffract/remap.py`` but with PXRDGen's grid / normalization.
"""

from __future__ import annotations

import numpy as np

# ---- Model input contract (Methods: "2θ range 5°-80°, step 0.01°", Cu Kα) ----
LAMBDA_CU_KA1 = 1.540598  # Å, Cu Kα1 (the peak-position reference line)
TWO_THETA_MIN = 5.0       # degrees, inclusive
TWO_THETA_STEP = 0.01     # degrees
N_POINTS = 7500           # -> covers [5.00, 79.99]


def model_two_theta_grid() -> np.ndarray:
    """The model's input abscissa: 7500 points at 5.00, 5.01, …, 79.99°."""
    return TWO_THETA_MIN + TWO_THETA_STEP * np.arange(N_POINTS, dtype=np.float64)


def to_two_theta_cu(two_theta_deg: np.ndarray, wavelength_src: float) -> np.ndarray:
    """Map source 2θ (at ``wavelength_src``) to the equivalent 2θ at Cu Kα1.

    sin θ_cu = (λ_cu / λ_src) sin θ_src. Reflections with no real Cu Kα1 angle
    (d < λ_cu/2, i.e. sin θ_cu > 1) map to NaN and are dropped by the caller.
    """
    theta = np.deg2rad(np.asarray(two_theta_deg, dtype=np.float64)) / 2.0
    s = (LAMBDA_CU_KA1 / wavelength_src) * np.sin(theta)
    s = np.where(s <= 1.0, s, np.nan)
    return np.rad2deg(2.0 * np.arcsin(s))


# Precompute once — the target grid never changes.
_TT_MODEL = model_two_theta_grid()


def remap_pattern(
    two_theta_deg: np.ndarray,
    intensity: np.ndarray,
    wavelength: float,
) -> tuple[np.ndarray, float]:
    """Resample ``(2θ, intensity)`` collected at ``wavelength`` onto PXRDGen's grid.

    Returns ``(x, coverage)`` where ``x`` is a float32 array of shape (7500,)
    normalized so its maximum is 1 on the Cu Kα1 2θ [5°,80°) grid, and
    ``coverage`` is the fraction of that grid actually spanned by the source
    pattern (bins outside the measured range are zero-filled). Negative
    intensities (from background subtraction) are floored at zero first.
    """
    tt = np.asarray(two_theta_deg, dtype=np.float64)
    y = np.asarray(intensity, dtype=np.float64)
    if tt.shape != y.shape or tt.ndim != 1:
        raise ValueError("two_theta and intensity must be 1-D arrays of equal length")

    # Move every measured point onto the Cu Kα1 2θ scale, drop non-physical maps.
    ttc = to_two_theta_cu(tt, wavelength)
    ok = np.isfinite(ttc)
    ttc, y = ttc[ok], y[ok]
    if ttc.size == 0:
        return np.zeros(N_POINTS, dtype=np.float32), 0.0

    # Sort ascending in the Cu 2θ coordinate (np.interp needs increasing x).
    order = np.argsort(ttc)
    ttc_sorted = ttc[order]
    y_sorted = y[order]

    # Linear resample onto the fixed model grid; zero-fill outside the range.
    x = np.interp(_TT_MODEL, ttc_sorted, y_sorted, left=0.0, right=0.0)

    lo, hi = float(ttc_sorted[0]), float(ttc_sorted[-1])
    coverage = float(((_TT_MODEL >= lo) & (_TT_MODEL <= hi)).mean())

    # Preprocessing (Methods): floor negatives at zero, normalize by max peak.
    x = np.clip(x, 0.0, None)
    peak = x.max()
    if peak > 0:
        x = x / peak

    return x.astype(np.float32), coverage
