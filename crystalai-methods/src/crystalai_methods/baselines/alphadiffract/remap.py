"""Remap an experimental PXRD pattern onto OpenAlphaDiffract's input grid.

The model (see ``vendor/model.py``) takes 8192 intensities on a grid that is
**equally spaced in 2θ over [5°, 20°] at monochromatic 20 keV**
(λ = 12.39842/20 = 0.61992 Å). It has no wavelength input, so a pattern collected
at a *different* wavelength must first be moved onto that 20 keV 2θ scale.

We do this per data point via Bragg's law (wavelength-invariant d-spacing):

    d = λ_src / (2 sin θ_src)          # our reflection's d
    sin θ_20 = λ_20 / (2 d)            # same reflection's angle at 20 keV
             = (λ_20 / λ_src) sin θ_src

then linearly resample the intensity onto the model's fixed 2θ grid and normalize
to [0, 100]. For genuine 20 keV data this is the identity. NumPy only (no torch
in the remap math).
"""

from __future__ import annotations

import numpy as np

# ---- Model input contract (from the model card / how training data was made) ----
# λ[Å] = 12.39842 / E[keV]; the model was trained at E = 20 keV monochromatic.
PHOTON_ENERGY_KEV = 20.0
LAMBDA_MODEL = 12.39842 / PHOTON_ENERGY_KEV  # 0.619921 Å
TWO_THETA_MIN = 5.0   # degrees
TWO_THETA_MAX = 20.0  # degrees
N_POINTS = 8192


def model_two_theta_grid() -> np.ndarray:
    """The model's input abscissa: 8192 points equally spaced in 2θ [5°, 20°]."""
    return np.linspace(TWO_THETA_MIN, TWO_THETA_MAX, N_POINTS)


def two_theta_to_d(two_theta_deg: np.ndarray, wavelength: float) -> np.ndarray:
    """Bragg's law: d = λ / (2 sin θ), θ = (2θ)/2. Returns d in Å."""
    theta = np.deg2rad(np.asarray(two_theta_deg, dtype=np.float64)) / 2.0
    return wavelength / (2.0 * np.sin(theta))


def model_d_grid() -> np.ndarray:
    """Physical d-spacings (Å) of the model's bins — for reporting the window.

    Monotonic decreasing with 2θ; spans d ∈ [1.785, 7.106] Å for [5°,20°]@20 keV.
    """
    return two_theta_to_d(model_two_theta_grid(), LAMBDA_MODEL)


def to_two_theta_20kev(two_theta_deg: np.ndarray, wavelength_src: float) -> np.ndarray:
    """Map source 2θ (at ``wavelength_src``) to the equivalent 2θ at 20 keV.

    sin θ_20 = (λ_20 / λ_src) sin θ_src. Reflections with no real 20 keV angle
    (d < λ_20/2) map to NaN and are dropped by the caller.
    """
    theta = np.deg2rad(np.asarray(two_theta_deg, dtype=np.float64)) / 2.0
    s = (LAMBDA_MODEL / wavelength_src) * np.sin(theta)
    s = np.where(s <= 1.0, s, np.nan)
    return np.rad2deg(2.0 * np.arcsin(s))


# Precompute once — the target grid never changes.
_TT_MODEL = model_two_theta_grid()


def remap_pattern(
    two_theta_deg: np.ndarray,
    intensity: np.ndarray,
    wavelength: float,
) -> tuple[np.ndarray, float]:
    """Resample ``(2θ, intensity)`` collected at ``wavelength`` onto the model grid.

    Returns ``(x, coverage)`` where ``x`` is a float32 array of shape (8192,)
    normalized to [0, 100] on the model's 2θ [5°,20°]@20 keV grid, and
    ``coverage`` is the fraction of that grid actually spanned by the source
    pattern (bins outside the measured range are zero-filled).
    """
    tt = np.asarray(two_theta_deg, dtype=np.float64)
    y = np.asarray(intensity, dtype=np.float64)
    if tt.shape != y.shape or tt.ndim != 1:
        raise ValueError("two_theta and intensity must be 1-D arrays of equal length")

    # Move every measured point onto the 20 keV 2θ scale, drop non-physical maps.
    tt20 = to_two_theta_20kev(tt, wavelength)
    ok = np.isfinite(tt20)
    tt20, y = tt20[ok], y[ok]

    # Sort ascending in the 20 keV 2θ coordinate (np.interp needs increasing x).
    order = np.argsort(tt20)
    tt20_sorted = tt20[order]
    y_sorted = y[order]

    # Linear resample onto the fixed model grid; zero-fill outside the range.
    x = np.interp(_TT_MODEL, tt20_sorted, y_sorted, left=0.0, right=0.0)

    lo, hi = float(tt20_sorted[0]), float(tt20_sorted[-1])
    coverage = float(((_TT_MODEL >= lo) & (_TT_MODEL <= hi)).mean())

    # Preprocessing (model card): floor negatives at zero, then rescale to [0,100].
    x = np.clip(x, 0.0, None)
    span = x.max() - x.min()
    x = (x - x.min()) / (span + 1e-10) * 100.0

    return x.astype(np.float32), coverage
