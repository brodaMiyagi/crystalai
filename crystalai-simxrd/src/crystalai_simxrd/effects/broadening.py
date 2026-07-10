"""Crystallite-size and microstrain broadening.

Size (Scherrer, Lorentzian): ``β_L = Kλ / (D cosθ)``.
Strain (Williamson-Hall, Gaussian): ``β_G = 4ε tanθ``.
Returned separately so they combine with the instrumental Caglioti widths —
Gaussian in quadrature, Lorentzian linearly — before the TCH pseudo-Voigt.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def scherrer_fwhm(
    two_theta: NDArray, wavelength: float, crystallite_size_nm: float, k: float = 0.9
) -> NDArray:
    """Lorentzian FWHM (deg) from finite crystallite size ``D`` (nm). ∞ → 0."""
    tt = np.asarray(two_theta, dtype=np.float64)
    if crystallite_size_nm <= 0 or not np.isfinite(crystallite_size_nm):
        return np.zeros_like(tt)
    cos_theta = np.clip(np.cos(np.deg2rad(tt / 2.0)), 1e-15, None)
    return np.rad2deg(k * wavelength / (crystallite_size_nm * 10.0 * cos_theta))


def strain_fwhm(two_theta: NDArray, microstrain: float) -> NDArray:
    """Gaussian FWHM (deg) from isotropic microstrain ε (dimensionless)."""
    tt = np.asarray(two_theta, dtype=np.float64)
    if microstrain <= 0:
        return np.zeros_like(tt)
    return np.rad2deg(4.0 * microstrain * np.tan(np.deg2rad(tt / 2.0)))


def size_strain_fwhm(
    two_theta: NDArray, wavelength: float,
    crystallite_size_nm: float = np.inf, microstrain: float = 0.0, k: float = 0.9,
) -> tuple[NDArray, NDArray]:
    """Return ``(extra_gaussian, extra_lorentzian)`` FWHM (deg): strain, size."""
    return (
        strain_fwhm(two_theta, microstrain),
        scherrer_fwhm(two_theta, wavelength, crystallite_size_nm, k),
    )
