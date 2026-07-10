"""Peak profile functions — Gaussian, Lorentzian, pseudo-Voigt, Thompson-Cox-Hastings PV.

All **area-normalized** (∫ profile dx = 1) on a generic x-axis, so a peak's height is
set by its integrated intensity (|F|²·LP·…), not chosen. TCH-PV is the production core
(peak widths from Caglioti Gaussian + size/strain Lorentzian, combined by `tch_mix`).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

_FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def gaussian(x: NDArray, center: float, fwhm: float) -> NDArray:
    """Area-normalized Gaussian."""
    sigma = fwhm * _FWHM_TO_SIGMA
    return np.exp(-0.5 * ((x - center) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))


def lorentzian(x: NDArray, center: float, fwhm: float) -> NDArray:
    """Area-normalized Lorentzian (Cauchy)."""
    gamma = fwhm / 2.0
    return (gamma / np.pi) / ((x - center) ** 2 + gamma**2)


def pseudo_voigt(x: NDArray, center: float, fwhm: float, eta: float) -> NDArray:
    """Pseudo-Voigt ``η·L + (1-η)·G`` (shared FWHM), area-normalized."""
    return eta * lorentzian(x, center, fwhm) + (1.0 - eta) * gaussian(x, center, fwhm)


def tch_mix(fwhm_g: float, fwhm_l: float) -> tuple[float, float]:
    """Thompson-Cox-Hastings combination of Gaussian + Lorentzian widths → (fwhm, η).

    (Thompson, Cox & Hastings, J. Appl. Cryst. 20, 79 (1987).)
    """
    fg, fl = fwhm_g, fwhm_l
    fwhm = (
        fg**5
        + 2.69269 * fg**4 * fl
        + 2.42843 * fg**3 * fl**2
        + 4.47163 * fg**2 * fl**3
        + 0.07842 * fg * fl**4
        + fl**5
    ) ** 0.2
    if fwhm < 1e-15:
        return 0.0, 0.0
    r = fl / fwhm
    eta = min(1.0, max(0.0, 1.36603 * r - 0.47719 * r**2 + 0.11116 * r**3))
    return fwhm, eta


def thompson_cox_hastings_pseudo_voigt(
    x: NDArray, center: float, fwhm_g: float, fwhm_l: float
) -> NDArray:
    """TCH pseudo-Voigt from separate Gaussian and Lorentzian FWHM."""
    fwhm, eta = tch_mix(fwhm_g, fwhm_l)
    return pseudo_voigt(x, center, fwhm, eta)
