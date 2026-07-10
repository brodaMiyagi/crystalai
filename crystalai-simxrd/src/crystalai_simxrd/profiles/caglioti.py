"""Caglioti instrumental broadening + Lorentzian sample broadening.

Gaussian FWHM (2θ): ``FWHM_G² = U·tan²θ + V·tanθ + W``.
Lorentzian FWHM (2θ): ``FWHM_L = X·tanθ + Y/cosθ``.
These feed the Thompson-Cox-Hastings pseudo-Voigt (peak_shapes). Randomizing
`(U,V,W)` over the presets below spans sharp synchrotron → broad lab instruments.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class InstrumentParameters:
    """Caglioti Gaussian (U, V, W) + Lorentzian (X, Y) coefficients (degrees)."""

    U: float
    V: float
    W: float
    X: float = 0.0
    Y: float = 0.0
    name: str = ""


GENERIC_BRAGG_BRENTANO = InstrumentParameters(0.0143, -0.0104, 0.0097, name="Generic Bragg-Brentano")
STADI_P = InstrumentParameters(0.0060, -0.0045, 0.0040, name="STOE STADI-P (Debye-Scherrer)")
SIMXRD_DEFAULT = InstrumentParameters(0.0100, -0.0080, 0.0080, name="SimXRD default")


def caglioti_fwhm_g(two_theta: NDArray | float, U: float, V: float, W: float) -> NDArray:
    """Gaussian FWHM (deg) = √(U·tan²θ + V·tanθ + W), clamped ≥ 0."""
    tan_theta = np.tan(np.deg2rad(np.asarray(two_theta, dtype=np.float64) / 2.0))
    return np.sqrt(np.maximum(U * tan_theta**2 + V * tan_theta + W, 0.0))


def lorentzian_fwhm(two_theta: NDArray | float, X: float, Y: float) -> NDArray:
    """Lorentzian FWHM (deg) = X·tanθ + Y/cosθ."""
    theta = np.deg2rad(np.asarray(two_theta, dtype=np.float64) / 2.0)
    return X * np.tan(theta) + Y / np.cos(theta)


def instrument_fwhm(
    two_theta: NDArray | float, params: InstrumentParameters
) -> tuple[NDArray, NDArray]:
    """Return ``(fwhm_g, fwhm_l)`` (deg) at each 2θ for an instrument preset."""
    return (
        caglioti_fwhm_g(two_theta, params.U, params.V, params.W),
        lorentzian_fwhm(two_theta, params.X, params.Y),
    )
