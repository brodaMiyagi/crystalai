"""Lorentz-polarization factor for Bragg-Brentano (CW) geometry.

``LP(2θ) = (1 + cos²2θ) / (sin²θ · cosθ)`` — a 2θ-domain quantity, so it is applied
after mapping each reflection's d to 2θ at the working wavelength, before the profile
is built (SIMXRD_ROADMAP §5 order).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def lorentz_polarization_factor(two_theta: NDArray | float) -> NDArray | float:
    """LP factor for Bragg-Brentano geometry. ``two_theta`` in degrees."""
    tt = np.deg2rad(np.asarray(two_theta, dtype=np.float64))
    theta = tt / 2.0
    return (1.0 + np.cos(tt) ** 2) / (np.sin(theta) ** 2 * np.cos(theta))


def lorentz_polarization_factor_monochromator(
    two_theta: NDArray | float, two_theta_mono: float = 26.6
) -> NDArray | float:
    """LP factor including a crystal monochromator (default graphite 002, 2θ_m=26.6°)."""
    tt = np.deg2rad(np.asarray(two_theta, dtype=np.float64))
    theta = tt / 2.0
    cos2_mono = np.cos(np.deg2rad(two_theta_mono)) ** 2
    return (1.0 + cos2_mono * np.cos(tt) ** 2) / (np.sin(theta) ** 2 * np.cos(theta))
