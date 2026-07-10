"""Coordinate domains and transforms: 2θ ↔ d ↔ log₁₀(d), with Jacobians.

`log(d)` is the production/training coordinate (DESIGN_DECISIONS §1); 2θ is the
instrument's native domain where profiles are built (§1a); d is the intermediate.
`DomainGrid` tags a 1-D array with its domain + optional wavelength.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray


class Domain(Enum):
    TWO_THETA = "two_theta"
    D_SPACING = "d_spacing"
    LOG_D = "log_d"  # log10(d / Å) — production/training coordinate


# --------------------------------------------------------------------------- #
# Coordinate transforms (angles in degrees, d in Å, λ in Å)
# --------------------------------------------------------------------------- #
def two_theta_to_d(two_theta: NDArray | float, wavelength: float) -> NDArray | float:
    """2θ (deg) → d (Å), Bragg."""
    theta_rad = np.deg2rad(np.asarray(two_theta, dtype=np.float64) / 2.0)
    return wavelength / (2.0 * np.sin(theta_rad))


def d_to_two_theta(d: NDArray | float, wavelength: float) -> NDArray | float:
    """d (Å) → 2θ (deg), Bragg (sinθ clipped to [-1, 1])."""
    d_arr = np.asarray(d, dtype=np.float64)
    sin_theta = np.clip(wavelength / (2.0 * d_arr), -1.0, 1.0)
    return np.rad2deg(2.0 * np.arcsin(sin_theta))


def d_to_log_d(d: NDArray | float) -> NDArray | float:
    """d (Å) → log₁₀(d). Wavelength-independent."""
    return np.log10(np.asarray(d, dtype=np.float64))


def log_d_to_d(log_d: NDArray | float) -> NDArray | float:
    """log₁₀(d) → d (Å)."""
    return np.power(10.0, np.asarray(log_d, dtype=np.float64))


def two_theta_to_log_d(two_theta: NDArray | float, wavelength: float) -> NDArray | float:
    """2θ (deg) → log₁₀(d)."""
    return d_to_log_d(two_theta_to_d(two_theta, wavelength))


def log_d_to_two_theta(log_d: NDArray | float, wavelength: float) -> NDArray | float:
    """log₁₀(d) → 2θ (deg)."""
    return d_to_two_theta(log_d_to_d(log_d), wavelength)


# --------------------------------------------------------------------------- #
# Jacobians for intensity-conserving resampling: I_target = I_source · |dx_src/dx_tgt|
# --------------------------------------------------------------------------- #
def jacobian_d_per_two_theta(two_theta: NDArray | float, wavelength: float) -> NDArray | float:
    """|dd/d(2θ)| (per degree) — for 2θ → d intensity conservation."""
    theta_rad = np.deg2rad(np.asarray(two_theta, dtype=np.float64)) / 2.0
    # d = λ/(2 sinθ); dd/d(2θ) = -λ cosθ / (4 sin²θ). Per-degree factor folded via deg2rad.
    return np.deg2rad(wavelength * np.cos(theta_rad) / (4.0 * np.sin(theta_rad) ** 2))


def jacobian_logd_per_two_theta(two_theta: NDArray | float, wavelength: float) -> NDArray | float:
    """|d(log₁₀ d)/d(2θ)| (per degree) — for 2θ → log-d intensity conservation.

    ``d(log₁₀ d)/d(2θ) = (1/(d ln10)) · dd/d(2θ)``.
    """
    d = two_theta_to_d(two_theta, wavelength)
    return jacobian_d_per_two_theta(two_theta, wavelength) / (d * np.log(10.0))


# --------------------------------------------------------------------------- #
# DomainGrid
# --------------------------------------------------------------------------- #
@dataclass
class DomainGrid:
    """A 1-D grid tagged with its `Domain` and optional wavelength (Å)."""

    values: NDArray
    domain: Domain
    wavelength: float | None = None

    def __len__(self) -> int:
        return len(self.values)

    @property
    def step(self) -> float:
        """Mean step (assumes ~uniform spacing in this domain)."""
        return float(np.mean(np.diff(self.values)))

    def to_domain(self, target: Domain) -> "DomainGrid":
        """Return this grid's values in ``target`` domain (wavelength required for 2θ)."""
        if target == self.domain:
            return self
        # go via d-spacing
        if self.domain == Domain.TWO_THETA:
            if self.wavelength is None:
                raise ValueError("wavelength required to convert from 2θ")
            d = two_theta_to_d(self.values, self.wavelength)
        elif self.domain == Domain.D_SPACING:
            d = np.asarray(self.values, dtype=np.float64)
        else:  # LOG_D
            d = log_d_to_d(self.values)

        if target == Domain.D_SPACING:
            out = d
        elif target == Domain.LOG_D:
            out = d_to_log_d(d)
        else:  # TWO_THETA
            if self.wavelength is None:
                raise ValueError("wavelength required to convert to 2θ")
            out = d_to_two_theta(d, self.wavelength)
        return DomainGrid(values=out, domain=target, wavelength=self.wavelength)
