"""Domain-aware grid construction.

`make_log_d_grid` is the production grid: **uniform in log₁₀(d)** over a fixed
d-window (DESIGN_DECISIONS §1). `make_two_theta_grid` builds the internal 2θ grid
the profile is assembled on and the 2θ validation grid.
"""

from __future__ import annotations

import numpy as np

from ..core.domain import Domain, DomainGrid


def make_two_theta_grid(
    start: float = 5.0, stop: float = 150.0, step: float = 0.01,
    wavelength: float | None = None,
) -> DomainGrid:
    """Uniform grid in 2θ (degrees)."""
    values = np.arange(start, stop + step * 0.5, step)
    return DomainGrid(values=values, domain=Domain.TWO_THETA, wavelength=wavelength)


def make_log_d_grid(
    d_min: float = 0.7, d_max: float = 18.0, n_bins: int = 12000,
) -> DomainGrid:
    """Uniform grid in log₁₀(d) over ``[d_min, d_max]`` Å — the production coordinate.

    Defaults chosen empirically (DESIGN_DECISIONS §1): ``d_max=18`` matches the
    experimental median reach (17.7 Å) and captures the first reflection of ~87% of
    ICSD (the rest kept as realistic windowed patterns); ``d_min=0.7`` sits below all
    but ~4% of experimental patterns; ``n_bins=12000`` matches the dominant 0.010° 2θ
    step of the real data (RRUFF/opXRD) — finer undersamples nothing, coarser discards
    real resolution. Constant Δ(log d): every peak gets the same bins-per-FWHM.
    Wavelength-agnostic (log-d positions are wavelength-independent).
    """
    values = np.linspace(np.log10(d_min), np.log10(d_max), n_bins)
    return DomainGrid(values=values, domain=Domain.LOG_D, wavelength=None)
