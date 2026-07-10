"""Global 2θ zero-shift (sample-displacement / detector-zero miscalibration).

A single offset applied to **all** peak centres — preserves relative spacings and
the systematic-absence pattern (a whole-pattern instrument effect, distinct from
per-peak position perturbation; DESIGN_DECISIONS §7). Applied in 2θ before the peak
profiles are built, so in log-d it becomes a smooth angle-dependent distortion.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def apply_zero_shift(two_theta: NDArray, delta_deg: float) -> NDArray:
    """Add a constant offset ``delta_deg`` to all 2θ peak positions."""
    return np.asarray(two_theta, dtype=np.float64) + float(delta_deg)
