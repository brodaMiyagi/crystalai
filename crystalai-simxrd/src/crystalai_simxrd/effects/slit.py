"""Receiving-slit (top-hat) instrumental broadening.

Convolves the 2θ pattern with a rectangular slit of a given angular width. Optional:
Caglioti already carries instrumental Gaussian broadening, so enable the explicit
slit only when not double-counting (SIMXRD_ROADMAP §2.5).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def apply_slit(pattern: NDArray, step_deg: float, width_deg: float) -> NDArray:
    """Convolve a uniform-step 2θ pattern with a normalized top-hat of ``width_deg``."""
    if width_deg <= 0 or step_deg <= 0:
        return np.asarray(pattern, dtype=np.float64)
    n = max(1, int(round(width_deg / step_deg)))
    if n <= 1:
        return np.asarray(pattern, dtype=np.float64)
    kernel = np.ones(n) / n
    return np.convolve(np.asarray(pattern, dtype=np.float64), kernel, mode="same")
