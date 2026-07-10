"""Pattern normalization (SIMXRD_ROADMAP Phase 5.3).

``max`` (peak height → 1), ``area`` (integral → 1), or ``sqrt`` (variance-stabilizing
for Poisson-like counts, then max-normalized). Applied after the Poisson step and
before the relative-Gaussian noise (SIMXRD_ROADMAP §5 order).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def normalize(pattern: NDArray, method: str = "max") -> NDArray:
    """Normalize a pattern by ``method`` ∈ {'max', 'area', 'sqrt'}."""
    p = np.asarray(pattern, dtype=np.float64)
    if method == "max":
        m = np.max(p)
        return p / m if m > 0 else p
    if method == "area":
        a = np.sum(np.clip(p, 0.0, None))
        return p / a if a > 0 else p
    if method == "sqrt":
        q = np.sqrt(np.clip(p, 0.0, None))
        m = np.max(q)
        return q / m if m > 0 else q
    raise ValueError(f"unknown normalization method: {method!r}")
