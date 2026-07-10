"""Zero-shift alignment between two patterns on a shared grid.

Experimental patterns carry a sample-displacement zero-offset; before comparing to a
simulation we estimate and remove the best integer-bin shift by cross-correlation.
Small (a few bins) shifts only — this is calibration, not peak fitting.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def best_shift(obs: NDArray, calc: NDArray, max_shift: int = 20) -> int:
    """Integer bin shift ``k`` (apply to ``calc``) maximizing correlation with ``obs``."""
    o = np.asarray(obs, float)
    c = np.asarray(calc, float)
    o = o - o.mean()
    c = c - c.mean()
    best_k, best_corr = 0, -np.inf
    for k in range(-max_shift, max_shift + 1):
        cs = np.roll(c, k)
        corr = float(np.dot(o, cs))
        if corr > best_corr:
            best_corr, best_k = corr, k
    return best_k


def apply_shift(pattern: NDArray, k: int) -> NDArray:
    """Shift a pattern by ``k`` bins (edges zero-filled, not wrapped)."""
    p = np.asarray(pattern, float)
    out = np.zeros_like(p)
    if k == 0:
        return p.copy()
    if k > 0:
        out[k:] = p[:-k]
    else:
        out[:k] = p[-k:]
    return out
