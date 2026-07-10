"""March-Dollase preferred-orientation correction.

``P(α) = (r²cos²α + sin²α / r)^(-3/2)`` where α is the angle between each
reflection's plane normal and the texture axis, and r the March coefficient
(r=1 → random powder; r<1 → platelet; r>1 → needle). Uses the direct-lattice
metric to get α from `hkl`, so it works from a cached peak list (which carries the
metric and dominant `hkl`).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def march_dollase_factor(alpha: NDArray, r: float) -> NDArray:
    """March-Dollase factor at angle(s) ``alpha`` (radians)."""
    a = np.asarray(alpha, dtype=np.float64)
    return (r**2 * np.cos(a) ** 2 + np.sin(a) ** 2 / r) ** (-1.5)


def preferred_orientation_correction(
    hkls: list[tuple[int, int, int]],
    metric_tensor: NDArray,
    r: float = 1.0,
    axis: tuple[int, int, int] = (0, 0, 1),
) -> NDArray:
    """Per-reflection intensity correction for uniaxial texture along ``axis``.

    Returns all-ones for r≈1 (random powder). ``metric_tensor`` is the direct
    lattice metric (``BraggPeaks.metric_tensor``).
    """
    n = len(hkls)
    if abs(r - 1.0) < 1e-10 or metric_tensor is None or n == 0:
        return np.ones(n, dtype=np.float64)

    gstar = np.linalg.inv(np.asarray(metric_tensor, dtype=np.float64))  # reciprocal metric
    ax = np.array(axis, dtype=np.float64)          # texture PLANE (hkl) → reciprocal vector
    ax_norm = np.sqrt(ax @ gstar @ ax)

    out = np.ones(n, dtype=np.float64)
    for i, hkl in enumerate(hkls):
        h = np.array(hkl, dtype=np.float64)
        h_norm = np.sqrt(h @ gstar @ h)
        if h_norm < 1e-15 or ax_norm < 1e-15:
            continue
        # angle between two plane normals (both reciprocal vectors) via G*.
        cos_a = np.clip((h @ gstar @ ax) / (h_norm * ax_norm), -1.0, 1.0)
        out[i] = float(march_dollase_factor(np.arccos(cos_a), r))
    return out
