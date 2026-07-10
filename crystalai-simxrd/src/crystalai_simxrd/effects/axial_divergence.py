"""Axial-divergence asymmetry, parameterized by instrument geometry.

Finite axial (out-of-plane) divergence gives low-2θ peaks a characteristic
low-angle tail that vanishes at high angle. Parameterized by the physical geometry
(Finger-Cox-Jephcoat / Van Laar-Yelon): detector distance ``L``, source/sample
half-height ``H``, receiving-slit half-height ``S`` (mm). Applied to the assembled
2θ pattern before the log-d conversion (SIMXRD_ROADMAP §5).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def apply_axial_divergence(
    two_theta_grid: NDArray,
    pattern: NDArray,
    peak_positions: NDArray,
    *,
    L: float = 200.0,
    H: float = 2.0,
    S: float = 2.0,
    cutoff_two_theta: float = 60.0,
) -> NDArray:
    """Skew low-angle peaks toward lower 2θ (axial divergence).

    ``L/H/S`` are the detector-distance / half-heights (mm); the asymmetric shift
    scales with ``((H+S)/2L)²·cotθ``, so it is strong at low 2θ and negligible above
    ``cutoff_two_theta``. Non-destructive of integrated intensity (the modifier is
    mean-normalized per peak window).
    """
    x = np.asarray(two_theta_grid, dtype=np.float64)
    out = np.asarray(pattern, dtype=np.float64).copy()
    hl = ((H + S) / 2.0) / max(L, 1e-6)            # combined divergence half-angle scale

    for center in np.asarray(peak_positions, dtype=np.float64):
        if not (1.0 < center < cutoff_two_theta):
            continue
        cot = 1.0 / np.tan(np.deg2rad(center / 2.0))
        max_shift = -np.rad2deg(hl**2 * abs(cot))   # negative → low-angle tail
        if abs(max_shift) < 1e-4:
            continue
        half = max(abs(max_shift) * 3.0, 0.3)
        m = (x >= center + max_shift - half) & (x <= center + half)
        if not np.any(m):
            continue
        delta = x[m] - center
        w = np.ones(m.sum())
        tail = (delta < 0) & (delta >= max_shift)
        ratio = np.clip(delta[tail] / max_shift, 0.0, 1.0 - 1e-9)
        w[tail] = 1.0 / np.sqrt(1.0 - ratio)
        w[delta < max_shift] = 0.0
        seg = out[m]
        before = seg.sum()
        weighted = seg * w
        after = weighted.sum()
        if after > 0:                               # rescale so the window's integral is preserved
            out[m] = weighted * (before / after)
    return out
