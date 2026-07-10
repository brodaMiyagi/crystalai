"""Simple peak detection for validation (SIMXRD_ROADMAP Phase 5.4).

A thin ``scipy.signal.find_peaks`` wrapper — used to check simulated/experimental
peak positions, **not** in the production path.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def find_peak_positions(
    x: NDArray, intensity: NDArray, *, rel_height: float = 0.02, rel_prominence: float = 0.01,
) -> NDArray:
    """Return the x-positions of peaks above ``rel_height``·max with ``rel_prominence``·max."""
    from scipy.signal import find_peaks

    y = np.asarray(intensity, dtype=np.float64)
    m = float(np.max(y)) if y.size else 0.0
    if m <= 0:
        return np.array([])
    idx, _ = find_peaks(y, height=rel_height * m, prominence=rel_prominence * m)
    return np.asarray(x, dtype=np.float64)[idx]
