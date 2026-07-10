"""Debye-Waller thermal factor.

Overall isotropic temperature factor on intensity:
    DW = exp(-2·B·(sinθ/λ)²) = exp(-B / (2 d²))
Because ``sinθ/λ = 1/2d``, DW is **wavelength-independent** — it can be applied
directly to the d-space peak list (an on-the-fly thermal envelope, not baked into
the precomputed |F|²; DESIGN_DECISIONS §9). Damps low-d (high-angle) reflections.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def debye_waller_envelope(d_spacings: NDArray, b_iso: float) -> NDArray:
    """Multiplicative intensity factor exp(-B/(2 d²)) for each reflection (B in Å²)."""
    d = np.asarray(d_spacings, dtype=np.float64)
    if b_iso <= 0:
        return np.ones_like(d)
    return np.exp(-b_iso / (2.0 * d**2))


def b_iso_from_temperature(temperature_k: float) -> float:
    """Rough global B_iso (Å²) from temperature: B ≈ 0.005·T (≈1.5 Å² at 300 K)."""
    return max(0.0, 0.005 * float(temperature_k))
