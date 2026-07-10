"""Background models, applied in 2θ before the log-d conversion.

We train/align in the **background-subtracted** domain (DATA_ROADMAP §2), so the
default is a small **residual** background (imperfect subtraction) — a low-order,
low-amplitude Chebyshev that may go slightly negative. A full **physical**
background (Compton + air scatter + dark current) is available for raw-data
emulation or the arPLS-emulation residual mode.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def chebyshev_background(x: NDArray, coefficients: NDArray) -> NDArray:
    """Chebyshev series (first kind) over ``x`` mapped to [-1, 1]."""
    c = np.asarray(coefficients, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if x[-1] - x[0] < 1e-15:
        return np.full_like(x, c[0] if len(c) else 0.0)
    x_norm = 2.0 * (x - x[0]) / (x[-1] - x[0]) - 1.0
    return np.polynomial.chebyshev.chebval(x_norm, c)


def sample_residual_background(
    x: NDArray, peak_max: float, *, order: int = 4, rel_amplitude: float = 0.03,
    rng: np.random.Generator | None = None,
) -> NDArray:
    """A small, smooth residual baseline (~``rel_amplitude``·peak_max), may go negative.

    Models what remains after imperfect background subtraction; **not** clipped to ≥0.
    """
    rng = rng or np.random.default_rng()
    amp = rel_amplitude * max(peak_max, 1e-12)
    coeffs = rng.normal(0.0, 1.0, order + 1) / (1.0 + np.arange(order + 1) ** 1.5)
    bg = chebyshev_background(x, coeffs)
    m = np.max(np.abs(bg))
    return bg * (amp / m) if m > 0 else bg


def physical_background(
    two_theta: NDArray, wavelength: float, *, peak_max: float = 1.0,
    compton: float = 0.3, air: float = 0.1, dark: float = 0.05,
) -> NDArray:
    """Physical background: Compton (∝(sinθ/λ)²) + air scatter (∝1/sinθ) + dark current.

    Scaled so the overall level is ~``peak_max``-relative.
    """
    tt = np.asarray(two_theta, dtype=np.float64)
    theta = np.deg2rad(tt / 2.0)
    s2 = (np.sin(theta) / wavelength) ** 2
    inv_sin = 1.0 / np.clip(np.sin(theta), 1e-6, None)
    bg = compton * s2 / max(s2.max(), 1e-12) + air * inv_sin / inv_sin.max() + dark
    return bg * peak_max
