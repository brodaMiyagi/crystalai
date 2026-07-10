"""Counting (Poisson) and baseline (Gaussian) noise (AlphaDiffract, arXiv:2603.23367).

`(λmax, σrel)` are the noise-floor conditioning pair (DESIGN_DECISIONS §4a). Poisson
is a **count-domain** effect (variance = mean) applied in 2θ on the total counts
(profile + background), *before* the first normalization; relative Gaussian is applied
in log-d *after* normalization. See SIMXRD_ROADMAP §5.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def poisson_counting(
    pattern: NDArray, lambda_max: float, rng: np.random.Generator | None = None
) -> NDArray:
    """Normalization-invariant Poisson counting noise.

    ``I_pois = max(I)/λmax · Poisson(λmax · I/max(I))``. Small ``λmax`` → few counts →
    sharp √-scaled spikes; large ``λmax`` → clean. Negative inputs (residual-bg valleys)
    are floored at 0 for the counting step.
    """
    p = np.asarray(pattern, dtype=np.float64)
    pmax = np.max(p)
    if lambda_max <= 0 or pmax <= 0:
        return p
    rng = rng or np.random.default_rng()
    counts = rng.poisson(np.clip(lambda_max * p / pmax, 0.0, None))
    return counts * (pmax / lambda_max)


def relative_gaussian(
    pattern_normalized: NDArray, sigma_rel: float, rng: np.random.Generator | None = None
) -> NDArray:
    """Add ``N(0, σrel)`` to an already-normalized pattern (baseline/readout noise)."""
    if sigma_rel <= 0:
        return np.asarray(pattern_normalized, dtype=np.float64)
    rng = rng or np.random.default_rng()
    p = np.asarray(pattern_normalized, dtype=np.float64)
    return p + rng.normal(0.0, sigma_rel, size=p.shape)
