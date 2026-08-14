"""Counting (Poisson) and full-profile (Gaussian) noise (AlphaDiffract, arXiv:2603.23367).

`(λmax, σrel)` are the noise-floor conditioning pair (DESIGN_DECISIONS §4a). Both are
**2θ-domain** effects applied inside `simulate()`, before the single conversion to the
output domain (SIMXRD_ROADMAP §5, §5a) — there is no log-d-native noise anywhere on
this path. Poisson is count-domain (variance = mean) on the total 2θ counts (profile +
background); the Gaussian noise here (`relative_gaussian`) normalizes to [0,1], adds
`N(mean, std)`, and re-normalizes, all in 2θ.
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
    pattern_normalized: NDArray, sigma_rel: float, rng: np.random.Generator | None = None,
    mean: float = 0.0,
) -> NDArray:
    """Add ``N(mean, σrel)`` to an already-normalized pattern (baseline/readout noise).

    The single Gaussian-noise mechanism on the simulation path (SIMXRD_ROADMAP.md §5a),
    called from inside ``simulate()`` on the 2θ pattern, before any domain conversion —
    never natively in log-d. ``ProfileAugmentor`` uses ``mean=0`` (the σrel
    training-noise-floor conditioning case, DESIGN_DECISIONS.md §4a); a direct
    ``EffectConfig.gaussian_noise_mean/_std`` caller (e.g. the dashboard) can set
    either to a nonzero value.
    """
    sigma_rel = max(float(sigma_rel), 0.0)
    if sigma_rel <= 0 and mean == 0.0:
        return np.asarray(pattern_normalized, dtype=np.float64)
    rng = rng or np.random.default_rng()
    p = np.asarray(pattern_normalized, dtype=np.float64)
    return p + rng.normal(mean, sigma_rel, size=p.shape)
