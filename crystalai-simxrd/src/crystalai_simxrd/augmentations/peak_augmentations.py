"""Peak-list augmentation: model the hand-picked input (SIMXRD_ROADMAP §6, DD §7).

The peak-list channel's inference input is a manually-picked list, so we model *how a
human makes it* — not arbitrary corruption:

1. **Position error** — small per-peak selection jitter + a per-pattern off-center bias
   (the click lands on the profile shoulder), applied in log-d, sub-FWHM.
2. **Selective low-d dropping** — when peak-dense, drop preferentially from the low-d
   (high-2θ) forest, **protecting** the high-d (low-2θ) and strong peaks so the retained
   set still fixes the space group. (The full extinction-based manufactured-absence
   guard — never drop a diagnostic systematic-absence reflection — needs the SG and is
   deferred; the high-d/strong protection is the practical stand-in.)
3. **Spurious peaks** — 0–N additive impurity peaks.

Operates on a d-space peak list `(d_spacings, intensities)` and returns an augmented
one; the methods package bins it into the log-d peak histogram.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class PeakAugmentConfig:
    jitter_logd_sigma: float = 0.0015          # per-peak position jitter (log-d, sub-FWHM)
    off_center_logd: float = 0.001             # per-pattern systematic off-center bias (log-d)
    drop_enable: bool = True
    density_threshold: int = 40                # only thin when more peaks than this
    drop_fraction: tuple[float, float] = (0.0, 0.5)   # fraction of *droppable* peaks removed
    protect_high_d_frac: float = 0.5           # never drop the top fraction by d (low-2θ)
    protect_strong_frac: float = 0.10          # never drop the top fraction by intensity
    n_spurious: int = 3                        # 0..n additive spurious peaks
    spurious_rel_intensity: float = 0.10


class PeakAugmentor:
    """Callable: ``(d_spacings, intensities, rng) -> (d_aug, intensity_aug)``."""

    def __init__(self, config: PeakAugmentConfig | None = None):
        self.config = config or PeakAugmentConfig()

    def __call__(
        self, d_spacings: NDArray, intensities: NDArray,
        rng: np.random.Generator | None = None,
    ) -> tuple[NDArray, NDArray]:
        rng = rng or np.random.default_rng()
        c = self.config
        d = np.asarray(d_spacings, dtype=np.float64)
        inten = np.asarray(intensities, dtype=np.float64)
        if len(d) == 0:
            return d, inten

        # 1) position error (log-d): per-peak jitter + one per-pattern off-center bias
        log_d = np.log10(d)
        log_d = log_d + rng.normal(0.0, c.jitter_logd_sigma, len(d))
        log_d = log_d + rng.normal(0.0, c.off_center_logd)
        d = np.power(10.0, log_d)

        # 2) selective low-d dropping (protect high-d + strong)
        keep = np.ones(len(d), dtype=bool)
        if c.drop_enable and len(d) > c.density_threshold:
            hi_d_cut = np.quantile(d, 1.0 - c.protect_high_d_frac)      # keep d above this
            strong_cut = np.quantile(inten, 1.0 - c.protect_strong_frac)
            protected = (d >= hi_d_cut) | (inten >= strong_cut)
            droppable = np.where(~protected)[0]
            if len(droppable):
                frac = rng.uniform(*c.drop_fraction)
                n_drop = int(round(frac * len(droppable)))
                if n_drop > 0:
                    # drop preferentially from the lowest-d (highest-2θ) end
                    order = droppable[np.argsort(d[droppable])]
                    keep[order[:n_drop]] = False
        d, inten = d[keep], inten[keep]

        # 3) spurious peaks
        k = int(rng.integers(0, c.n_spurious + 1))
        if k > 0 and len(inten) > 0:
            lo, hi = float(np.min(d)), float(np.max(d))
            sp_d = rng.uniform(lo, hi, k)
            sp_i = rng.uniform(0.0, c.spurious_rel_intensity * float(np.max(inten)), k)
            d = np.concatenate([d, sp_d])
            inten = np.concatenate([inten, sp_i])

        order = np.argsort(d)
        return d[order], inten[order]
