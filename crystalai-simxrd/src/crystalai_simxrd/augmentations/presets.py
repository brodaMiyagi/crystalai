"""Named augmentation presets (SIMXRD_ROADMAP Phase 3.3).

- ``PRODUCTION`` — the training preset: wavelength + Caglioti + physical-effect
  randomization, matched d-range, Poisson+Gaussian noise, residual background.
- ``MILD`` — gentle ranges for early training / sanity.
- ``AGGRESSIVE`` — wide ranges, biased noisy (robustness stress).
- ``DIFCON_STYLE`` — narrow, Cu-Kα-centric, minimal effects (prior-paper-like).

Each returns ``(AugmentConfig, PeakAugmentConfig)`` via a factory so callers get a
fresh, mutable copy.
"""

from __future__ import annotations

from .peak_augmentations import PeakAugmentConfig
from .profile_augmentations import AugmentConfig


def production() -> tuple[AugmentConfig, PeakAugmentConfig]:
    return AugmentConfig(), PeakAugmentConfig()


def mild() -> tuple[AugmentConfig, PeakAugmentConfig]:
    a = AugmentConfig(
        wavelength=(1.0, 1.8), crystallite_nm=(80.0, 500.0), microstrain=(0.0, 0.0015),
        b_iso=(0.0, 1.0), march_r=(0.95, 1.06), zero_shift_deg=(-0.01, 0.01),
        background_rel_amplitude=(0.0, 0.02), lambda_max=(20.0, 100.0),
        sigma_rel=(1e-3, 2e-2),
    )
    p = PeakAugmentConfig(jitter_logd_sigma=0.0008, off_center_logd=0.0005,
                          drop_fraction=(0.0, 0.2), n_spurious=1)
    return a, p


def aggressive() -> tuple[AugmentConfig, PeakAugmentConfig]:
    a = AugmentConfig(
        wavelength=(0.4, 2.3), crystallite_nm=(8.0, 500.0), microstrain=(0.0, 0.006),
        b_iso=(0.0, 3.0), march_r=(0.6, 1.5), zero_shift_deg=(-0.04, 0.04),
        background_rel_amplitude=(0.0, 0.10), lambda_max=(1.0, 40.0),  # biased noisy
        sigma_rel=(5e-3, 1e-1),
    )
    p = PeakAugmentConfig(jitter_logd_sigma=0.003, off_center_logd=0.002,
                          drop_fraction=(0.0, 0.7), n_spurious=5,
                          spurious_rel_intensity=0.2)
    return a, p


def difcon_style() -> tuple[AugmentConfig, PeakAugmentConfig]:
    a = AugmentConfig(
        wavelength=(1.5406, 1.5406), crystallite_nm=None, microstrain=(0.0, 0.0),
        b_iso=(0.0, 0.0), march_r=(1.0, 1.0), zero_shift_deg=(0.0, 0.0),
        axial_divergence=False, background_choices=(None,),
        background_rel_amplitude=(0.0, 0.0), lambda_max=(50.0, 100.0),
        sigma_rel=(1e-3, 1e-2),
    )
    p = PeakAugmentConfig(drop_enable=False, jitter_logd_sigma=0.0, off_center_logd=0.0,
                          n_spurious=0)
    return a, p


PRESETS = {
    "PRODUCTION": production,
    "MILD": mild,
    "AGGRESSIVE": aggressive,
    "DIFCON_STYLE": difcon_style,
}


def get_preset(name: str) -> tuple[AugmentConfig, PeakAugmentConfig]:
    try:
        return PRESETS[name]()
    except KeyError:
        raise KeyError(f"unknown preset {name!r}; available: {list(PRESETS)}") from None
