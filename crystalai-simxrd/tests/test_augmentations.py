"""Phase-3: on-the-fly augmentation (profile + peak-list) behaves as specified."""

import numpy as np
import pytest

from crystalai_simxrd.augmentations.peak_augmentations import PeakAugmentConfig, PeakAugmentor
from crystalai_simxrd.augmentations.presets import PRESETS, get_preset
from crystalai_simxrd.augmentations.profile_augmentations import (
    AugmentConfig,
    ProfileAugmentor,
)
from crystalai_simxrd.core.bragg import compute_peak_list
from crystalai_simxrd.core.crystal import load_cif


@pytest.fixture
def quartz(cif_dir):
    return load_cif(cif_dir / "alpha_quartz.cif")


@pytest.mark.filterwarnings("ignore")
def test_profile_augmentor_valid_and_conditioned(quartz):
    cfg = AugmentConfig()
    aug = ProfileAugmentor(cfg)
    for i in range(6):
        ap = aug(quartz, np.random.default_rng(i))
        assert len(ap.x_axis) == cfg.n_log_d_bins
        assert np.isfinite(ap.intensity).all()
        assert ap.intensity.max() <= 1.0 + 5 * 0.1        # max-normalized (+ Gaussian tail)
        lam, sig = ap.noise_floor
        assert cfg.wavelength[0] <= ap.wavelength <= cfg.wavelength[1]
        assert cfg.lambda_max[0] <= lam <= cfg.lambda_max[1]
        assert cfg.sigma_rel[0] <= sig <= cfg.sigma_rel[1]


@pytest.mark.filterwarnings("ignore")
def test_profile_augmentor_is_stochastic(quartz):
    aug = ProfileAugmentor()
    a = aug(quartz, np.random.default_rng(1))
    b = aug(quartz, np.random.default_rng(2))
    assert not np.allclose(a.intensity, b.intensity)      # different draws differ
    assert a.noise_floor != b.noise_floor


@pytest.mark.filterwarnings("ignore")
def test_profile_augmentor_from_cached_peaks(quartz):
    """Augmenting a cached BraggPeaks must work (the training path)."""
    peaks = compute_peak_list(quartz, d_min=0.7)
    ap = ProfileAugmentor()(peaks, np.random.default_rng(0))
    assert np.isfinite(ap.intensity).all() and len(ap.x_axis) == 12000


def test_peak_augmentor_position_and_spurious():
    d = np.linspace(1.0, 5.0, 20)
    inten = np.ones_like(d)
    pa = PeakAugmentor(PeakAugmentConfig(drop_enable=False, n_spurious=3,
                                         jitter_logd_sigma=0.002))
    d2, i2 = pa(d, inten, np.random.default_rng(0))
    assert len(d2) >= len(d)                               # spurious added, none dropped
    # positions perturbed (not identical to originals)
    assert not np.allclose(np.sort(d2)[:len(d)], d, atol=1e-6)


def test_peak_augmentor_drops_low_d_protects_high_d():
    # dense peak list; strong at high d
    d = np.linspace(0.7, 6.0, 200)
    inten = np.linspace(1.0, 2.0, 200)                     # slightly stronger at high d
    pa = PeakAugmentor(PeakAugmentConfig(
        drop_enable=True, density_threshold=40, drop_fraction=(0.6, 0.6),
        protect_high_d_frac=0.5, n_spurious=0, jitter_logd_sigma=0.0, off_center_logd=0.0))
    d2, _ = pa(d, inten, np.random.default_rng(0))
    assert len(d2) < len(d)                                # dense → dropped
    assert d2.max() >= d.max() - 1e-9                      # highest-d peak protected
    # dropping concentrated at low d: the low half loses more than the high half
    lost_low = np.sum(d < np.median(d)) - np.sum(d2 < np.median(d))
    lost_high = np.sum(d >= np.median(d)) - np.sum(d2 >= np.median(d))
    assert lost_low > lost_high


def test_presets_load_and_differ():
    for name in PRESETS:
        a, p = get_preset(name)
        assert isinstance(a, AugmentConfig) and isinstance(p, PeakAugmentConfig)
    assert get_preset("DIFCON_STYLE")[0].axial_divergence is False
    with pytest.raises(KeyError):
        get_preset("nope")
