"""Phase-2: each physical effect does the physically-expected thing."""

import numpy as np
import pytest

from crystalai_simxrd.core.crystal import load_cif
from crystalai_simxrd.core.domain import Domain
from crystalai_simxrd.core.wavelengths import CU_KA1
from crystalai_simxrd.simulation.simulator import EffectConfig, simulate

WL = CU_KA1


@pytest.fixture
def ceo2(cif_dir):
    return load_cif(cif_dir / "CeO2.cif")


def _tt(st, effects=None):
    return simulate(st, WL, domain=Domain.TWO_THETA, effects=effects)


def test_debye_waller_damps_high_angle(ceo2):
    base = _tt(ceo2)
    dw = _tt(ceo2, EffectConfig(b_iso=1.5))
    ratio = dw.peak_intensities / base.peak_intensities
    d = base.peak_positions_d
    assert ratio.max() <= 1.0 + 1e-9                         # DW only damps
    assert ratio[np.argmin(d)] < ratio[np.argmax(d)] - 0.1  # low-d (high-angle) damped more


def test_size_and_strain_broaden(ceo2):
    base = _tt(ceo2)
    size = _tt(ceo2, EffectConfig(crystallite_size_nm=10))
    strain = _tt(ceo2, EffectConfig(microstrain=0.005))
    # broadening conserves area but lowers peak height
    assert size.intensity.max() < 0.5 * base.intensity.max()
    assert strain.intensity.max() < base.intensity.max()


def test_preferred_orientation(ceo2):
    base = _tt(ceo2)
    assert np.allclose(_tt(ceo2, EffectConfig(march_r=1.0)).peak_intensities,
                       base.peak_intensities)                # r=1 is identity
    po = _tt(ceo2, EffectConfig(march_r=0.6, march_axis=(1, 1, 1)))
    rel = po.peak_intensities / base.peak_intensities
    assert rel.max() / rel.min() > 1.2                       # texture reweights peaks


def test_zero_shift_moves_all_peaks(ceo2):
    base = _tt(ceo2)
    zs = _tt(ceo2, EffectConfig(zero_shift_deg=0.3))
    moved = zs.x_axis[np.argmax(zs.intensity)] - base.x_axis[np.argmax(base.intensity)]
    assert abs(moved - 0.3) < 0.02


def test_slit_broadens(ceo2):
    base = _tt(ceo2)
    slit = _tt(ceo2, EffectConfig(slit_width_deg=0.1))
    assert slit.intensity.max() < base.intensity.max()


def test_residual_background_small_and_signed(ceo2):
    base = _tt(ceo2)
    bg = _tt(ceo2, EffectConfig(background="residual", background_rel_amplitude=0.03, rng_seed=0))
    add = bg.intensity - base.intensity
    assert add.min() < 0.0                                   # residual may go negative
    assert np.max(np.abs(add)) <= 0.035 * base.intensity.max() + 1e-6


def test_axial_divergence_preserves_integrated_intensity(ceo2):
    base = _tt(ceo2)
    ax = _tt(ceo2, EffectConfig(axial_divergence=True, axial_L=150, axial_H=5, axial_S=3))
    assert not np.allclose(ax.intensity, base.intensity)     # it does something
    assert abs(ax.intensity.sum() - base.intensity.sum()) / base.intensity.sum() < 0.02


def test_full_effects_log_d_is_finite(ceo2):
    p = simulate(ceo2, WL, effects=EffectConfig(
        b_iso=1.0, crystallite_size_nm=50, microstrain=0.002, march_r=0.9,
        zero_shift_deg=0.1, axial_divergence=True, slit_width_deg=0.03,
        background="residual", rng_seed=1))
    assert len(p.x_axis) == 12000
    assert np.isfinite(p.intensity).all()


def test_full_profile_gaussian_noise_off_is_identity(ceo2):
    base = _tt(ceo2, EffectConfig(rng_seed=0))
    off = _tt(ceo2, EffectConfig(gaussian_noise_mean=0.0, gaussian_noise_std=0.0, rng_seed=0))
    assert np.allclose(base.intensity, off.intensity)


def test_full_profile_gaussian_noise_bounded_and_negative_in_two_theta(ceo2):
    noisy = _tt(ceo2, EffectConfig(gaussian_noise_mean=0.0, gaussian_noise_std=0.01, rng_seed=0))
    assert np.isclose(noisy.intensity.max(), 1.0)
    assert noisy.intensity.min() < 0.0          # ripples dip slightly negative, like residual bg


def test_full_profile_gaussian_noise_reproducible_with_seed(ceo2):
    a = _tt(ceo2, EffectConfig(gaussian_noise_mean=0.01, gaussian_noise_std=0.02, rng_seed=7))
    b = _tt(ceo2, EffectConfig(gaussian_noise_mean=0.01, gaussian_noise_std=0.02, rng_seed=7))
    assert np.allclose(a.intensity, b.intensity)


def test_full_profile_gaussian_noise_applied_in_two_theta_before_log_d_convert(ceo2):
    """The noise must be added to the 2θ-domain pattern before the single conversion
    to log-d — not natively in log-d — so the log-d output isn't re-bounded to [0,1]
    by the (non-normalizing) Jacobian resample (SIMXRD_ROADMAP §5a)."""
    p = simulate(ceo2, WL, domain=Domain.LOG_D,
                effects=EffectConfig(gaussian_noise_mean=0.0, gaussian_noise_std=0.01, rng_seed=0))
    assert len(p.x_axis) == 12000
    assert np.isfinite(p.intensity).all()
    assert p.intensity.max() > 1.0 + 1e-6      # Jacobian-resampled, not re-clamped to [0,1]
