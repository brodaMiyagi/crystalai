"""Unit tests for the PXRDGen input remap (no model / capsule needed)."""

from __future__ import annotations

import numpy as np
import pytest

from crystalai_methods.baselines.pxrdgen.remap import (
    LAMBDA_CU_KA1,
    N_POINTS,
    TWO_THETA_MIN,
    TWO_THETA_STEP,
    model_two_theta_grid,
    remap_pattern,
)


def test_grid_matches_shipped_contract():
    g = model_two_theta_grid()
    assert g.shape == (N_POINTS,)
    # 7500 points at 5.00, 5.01, …, 79.99 (the loader's arange(5,80.01,.01)[:-1]).
    np.testing.assert_allclose(g[0], TWO_THETA_MIN)
    np.testing.assert_allclose(g[-1], 79.99)
    np.testing.assert_allclose(np.diff(g), TWO_THETA_STEP)


def test_remap_output_contract():
    tt = np.linspace(5.0, 90.0, 4000)
    y = np.random.default_rng(0).random(tt.shape)
    x, cov = remap_pattern(tt, y, wavelength=LAMBDA_CU_KA1)
    assert x.shape == (N_POINTS,)
    assert x.dtype == np.float32
    assert x.min() >= 0.0
    np.testing.assert_allclose(x.max(), 1.0, atol=1e-6)  # normalized by max peak
    assert cov == pytest.approx(1.0)  # Cu Kα 5–90° fully spans [5,80)


def test_negatives_floored():
    tt = np.linspace(5.0, 90.0, 2000)
    y = np.full_like(tt, -3.0)
    x, _ = remap_pattern(tt, y, wavelength=LAMBDA_CU_KA1)
    assert x.min() >= 0.0 and x.max() == 0.0  # all-negative -> all-zero


def test_cu_is_near_identity_resample():
    # A Cu Kα peak stays at its own 2θ (remap to Cu Kα1 is ~identity).
    tt_peak = 30.0
    tt = np.linspace(5.0, 80.0, 7500)
    y = np.zeros_like(tt)
    y[np.argmin(np.abs(tt - tt_peak))] = 1.0
    x, _ = remap_pattern(tt, y, wavelength=LAMBDA_CU_KA1)
    tt_at_max = model_two_theta_grid()[int(np.argmax(x))]
    assert abs(tt_at_max - tt_peak) < 0.05


def test_wavelength_invariance_through_d():
    # Same reflection (same d) at Cu vs Mo must land in the same Cu-grid bin.
    d_target = 2.5
    bins = []
    for lam, tt_hi in [(LAMBDA_CU_KA1, 90.0), (0.7093, 40.0)]:
        tt_peak = np.rad2deg(2 * np.arcsin(lam / (2 * d_target)))
        tt = np.linspace(3.0, tt_hi, 8000)
        y = np.zeros_like(tt)
        y[np.argmin(np.abs(tt - tt_peak))] = 1.0
        x, _ = remap_pattern(tt, y, wavelength=lam)
        bins.append(int(np.argmax(x)))
    assert abs(bins[0] - bins[1]) <= 1


def test_mo_partial_coverage():
    # Mo Kα low-angle reflections map above the 5° Cu edge -> coverage < 1.
    tt = np.linspace(4.0, 55.0, 4000)
    y = np.ones_like(tt)
    x, cov = remap_pattern(tt, y, wavelength=0.7093)
    assert 0.0 < cov < 1.0
    assert x[0] == 0.0  # 5° Cu bin below the mapped Mo range -> zero-filled
