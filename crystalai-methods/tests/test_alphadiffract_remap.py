"""Unit tests for the OpenAlphaDiffract input remap (no model download needed)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from crystalai_methods.baselines.alphadiffract.remap import (
    LAMBDA_MODEL,
    N_POINTS,
    TWO_THETA_MAX,
    TWO_THETA_MIN,
    model_d_grid,
    remap_pattern,
)

FIXTURES = (
    Path(__file__).resolve().parents[2]
    / "crystalai-data" / "tests" / "fixtures" / "exp_subset"
)


def test_model_grid_shape_and_window():
    d = model_d_grid()
    assert d.shape == (N_POINTS,)
    # Monotonic decreasing in 2θ; endpoints match Bragg at the window edges.
    assert np.all(np.diff(d) < 0)
    np.testing.assert_allclose(d.max(), LAMBDA_MODEL / (2 * np.sin(np.deg2rad(TWO_THETA_MIN) / 2)))
    np.testing.assert_allclose(d.min(), LAMBDA_MODEL / (2 * np.sin(np.deg2rad(TWO_THETA_MAX) / 2)))
    np.testing.assert_allclose([d.min(), d.max()], [1.785, 7.106], atol=1e-2)


def test_remap_output_contract():
    # A Cu Kα pattern fully covering the model window (2θ 5..90°).
    tt = np.linspace(5.0, 90.0, 4000)
    y = np.random.default_rng(0).random(tt.shape)
    x, cov = remap_pattern(tt, y, wavelength=1.54056)
    assert x.shape == (N_POINTS,)
    assert x.dtype == np.float32
    assert x.min() >= 0.0 and x.max() <= 100.0 + 1e-4
    np.testing.assert_allclose(x.max(), 100.0, atol=1e-3)  # min-max hits 100
    assert cov == pytest.approx(1.0)  # Cu Kα 5–90° fully spans the d-window


def test_negatives_floored():
    tt = np.linspace(5.0, 90.0, 2000)
    y = np.full_like(tt, -3.0)  # all-negative bgsub noise
    x, _ = remap_pattern(tt, y, wavelength=1.54056)
    assert x.min() >= 0.0


def test_partial_coverage_zerofill():
    # Scan 2θ 5–30° at Cu Kα maps to d ∈ [2.98, 17.7] Å: it straddles the model
    # window [1.785, 7.106], covering the large-d part but not d < 2.98 Å.
    tt = np.linspace(5.0, 30.0, 1000)
    y = np.ones_like(tt)
    x, cov = remap_pattern(tt, y, wavelength=1.54056)
    assert 0.0 < cov < 1.0
    assert x[-1] == 0.0  # last bin = smallest d = highest angle, outside scan


def test_peak_lands_at_correct_d():
    # A single sharp peak at a known d should map to the nearest model bin.
    d_target = 3.0  # Å, inside [1.785, 7.106]
    lam = 1.54056
    tt_peak = np.rad2deg(2 * np.arcsin(lam / (2 * d_target)))
    tt = np.linspace(5.0, 90.0, 8000)
    y = np.zeros_like(tt)
    y[np.argmin(np.abs(tt - tt_peak))] = 1.0
    x, _ = remap_pattern(tt, y, wavelength=lam)
    d_grid = model_d_grid()
    d_at_max = d_grid[int(np.argmax(x))]
    assert abs(d_at_max - d_target) < 0.02


def test_wavelength_invariance():
    # Same structure (same d-peak) measured at two wavelengths must land in the
    # same model bin — the whole point of remapping through d.
    d_target = 2.5
    peaks_bin = []
    for lam, tt_hi in [(1.54056, 90.0), (0.7107, 40.0)]:  # Cu Kα, Mo Kα
        tt_peak = np.rad2deg(2 * np.arcsin(lam / (2 * d_target)))
        tt = np.linspace(3.0, tt_hi, 8000)
        y = np.zeros_like(tt)
        y[np.argmin(np.abs(tt - tt_peak))] = 1.0
        x, _ = remap_pattern(tt, y, wavelength=lam)
        peaks_bin.append(int(np.argmax(x)))
    assert abs(peaks_bin[0] - peaks_bin[1]) <= 1


@pytest.mark.skipif(not FIXTURES.exists(), reason="exp_subset fixtures not present")
def test_remap_on_real_fixture():
    from crystalai_data.xrddata.database import read_xy

    p = FIXTURES / "patterns" / "lab" / "RWTH-A_ML2" / "bgsub.xy"
    xs, ys = read_xy(p)
    x, cov = remap_pattern(np.asarray(xs), np.asarray(ys), wavelength=1.54056)
    assert x.shape == (N_POINTS,)
    assert x.min() >= 0.0 and x.max() == pytest.approx(100.0, abs=1e-3)
    assert cov > 0.5  # a full lab scan covers most of the window
