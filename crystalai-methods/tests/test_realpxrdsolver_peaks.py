"""Unit tests for the RealPXRDSolver manual-peak remap (no model needed).

The model consumes an integer-2θ-degree position embedding (nn.Embedding(180),
peaks floored via ``.long()``); these tests exercise the ``prepare.remap_peak_list``
step that puts our manual peaks onto the Cu Kα1 2θ scale before that flooring.
"""

from __future__ import annotations

import numpy as np

from crystalai_methods.baselines.pxrdgen.remap import LAMBDA_CU_KA1
from crystalai_methods.baselines.realpxrdsolver.prepare import remap_peak_list


def _write(tmp_path, rows, header="# view=peaks\n"):
    p = tmp_path / "peaks.xy"
    with open(p, "w") as f:
        f.write(header)
        for x, y in rows:
            f.write(f"{x} {y}\n")
    return p


def test_cu_peaks_are_near_identity(tmp_path):
    rows = [(16.15, 2.45), (26.15, 100.0), (27.30, 16.5), (45.0, 30.0)]
    p = _write(tmp_path, rows)
    out = remap_peak_list(p, LAMBDA_CU_KA1)
    assert out.shape == (4, 2)
    # Cu-collected peaks map onto ~themselves; ascending in 2θ.
    np.testing.assert_allclose(out[:, 0], sorted(r[0] for r in rows), atol=0.01)
    assert np.all(np.diff(out[:, 0]) > 0)


def test_mo_peak_shifts_to_larger_cu_angle(tmp_path):
    # A reflection measured at Mo (short λ) sits at a small 2θ; at Cu it moves out.
    p = _write(tmp_path, [(10.0, 100.0)], header="")
    out = remap_peak_list(p, 0.7093)  # Mo Kα1
    assert out[0, 0] > 10.0
    assert out[0, 1] == 100.0  # intensity preserved


def test_nonphysical_and_zero_intensity_dropped(tmp_path):
    # d < λ_cu/2 has no Cu Kα1 angle -> dropped; zero-intensity peaks dropped too.
    p = _write(tmp_path, [(5.0, 0.0), (170.0, 50.0), (30.0, 80.0)], header="")
    out = remap_peak_list(p, 0.7093)  # Mo: 170° maps to sinθ>1 -> NaN -> dropped
    assert out.shape == (1, 2)  # only the 30° peak survives
    assert out[0, 0] > 30.0  # Mo -> Cu shifts it to a larger angle
    assert out[0, 1] == 80.0


def test_empty_returns_none(tmp_path):
    p = _write(tmp_path, [(5.0, 0.0)], header="")  # only a zero-intensity peak
    assert remap_peak_list(p, LAMBDA_CU_KA1) is None
