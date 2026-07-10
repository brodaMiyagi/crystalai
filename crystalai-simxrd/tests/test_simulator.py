"""Phase-1 checkpoint: the profile → log-d pipeline.

Validates that the 2θ-assembly → single-log-d-conversion path (a) places every
reflection at its correct log₁₀(d), (b) conserves integrated intensity across the
2θ↔log-d resample (Jacobian correct), and (c) keeps 2θ peak positions on pymatgen.
"""

import numpy as np
import pytest

pytest.importorskip("scipy")
from scipy.signal import find_peaks  # noqa: E402

from crystalai_simxrd.core.crystal import load_cif  # noqa: E402
from crystalai_simxrd.core.domain import Domain  # noqa: E402
from crystalai_simxrd.core.wavelengths import CU_KA1  # noqa: E402
from crystalai_simxrd.simulation.simulator import simulate  # noqa: E402

WL = CU_KA1


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("name", ["Si", "CeO2", "LaB6"])
def test_log_d_positions_and_area_conservation(name, cif_dir):
    st = load_cif(cif_dir / f"{name}.cif")
    p = simulate(st, WL, domain=Domain.LOG_D)
    p2 = simulate(st, WL, domain=Domain.TWO_THETA)

    # (a) every accessible reflection appears at its log10(d)
    step = p.x_axis[1] - p.x_axis[0]
    peaks, _ = find_peaks(p.intensity, height=p.intensity.max() * 0.01)
    found = p.x_axis[peaks]
    for t in np.log10(p.peak_positions_d):
        assert np.min(np.abs(found - t)) < 2 * step, f"reflection at log-d={t:.3f} not found"

    # (b) integrated intensity conserved across 2θ ↔ log-d
    area_tt = np.sum(p2.intensity) * (p2.x_axis[1] - p2.x_axis[0])
    area_ld = np.sum(p.intensity) * step
    assert abs(area_ld - area_tt) / area_tt < 0.02, "area not conserved across resample"


@pytest.mark.filterwarnings("ignore")
def test_two_theta_positions_match_pymatgen(cif_dir):
    pytest.importorskip("pymatgen")
    from pymatgen.analysis.diffraction.xrd import XRDCalculator

    st = load_cif(cif_dir / "Si.cif")
    p2 = simulate(st, WL, domain=Domain.TWO_THETA, internal_step=0.01)
    ref = XRDCalculator(wavelength=WL).get_pattern(st, two_theta_range=(5, 150))
    peaks, _ = find_peaks(p2.intensity, height=p2.intensity.max() * 0.01)
    found = p2.x_axis[peaks]
    for x in ref.x:
        # find_peaks quantizes to the 0.01° grid → allow ~1.5 bins
        assert np.min(np.abs(found - x)) < 0.016, f"pymatgen peak {x:.3f}° not reproduced"


@pytest.mark.filterwarnings("ignore")
def test_simulate_from_cached_peak_list(cif_dir):
    """Simulator must accept a cached BraggPeaks (the training path) identically."""
    from crystalai_simxrd.core.bragg import compute_peak_list

    st = load_cif(cif_dir / "Si.cif")
    peaks = compute_peak_list(st, d_min=0.7)
    from_struct = simulate(st, WL, domain=Domain.LOG_D)
    from_cache = simulate(peaks, WL, domain=Domain.LOG_D)
    assert np.allclose(from_struct.intensity, from_cache.intensity)
