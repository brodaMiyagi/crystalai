"""Phase 5: bragg_peaks store round-trip + simulate-from-store equivalence."""

import numpy as np
import pytest

from crystalai_simxrd.core.bragg import compute_peak_list
from crystalai_simxrd.core.crystal import load_cif
from crystalai_simxrd.core.domain import Domain
from crystalai_simxrd.io import bragg_store
from crystalai_simxrd.io.bragg_store import BraggStore
from crystalai_simxrd.simulation.simulator import simulate


def test_serialize_roundtrip(cif_dir):
    peaks = compute_peak_list(load_cif(cif_dir / "CeO2.cif"), d_min=0.7)
    back = bragg_store.deserialize(bragg_store.serialize(peaks))
    assert np.array_equal(back.d_spacings, peaks.d_spacings)      # d is float64 → exact
    assert np.allclose(back.f_squared, peaks.f_squared, rtol=1e-3)  # f2 float32
    assert back.hkls == peaks.hkls
    assert np.allclose(back.metric_tensor, peaks.metric_tensor)


def test_store_write_read_and_simulate(cif_dir, tmp_path):
    names = ["Si", "NaCl", "CeO2"]
    peaks = {i: compute_peak_list(load_cif(cif_dir / f"{n}.cif"), d_min=0.7)
             for i, n in enumerate(names)}
    db = tmp_path / "bragg.sqlite"
    conn = bragg_store.connect(db)
    bragg_store.create_schema(conn)
    bragg_store.write_many(conn, [(i, p, 0.7) for i, p in peaks.items()])
    conn.close()

    store = BraggStore(db)
    assert len(store) == 3
    assert 1 in store and 99 not in store
    for i, n in enumerate(names):
        st = load_cif(cif_dir / f"{n}.cif")
        from_store = simulate(store.get(i), 1.5406, domain=Domain.TWO_THETA)
        from_struct = simulate(st, 1.5406, domain=Domain.TWO_THETA)
        assert np.allclose(from_store.intensity, from_struct.intensity, rtol=1e-4)


def test_public_api_imports():
    import crystalai_simxrd as sx

    for name in ("Simulator", "simulate", "SimulatedPattern", "EffectConfig",
                 "BraggPeaks", "compute_peak_list", "BraggStore",
                 "ProfileAugmentor", "PeakAugmentor", "Domain", "get_preset"):
        assert hasattr(sx, name), name


@pytest.mark.filterwarnings("ignore")
def test_simulator_class(cif_dir):
    from crystalai_simxrd import Simulator

    sim = Simulator()
    p = sim(load_cif(cif_dir / "Si.cif"), 1.5406)
    assert len(p.x_axis) == 12000 and np.isfinite(p.intensity).all()
