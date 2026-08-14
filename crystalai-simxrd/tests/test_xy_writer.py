"""``write_xy``'s ``scale_to_100`` option: strongest line in the saved range -> 100."""

import numpy as np

from crystalai_simxrd.core.domain import Domain
from crystalai_simxrd.io.xy_writer import write_xy
from crystalai_simxrd.profiles.caglioti import SIMXRD_DEFAULT
from crystalai_simxrd.simulation.simulator import EffectConfig

X = np.array([10.0, 20.0, 30.0, 40.0])
Y = np.array([5.0, 40.0, 10.0, 2.0])


def _write(tmp_path, **kw):
    path = write_xy(
        tmp_path / "out.xy",
        x=X, y=Y, domain=Domain.TWO_THETA, wavelength=1.5406,
        structure_name="test", n_peaks=4,
        effects=EffectConfig(), instrument=SIMXRD_DEFAULT,
        **kw,
    )
    return path.read_text()


def test_scale_to_100_sets_max_peak_to_100(tmp_path):
    text = _write(tmp_path, scale_to_100=True)
    rows = [ln.split() for ln in text.splitlines() if ln and not ln.startswith("#")]
    values = [float(y) for _, y in rows]
    assert max(values) == 100.0
    assert values == [v / Y.max() * 100.0 for v in Y]


def test_scale_to_100_off_by_default_preserves_raw_scale(tmp_path):
    text = _write(tmp_path)
    rows = [ln.split() for ln in text.splitlines() if ln and not ln.startswith("#")]
    values = [float(y) for _, y in rows]
    assert values == list(Y)


def test_scale_to_100_header_line_present_only_when_enabled(tmp_path):
    scaled = _write(tmp_path, scale_to_100=True)
    unscaled = _write(tmp_path, scale_to_100=False)
    assert "scaled to 0-100" in scaled
    assert "scaled to 0-100" not in unscaled


def test_scale_to_100_applies_after_crop(tmp_path):
    # crop drops the global-max point (y=40 at x=20); the 0-100 scale should key off
    # the strongest line *within* the cropped range (y=10 at x=30), not the full array.
    text = _write(tmp_path, crop=(25.0, 45.0), scale_to_100=True)
    rows = [ln.split() for ln in text.splitlines() if ln and not ln.startswith("#")]
    values = [float(y) for _, y in rows]
    assert max(values) == 100.0
    assert values == [100.0, 20.0]
