"""Tests for the CS-stratified ICSD split assignment."""

import pytest
from pymatgen.core import Lattice, Structure

from crystalai_data.crystals import database as db
from crystalai_data.crystals.assign_split import _slice_counts, assign


def _make_db(tmp_path, name="c.sqlite"):
    path = tmp_path / name
    conn = db.connect(path)
    db.create_schema(conn)
    return conn


def _insert_icsd(conn, cs, sg, n, start=0):
    """Insert ``n`` trivial icsd rows with a given crystal_system."""
    s = Structure(Lattice.cubic(4.0), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    for i in range(start, start + n):
        row = db.compute_metadata(
            s, source="icsd", source_id=f"{cs}_{i}", space_group=sg,
            crystal_system=cs, source_file_id=str(i),
        )
        db.insert_crystal(conn, row)


def test_slice_counts_sums_exactly():
    for n in (0, 1, 2, 3, 7, 10, 99, 1000):
        t, v, te = _slice_counts(n, (0.8, 0.1, 0.1))
        assert t + v + te == n
        assert min(t, v, te) >= 0


def test_stratified_proportions(tmp_path):
    conn = _make_db(tmp_path)
    _insert_icsd(conn, cs=7, sg=225, n=1000)
    _insert_icsd(conn, cs=1, sg=1, n=100)
    summary = assign(conn, ratios=(0.8, 0.1, 0.1), seed=1)

    assert summary[7] == (800, 100, 100)
    assert summary[1] == (80, 10, 10)
    # every system present in every fold
    for split in ("train", "val", "test"):
        cs_in_fold = {
            r[0] for r in conn.execute(
                "SELECT DISTINCT crystal_system FROM crystals WHERE split=?", (split,)
            )
        }
        assert cs_in_fold == {1, 7}


def test_determinism_same_seed(tmp_path):
    def build_and_assign(name, seed):
        conn = _make_db(tmp_path, name)
        _insert_icsd(conn, cs=7, sg=225, n=200)
        _insert_icsd(conn, cs=3, sg=20, n=50)
        assign(conn, seed=seed)
        return {
            r[0]: r[1] for r in conn.execute("SELECT source_id, split FROM crystals")
        }

    a = build_and_assign("a.sqlite", seed=42)
    b = build_and_assign("b.sqlite", seed=42)
    c = build_and_assign("c2.sqlite", seed=7)
    assert a == b           # same seed -> identical assignment
    assert a != c           # different seed -> different assignment


def test_only_fills_nulls_by_default(tmp_path):
    conn = _make_db(tmp_path)
    _insert_icsd(conn, cs=7, sg=225, n=100)
    # pre-set one row to a sentinel split
    conn.execute("UPDATE crystals SET split='train' WHERE id=1")
    conn.commit()
    assign(conn)  # default: only null rows
    # the pre-set row keeps its value, was not part of the stratified slice
    assert conn.execute("SELECT split FROM crystals WHERE id=1").fetchone()[0] == "train"
    assert conn.execute("SELECT COUNT(*) FROM crystals WHERE split IS NULL").fetchone()[0] == 0


def test_does_not_touch_other_sources(tmp_path):
    conn = _make_db(tmp_path)
    _insert_icsd(conn, cs=7, sg=225, n=50)
    # an mp-20 row with native split
    s = Structure(Lattice.cubic(4.0), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    db.insert_crystal(conn, db.compute_metadata(
        s, source="mp-20", source_id="mp-1", space_group=225, crystal_system=7,
        split="val",
    ))
    assign(conn, source="icsd")
    assert conn.execute(
        "SELECT split FROM crystals WHERE source='mp-20'"
    ).fetchone()[0] == "val"
    # mp-20 not double-assigned
    assert conn.execute(
        "SELECT COUNT(*) FROM crystals WHERE source='icsd' AND split IS NULL"
    ).fetchone()[0] == 0


def test_ratios_must_sum_to_one(tmp_path):
    conn = _make_db(tmp_path)
    _insert_icsd(conn, cs=7, sg=225, n=10)
    with pytest.raises(ValueError):
        assign(conn, ratios=(0.8, 0.1, 0.2))
