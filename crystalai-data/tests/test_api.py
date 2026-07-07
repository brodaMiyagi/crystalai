"""Tests for the CrystalDatabase query API against a small built database."""

import pytest
from pymatgen.core import Lattice, Structure

from crystalai_data.crystals import database as db
from crystalai_data.crystals.api import CrystalDatabase


def _ordered(species, coords, a=4.0):
    return Structure(Lattice.cubic(a), species, coords)


@pytest.fixture
def db_path(tmp_path):
    """Build a small file-backed DB with a mix of sources/elements/disorder."""
    path = tmp_path / "crystals.sqlite"
    conn = db.connect(path)
    db.create_schema(conn)

    rows = [
        # TiO2 — icsd, ordered, exact {O,Ti}, split null
        db.compute_metadata(
            _ordered(["Ti", "O", "O"], [[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]]),
            source="icsd", source_id="100", space_group=136, crystal_system=4,
            source_file_id="9001",
        ),
        # Fe2O3 — mp-20, ordered, contains {Fe,O}, split train, 5 atoms
        db.compute_metadata(
            _ordered(["Fe", "Fe", "O", "O", "O"],
                     [[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0], [0.5, 0.5, 0], [0, 0, 0.5]]),
            source="mp-20", source_id="mp-1", space_group=167, crystal_system=5,
            split="train",
        ),
        # NaCl — mp-20, ordered, no oxygen, split test
        db.compute_metadata(
            _ordered(["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]),
            source="mp-20", source_id="mp-2", space_group=225, crystal_system=7,
            split="test",
        ),
        # Mn/Cr-O — icsd, DISORDERED, contains O
        db.compute_metadata(
            Structure(Lattice.cubic(4.0), [{"Mn": 0.5, "Cr": 0.5}, {"O": 1.0}],
                      [[0, 0, 0], [0.5, 0.5, 0.5]]),
            source="icsd", source_id="101", space_group=225, crystal_system=7,
            source_file_id="9002",
        ),
    ]
    for r in rows:
        db.insert_crystal(conn, r)
    conn.close()
    return path


@pytest.fixture
def cdb(db_path):
    with CrystalDatabase(db_path) as d:
        yield d


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        CrystalDatabase(tmp_path / "nope.sqlite")


def test_get_and_missing(cdb):
    s = cdb.get(1)
    assert isinstance(s, Structure)
    assert s.composition.reduced_formula == "TiO2"
    with pytest.raises(KeyError):
        cdb.get(9999)


def test_get_cif_roundtrips_to_structure(cdb):
    cif = cdb.get_cif(1)
    assert "data_" in cif
    assert Structure.from_str(cif, fmt="cif").composition.reduced_formula == "TiO2"


def test_get_metadata(cdb):
    m = cdb.get_metadata(2)
    assert m["source"] == "mp-20"
    assert m["space_group"] == 167
    assert m["split"] == "train"
    assert "structure_blob" not in m


def test_count_and_filters(cdb):
    assert cdb.count() == 4
    assert cdb.count(source="icsd") == 2
    assert cdb.count(source="mp-20") == 2
    assert cdb.count(split="train") == 1
    assert cdb.count(exclude_disordered=True) == 3
    assert cdb.count(crystal_system=7) == 2
    assert cdb.count(source="icsd", exclude_disordered=True) == 1
    assert cdb.count(max_atoms=2) == 2  # NaCl + disordered MnCrO


def test_ids_ordered(cdb):
    assert cdb.ids(source="mp-20") == [2, 3]
    assert cdb.ids() == [1, 2, 3, 4]


def test_iter_all_filters(cdb):
    assert sum(1 for _ in cdb.iter_all()) == 4
    icsd = list(cdb.iter_all(source="icsd"))
    assert len(icsd) == 2 and all(isinstance(s, Structure) for s in icsd)
    assert sum(1 for _ in cdb.iter_all(source="mp-20", split="test")) == 1
    assert sum(1 for _ in cdb.iter_all(exclude_disordered=True)) == 3


def test_iter_by_space_group_and_cs(cdb):
    assert sum(1 for _ in cdb.iter_by_space_group(225)) == 2
    assert sum(1 for _ in cdb.iter_by_crystal_system(4)) == 1


def test_query_by_elements_contains_all(cdb):
    ids = set(cdb.query_by_elements(contains_all=["Fe", "O"]))
    assert ids == {2}
    # 'O'-containing: TiO2, Fe2O3, MnCrO  (not NaCl)
    assert set(cdb.query_by_elements(contains_all=["O"])) == {1, 2, 4}


def test_query_by_elements_exact(cdb):
    assert set(cdb.query_by_elements(exact=["Ti", "O"])) == {1}
    assert set(cdb.query_by_elements(exact=["Fe", "O"])) == {2}  # Fe2O3 is exactly {Fe,O}
    assert set(cdb.query_by_elements(exact=["Na", "Cl"])) == {3}
    assert set(cdb.query_by_elements(exact=["O"])) == set()  # no single-element O phase


def test_query_by_elements_exclude(cdb):
    # everything without O -> only NaCl
    assert set(cdb.query_by_elements(exclude=["O"])) == {3}
    # contains Fe but exclude O -> none (Fe only appears with O here)
    assert set(cdb.query_by_elements(contains_all=["Fe"], exclude=["O"])) == set()


def test_query_by_elements_requires_arg(cdb):
    with pytest.raises(ValueError):
        list(cdb.query_by_elements())


def test_to_frame(cdb):
    frame = cdb.to_frame(source="mp-20")
    assert list(frame["source"].unique()) == ["mp-20"]
    assert len(frame) == 2
    assert "structure_blob" not in frame.columns
    assert {"space_group", "split", "n_atoms"} <= set(frame.columns)
    assert len(cdb.to_frame(limit=1)) == 1
