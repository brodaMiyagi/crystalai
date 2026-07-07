"""Tests for serialization, schema, metadata extraction, and inserts."""

import pytest
from pymatgen.core import Lattice, Structure

from crystalai_data.crystals import database as db


@pytest.fixture
def ordered_structure():
    # Rutile-like TiO2: ordered, tetragonal cell.
    lat = Lattice.tetragonal(4.59, 2.96)
    return Structure(
        lat,
        ["Ti", "Ti", "O", "O", "O", "O"],
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.3, 0.3, 0.0],
            [0.7, 0.7, 0.0],
            [0.2, 0.8, 0.5],
            [0.8, 0.2, 0.5],
        ],
    )


@pytest.fixture
def disordered_structure():
    # Mixed Mn/Cr site (fractional occupancy) + a partially-occupied O site.
    lat = Lattice.cubic(4.0)
    return Structure(
        lat,
        [{"Mn": 0.67, "Cr": 0.33}, {"O": 0.5}],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )


# --------------------------------------------------------------------------- #
# Serialization round-trip
# --------------------------------------------------------------------------- #
def test_roundtrip_ordered(ordered_structure):
    out = db.deserialize(db.serialize(ordered_structure))
    assert len(out) == len(ordered_structure)
    assert out.composition.reduced_formula == ordered_structure.composition.reduced_formula
    assert out.lattice.matrix == pytest.approx(ordered_structure.lattice.matrix)


def test_roundtrip_preserves_fractional_occupancy(disordered_structure):
    out = db.deserialize(db.serialize(disordered_structure))
    assert not out.is_ordered
    occ = out[0].species.as_dict()
    assert occ["Mn"] == pytest.approx(0.67)
    assert occ["Cr"] == pytest.approx(0.33)
    assert out[1].species.as_dict()["O"] == pytest.approx(0.5)


def test_blob_is_gzip(ordered_structure):
    blob = db.serialize(ordered_structure)
    assert blob[:2] == b"\x1f\x8b"  # gzip magic


# --------------------------------------------------------------------------- #
# Metadata extraction
# --------------------------------------------------------------------------- #
def test_compute_metadata_ordered(ordered_structure):
    row = db.compute_metadata(
        ordered_structure, source="icsd", source_id="999",
        space_group=136, crystal_system=4, source_file_id="12345",
    )
    assert row.formula == "TiO2"
    assert row.anonymized_formula == "AB2"
    assert row.chemsys == "-O-Ti-"
    assert row.n_elements == 2
    assert row.n_atoms == 6 and row.n_sites == 6
    assert row.is_disordered == 0
    assert row.n_atoms_le_20 == 1
    assert row.space_group == 136 and row.crystal_system == 4
    assert dict(row.elements) == {"Ti": pytest.approx(1.0), "O": pytest.approx(2.0)}
    assert row.c == pytest.approx(2.96)


def test_compute_metadata_strips_oxidation_states():
    # Fe in two oxidation states (Fe2+/Fe3+) on different sites, plus O2-:
    # element columns must key on bare elements, not double-count Fe.
    lat = Lattice.cubic(8.0)
    s = Structure(
        lat,
        ["Fe2+", "Fe3+", "Fe3+", "O2-", "O2-", "O2-", "O2-"],
        [[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0], [0.25, 0.25, 0.25],
         [0.75, 0.75, 0.25], [0.25, 0.75, 0.75], [0.75, 0.25, 0.75]],
    )
    row = db.compute_metadata(
        s, source="icsd", source_id="1", space_group=227, crystal_system=7,
    )
    assert row.n_elements == 2
    assert row.chemsys == "-Fe-O-"
    assert sorted(e for e, _ in row.elements) == ["Fe", "O"]
    assert dict(row.elements) == {"Fe": pytest.approx(3.0), "O": pytest.approx(4.0)}


def test_compute_metadata_disordered_flags(disordered_structure):
    row = db.compute_metadata(
        disordered_structure, source="icsd", source_id="1",
        space_group=225, crystal_system=7,
    )
    assert row.is_disordered == 1
    # 0.67+0.33 + 0.5 = 1.5 atoms -> rounds to 2; 2 distinct sites.
    assert row.n_sites == 2
    assert row.n_atoms == 2
    assert row.source_file_id is None


# --------------------------------------------------------------------------- #
# Schema + insert
# --------------------------------------------------------------------------- #
@pytest.fixture
def conn():
    c = db.connect(":memory:")
    db.create_schema(c)
    yield c
    c.close()


def test_insert_and_read_back(conn, ordered_structure):
    row = db.compute_metadata(
        ordered_structure, source="icsd", source_id="600252",
        space_group=136, crystal_system=4, source_file_id="100000",
    )
    cid = db.insert_crystal(conn, row)
    assert cid is not None

    got = conn.execute(
        "SELECT source, source_id, source_file_id, space_group, crystal_system, "
        "formula, n_atoms, is_disordered FROM crystals WHERE id=?", (cid,)
    ).fetchone()
    assert got == ("icsd", "600252", "100000", 136, 4, "TiO2", 6, 0)

    blob = conn.execute(
        "SELECT structure_blob FROM crystals WHERE id=?", (cid,)
    ).fetchone()[0]
    assert len(db.deserialize(blob)) == 6

    els = conn.execute(
        "SELECT element, amount FROM crystal_elements WHERE crystal_id=? "
        "ORDER BY element", (cid,)
    ).fetchall()
    assert els == [("O", 2.0), ("Ti", 1.0)]


def test_unique_constraint_dedup(conn, ordered_structure):
    row = db.compute_metadata(
        ordered_structure, source="icsd", source_id="600252",
        space_group=136, crystal_system=4, source_file_id="100000",
    )
    assert db.insert_crystal(conn, row) is not None
    # Same (source, source_id) -> ignored, returns None, no duplicate element rows.
    assert db.insert_crystal(conn, row) is None
    assert conn.execute("SELECT COUNT(*) FROM crystals").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM crystal_elements").fetchone()[0] == 2


def test_insert_many_batches_and_skips_dupes(conn, ordered_structure, disordered_structure):
    r1 = db.compute_metadata(
        ordered_structure, source="icsd", source_id="A",
        space_group=136, crystal_system=4, source_file_id="1",
    )
    r2 = db.compute_metadata(
        disordered_structure, source="icsd", source_id="B",
        space_group=225, crystal_system=7, source_file_id="2",
    )
    inserted = db.insert_many(conn, [r1, r2, r1])  # r1 repeated -> one ignored
    assert inserted == 2
    assert conn.execute("SELECT COUNT(*) FROM crystals").fetchone()[0] == 2


def test_existing_source_file_ids(conn, ordered_structure):
    row = db.compute_metadata(
        ordered_structure, source="icsd", source_id="X",
        space_group=136, crystal_system=4, source_file_id="42",
    )
    db.insert_crystal(conn, row)
    assert db.existing_source_file_ids(conn, "icsd") == {"42"}
    assert db.existing_source_file_ids(conn, "mp-20") == set()


def test_split_set_at_ingest_and_read_back(conn, ordered_structure):
    # MP-20 path: split set at ingest, source_file_id null, dedup on source_id.
    row = db.compute_metadata(
        ordered_structure, source="mp-20", source_id="mp-1234",
        space_group=136, crystal_system=4, source_file_id=None, split="val",
    )
    assert row.split == "val"
    cid = db.insert_crystal(conn, row)
    got = conn.execute(
        "SELECT source, split, source_file_id FROM crystals WHERE id=?", (cid,)
    ).fetchone()
    assert got == ("mp-20", "val", None)


def test_split_defaults_null_for_icsd(conn, ordered_structure):
    row = db.compute_metadata(
        ordered_structure, source="icsd", source_id="Y",
        space_group=136, crystal_system=4, source_file_id="7",
    )
    assert row.split is None
    cid = db.insert_crystal(conn, row)
    assert conn.execute("SELECT split FROM crystals WHERE id=?", (cid,)).fetchone()[0] is None


def test_existing_source_ids(conn, ordered_structure, disordered_structure):
    db.insert_crystal(conn, db.compute_metadata(
        ordered_structure, source="mp-20", source_id="mp-1",
        space_group=136, crystal_system=4, split="train",
    ))
    db.insert_crystal(conn, db.compute_metadata(
        disordered_structure, source="mp-20", source_id="mp-2",
        space_group=225, crystal_system=7, split="test",
    ))
    assert db.existing_source_ids(conn, "mp-20") == {"mp-1", "mp-2"}
    assert db.existing_source_ids(conn, "icsd") == set()
