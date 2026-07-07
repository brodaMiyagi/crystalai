"""SQLite storage + pymatgen ``Structure`` serialization for the crystals DB.

One table (``crystals``) holds every ICSD and MP-20 structure as a
gzip-compressed compact dict alongside indexed metadata columns; a junction
table (``crystal_elements``) backs relational composition queries. See
DATA_ROADMAP.md §1 for the schema rationale and column roles.

This module owns the *storage* concern only: serialization, schema DDL,
connection setup, metadata extraction from a ``Structure``, and the
single-transaction insert of a crystal plus its element rows. The ingest scripts
(``convert_icsd.py`` / ``convert_mp20.py``) own *where the rows come from*.
"""

from __future__ import annotations

import gzip
import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from pymatgen.core import Lattice, Structure

__all__ = [
    "serialize",
    "deserialize",
    "structure_to_compact_dict",
    "compact_dict_to_structure",
    "compute_metadata",
    "CrystalRow",
    "connect",
    "create_schema",
    "insert_crystal",
    "insert_many",
    "existing_source_file_ids",
    "existing_source_ids",
    "SCHEMA_SQL",
]


# --------------------------------------------------------------------------- #
# Compact Structure serialization (DATA_ROADMAP.md §1)
# --------------------------------------------------------------------------- #
def structure_to_compact_dict(s: Structure) -> dict:
    return {
        "lattice": s.lattice.matrix.tolist(),
        "sites": [
            {
                "species": [
                    {"el": sp.symbol, "occ": occ} for sp, occ in site.species.items()
                ],
                "abc": list(site.frac_coords),
            }
            for site in s
        ],
    }


def compact_dict_to_structure(d: dict) -> Structure:
    return Structure(
        lattice=Lattice(d["lattice"]),
        species=[
            {sp["el"]: sp["occ"] for sp in site["species"]} for site in d["sites"]
        ],
        coords=[site["abc"] for site in d["sites"]],
    )


def serialize(s: Structure) -> bytes:
    """Structure -> gzipped compact-dict JSON bytes (preserves occupancies)."""
    return gzip.compress(json.dumps(structure_to_compact_dict(s)).encode())


def deserialize(blob: bytes) -> Structure:
    """Inverse of :func:`serialize`."""
    return compact_dict_to_structure(json.loads(gzip.decompress(blob).decode()))


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS crystals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source          TEXT    NOT NULL,
    source_id       TEXT    NOT NULL,
    source_file_id  TEXT,
    structure_blob  BLOB    NOT NULL,

    space_group     INTEGER NOT NULL,
    crystal_system  INTEGER NOT NULL,

    formula             TEXT    NOT NULL,
    chemsys             TEXT    NOT NULL,
    anonymized_formula  TEXT    NOT NULL,
    n_elements          INTEGER NOT NULL,
    n_atoms             INTEGER NOT NULL,
    n_sites             INTEGER NOT NULL,

    volume          REAL    NOT NULL,
    density         REAL    NOT NULL,
    a               REAL    NOT NULL,
    b               REAL    NOT NULL,
    c               REAL    NOT NULL,
    alpha           REAL    NOT NULL,
    beta            REAL    NOT NULL,
    gamma           REAL    NOT NULL,

    is_disordered   INTEGER NOT NULL,
    n_atoms_le_20   INTEGER NOT NULL,

    split           TEXT,

    UNIQUE(source, source_id)
);

CREATE INDEX IF NOT EXISTS idx_source         ON crystals(source);
CREATE INDEX IF NOT EXISTS idx_space_group    ON crystals(space_group);
CREATE INDEX IF NOT EXISTS idx_crystal_system ON crystals(crystal_system);
CREATE INDEX IF NOT EXISTS idx_formula        ON crystals(formula);
CREATE INDEX IF NOT EXISTS idx_chemsys        ON crystals(chemsys);
CREATE INDEX IF NOT EXISTS idx_anon_formula   ON crystals(anonymized_formula);
CREATE INDEX IF NOT EXISTS idx_n_atoms_le_20  ON crystals(n_atoms_le_20);
CREATE INDEX IF NOT EXISTS idx_split          ON crystals(split);

CREATE TABLE IF NOT EXISTS crystal_elements (
    crystal_id  INTEGER NOT NULL,
    element     TEXT    NOT NULL,
    amount      REAL    NOT NULL,
    FOREIGN KEY (crystal_id) REFERENCES crystals(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_ce_element ON crystal_elements(element);
CREATE INDEX IF NOT EXISTS idx_ce_crystal ON crystal_elements(crystal_id);
"""

# Column order for the crystals INSERT (id/split omitted: autoincrement / set later).
_INSERT_COLUMNS = (
    "source",
    "source_id",
    "source_file_id",
    "structure_blob",
    "space_group",
    "crystal_system",
    "formula",
    "chemsys",
    "anonymized_formula",
    "n_elements",
    "n_atoms",
    "n_sites",
    "volume",
    "density",
    "a",
    "b",
    "c",
    "alpha",
    "beta",
    "gamma",
    "is_disordered",
    "n_atoms_le_20",
    "split",
)

_INSERT_CRYSTAL_SQL = (
    f"INSERT OR IGNORE INTO crystals ({', '.join(_INSERT_COLUMNS)}) "
    f"VALUES ({', '.join('?' for _ in _INSERT_COLUMNS)})"
)


# --------------------------------------------------------------------------- #
# Row container + metadata extraction
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class CrystalRow:
    """A fully-prepared crystals row plus its element list, ready to insert.

    ``elements`` is the list of ``(element_symbol, per-formula-unit amount)``
    pairs written to ``crystal_elements`` in the same transaction.
    """

    source: str
    source_id: str
    source_file_id: str | None
    structure_blob: bytes
    space_group: int
    crystal_system: int
    formula: str
    chemsys: str
    anonymized_formula: str
    n_elements: int
    n_atoms: int
    n_sites: int
    volume: float
    density: float
    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float
    is_disordered: int
    n_atoms_le_20: int
    split: str | None = None
    elements: list[tuple[str, float]] = field(default_factory=list)

    def _crystal_values(self) -> tuple:
        return tuple(getattr(self, col) for col in _INSERT_COLUMNS)


def _chemsys_sentinel(symbols: list[str]) -> str:
    """Element set as a sentinel-delimited string, e.g. ['Ti','O'] -> '-O-Ti-'."""
    return "-" + "-".join(sorted(symbols)) + "-"


def compute_metadata(
    structure: Structure,
    *,
    source: str,
    source_id: str,
    space_group: int,
    crystal_system: int,
    source_file_id: str | None = None,
    split: str | None = None,
) -> CrystalRow:
    """Derive all query columns + element rows from a parsed ``Structure``.

    The caller supplies the authoritative ``space_group`` (read from the source,
    not re-derived) and the matching ``crystal_system`` (from ``sg_to_cs``).
    Everything else — composition keys, cell scalars, disorder/size flags, the
    serialized blob and the element list — is computed here from the structure.
    ``split`` is set at ingest for MP-20 (native train/val/test); null for ICSD
    (assigned later by a CS-stratified split).
    """
    # Strip oxidation states and sum per element: ICSD CIFs decorate sites with
    # ionic species (Hg2+, S2-), and the same element can appear in two oxidation
    # states on different sites. The query columns and the element junction table
    # must key on the bare element ('Hg', not 'Hg2+') for membership queries to
    # work and for chemsys/n_elements not to double-count. This also matches the
    # blob, whose serializer stores bare element symbols (DATA_ROADMAP.md §1).
    comp = structure.composition.element_composition
    symbols = [el.symbol for el in comp.elements]

    # Total atoms in the as-parsed cell (fractional under disorder -> round to an
    # integer count); n_sites is the distinct crystallographic site count.
    n_atoms = int(round(comp.num_atoms))
    n_sites = len(structure)

    abc = structure.lattice.abc
    angles = structure.lattice.angles

    # Per-formula-unit element amounts (fractional under disorder).
    reduced = comp.reduced_composition.as_dict()
    elements = [(sym, float(amt)) for sym, amt in reduced.items()]

    return CrystalRow(
        source=source,
        source_id=str(source_id),
        source_file_id=None if source_file_id is None else str(source_file_id),
        structure_blob=serialize(structure),
        space_group=int(space_group),
        crystal_system=int(crystal_system),
        formula=comp.reduced_formula,
        chemsys=_chemsys_sentinel(symbols),
        anonymized_formula=comp.anonymized_formula,
        n_elements=len(symbols),
        n_atoms=n_atoms,
        n_sites=n_sites,
        volume=float(structure.volume),
        density=float(structure.density),
        a=float(abc[0]),
        b=float(abc[1]),
        c=float(abc[2]),
        alpha=float(angles[0]),
        beta=float(angles[1]),
        gamma=float(angles[2]),
        is_disordered=0 if structure.is_ordered else 1,
        n_atoms_le_20=1 if n_atoms <= 20 else 0,
        split=split,
        elements=elements,
    )


# --------------------------------------------------------------------------- #
# Connection + insert
# --------------------------------------------------------------------------- #
def connect(path: str | Path, *, bulk: bool = False) -> sqlite3.Connection:
    """Open (creating if needed) the crystals SQLite database.

    ``bulk=True`` trades crash-durability for write throughput during a one-shot
    ingest (``synchronous=OFF``, in-memory journal). The default is safe.
    Foreign keys are enabled so ``crystal_elements`` cascades on delete.
    """
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA foreign_keys = ON")
    if bulk:
        conn.execute("PRAGMA synchronous = OFF")
        conn.execute("PRAGMA journal_mode = MEMORY")
        conn.execute("PRAGMA temp_store = MEMORY")
    return conn


def create_schema(conn: sqlite3.Connection) -> None:
    """Create tables + indexes if absent (idempotent)."""
    conn.executescript(SCHEMA_SQL)
    conn.commit()


def insert_crystal(conn: sqlite3.Connection, row: CrystalRow) -> int | None:
    """Insert one crystal and its element rows in a single transaction.

    Returns the new ``crystals.id``, or ``None`` if the row was a duplicate on
    ``UNIQUE(source, source_id)`` (``INSERT OR IGNORE`` skipped it). Commits on
    success; rolls back on error so a partial crystal/element write never lands.
    """
    try:
        cur = conn.execute(_INSERT_CRYSTAL_SQL, row._crystal_values())
        if cur.rowcount == 0:  # duplicate ignored
            conn.rollback()
            return None
        crystal_id = cur.lastrowid
        if row.elements:
            conn.executemany(
                "INSERT INTO crystal_elements (crystal_id, element, amount) "
                "VALUES (?, ?, ?)",
                [(crystal_id, el, amt) for el, amt in row.elements],
            )
        conn.commit()
        return crystal_id
    except Exception:
        conn.rollback()
        raise


def insert_many(conn: sqlite3.Connection, rows: list[CrystalRow]) -> int:
    """Insert a batch of crystals + their element rows in one transaction.

    Bulk path for the ingest scripts: a single commit amortizes the per-row cost
    across the whole batch. Duplicates on ``UNIQUE(source, source_id)`` are
    skipped (``INSERT OR IGNORE``). Returns the number of rows actually inserted.
    Rolls back the entire batch on error.
    """
    inserted = 0
    try:
        for row in rows:
            cur = conn.execute(_INSERT_CRYSTAL_SQL, row._crystal_values())
            if cur.rowcount == 0:  # duplicate ignored
                continue
            inserted += 1
            if row.elements:
                crystal_id = cur.lastrowid
                conn.executemany(
                    "INSERT INTO crystal_elements (crystal_id, element, amount) "
                    "VALUES (?, ?, ?)",
                    [(crystal_id, el, amt) for el, amt in row.elements],
                )
        conn.commit()
        return inserted
    except Exception:
        conn.rollback()
        raise


def existing_source_file_ids(conn: sqlite3.Connection, source: str) -> set[str]:
    """Return the set of non-null ``source_file_id`` values already in the DB
    for ``source`` — lets an ingest resume cheaply by skipping done files."""
    cur = conn.execute(
        "SELECT source_file_id FROM crystals "
        "WHERE source = ? AND source_file_id IS NOT NULL",
        (source,),
    )
    return {r[0] for r in cur}


def existing_source_ids(conn: sqlite3.Connection, source: str) -> set[str]:
    """Return the set of ``source_id`` values already in the DB for ``source``.

    The dedup/resume key for sources without an on-disk file id (MP-20, whose
    ``source_id`` is the MP material id and ``source_file_id`` is null)."""
    cur = conn.execute(
        "SELECT source_id FROM crystals WHERE source = ?", (source,)
    )
    return {r[0] for r in cur}
