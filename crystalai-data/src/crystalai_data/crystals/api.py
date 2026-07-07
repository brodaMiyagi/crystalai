"""``CrystalDatabase`` — the read query interface over ``crystals.sqlite``.

This is the contract the downstream packages consume (DATA_ROADMAP.md §1 Query
API, §4): ``crystalai_simxrd`` iterates structures to simulate; ``crystalai_methods``
selects Track-A / Track-B pools by ``source`` / ``max_atoms`` / ``split`` and does
matched-sample lookups. Nothing here writes — ingest is the converters' job.

The Track-A / Track-B and ICSD / MP-20 distinctions are **query filters on the one
table**, not separate stores::

    db.iter_all(source="icsd")                       # Track A: full ICSD pool
    db.iter_all(max_atoms=20, exclude_disordered=True)  # Track B internal subset
    db.iter_all(source="mp-20", max_atoms=20, split="train")  # released generator set

Structure iterators yield pymatgen ``Structure`` objects with disorder preserved,
ready for the simXRD Bragg calculator. For the training DataLoader's random-access
pattern, build an id list once with :meth:`ids` and point-look-up with :meth:`get`;
each worker process should open its own ``CrystalDatabase`` (its own read-only
connection).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterator

from pymatgen.core import Structure

from .database import deserialize

__all__ = ["CrystalDatabase"]

# Every metadata column except the structure blob (the blob is unreadable in a
# DataFrame; to_frame is metadata-only by design — DATA_ROADMAP.md §1).
_METADATA_COLUMNS = (
    "id, source, source_id, source_file_id, space_group, crystal_system, "
    "formula, chemsys, anonymized_formula, n_elements, n_atoms, n_sites, "
    "volume, density, a, b, c, alpha, beta, gamma, "
    "is_disordered, n_atoms_le_20, split"
)


def _placeholders(n: int) -> str:
    return ", ".join("?" for _ in range(n))


class CrystalDatabase:
    """Read-only query layer over a crystals SQLite database.

    Open with a path; use as a context manager or call :meth:`close`. The
    connection is opened read-only (``mode=ro``) so the API can never corrupt the
    file — and reads over the NFS-hosted DB are safe (only live writes aren't).
    """

    def __init__(self, path: str | Path):
        self.path = str(path)
        if not Path(self.path).exists():
            raise FileNotFoundError(self.path)
        # check_same_thread=False: tolerate use from a DataLoader worker thread;
        # read-only so concurrent reads are safe.
        self._conn = sqlite3.connect(
            f"file:{self.path}?mode=ro", uri=True, check_same_thread=False
        )

    # -- lifecycle --------------------------------------------------------- #
    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "CrystalDatabase":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- single-row access ------------------------------------------------- #
    def get(self, id: int) -> Structure:
        """Deserialize one structure by primary key. ``KeyError`` if absent."""
        row = self._conn.execute(
            "SELECT structure_blob FROM crystals WHERE id = ?", (id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"no crystal with id={id}")
        return deserialize(row[0])

    def get_cif(self, id: int) -> str:
        """Reconstruct a CIF string for one structure (manual inspection only).

        Written from the stored cell in P1 (no symmetry re-analysis); the
        authoritative space group lives in the ``space_group`` column.
        """
        from pymatgen.io.cif import CifWriter

        return str(CifWriter(self.get(id)))

    def get_metadata(self, id: int) -> dict:
        """Return the metadata columns (no blob) for one row as a dict."""
        cur = self._conn.execute(
            f"SELECT {_METADATA_COLUMNS} FROM crystals WHERE id = ?", (id,)
        )
        row = cur.fetchone()
        if row is None:
            raise KeyError(f"no crystal with id={id}")
        return dict(zip([d[0] for d in cur.description], row, strict=True))

    # -- filtering --------------------------------------------------------- #
    @staticmethod
    def _build_where(
        *,
        source: str | None = None,
        space_group: int | None = None,
        crystal_system: int | None = None,
        max_atoms: int | None = None,
        min_atoms: int | None = None,
        split: str | None = None,
        exclude_disordered: bool = False,
        formula: str | None = None,
        anonymized_formula: str | None = None,
        chemsys: str | None = None,
    ) -> tuple[str, list]:
        clauses: list[str] = []
        params: list = []

        def add(clause: str, value=None) -> None:
            clauses.append(clause)
            if value is not None:
                params.append(value)

        if source is not None:
            add("source = ?", source)
        if space_group is not None:
            add("space_group = ?", int(space_group))
        if crystal_system is not None:
            add("crystal_system = ?", int(crystal_system))
        if max_atoms is not None:
            add("n_atoms <= ?", int(max_atoms))
        if min_atoms is not None:
            add("n_atoms >= ?", int(min_atoms))
        if split is not None:
            add("split = ?", split)
        if exclude_disordered:
            add("is_disordered = 0")
        if formula is not None:
            add("formula = ?", formula)
        if anonymized_formula is not None:
            add("anonymized_formula = ?", anonymized_formula)
        if chemsys is not None:
            add("chemsys = ?", chemsys)

        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        return where, params

    def ids(self, **filters) -> list[int]:
        """All ids matching ``filters`` (see :meth:`_build_where`), ascending.

        The id list a training DataLoader builds once and indexes with :meth:`get`.
        """
        where, params = self._build_where(**filters)
        cur = self._conn.execute(
            f"SELECT id FROM crystals{where} ORDER BY id", params
        )
        return [r[0] for r in cur]

    def count(self, **filters) -> int:
        """Number of rows matching ``filters``."""
        where, params = self._build_where(**filters)
        return self._conn.execute(
            f"SELECT COUNT(*) FROM crystals{where}", params
        ).fetchone()[0]

    def iter_all(
        self,
        source: str | None = None,
        max_atoms: int | None = None,
        split: str | None = None,
        exclude_disordered: bool = False,
    ) -> Iterator[Structure]:
        """Main training iterator: stream ``Structure`` objects matching filters.

        ``source='icsd'`` is the Track-A pool; ``max_atoms=20`` the Track-B
        size-bounded subset; the filters compose (e.g. ``source='mp-20',
        max_atoms=20`` for the redistributable generator set).
        """
        yield from self._iter_structures(
            source=source,
            max_atoms=max_atoms,
            split=split,
            exclude_disordered=exclude_disordered,
        )

    def iter_by_space_group(self, sg: int) -> Iterator[Structure]:
        yield from self._iter_structures(space_group=sg)

    def iter_by_crystal_system(self, cs: int) -> Iterator[Structure]:
        yield from self._iter_structures(crystal_system=cs)

    def _iter_structures(self, **filters) -> Iterator[Structure]:
        where, params = self._build_where(**filters)
        # Separate cursor so the generator streams blobs lazily (no full
        # materialization), independent of other queries on the connection.
        cur = self._conn.execute(
            f"SELECT structure_blob FROM crystals{where} ORDER BY id", params
        )
        for (blob,) in cur:
            yield deserialize(blob)

    # -- composition membership (crystal_elements join) -------------------- #
    def query_by_elements(
        self,
        contains_all: list[str] | None = None,
        exact: list[str] | None = None,
        exclude: list[str] | None = None,
    ) -> Iterator[int]:
        """Composition membership queries via the ``crystal_elements`` join.

        - ``contains_all={'Fe','O'}`` — has *at least* these elements.
        - ``exact={'Ti','O'}`` — has *exactly* this element set, nothing else.
        - ``exclude={'C'}`` — has *none* of these (combinable with the above).

        Yields crystal ids. Pass at least one argument. These go through the
        indexed junction table rather than substring-matching ``chemsys`` (which
        is buggy: 'O' matches 'Os'). ``exact`` takes precedence over
        ``contains_all`` if both are given (exact is the stricter constraint).
        """
        if not (contains_all or exact or exclude):
            raise ValueError("query_by_elements needs at least one of "
                             "contains_all / exact / exclude")

        params: list = []
        if exact:
            els = sorted(set(exact))
            base = (
                "SELECT crystal_id FROM crystal_elements GROUP BY crystal_id "
                f"HAVING COUNT(DISTINCT element) = ? AND "
                f"SUM(element IN ({_placeholders(len(els))})) = COUNT(*)"
            )
            params += [len(els), *els]
        elif contains_all:
            els = sorted(set(contains_all))
            base = (
                f"SELECT crystal_id FROM crystal_elements "
                f"WHERE element IN ({_placeholders(len(els))}) "
                "GROUP BY crystal_id HAVING COUNT(DISTINCT element) = ?"
            )
            params += [*els, len(els)]
        else:
            base = "SELECT id AS crystal_id FROM crystals"

        if exclude:
            exl = sorted(set(exclude))
            sql = (
                f"SELECT crystal_id FROM ({base}) WHERE crystal_id NOT IN "
                f"(SELECT crystal_id FROM crystal_elements "
                f"WHERE element IN ({_placeholders(len(exl))}))"
            )
            params += exl
        else:
            sql = base

        for (cid,) in self._conn.execute(sql, params):
            yield cid

    # -- analytics --------------------------------------------------------- #
    def to_frame(self, limit: int | None = None, **filters):
        """Metadata-only DataFrame (no blobs) for exploratory analysis.

        Thin wrapper over ``pd.read_sql_query``; accepts the same filter kwargs
        as :meth:`count`. The structure blob is intentionally excluded.
        """
        import pandas as pd

        where, params = self._build_where(**filters)
        sql = f"SELECT {_METADATA_COLUMNS} FROM crystals{where} ORDER BY id"
        if limit is not None:
            sql += " LIMIT ?"
            params = [*params, int(limit)]
        return pd.read_sql_query(sql, self._conn, params=params)
