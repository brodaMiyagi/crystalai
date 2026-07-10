"""Fork-safe store for precomputed Bragg peak lists (DESIGN_DECISIONS §9).

The wavelength-independent ``{d, |F|², hkl, metric}`` per structure is precomputed
once (the expensive, heavy-tailed step) and cached here, keyed by ``cif_id`` (the
``crystals.sqlite`` id). At training time the DataLoader loads a peak list by id and
runs the cheap on-the-fly stage (profile build + effects + augmentation).

Storage mirrors the crystals DB's fork-safe design (DATA_ROADMAP §1): one SQLite
table of gzip-compressed ``.npz`` blobs, each worker holding its own read-only
connection — **not** 257k individual files (which defeat DataLoader random access).
"""

from __future__ import annotations

import gzip
import io
import os
import sqlite3
from pathlib import Path

import numpy as np

from ..core.bragg import BraggPeaks

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS bragg_peaks (
    cif_id  INTEGER PRIMARY KEY,
    d_min   REAL    NOT NULL,
    n_peaks INTEGER NOT NULL,
    blob    BLOB    NOT NULL
);
"""


# --------------------------------------------------------------------------- #
# (de)serialization
# --------------------------------------------------------------------------- #
def serialize(peaks: BraggPeaks) -> bytes:
    """BraggPeaks → gzipped npz bytes (d float64 — positions must be exact; f2 float32;
    hkl int16; metric f64)."""
    hkl = np.asarray(peaks.hkls, dtype=np.int16).reshape(-1, 3) if peaks.hkls else np.zeros((0, 3), np.int16)
    metric = np.asarray(peaks.metric_tensor, dtype=np.float64) if peaks.metric_tensor is not None else np.zeros((3, 3))
    buf = io.BytesIO()
    np.savez(
        buf,
        d=np.asarray(peaks.d_spacings, np.float64),
        f2=np.asarray(peaks.f_squared, np.float32),
        hkl=hkl,
        metric=metric,
    )
    return gzip.compress(buf.getvalue())


def deserialize(blob: bytes) -> BraggPeaks:
    """Inverse of :func:`serialize`."""
    with np.load(io.BytesIO(gzip.decompress(blob))) as z:
        hkl = z["hkl"]
        return BraggPeaks(
            d_spacings=z["d"].astype(np.float64),
            f_squared=z["f2"].astype(np.float64),
            hkls=[tuple(int(v) for v in row) for row in hkl],
            metric_tensor=z["metric"].astype(np.float64),
        )


# --------------------------------------------------------------------------- #
# writer helpers
# --------------------------------------------------------------------------- #
def connect(path: str | Path, *, bulk: bool = False) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    if bulk:
        conn.execute("PRAGMA synchronous = OFF")
        conn.execute("PRAGMA journal_mode = MEMORY")
    return conn


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)
    conn.commit()


def write_many(conn: sqlite3.Connection, items: list[tuple[int, BraggPeaks, float]]) -> int:
    """Insert/replace ``(cif_id, peaks, d_min)`` rows in one transaction."""
    conn.executemany(
        "INSERT OR REPLACE INTO bragg_peaks (cif_id, d_min, n_peaks, blob) VALUES (?,?,?,?)",
        [(int(cid), float(dmin), len(pk), serialize(pk)) for cid, pk, dmin in items],
    )
    conn.commit()
    return len(items)


def write_serialized(conn: sqlite3.Connection, items: list[tuple[int, bytes, int, float]]) -> int:
    """Insert/replace pre-serialized ``(cif_id, blob, n_peaks, d_min)`` rows (batch path)."""
    conn.executemany(
        "INSERT OR REPLACE INTO bragg_peaks (cif_id, d_min, n_peaks, blob) VALUES (?,?,?,?)",
        [(int(cid), float(dmin), int(n), blob) for cid, blob, n, dmin in items],
    )
    conn.commit()
    return len(items)


def existing_ids(conn: sqlite3.Connection) -> set[int]:
    return {r[0] for r in conn.execute("SELECT cif_id FROM bragg_peaks")}


# --------------------------------------------------------------------------- #
# reader (fork-safe: one connection per process)
# --------------------------------------------------------------------------- #
class BraggStore:
    """Read-only reader; opens a per-process connection on first use (fork-safe)."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        self._conn: sqlite3.Connection | None = None
        self._pid: int | None = None

    def _c(self) -> sqlite3.Connection:
        if self._conn is None or self._pid != os.getpid():
            self._conn = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
            self._pid = os.getpid()
        return self._conn

    def get(self, cif_id: int) -> BraggPeaks | None:
        row = self._c().execute(
            "SELECT blob FROM bragg_peaks WHERE cif_id = ?", (int(cif_id),)
        ).fetchone()
        return None if row is None else deserialize(row[0])

    def __contains__(self, cif_id: int) -> bool:
        return self._c().execute(
            "SELECT 1 FROM bragg_peaks WHERE cif_id = ?", (int(cif_id),)
        ).fetchone() is not None

    def __len__(self) -> int:
        return self._c().execute("SELECT COUNT(*) FROM bragg_peaks").fetchone()[0]
