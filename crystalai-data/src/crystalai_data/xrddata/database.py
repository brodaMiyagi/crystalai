"""Unified experimental-PXRD store: canonical text files + one CSV index.

Every experimental pattern (RRUFF, opXRD, internal lab) is normalized into one
layout under a single store root (``exp_data/`` by default, gitignored and
published to HuggingFace separately). Each pattern owns a directory::

    patterns/<source>/<source_id>/
        raw.xy        # as-measured, native coordinate (required)
        bgsub.xy      # background-subtracted (absent => null)
        peaks.xy      # peak list, SAME coordinate as raw (absent => null)
        structure.cif # phase(s); multi-block if >1 phase (absent => null)

and one row in ``index.csv``. The canonical files are the source of truth;
``index.csv`` is rebuildable from them. Nothing is resampled or re-binned at
ingest — all binning / coordinate conversion / background subtraction is
deferred to the downstream ``__getitem__`` (DATA_ROADMAP.md §2).

This module owns the *storage* concern only: the index schema, the canonical
file writers, the source-scoped index merge, and the ``XRDDatabase`` resolver.
The per-source ``ingest_*.py`` scripts own *where the rows come from*.

CSV (not SQLite) is deliberate: at ~5k rows the index loads once into a
DataFrame served from memory, so there is no random-access-across-processes
problem to solve and CSV stays the lighter, greppable choice (DATA_ROADMAP.md §2).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import pandas as pd

__all__ = [
    "INDEX_COLUMNS",
    "IndexRow",
    "default_store",
    "pattern_dir",
    "write_xy",
    "write_cif",
    "read_xy",
    "read_index",
    "write_source_rows",
    "XRDDatabase",
]

# Column order for index.csv (DATA_ROADMAP.md §2 "Unified index schema").
# ``id`` is assigned at write time (deterministic, see write_source_rows) and is
# not part of IndexRow, which the ingesters build before ids are known.
INDEX_COLUMNS = (
    "id",
    "source",
    "source_id",
    "raw_path",
    "bgsub_path",
    "autobg",
    "peaks_path",
    "x_coord",
    "wavelength_A",
    "format",
    "crystal_system",
    "space_group",
    "a",
    "b",
    "c",
    "alpha",
    "beta",
    "gamma",
    "formula",
    "n_phases",
    "cif_id",
    "cif_path",
    "quality_flag",
    "audit",
    "notes",
)


@dataclass(slots=True)
class IndexRow:
    """One experimental pattern's index row (file paths relative to the store root).

    Built by the ingesters *before* ids exist; ``id`` is assigned deterministically
    at write time. Nullable label columns default to ``None`` so a labels-only or
    unlabelled pattern is expressed by leaving them unset.
    """

    source: str  # 'rruff' | 'opxrd' | 'lab'
    source_id: str  # unique within source; (source, source_id) is the store key
    raw_path: str  # as-measured .xy, relative to store root (required)
    x_coord: str  # 'two_theta' | 'd' | 'tof'
    format: str  # raw-loader tag ('xy', 'xrdml', 'raw_stadi', ...)
    bgsub_path: str | None = None
    # Provenance of bgsub_path: 1 = auto-generated (pybaselines, file named
    # ``bgsub_autobg.xy``); 0 = native/curated (e.g. RRUFF XY_Processed,
    # ``bgsub.xy``); null = no background-subtracted view exists.
    autobg: int | None = None
    peaks_path: str | None = None
    wavelength_A: float | None = None
    crystal_system: int | None = None
    space_group: int | None = None
    a: float | None = None
    b: float | None = None
    c: float | None = None
    alpha: float | None = None
    beta: float | None = None
    gamma: float | None = None
    formula: str | None = None
    n_phases: int = 1
    cif_id: int | None = None
    cif_path: str | None = None
    quality_flag: str = "clean"  # clean | noisy | multi_phase | amorphous
    # Wavelength-audit verdict, set by audit.py (null until audited):
    #   'pass' | 'missing_wavelength' | 'suspicious_wavelength'. Training filters
    #   to 'pass'; 'missing_wavelength' rows are retained for the wavelength-sweep
    #   experiments (angle-dispersive patterns testable against assumed Kα lines).
    audit: str | None = None
    notes: str | None = None

    def as_record(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Store paths
# --------------------------------------------------------------------------- #
def default_store() -> Path:
    """Default store root: ``crystalai-data/exp_data/``.

    ``.../src/crystalai_data/xrddata/database.py`` -> ``parents[3]`` is the
    ``crystalai-data`` package directory. Overridable by the ingesters / apps via
    an explicit path or the ``CRYSTALAI_EXP_DATA_FOLDER`` env key.
    """
    return Path(__file__).resolve().parents[3] / "exp_data"


def resolve_store(env: dict[str, str], root: Path) -> Path:
    """Experimental-store root from ``CRYSTALAI_EXP_DATA_FOLDER`` (resolved relative to the
    monorepo ``root`` when not absolute), or :func:`default_store` when the key is unset."""
    val = env.get("CRYSTALAI_EXP_DATA_FOLDER")
    if not val:
        return default_store()
    p = Path(val)
    return p if p.is_absolute() else (root / p)


def pattern_dir(store: Path, source: str, source_id: str) -> Path:
    """Absolute directory for one pattern's canonical files (created on demand)."""
    d = Path(store) / "patterns" / source / source_id
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------- #
# Canonical file IO — verbatim, never resampled
# --------------------------------------------------------------------------- #
def write_xy(
    path: Path,
    xs: Sequence[float],
    ys: Sequence[float],
    *,
    header: Sequence[str] | None = None,
) -> None:
    """Write a two-column ``.xy`` file, values in their native coordinate.

    ``repr`` is used per value so the float round-trips exactly — the store is
    lossless (no re-binning, no precision loss). Header lines are written as
    ``#``-prefixed comments and ignored by :func:`read_xy`.
    """
    if len(xs) != len(ys):
        raise ValueError(f"xs/ys length mismatch: {len(xs)} vs {len(ys)}")
    lines: list[str] = []
    if header:
        lines.extend(f"# {h}" for h in header)
    lines.extend(f"{x!r} {y!r}" for x, y in zip(xs, ys, strict=True))
    path.write_text("\n".join(lines) + "\n")


def write_cif(path: Path, text: str) -> None:
    """Write CIF text verbatim (single- or multi-block)."""
    path.write_text(text if text.endswith("\n") else text + "\n")


def read_xy(path: Path) -> tuple[list[float], list[float]]:
    """Read a canonical ``.xy`` file back into (xs, ys), skipping ``#`` comments."""
    xs: list[float] = []
    ys: list[float] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").split()
        if len(parts) < 2:
            continue
        xs.append(float(parts[0]))
        ys.append(float(parts[1]))
    return xs, ys


# --------------------------------------------------------------------------- #
# Index IO
# --------------------------------------------------------------------------- #
def index_path(store: Path) -> Path:
    return Path(store) / "index.csv"


def read_index(store: Path) -> pd.DataFrame:
    """Load ``index.csv`` (empty, correctly-typed frame if absent)."""
    p = index_path(store)
    if not p.is_file():
        return pd.DataFrame(columns=list(INDEX_COLUMNS))
    return pd.read_csv(p)


def write_source_rows(store: Path, source: str, rows: Iterable[IndexRow]) -> int:
    """Replace all rows for ``source`` with ``rows`` and rewrite ``index.csv``.

    An ingester is idempotent at the source granularity: re-running it drops the
    source's previous rows and writes the fresh set, leaving other sources
    untouched. Ids are (re)assigned deterministically over the combined table —
    sorted by ``(source, source_id)`` then numbered 0..N-1 — so a full rebuild
    from the same inputs reproduces the same ids. Returns the number of rows
    written for ``source``.
    """
    store = Path(store)
    store.mkdir(parents=True, exist_ok=True)

    existing = read_index(store)
    kept = existing[existing["source"] != source] if len(existing) else existing

    new_df = pd.DataFrame([r.as_record() for r in rows])
    n_written = len(new_df)

    frames = [f for f in (kept, new_df) if len(f)]
    combined = pd.concat(frames, ignore_index=True) if frames else new_df
    # Deterministic id assignment: stable sort by (source, source_id).
    if "id" in combined.columns:
        combined = combined.drop(columns=["id"])
    combined = combined.sort_values(
        ["source", "source_id"], kind="stable"
    ).reset_index(drop=True)
    combined.insert(0, "id", range(len(combined)))

    # Enforce the canonical column order (any missing column added as null).
    for col in INDEX_COLUMNS:
        if col not in combined.columns:
            combined[col] = pd.NA
    combined = combined[list(INDEX_COLUMNS)]

    combined.to_csv(index_path(store), index=False)
    return n_written


def write_index_frame(store: Path, df: pd.DataFrame) -> None:
    """Persist a full index DataFrame back to ``index.csv``, column order enforced.

    Used by cross-source passes (e.g. the background generator) that mutate
    existing rows in place rather than replacing a whole source. The frame must
    already carry ``id`` (these passes never renumber). Any missing canonical
    column is added as null.
    """
    df = df.copy()
    for col in INDEX_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df[list(INDEX_COLUMNS)].to_csv(index_path(Path(store)), index=False)


# --------------------------------------------------------------------------- #
# Resolver
# --------------------------------------------------------------------------- #
class XRDDatabase:
    """Read-only view over the experimental store: index frame + file resolver.

    The index is loaded once into a DataFrame (served from memory). Pattern
    arrays are read on demand from the canonical ``.xy`` files by ``id``.
    """

    def __init__(self, store: str | Path | None = None):
        self.store = Path(store) if store is not None else default_store()
        self.index = read_index(self.store)

    def __len__(self) -> int:
        return len(self.index)

    def to_frame(self, **filters) -> pd.DataFrame:
        """Metadata frame, optionally filtered by exact column matches.

        e.g. ``to_frame(source='rruff')`` or ``to_frame(x_coord='two_theta')``.
        """
        df = self.index
        for col, val in filters.items():
            df = df[df[col] == val]
        return df.reset_index(drop=True)

    def row(self, pattern_id: int) -> pd.Series:
        hit = self.index[self.index["id"] == pattern_id]
        if hit.empty:
            raise KeyError(f"no pattern with id={pattern_id}")
        return hit.iloc[0]

    def _resolve(self, rel: object) -> Path | None:
        if rel is None or (isinstance(rel, float) and pd.isna(rel)):
            return None
        return self.store / str(rel)

    def load_pattern(
        self, pattern_id: int, which: str = "raw"
    ) -> tuple[list[float], list[float]]:
        """Load a pattern's (x, y) from its canonical file.

        ``which`` selects the view: ``'raw'``, ``'bgsub'`` or ``'peaks'``.
        Raises ``FileNotFoundError`` if that view is absent for the pattern.
        """
        col = {"raw": "raw_path", "bgsub": "bgsub_path", "peaks": "peaks_path"}[which]
        path = self._resolve(self.row(pattern_id)[col])
        if path is None:
            raise FileNotFoundError(f"pattern {pattern_id} has no '{which}' view")
        return read_xy(path)

    def cif_text(self, pattern_id: int) -> str | None:
        """Standalone CIF text for a pattern, or None if it has no stored structure."""
        path = self._resolve(self.row(pattern_id)["cif_path"])
        return None if path is None else path.read_text()

    def iter_ids(self, **filters) -> Iterator[int]:
        yield from self.to_frame(**filters)["id"].tolist()


# Keep the dataclass field set and the CSV column set in sync at import time —
# a drift between them is a schema bug that should fail loudly, not silently.
_row_fields = {f.name for f in fields(IndexRow)}
_missing = _row_fields - set(INDEX_COLUMNS)
assert not _missing, f"IndexRow fields absent from INDEX_COLUMNS: {_missing}"
