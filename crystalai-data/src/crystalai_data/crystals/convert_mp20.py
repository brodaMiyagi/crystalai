"""Ingest the MP-20 train/val/test CSVs into ``crystals.sqlite``.

MP-20 is the DFT-relaxed Materials Project subset (<=20 atoms/cell, ordered,
redistributable) added solely for Track B's public generator release. It shares
the one ``crystals`` table with ICSD, distinguished by ``source='mp-20'``. See
DATA_ROADMAP.md §1 (convert_mp20) and DESIGN_DECISIONS.md §2/§8.

Each CSV row carries a full CIF string plus an authoritative ``spacegroup.number``.
Contract this implements:

  * Space group is **read from the CSV field** (``spacegroup.number``), *not* the
    CIF text: MP-20 CIFs are P1-expanded (``_symmetry_Int_Tables_number 1`` with
    every atom listed), so the CIF's internal symmetry is meaningless here.
  * Crystal system is **derived** from that SG via ``sg_to_cs``.
  * ``source_id`` is the MP material id (``mp-1234``); ``source_file_id`` is null.
  * ``split`` is set from the source file (train/val/test) at ingest — honoring
    MP-20's native split is a leakage-safety requirement (DESIGN_DECISIONS.md §8).
  * ``n_atoms_le_20`` is 1 by construction; a structure that parses to >20 atoms
    is flagged (but still inserted) as a data surprise.

Run (from the package directory)::

    uv run python -m crystalai_data.crystals.convert_mp20            # uses .env
    uv run python -m crystalai_data.crystals.convert_mp20 --limit 200
"""

from __future__ import annotations

import argparse
import os
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from pymatgen.core import Structure

from . import database as db
from ._ingest import load_dotenv, local_build, repo_root
from .symmetry import sg_to_cs

_SPLIT_FILES = ("train", "val", "test")
_SG_COLUMN = "spacegroup.number"
_ID_COLUMN = "material_id"
_CIF_COLUMN = "cif"


@dataclass(slots=True)
class _WorkItem:
    cif: str
    source_id: str
    sg: int
    split: str


def _process_row(item: _WorkItem) -> tuple[str, str, object]:
    """Parse one MP-20 row into a CrystalRow.

    Returns ``("ok", mp_id, (row, over20))`` or ``("skip", mp_id, reason)``.
    Runs in worker processes — self-contained and pickle-safe.
    """
    try:
        if not 1 <= item.sg <= 230:
            return ("skip", item.source_id, f"bad_sg:{item.sg}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            structure = Structure.from_str(item.cif, fmt="cif")

        row = db.compute_metadata(
            structure,
            source="mp-20",
            source_id=item.source_id,
            space_group=item.sg,
            crystal_system=sg_to_cs(item.sg),
            source_file_id=None,
            split=item.split,
        )
        return ("ok", item.source_id, (row, row.n_atoms_le_20 == 0))
    except Exception as exc:  # noqa: BLE001 — one bad row must not kill the run
        return ("skip", item.source_id, f"error:{type(exc).__name__}:{exc}")


def _load_split_items(
    data_dir: Path, skip_ids: set[str]
) -> tuple[list[_WorkItem], Counter]:
    """Read the three CSVs into work items, skipping already-ingested ids."""
    import pandas as pd

    items: list[_WorkItem] = []
    per_split: Counter = Counter()
    for split in _SPLIT_FILES:
        path = data_dir / f"{split}.csv"
        if not path.is_file():
            print(f"[mp-20] WARNING: missing {path.name}, skipping that split")
            continue
        df = pd.read_csv(path, usecols=[_ID_COLUMN, _CIF_COLUMN, _SG_COLUMN])
        per_split[split] = len(df)
        for mp_id, cif, sg in zip(
            df[_ID_COLUMN], df[_CIF_COLUMN], df[_SG_COLUMN], strict=True
        ):
            mp_id = str(mp_id)
            if mp_id in skip_ids:
                continue
            items.append(_WorkItem(cif=str(cif), source_id=mp_id, sg=int(sg), split=split))
    return items, per_split


def convert(
    *,
    data_dir: Path,
    db_path: Path,
    workers: int,
    batch_size: int,
    limit: int | None,
    work_dir: Path | None = None,
) -> None:
    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover
        def tqdm(x, **k):  # type: ignore
            return x

    print(f"[mp-20] data_dir     = {data_dir}")

    stats: Counter = Counter()
    split_inserted: Counter = Counter()
    over20 = 0
    inserted_total = 0

    with local_build(db_path, work_dir, bulk=True) as conn:
        skip_ids = db.existing_source_ids(conn, "mp-20")
        if skip_ids:
            print(f"[mp-20] resume: {len(skip_ids):,} MP-20 rows present, skipping")

        items, per_split = _load_split_items(data_dir, skip_ids)
        print(f"[mp-20] source rows  = {dict(per_split)} (total {sum(per_split.values()):,})")
        if limit is not None:
            items = items[:limit]
        print(f"[mp-20] to process   = {len(items):,} rows")

        batch: list[db.CrystalRow] = []

        def flush() -> None:
            nonlocal inserted_total, batch
            if batch:
                inserted_total += db.insert_many(conn, batch)
                batch = []

        def handle(result: tuple[str, str, object]) -> None:
            nonlocal over20
            status, _mp_id, payload = result
            if status == "ok":
                row, is_over20 = payload  # type: ignore[misc]
                if is_over20:
                    over20 += 1
                split_inserted[row.split] += 1
                batch.append(row)
                if len(batch) >= batch_size:
                    flush()
            else:
                stats[str(payload).split(":", 1)[0]] += 1

        if workers > 1:
            import multiprocessing as mp

            with mp.Pool(workers) as pool:
                for result in tqdm(
                    pool.imap_unordered(_process_row, items, chunksize=32),
                    total=len(items),
                    desc="[mp-20] ingest",
                ):
                    handle(result)
        else:
            for item in tqdm(items, desc="[mp-20] ingest"):
                handle(_process_row(item))

        flush()

    print(f"\n[mp-20] inserted     = {inserted_total:,} {dict(split_inserted)}")
    if stats:
        print(f"[mp-20] skipped      = {dict(stats)}")
    if over20:
        print(f"[mp-20] WARNING: {over20:,} inserted rows have >20 atoms (n_atoms_le_20=0)")


def main(argv: list[str] | None = None) -> None:
    root = repo_root()
    env = load_dotenv(root / ".env")

    data_default = env.get("MP20_DATA_FOLDER")
    db_default = env.get("CRYSTALS_DB", str(root / "crystalai-data" / "crystals.sqlite"))

    p = argparse.ArgumentParser(description="Ingest MP-20 CSVs into crystals.sqlite")
    p.add_argument("--data-dir", default=data_default, type=str, required=data_default is None)
    p.add_argument("--db", default=db_default, type=str)
    p.add_argument(
        "--work-dir",
        default=None,
        type=str,
        help="local-disk dir to build the DB in before publishing to --db "
        "(default: system temp). The DB must not be written live on NFS.",
    )
    p.add_argument("--workers", default=max(1, (os.cpu_count() or 2) - 1), type=int)
    p.add_argument("--batch-size", default=2000, type=int)
    p.add_argument("--limit", default=None, type=int, help="process at most N rows")
    args = p.parse_args(argv)

    convert(
        data_dir=Path(args.data_dir),
        db_path=Path(args.db),
        workers=args.workers,
        batch_size=args.batch_size,
        limit=args.limit,
        work_dir=Path(args.work_dir) if args.work_dir else None,
    )


if __name__ == "__main__":
    main()
