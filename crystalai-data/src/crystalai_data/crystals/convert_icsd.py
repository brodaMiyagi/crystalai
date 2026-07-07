"""Ingest the ICSD CIF collection into ``crystals.sqlite``.

Walks the ICSD CIF directory (files named ``icsdid_<icsd_id>.cif``) and, for each
loadable CIF, writes one ``crystals`` row (``source='icsd'``) plus its
``crystal_elements`` rows. See DATA_ROADMAP.md §1 (conversion pipeline) for the
contract this implements:

  * Space group is **read from the CIF text** — the integer ``_space_group_IT_number``
    when present, else the Hermann-Mauguin symbol via ``hm_symbol_to_sg_number``.
    It is never re-derived (no spglib).
  * Crystal system is **derived** from the space group via ``sg_to_cs``, so SG/CS
    can't disagree.
  * ``source_id`` is the ICSD *collection code*; ``source_file_id`` is the
    *icsd_id* (the on-disk filename). The two are bridged by the prior-paper
    mapping CSV (``icsd_sim_labels.csv``: ``ICSD_ID,SG,EG,CS,COLLECTION_CODE,...``).
  * Disorder (fractional occupancies) is preserved exactly — never majority-
    replaced or supercell-expanded.

Run (from the package directory)::

    uv run python -m crystalai_data.crystals.convert_icsd            # uses .env
    uv run python -m crystalai_data.crystals.convert_icsd --limit 500
"""

from __future__ import annotations

import argparse
import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

from pymatgen.io.cif import CifParser

from . import database as db
from ._ingest import load_dotenv, local_build, repo_root
from .symmetry import hm_symbol_to_sg_number, sg_to_cs

_CIF_NAME_RE = re.compile(r"icsdid_(\d+)\.cif$")
_IT_NUMBER_RE = re.compile(r"_space_group_IT_number\s+(\d+)")
_HM_RE = re.compile(
    r"_(?:space_group_name_H-M_alt|symmetry_space_group_name_H-M)\s+(.+)"
)


# --------------------------------------------------------------------------- #
# Label CSV (collection-code bridge + SG fallback)
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class _LabelRow:
    collection_code: str
    sg: int | None


def _load_labels(csv_path: Path) -> dict[str, _LabelRow]:
    """``icsd_id`` (str) -> (collection_code, reported SG). Pure-stdlib csv."""
    import csv

    out: dict[str, _LabelRow] = {}
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            icsd_id = (r.get("ICSD_ID") or "").strip()
            code = (r.get("COLLECTION_CODE") or "").strip()
            if not icsd_id or not code:
                continue
            sg_raw = (r.get("SG") or "").strip()
            sg = int(sg_raw) if sg_raw.isdigit() and 1 <= int(sg_raw) <= 230 else None
            out[icsd_id] = _LabelRow(collection_code=code, sg=sg)
    return out


# --------------------------------------------------------------------------- #
# Per-CIF work
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class _WorkItem:
    path: str
    icsd_id: str
    collection_code: str
    csv_sg: int | None


def _space_group_from_text(text: str) -> tuple[int | None, str]:
    """Read SG from CIF text. Returns (sg_or_None, provenance)."""
    m = _IT_NUMBER_RE.search(text)
    if m:
        sg = int(m.group(1))
        if 1 <= sg <= 230:
            return sg, "it_number"
    m = _HM_RE.search(text)
    if m:
        symbol = m.group(1).strip().strip("'\"")
        try:
            return hm_symbol_to_sg_number(symbol), "hm_symbol"
        except ValueError:
            pass
    return None, "none"


def _process_cif(item: _WorkItem) -> tuple[str, str, object]:
    """Parse one CIF into a CrystalRow.

    Returns ``("ok", icsd_id, CrystalRow)`` or ``("skip", icsd_id, reason)``.
    Runs in worker processes, so it must be self-contained and pickle-safe.
    """
    try:
        text = Path(item.path).read_text(errors="replace")

        sg, prov = _space_group_from_text(text)
        if sg is None:
            sg, prov = item.csv_sg, "csv_fallback"
        if sg is None:
            return ("skip", item.icsd_id, "no_space_group")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # occupancy_tolerance high so disordered sites (occ sums > 1 within
            # rounding) parse instead of raising; cell kept as-parsed.
            parser = CifParser(item.path, occupancy_tolerance=100.0)
            structures = parser.parse_structures(primitive=False)
        if not structures:
            return ("skip", item.icsd_id, "no_structure")
        structure = structures[0]

        row = db.compute_metadata(
            structure,
            source="icsd",
            source_id=item.collection_code,
            space_group=sg,
            crystal_system=sg_to_cs(sg),
            source_file_id=item.icsd_id,
        )
        return ("ok", item.icsd_id, (row, prov))
    except Exception as exc:  # noqa: BLE001 — one bad CIF must not kill the run
        return ("skip", item.icsd_id, f"error:{type(exc).__name__}:{exc}")


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def _iter_work(
    cif_dir: Path, labels: dict[str, _LabelRow], skip_ids: set[str]
) -> tuple[list[_WorkItem], int, int]:
    """Build the work list. Returns (items, n_no_mapping, n_already_done)."""
    items: list[_WorkItem] = []
    n_no_mapping = 0
    n_done = 0
    with os.scandir(cif_dir) as it:
        for entry in it:
            m = _CIF_NAME_RE.search(entry.name)
            if not m:
                continue
            icsd_id = m.group(1)
            if icsd_id in skip_ids:
                n_done += 1
                continue
            label = labels.get(icsd_id)
            if label is None:
                n_no_mapping += 1
                continue
            items.append(
                _WorkItem(
                    path=entry.path,
                    icsd_id=icsd_id,
                    collection_code=label.collection_code,
                    csv_sg=label.sg,
                )
            )
    return items, n_no_mapping, n_done


def convert(
    *,
    cif_dir: Path,
    labels_csv: Path,
    db_path: Path,
    workers: int,
    batch_size: int,
    limit: int | None,
    work_dir: Path | None = None,
) -> None:
    from collections import Counter

    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover
        def tqdm(x, **k):  # type: ignore
            return x

    print(f"[icsd] cif_dir       = {cif_dir}")
    print(f"[icsd] labels_csv    = {labels_csv}")
    labels = _load_labels(labels_csv)
    print(f"[icsd] labels loaded = {len(labels):,} icsd_id -> collection_code")

    stats: Counter[str] = Counter()
    prov_stats: Counter[str] = Counter()
    inserted_total = 0

    # local_build hands back a schema-ready connection on local disk and copies
    # the finished DB to db_path on clean exit (NFS-safe; see _ingest).
    with local_build(db_path, work_dir, bulk=True) as conn:
        skip_ids = db.existing_source_file_ids(conn, "icsd")
        if skip_ids:
            print(f"[icsd] resume: {len(skip_ids):,} ICSD rows present, skipping")

        items, n_no_mapping, n_done = _iter_work(cif_dir, labels, skip_ids)
        if limit is not None:
            items = items[:limit]
        print(
            f"[icsd] to process    = {len(items):,} CIFs "
            f"({n_no_mapping:,} without a CSV mapping, {n_done:,} already done)"
        )

        batch: list[db.CrystalRow] = []

        def flush() -> None:
            nonlocal inserted_total, batch
            if batch:
                inserted_total += db.insert_many(conn, batch)
                batch = []

        def handle(result: tuple[str, str, object]) -> None:
            status, _icsd_id, payload = result
            if status == "ok":
                row, prov = payload  # type: ignore[misc]
                prov_stats[prov] += 1
                batch.append(row)
                if len(batch) >= batch_size:
                    flush()
            else:
                stats[str(payload).split(":", 1)[0]] += 1

        if workers > 1:
            import multiprocessing as mp

            with mp.Pool(workers) as pool:
                for result in tqdm(
                    pool.imap_unordered(_process_cif, items, chunksize=32),
                    total=len(items),
                    desc="[icsd] ingest",
                ):
                    handle(result)
        else:
            for item in tqdm(items, desc="[icsd] ingest"):
                handle(_process_cif(item))

        flush()

    print(f"\n[icsd] inserted      = {inserted_total:,}")
    print(f"[icsd] sg provenance = {dict(prov_stats)}")
    if stats:
        print(f"[icsd] skipped       = {dict(stats)}")
    print(f"[icsd] no CSV mapping = {n_no_mapping:,} (not inserted)")


def main(argv: list[str] | None = None) -> None:
    root = repo_root()
    env = load_dotenv(root / ".env")

    cif_default = env.get("ICSD_CIF_FOLDER")
    labels_default = env.get("ICSD_LABELS_CSV")
    if labels_default is None and cif_default:
        # icsd_sim_labels.csv sits two levels above the CIF .../cif_ICSD/data dir.
        labels_default = str(Path(cif_default).resolve().parents[1] / "icsd_sim_labels.csv")
    db_default = env.get("CRYSTALS_DB", str(root / "crystalai-data" / "crystals.sqlite"))

    p = argparse.ArgumentParser(description="Ingest ICSD CIFs into crystals.sqlite")
    p.add_argument("--cif-dir", default=cif_default, type=str, required=cif_default is None)
    p.add_argument("--labels-csv", default=labels_default, type=str, required=labels_default is None)
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
    p.add_argument("--limit", default=None, type=int, help="process at most N CIFs")
    args = p.parse_args(argv)

    convert(
        cif_dir=Path(args.cif_dir),
        labels_csv=Path(args.labels_csv),
        db_path=Path(args.db),
        workers=args.workers,
        batch_size=args.batch_size,
        limit=args.limit,
        work_dir=Path(args.work_dir) if args.work_dir else None,
    )


if __name__ == "__main__":
    main()
