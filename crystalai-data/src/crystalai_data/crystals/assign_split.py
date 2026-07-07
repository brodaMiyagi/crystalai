"""Assign a crystal-system-stratified train/val/test split to ICSD rows.

MP-20 carries its native split from ingest; ICSD rows are null until this pass
fills them (DATA_ROADMAP.md §1). The split is **stratified by crystal_system**:
each of the 7 systems is split into the same train/val/test proportions, so the
rare systems (triclinic, trigonal) are represented in every fold — matching the
CS-gated SG head the methods package assumes (METHODS_ROADMAP.md §2.1).

Deterministic given ``--seed``: within each crystal system the id list is sorted
then shuffled with a seeded RNG and sliced by the ratios, so re-running on the
same row set reproduces the assignment exactly.

Run (from the package directory)::

    uv run python -m crystalai_data.crystals.assign_split            # 80/10/10, icsd
    uv run python -m crystalai_data.crystals.assign_split --ratios 0.7 0.15 0.15
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np

from ._ingest import load_dotenv, local_build, repo_root

_SPLITS = ("train", "val", "test")


def _slice_counts(n: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    """Partition ``n`` into (train, val, test) counts summing to exactly ``n``."""
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)
    n_test = n - n_train - n_val
    return n_train, n_val, n_test


def assign(
    conn: sqlite3.Connection,
    *,
    source: str = "icsd",
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    reassign: bool = False,
) -> dict[int, tuple[int, int, int]]:
    """Fill ``split`` for ``source`` rows, stratified by crystal system.

    By default only rows with a null ``split`` are assigned (safe to re-run);
    ``reassign=True`` overwrites existing splits for ``source``. Returns a
    ``{crystal_system: (n_train, n_val, n_test)}`` summary.
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1, got {ratios} (sum {sum(ratios)})")

    rng = np.random.default_rng(seed)
    where_null = "" if reassign else " AND split IS NULL"
    css = [
        r[0]
        for r in conn.execute(
            f"SELECT DISTINCT crystal_system FROM crystals "
            f"WHERE source = ?{where_null} ORDER BY crystal_system",
            (source,),
        )
    ]

    summary: dict[int, tuple[int, int, int]] = {}
    updates: list[tuple[str, int]] = []
    for cs in css:
        ids = np.array(
            [
                r[0]
                for r in conn.execute(
                    f"SELECT id FROM crystals "
                    f"WHERE source = ? AND crystal_system = ?{where_null} ORDER BY id",
                    (source, cs),
                )
            ]
        )
        rng.shuffle(ids)
        n_train, n_val, n_test = _slice_counts(len(ids), ratios)
        bounds = {"train": (0, n_train),
                  "val": (n_train, n_train + n_val),
                  "test": (n_train + n_val, len(ids))}
        for split, (lo, hi) in bounds.items():
            updates.extend((split, int(i)) for i in ids[lo:hi])
        summary[cs] = (n_train, n_val, n_test)

    conn.executemany("UPDATE crystals SET split = ? WHERE id = ?", updates)
    conn.commit()
    return summary


def run(
    *,
    db_path: Path,
    source: str,
    ratios: tuple[float, float, float],
    seed: int,
    reassign: bool,
    work_dir: Path | None,
) -> None:
    from .symmetry import cs_number_to_name

    print(f"[split] source={source} ratios={ratios} seed={seed} reassign={reassign}")
    with local_build(db_path, work_dir, bulk=False) as conn:
        before = conn.execute(
            "SELECT COUNT(*) FROM crystals WHERE source = ? AND split IS NULL",
            (source,),
        ).fetchone()[0]
        print(f"[split] {source} rows with null split before: {before:,}")

        summary = assign(
            conn, source=source, ratios=ratios, seed=seed, reassign=reassign
        )

        print("[split] per crystal system (train / val / test):")
        totals = [0, 0, 0]
        for cs in sorted(summary):
            t, v, te = summary[cs]
            totals = [totals[0] + t, totals[1] + v, totals[2] + te]
            print(f"  {cs} {cs_number_to_name(cs):<13} {t:>8,} {v:>7,} {te:>7,}")
        print(f"  {'TOTAL':<15} {totals[0]:>8,} {totals[1]:>7,} {totals[2]:>7,}")

        remaining = conn.execute(
            "SELECT COUNT(*) FROM crystals WHERE source = ? AND split IS NULL",
            (source,),
        ).fetchone()[0]
        print(f"[split] {source} rows still null after: {remaining:,}")


def main(argv: list[str] | None = None) -> None:
    root = repo_root()
    env = load_dotenv(root / ".env")
    db_default = env.get("CRYSTALS_DB", str(root / "crystalai-data" / "crystals.sqlite"))

    p = argparse.ArgumentParser(description="Assign CS-stratified split to ICSD rows")
    p.add_argument("--db", default=db_default, type=str)
    p.add_argument("--source", default="icsd", type=str)
    p.add_argument(
        "--ratios", nargs=3, type=float, default=[0.8, 0.1, 0.1],
        metavar=("TRAIN", "VAL", "TEST"), help="must sum to 1.0",
    )
    p.add_argument("--seed", default=42, type=int)
    p.add_argument(
        "--reassign", action="store_true",
        help="overwrite existing splits for this source (default: only fill nulls)",
    )
    p.add_argument("--work-dir", default=None, type=str)
    args = p.parse_args(argv)

    run(
        db_path=Path(args.db),
        source=args.source,
        ratios=tuple(args.ratios),
        seed=args.seed,
        reassign=args.reassign,
        work_dir=Path(args.work_dir) if args.work_dir else None,
    )


if __name__ == "__main__":
    main()
