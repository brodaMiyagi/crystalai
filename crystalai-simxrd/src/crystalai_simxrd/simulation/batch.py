"""Batch Bragg-peak precompute (Phase 5.1, DESIGN_DECISIONS §9).

Walks ``crystals.sqlite`` and writes the wavelength-independent ``{d, |F|², hkl,
metric}`` peak list per structure into a fork-safe `bragg_peaks` store. This is the
expensive, heavy-tailed step, paid once; the training DataLoader then reads a peak
list by ``cif_id`` and runs the cheap on-the-fly stage. Resumable and multiprocess.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from ..core.bragg import compute_peak_list
from ..io import bragg_store


def _compute_one(args: tuple[int, bytes, float]) -> tuple[int, bytes, int]:
    """Worker: deserialize a crystals blob → peak list → serialized bytes."""
    from crystalai_data.crystals import database as cdb  # imported in worker

    cif_id, structure_blob, d_min = args
    peaks = compute_peak_list(cdb.deserialize(structure_blob), d_min=d_min)
    return cif_id, bragg_store.serialize(peaks), len(peaks)


def precompute_bragg_store(
    crystals_db: str | Path,
    out_store: str | Path,
    *,
    d_min: float = 0.7,
    source: str | None = "icsd",
    limit: int | None = None,
    max_n_sites: int | None = None,
    workers: int = 1,
    batch_size: int = 200,
    resume: bool = True,
) -> int:
    """Precompute peak lists for structures in ``crystals_db`` into ``out_store``.

    ``source`` filters (e.g. ``'icsd'``); ``max_n_sites`` skips the giant-cell tail
    (they OOM / take minutes — handle separately). Returns the number written.
    """
    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover
        def tqdm(x, **k):  # type: ignore
            return x

    cin = sqlite3.connect(f"file:{crystals_db}?mode=ro", uri=True)
    where = []
    if source:
        where.append(f"source = '{source}'")
    if max_n_sites:
        where.append(f"n_sites <= {int(max_n_sites)}")
    clause = (" WHERE " + " AND ".join(where)) if where else ""
    ids = [r[0] for r in cin.execute(f"SELECT id FROM crystals{clause}")]
    if limit:
        ids = ids[:limit]

    out = bragg_store.connect(out_store, bulk=True)
    bragg_store.create_schema(out)
    done = bragg_store.existing_ids(out) if resume else set()
    todo = [i for i in ids if i not in done]
    print(f"[precompute] structures={len(ids):,}  already done={len(done):,}  to do={len(todo):,}")

    def _iter_args():
        for cid in todo:
            blob = cin.execute("SELECT structure_blob FROM crystals WHERE id=?", (cid,)).fetchone()[0]
            yield (cid, blob, d_min)

    written = 0
    buf: list[tuple[int, bytes, int, float]] = []

    def _flush():
        nonlocal written
        if buf:
            bragg_store.write_serialized(out, buf)
            written += len(buf)
            buf.clear()

    if workers > 1:
        import multiprocessing as mp

        with mp.Pool(workers) as pool:
            for cid, blob, n in tqdm(pool.imap_unordered(_compute_one, _iter_args(), chunksize=8),
                                     total=len(todo), desc="[precompute]"):
                buf.append((cid, blob, n, d_min))
                if len(buf) >= batch_size:
                    _flush()
    else:
        for args in tqdm(_iter_args(), total=len(todo), desc="[precompute]"):
            cid, blob, n = _compute_one(args)
            buf.append((cid, blob, n, d_min))
            if len(buf) >= batch_size:
                _flush()
    _flush()
    out.close()
    print(f"[precompute] wrote {written:,} peak lists -> {out_store}")
    return written
