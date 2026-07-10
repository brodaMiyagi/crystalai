"""Automatic PXRD background estimation + subtraction (GSAS-II AutoBkg style).

The core is a thin, reusable wrapper over `pybaselines
<https://pybaselines.readthedocs.io/>`_ — the same library GSAS-II's *AutoBkg*
tool builds on (https://advancedphotonsource.github.io/GSAS-II-tutorials/AutoBkg/).
It is intended as the one background primitive for CrystalAI in general:
``crystalai-methods`` can call :func:`auto_background` on-the-fly in a Dataset's
``__getitem__``, and the store-walking :func:`generate` precomputes a
``bgsub_autobg.xy`` view for every experimental pattern that ships no native
background-subtracted profile.

Default method is asymmetrically-reweighted penalized least squares (**arPLS**), a
Whittaker-smoothing baseline that is robust for powder patterns (broad, smoothly
varying background under sharp Bragg peaks) and is one of the pybaselines methods
GSAS-II exposes. The method and its smoothness ``lam`` are configurable so callers
can match a specific GSAS-II AutoBkg setting or tune per-instrument.

Provenance in the store (DATA_ROADMAP.md §2): auto-generated backgrounds are
written to ``bgsub_autobg.xy`` (never ``bgsub.xy``, which is reserved for a
source's own curated profile such as RRUFF's ``XY_Processed``) and flagged with
``autobg=1`` in the index; native profiles carry ``autobg=0``.
"""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import numpy as np

from ..crystals._ingest import load_dotenv, repo_root
from . import database as db

__all__ = ["auto_background", "generate"]

# arPLS smoothness. Larger => stiffer (smoother) baseline. 1e6 chosen against
# RRUFF ground truth: auto-corrected vs RRUFF's own background-subtracted profile
# correlates ~0.996 (min 0.98) at lam=1e6 across sampled minerals, slightly better
# than 1e5 while staying flexible enough for structured synchrotron backgrounds.
# Exposed so callers can match a specific GSAS-II AutoBkg setting or tune per run.
_DEFAULT_METHOD = "arpls"
_DEFAULT_LAM = 1e6


def auto_background(
    x,
    y,
    *,
    method: str = _DEFAULT_METHOD,
    lam: float = _DEFAULT_LAM,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the background of one pattern and return ``(baseline, corrected)``.

    ``x`` / ``y`` are the pattern in its native coordinate (nothing is resampled).
    ``method`` names any :class:`pybaselines.Baseline` method (default ``arpls``);
    ``lam`` is forwarded to the Whittaker methods that take it, other ``kwargs`` are
    passed through. ``corrected = y - baseline`` (not clipped: a slightly negative
    tail is meaningful noise, and any clipping is a downstream choice).
    """
    from pybaselines import Baseline

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    fitter = Baseline(x_data=x)
    fn = getattr(fitter, method, None)
    if fn is None:
        raise ValueError(f"unknown pybaselines baseline method: {method!r}")

    # Whittaker methods (arpls, iarpls, asls, ...) take ``lam``; others don't.
    # pybaselines methods are decorated, so introspect the real signature (not
    # ``__code__``, which is the wrapper's and omits ``lam``).
    call_kwargs = dict(kwargs)
    if "lam" in inspect.signature(fn).parameters:
        call_kwargs.setdefault("lam", lam)
    baseline, _ = fn(y, **call_kwargs)
    return baseline, y - baseline


def generate(
    store: Path,
    *,
    method: str = _DEFAULT_METHOD,
    lam: float = _DEFAULT_LAM,
    overwrite: bool = False,
    limit: int | None = None,
) -> None:
    """Precompute ``bgsub_autobg.xy`` for every store pattern missing a background.

    Walks ``index.csv``; for each row that has a ``raw`` view but no ``bgsub_path``
    (i.e. the source shipped no curated background-subtracted profile), estimates
    the background from ``raw.xy``, writes the corrected profile alongside it, and
    sets ``bgsub_path`` + ``autobg=1``. Rows already carrying a native background
    (``autobg=0``) are left untouched unless ``overwrite`` is set. Idempotent:
    re-running only fills rows still missing a background.

    Run this **after** the ingesters — an ingester rewrites its source's rows and
    would drop the ``autobg`` fill, so the order is ingest → ingest → background.
    """
    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover
        def tqdm(x, **k):  # type: ignore
            return x

    store = Path(store)
    idx = db.read_index(store)
    if idx.empty:
        raise SystemExit(f"[autobg] empty/absent index at {store}")

    needs = idx["bgsub_path"].isna()
    if overwrite:
        # also redo previously auto-generated ones (keep native autobg=0 intact)
        needs = needs | (idx["autobg"] == 1)
    todo = idx[needs & idx["raw_path"].notna()]
    if limit is not None:
        todo = todo.head(limit)

    print(f"[autobg] store   = {store}")
    print(f"[autobg] method  = {method} (lam={lam})")
    print(f"[autobg] total patterns = {len(idx):,}  missing background = {len(todo):,}")

    done = 0
    fail = 0
    for row in tqdm(list(todo.itertuples()), desc="[autobg]"):
        try:
            x, y = db.read_xy(store / row.raw_path)
            _, corrected = auto_background(x, y, method=method, lam=lam)
            rel = f"patterns/{row.source}/{row.source_id}/bgsub_autobg.xy"
            db.write_xy(
                store / rel,
                x,
                corrected.tolist(),
                header=[
                    f"source={row.source}",
                    f"source_id={row.source_id}",
                    f"x_coord={row.x_coord}",
                    "view=bgsub",
                    f"autobg={method} lam={lam}",
                ],
            )
            idx.loc[idx["id"] == row.id, "bgsub_path"] = rel
            idx.loc[idx["id"] == row.id, "autobg"] = 1
            done += 1
        except Exception as exc:  # noqa: BLE001 — one bad pattern must not kill the run
            fail += 1
            print(f"[autobg] fail id={row.id} {row.source_id}: {type(exc).__name__}: {exc}")

    db.write_index_frame(store, idx)
    print(f"\n[autobg] backgrounds written = {done:,}  failed = {fail:,}")
    total_bg = idx["bgsub_path"].notna().sum()
    print(f"[autobg] patterns with a background now = {total_bg:,}/{len(idx):,} "
          f"(native {int((idx['autobg'] == 0).sum()):,}, auto {int((idx['autobg'] == 1).sum()):,})")


def main(argv: list[str] | None = None) -> None:
    root = repo_root()
    env = load_dotenv(root / ".env")
    store_default = str(db.resolve_store(env, root))

    p = argparse.ArgumentParser(description="Precompute auto-backgrounds over the experimental store")
    p.add_argument("--store", default=store_default, type=str)
    p.add_argument("--method", default=_DEFAULT_METHOD, type=str, help="pybaselines Baseline method")
    p.add_argument("--lam", default=_DEFAULT_LAM, type=float, help="arPLS/Whittaker smoothness")
    p.add_argument("--overwrite", action="store_true", help="also recompute existing auto backgrounds")
    p.add_argument("--limit", default=None, type=int)
    args = p.parse_args(argv)

    generate(
        Path(args.store),
        method=args.method,
        lam=args.lam,
        overwrite=args.overwrite,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
