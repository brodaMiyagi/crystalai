"""Build the committed test fixture ``tests/fixtures/exp_subset`` from the exp_data store.

Curates a small, diverse subset (per source) spanning peak count, 2θ bin size,
max_background/max_raw ratio, background provenance (native vs arPLS-auto), and label
type. Writes a valid mini-store (index.csv + patterns/) loadable with
``XRDDatabase("tests/fixtures/exp_subset")``. Re-run after the store changes.

    uv run python scripts/build_exp_subset.py
"""

from __future__ import annotations

import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from crystalai_data.xrddata import XRDDatabase
from crystalai_data.xrddata import database as db

warnings.simplefilter("ignore")


def _pattern_stats(store: Path, row) -> dict | None:
    """Per-pattern: 2θ step, peak count, max_background/max_raw ratio."""
    try:
        xr, yr = db.read_xy(store / row.raw_path)
    except Exception:
        return None
    xr, yr = np.array(xr), np.array(yr)
    if len(xr) < 5:
        return None
    step = float(np.median(np.diff(xr)))
    max_raw = float(np.nanmax(yr))
    ratio, n_peaks = np.nan, 0
    if isinstance(row.bgsub_path, str):
        try:
            xb, yb = db.read_xy(store / row.bgsub_path)
            ybi = np.interp(xr, np.array(xb), np.array(yb))
            ratio = float(np.nanmax(yr - ybi) / max_raw) if max_raw > 0 else np.nan
            peaks, _ = find_peaks(
                ybi, height=max(np.nanmax(ybi) * 0.05, 0),
                prominence=np.nanmax(ybi) * 0.03, distance=max(1, int(0.1 / step)),
            )
            n_peaks = len(peaks)
        except Exception:
            pass
    sub = row.source_id.split("_")[1] if row.source == "opxrd" else row.source
    return dict(id=int(row.id), source=row.source, sub=sub, step=step,
                n_peaks=n_peaks, ratio=ratio)


def _pick(g: pd.DataFrame, targets: list[float], min_peak_q: float = 0.5) -> list[int]:
    """High-peak patterns nearest each target ratio (diverse ratio coverage)."""
    gg = g[g.n_peaks >= g.n_peaks.quantile(min_peak_q)]
    gg = gg if len(gg) else g
    chosen: list[int] = []
    for tr in targets:
        cand = gg[~gg.id.isin(chosen)]
        if cand.empty:
            break
        score = (cand.ratio - tr).abs() - 1e-4 * cand.n_peaks
        chosen.append(int(cand.loc[score.idxmin(), "id"]))
    return chosen


def main() -> None:
    store = db.default_store()
    idx = db.read_index(store)
    stats = pd.DataFrame(
        s for s in (_pattern_stats(store, r) for r in idx.itertuples()) if s
    )
    stats = stats[stats.ratio.notna() & np.isfinite(stats.ratio)]

    sel = _pick(stats[stats.source == "rruff"], [0.02, 0.06, 0.12, 0.25, 0.40])
    sel += _pick(stats[stats.source == "lab"], [0.03, 0.09, 0.20, 0.42])
    cnrs = stats[stats["sub"] == "CNRS"]
    sel += [int(cnrs.loc[(cnrs.step - 0.0025).abs().idxmin(), "id"])]
    sel += _pick(cnrs, [0.12, 0.26])
    sel += _pick(stats[stats["sub"] == "HKUST-A"], [0.05, 0.20])
    hb = stats[stats["sub"] == "HKUST-B"]
    sel += [int(hb.loc[hb.n_peaks.idxmax(), "id"])]
    sel = list(dict.fromkeys(sel))

    out = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "exp_subset"
    if out.exists():
        shutil.rmtree(out)
    (out / "patterns").mkdir(parents=True)
    by_id = idx.set_index("id")
    rows = []
    for pid in sel:
        row = by_id.loc[pid].to_dict()
        row["id"] = pid
        dst = out / "patterns" / row["source"] / row["source_id"]
        dst.mkdir(parents=True, exist_ok=True)
        for col in ("raw_path", "bgsub_path", "peaks_path", "cif_path"):
            rel = row.get(col)
            if isinstance(rel, str):
                shutil.copy2(store / rel, dst / Path(rel).name)
        rows.append(row)
    pd.DataFrame(rows)[list(db.INDEX_COLUMNS)].to_csv(out / "index.csv", index=False)

    xdb = XRDDatabase(out)
    print(f"exp_subset: {len(xdb)} patterns -> {out}")
    print(xdb.index.source.value_counts().to_dict())


if __name__ == "__main__":
    main()
