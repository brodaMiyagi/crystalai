"""Evaluate the OpenAlphaDiffract baseline over our experimental PXRD store.

Selection (verified counts, see plan): ``(audit=='pass' OR source=='lab')`` with
non-null wavelength, space group, and a background-subtracted view. Scored in two
sets by background provenance:
  * MANUAL/native bg (autobg=0): RRUFF curated + lab manually-verified  (~1114)
  * AUTO bg        (autobg=1): RRUFF + opXRD pybaselines auto-bg         (~2456)

Each pattern is remapped (Bragg d-spacing) onto the model's fixed
2θ∈[5,20]°@20 keV / 8192-bin grid, then run through the frozen model.

    python -m crystalai_methods.baselines.alphadiffract.run_eval [--limit N]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from crystalai_data.xrddata.database import XRDDatabase

from .infer import DEFAULT_MODEL_DIR, load_model, predict_batch
from .metrics import LP_NAMES, format_report
from .remap import remap_pattern


def select_rows(db: XRDDatabase) -> pd.DataFrame:
    """Apply the scorable-set filter and tag each row with its background set."""
    df = db.to_frame()
    keep = (df["audit"].astype("string") == "pass") | (df["source"] == "lab")
    df = df[keep].copy()
    for col in ("wavelength_A", "space_group", "bgsub_path"):
        df = df[df[col].notna()]
    df = df.reset_index(drop=True)
    df["bg_set"] = np.where(df["autobg"].astype(float) == 0.0, "manual", "auto")
    return df


def build_inputs(db: XRDDatabase, rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Load + remap every selected pattern. Returns (X[N,8192], coverage[N])."""
    X = np.zeros((len(rows), 8192), dtype=np.float32)
    coverage = np.zeros(len(rows), dtype=np.float64)
    for i, row in enumerate(tqdm(rows.itertuples(index=False), total=len(rows),
                                 desc="remap", unit="pat")):
        xs, ys = db.load_pattern(int(row.id), which="bgsub")
        x, cov = remap_pattern(np.asarray(xs), np.asarray(ys), float(row.wavelength_A))
        X[i] = x
        coverage[i] = cov
    return X, coverage


def run_inference(model, X: np.ndarray, batch_size: int, device: str) -> dict:
    out: dict[str, list] = {}
    for start in tqdm(range(0, len(X), batch_size), desc="infer", unit="batch"):
        pred = predict_batch(model, X[start:start + batch_size], device=device)
        for k, v in pred.items():
            out.setdefault(k, []).append(v)
    return {k: np.concatenate(v, axis=0) for k, v in out.items()}


def assemble_results(rows: pd.DataFrame, coverage: np.ndarray, pred: dict) -> pd.DataFrame:
    res = pd.DataFrame({
        "id": rows["id"].to_numpy(),
        "source": rows["source"].to_numpy(),
        "source_id": rows["source_id"].to_numpy(),
        "bg_set": rows["bg_set"].to_numpy(),
        "wavelength_A": rows["wavelength_A"].to_numpy(dtype=float),
        "coverage": coverage,
        "cs_true": rows["crystal_system"].to_numpy(dtype=float).astype(int),
        "sg_true": rows["space_group"].to_numpy(dtype=float).astype(int),
        "cs_pred": pred["cs_pred"],
        "cs_conf": pred["cs_conf"],
        "sg_pred": pred["sg_pred"],
        "sg_conf": pred["sg_conf"],
        "sg_top5": list(pred["sg_top5"]),
    })
    for j, name in enumerate(LP_NAMES):
        res[f"lp_true_{name}"] = rows[name].to_numpy(dtype=float)
        res[f"lp_pred_{name}"] = pred["lp"][:, j]
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", type=str, default=None,
                    help="experimental store root (default: crystalai_data default)")
    ap.add_argument("--model-dir", type=str, default=str(DEFAULT_MODEL_DIR))
    ap.add_argument("--limit", type=int, default=None,
                    help="only evaluate the first N selected patterns (smoke test)")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--out", type=str,
                    default=str(DEFAULT_MODEL_DIR / "results.csv"))
    args = ap.parse_args()

    db = XRDDatabase(args.store)
    rows = select_rows(db)
    if args.limit:
        rows = rows.head(args.limit).reset_index(drop=True)

    n_manual = int((rows["bg_set"] == "manual").sum())
    n_auto = int((rows["bg_set"] == "auto").sum())
    print(f"Selected {len(rows)} patterns  (manual={n_manual}, auto={n_auto})")

    model = load_model(args.model_dir, device=args.device)
    X, coverage = build_inputs(db, rows)
    pred = run_inference(model, X, args.batch_size, args.device)
    res = assemble_results(rows, coverage, pred)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    to_save = res.copy()
    to_save["sg_top5"] = ["|".join(map(str, row)) for row in res["sg_top5"]]
    to_save.to_csv(args.out, index=False)
    print(f"\nWrote per-pattern predictions -> {args.out}\n")

    if args.limit and args.limit <= 20:
        # Smoke-test view: show predictions next to ground truth.
        cols = ["source_id", "bg_set", "coverage", "cs_true", "cs_pred",
                "sg_true", "sg_pred", "sg_conf"]
        with pd.option_context("display.width", 140, "display.max_columns", None):
            print(res[cols].to_string(index=False))
        print()

    print(format_report(res))


if __name__ == "__main__":
    main()
