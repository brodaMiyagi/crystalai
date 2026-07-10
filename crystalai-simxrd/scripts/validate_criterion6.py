"""Criterion #6 (SIMXRD_ROADMAP §7) — noise-aware sim-vs-experimental validation.

For each matched (structure, experimental pattern) pair in `crystalai-data`'s exp_subset,
refine the forward-simulation effects and compare to the RAW counts in 2θ with a jointly-
fitted background (`compare_structure_to_raw`). Report Rwp, the counting-noise floor,
GoF = Rwp/floor, the Rietveld-partition R_Bragg (structural bug-detector), and cosine.

GATE: >= 5 pairs with GoF < 4. (R_Bragg is reported, not gated — see DESIGN_DECISIONS §10.)

    uv run python scripts/validate_criterion6.py

Lab pairs resolve their structure from crystals.sqlite via cif_id; opXRD pairs use the
inline structure.cif. Paths come from the repo-root .env (CRYSTALAI_EXP_DATA_FOLDER,
CRYSTALS_DB) with sensible fallbacks, or pass --exp-subset / --crystals-db.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

GOF_GATE = 4.0
MIN_PASS = 5


def _make_eff(p):
    from crystalai_simxrd.simulation.simulator import EffectConfig

    ls, b, r, z = p
    return EffectConfig(
        b_iso=float(np.clip(b, 0, 3)),
        crystallite_size_nm=float(np.clip(np.exp(ls), 15, 900)),
        march_r=float(np.clip(r, 0.55, 1.35)),
        zero_shift_deg=float(np.clip(z, -0.1, 0.1)),
    )


def _refine(st, x, y, wl):
    """Refine {size, B, March-r, zero-shift} minimizing Rwp; return the best result dict."""
    from crystalai_simxrd.comparison.compare import compare_structure_to_raw

    def obj(p):
        return compare_structure_to_raw(st, x, y, wl, effects=_make_eff(p))["rwp"]

    best = None
    for start in ([np.log(80), 0.5, 1.0, 0.0], [np.log(150), 1.0, 0.85, 0.0],
                  [np.log(40), 0.3, 1.1, 0.0]):
        res = minimize(obj, start, method="Nelder-Mead",
                       options=dict(maxiter=200, xatol=1e-2, fatol=1e-2))
        r = compare_structure_to_raw(st, x, y, wl, effects=_make_eff(res.x))
        if best is None or r["rwp"] < best["rwp"]:
            best = r
    return best


def _pairs(exp_subset: Path, crystals_db: Path):
    """Yield (name, structure, two_theta, raw_intensity, wavelength) for matched pairs."""
    from crystalai_data.xrddata.database import XRDDatabase
    from pymatgen.core import Structure

    xdb = XRDDatabase(exp_subset)
    # lab pairs (structure from crystals.sqlite via cif_id = primary key)
    if crystals_db.exists():
        from crystalai_data.crystals.api import CrystalDatabase

        cdb = CrystalDatabase(crystals_db)
        for _, r in xdb.to_frame(source="lab").iterrows():
            if not np.isfinite(r["cif_id"]) or not np.isfinite(r["wavelength_A"]):
                continue
            x, y = xdb.load_pattern(int(r["id"]), "raw")
            yield (r["source_id"], cdb.get(int(r["cif_id"])),
                   np.asarray(x), np.asarray(y), float(r["wavelength_A"]))
    # opXRD pairs (inline structure.cif)
    for _, r in xdb.to_frame(source="opxrd").iterrows():
        if not isinstance(r["cif_path"], str) or not np.isfinite(r["wavelength_A"]):
            continue
        x, y = xdb.load_pattern(int(r["id"]), "raw")
        yield (r["source_id"], Structure.from_str(xdb.cif_text(int(r["id"])), fmt="cif"),
               np.asarray(x), np.asarray(y), float(r["wavelength_A"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-subset", default=None)
    ap.add_argument("--crystals-db", default=None)
    args = ap.parse_args()

    from crystalai_data.crystals._ingest import load_dotenv, repo_root

    root = repo_root()
    env = load_dotenv(root / ".env")
    exp_subset = Path(args.exp_subset) if args.exp_subset else (
        root / "crystalai-data" / "tests" / "fixtures" / "exp_subset")
    crystals_db = Path(args.crystals_db) if args.crystals_db else Path(
        env.get("CRYSTALS_DB", str(root / "crystalai-data" / "crystals.sqlite")))

    print(f"exp_subset  = {exp_subset}")
    print(f"crystals_db = {crystals_db} ({'found' if crystals_db.exists() else 'MISSING'})\n")
    print("%-26s %7s %8s %6s %9s %6s  gate" % (
        "pattern", "Rwp", "floor", "GoF", "R_Bragg", "cos"))

    n_pass = 0
    n_total = 0
    for name, st, x, y, wl in _pairs(exp_subset, crystals_db):
        n_total += 1
        b = _refine(st, x, y, wl)
        ok = b["gof"] < GOF_GATE
        n_pass += ok
        print("%-26s %6.1f%% %7.1f%% %6.2f %8.1f%% %6.3f  %s" % (
            name, b["rwp"], b["rwp_noise_floor"], b["gof"], b["r_bragg"], b["cosine"],
            "PASS" if ok else ""))

    print(f"\n{n_pass}/{n_total} pairs achieve GoF < {GOF_GATE:g}"
          f"  (criterion #6 needs >= {MIN_PASS})")
    print("CRITERION #6:", "MET" if n_pass >= MIN_PASS else "NOT MET")
    raise SystemExit(0 if n_pass >= MIN_PASS else 1)


if __name__ == "__main__":
    main()
