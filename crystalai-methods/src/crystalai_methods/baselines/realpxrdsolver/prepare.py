"""Build RealPXRDSolver inputs from an experimental-store source.

For each selected row we resolve the ground-truth structure and its full-cell
composition (same convention as the PXRDGen baseline — the generator needs the
exact atom multiset, not the reduced formula) and write up to two input variants:

  * ``inputs_raw/<id>.dat``   — 2-col ``(Cu-Kα1 2θ grid, intensity)`` produced by
    Bragg-remapping the **raw** pattern onto the Cu Kα1 2θ [5°,80°) grid
    (``pxrdgen.remap``). ``run_eval.py --peak-mode raw`` feeds this through
    RealPXRDSolver's *own* ``subtract_polynomial_background`` + ``search_peaks``.

  * ``inputs_peaks/<id>.dat`` — 2-col ``(Cu-Kα1 2θ, intensity)`` from **our manual
    peak list** (``peaks.xy``), remapped peak-by-peak to Cu Kα1. ``run_eval.py
    --peak-mode manual`` uses it directly as the sparse d-I list. Written only for
    rows that ship a ``peaks_path`` (lab patterns).

Ground truth: lab rows resolve the matched ICSD entry via ``cif_id``; opXRD rows
carry no ICSD match, so GT comes from the shipped ``cif_path``. Disordered
(partial-occupancy) cells are skipped — the generator emits ordered cells.

Run inside the PXRDGen venv (has pymatgen). NumPy-only remap math.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from crystalai_methods.baselines.pxrdgen.remap import (
    model_two_theta_grid,
    remap_pattern,
    to_two_theta_cu,
)


def cell_formula(structure) -> tuple[str, int]:
    """Full-cell composition string (e.g. 'Ti4O8') + atom count of the primitive.

    Raises ``ValueError`` for disordered (partial-occupancy) structures.
    """
    prim = structure.get_primitive_structure()
    if not prim.is_ordered:
        raise ValueError("disordered structure (partial occupancy)")
    counts = Counter(sp.symbol for sp in prim.species)
    formula = "".join(f"{el}{counts[el]}" for el in sorted(counts))
    return formula, len(prim)


def remap_peak_list(path: Path, wavelength: float) -> np.ndarray | None:
    """Read a ``peaks.xy`` (2θ, intensity) list and remap 2θ onto Cu Kα1.

    Returns an ascending 2-col ``(2θ_cu, intensity)`` array, or ``None`` if the
    file has no usable peaks. Peaks with no physical Cu Kα1 angle are dropped.
    """
    peaks = np.loadtxt(path)  # comments='#' by default -> header skipped
    if peaks.ndim == 1:  # a single peak row loads as shape (2,)
        peaks = peaks.reshape(1, -1)
    if peaks.ndim != 2 or peaks.shape[0] == 0:
        return None
    tt, inten = peaks[:, 0], peaks[:, 1]
    ttc = to_two_theta_cu(tt, wavelength)
    ok = np.isfinite(ttc) & (inten > 0)
    ttc, inten = ttc[ok], inten[ok]
    if ttc.size == 0:
        return None
    order = np.argsort(ttc)
    return np.column_stack([ttc[order], inten[order]])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="../crystalai-data",
                    help="crystalai-data dir (holds exp_data/ and crystals.sqlite)")
    ap.add_argument("--out", required=True, help="output dir for inputs + manifest")
    ap.add_argument("--source", default="lab", help="exp_data source to prepare")
    ap.add_argument("--crystals-src", action="append", default=None,
                    help="add crystalai_data src to sys.path (repeatable)")
    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    for p in (args.crystals_src or [str(data_root / "src")]):
        sys.path.insert(0, p)
    from crystalai_data.crystals.api import CrystalDatabase  # noqa: E402
    from pymatgen.core.structure import Structure  # noqa: E402

    exp = data_root / "exp_data"
    db = CrystalDatabase(str(data_root / "crystals.sqlite"))

    def load_structure(r):
        if r["cif_id"]:
            return db.get(int(float(r["cif_id"])))
        if r["cif_path"]:
            return Structure.from_file(exp / r["cif_path"])
        raise ValueError("no cif_id or cif_path")

    out = Path(args.out).resolve()
    (out / "inputs_raw").mkdir(parents=True, exist_ok=True)
    (out / "inputs_peaks").mkdir(parents=True, exist_ok=True)
    (out / "gt").mkdir(parents=True, exist_ok=True)
    grid = model_two_theta_grid()

    rows = [r for r in csv.DictReader(open(exp / "index.csv"))
            if r["source"] == args.source]
    manifest = []
    for r in rows:
        sid = r["source_id"]
        raw = exp / (r["raw_path"] or "")
        if not (r["cif_id"] or r["cif_path"]) or not raw.exists() or not r["wavelength_A"]:
            print(f"skip {sid}: missing structure/raw/wavelength")
            continue
        wl = float(r["wavelength_A"])
        try:
            structure = load_structure(r)
            formula, n_atoms = cell_formula(structure)
        except Exception as e:  # noqa: BLE001
            print(f"skip {sid}: {e}")
            continue

        # search_peaks input: remap the RAW pattern onto the Cu Kα1 2θ grid.
        pattern = np.loadtxt(raw)
        y, coverage = remap_pattern(pattern[:, 0], pattern[:, 1], wl)
        np.savetxt(out / "inputs_raw" / f"{sid}.dat",
                   np.column_stack([grid, y]), fmt="%.6f")

        # manual-peak input: remap our peak list (lab only).
        has_peaks = 0
        peaks_path = exp / (r["peaks_path"] or "")
        if r["peaks_path"] and peaks_path.exists():
            pk = remap_peak_list(peaks_path, wl)
            if pk is not None:
                np.savetxt(out / "inputs_peaks" / f"{sid}.dat", pk, fmt="%.6f")
                has_peaks = 1

        structure.to(filename=str(out / "gt" / f"{sid}.cif"), fmt="cif")
        manifest.append({
            "id": sid, "formula": formula, "n_atoms": n_atoms,
            "reduced_formula": r["formula"], "wavelength_A": r["wavelength_A"],
            "coverage": round(coverage, 4), "has_peaks": has_peaks,
            "cif_id": r["cif_id"], "space_group": r["space_group"],
        })
        print(f"{sid}: {formula} (n={n_atoms}) cov={coverage:.3f} peaks={has_peaks}")

    fields = ["id", "formula", "n_atoms", "reduced_formula", "wavelength_A",
              "coverage", "has_peaks", "cif_id", "space_group"]
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(manifest)
    print(f"\nWrote {len(manifest)} inputs to {out} "
          f"({sum(m['has_peaks'] for m in manifest)} with manual peaks)")


if __name__ == "__main__":
    main()
