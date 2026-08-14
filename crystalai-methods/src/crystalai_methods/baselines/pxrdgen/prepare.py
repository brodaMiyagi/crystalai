"""Build PXRDGen inputs from the experimental store's lab patterns.

For each ``source=='lab'`` row we:
  1. load the background-subtracted pattern,
  2. remap it onto PXRDGen's Cu Kα1 2θ [5°,80°) 7500-point grid (``remap.py``),
  3. resolve the matched ICSD structure from ``crystals.sqlite`` (``cif_id``),
     reduce it to its primitive cell, and read off the **full-cell composition**
     (PXRDGen needs the exact atom multiset, MP-20 convention, not the reduced
     formula), and
  4. write ``<id>.dat`` (7500 intensities), the ground-truth CIF, and a manifest
     row (formula, n_atoms, wavelength, coverage).

Run inside the dedicated PXRDGen venv (has pymatgen). Writes into an output dir
that ``run_eval.py`` then consumes.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from crystalai_methods.baselines.pxrdgen.remap import remap_pattern


def cell_formula(structure) -> tuple[str, int]:
    """Full-cell composition string (e.g. 'Cr4O6') + atom count of the primitive.

    Raises ``ValueError`` for disordered (partial-occupancy) structures — PXRDGen
    generates ordered structures and cannot consume a fractional atom multiset.
    """
    prim = structure.get_primitive_structure()
    if not prim.is_ordered:
        raise ValueError("disordered structure (partial occupancy)")
    counts = Counter(sp.symbol for sp in prim.species)
    formula = "".join(f"{el}{counts[el]}" for el in sorted(counts))
    return formula, len(prim)


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
        """ICSD structures via cif_id (lab); opXRD via its shipped cif_path."""
        if r["cif_id"]:
            return db.get(int(float(r["cif_id"])))
        if r["cif_path"]:
            return Structure.from_file(exp / r["cif_path"])
        raise ValueError("no cif_id or cif_path")
    out = Path(args.out).resolve()
    (out / "inputs").mkdir(parents=True, exist_ok=True)
    (out / "gt").mkdir(parents=True, exist_ok=True)

    rows = [r for r in csv.DictReader(open(exp / "index.csv"))
            if r["source"] == args.source]
    manifest = []
    for r in rows:
        sid = r["source_id"]
        bg = exp / (r["bgsub_path"] or "")
        if not (r["cif_id"] or r["cif_path"]) or not bg.exists() or not r["wavelength_A"]:
            print(f"skip {sid}: missing structure/bgsub/wavelength")
            continue
        pattern = np.loadtxt(bg)
        x, coverage = remap_pattern(pattern[:, 0], pattern[:, 1],
                                    float(r["wavelength_A"]))
        try:
            structure = load_structure(r)
            formula, n_atoms = cell_formula(structure)
        except Exception as e:  # noqa: BLE001
            print(f"skip {sid}: {e}")
            continue

        np.savetxt(out / "inputs" / f"{sid}.dat", x, fmt="%.7f")
        structure.to(filename=str(out / "gt" / f"{sid}.cif"), fmt="cif")
        manifest.append({
            "id": sid, "formula": formula, "n_atoms": n_atoms,
            "reduced_formula": r["formula"], "wavelength_A": r["wavelength_A"],
            "coverage": round(coverage, 4), "cif_id": r["cif_id"],
            "space_group": r["space_group"],
        })
        print(f"{sid}: {formula} (n={n_atoms}) cov={coverage:.3f}")

    fields = ["id", "formula", "n_atoms", "reduced_formula", "wavelength_A",
              "coverage", "cif_id", "space_group"]
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(manifest)
    print(f"\nWrote {len(manifest)} inputs to {out}")


if __name__ == "__main__":
    main()
