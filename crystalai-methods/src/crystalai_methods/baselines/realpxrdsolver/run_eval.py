"""Run RealPXRDSolver on prepared inputs and score structure matches.

Loads the trained ``pxrd-atom25`` generator (peak-set BERT + flow decoder, the
XRD+formula variant — no lattice input, the analog of PXRDGen's flow_CNN) once,
then for each manifest row builds the sparse d-I peak list, draws ``--num-evals``
candidate structures, and compares each against the ground truth with pymatgen's
``StructureMatcher`` under two settings:

  * ``default`` — StructureMatcher() defaults (ltol=0.2, stol=0.3, angle_tol=5)
  * ``paper``   — RealPXRDSolver's own setting (ltol=0.3, stol=0.5, angle_tol=10;
                  see ``scripts/compute.py`` in the upstream repo)

Two ways to obtain the peak list (``--peak-mode``):

  * ``raw``    — feed ``inputs_raw/<id>.dat`` through RealPXRDSolver's *own*
                 ``subtract_polynomial_background`` + ``search_peaks`` (imported
                 verbatim from the upstream ``scripts/sample_flow.py``).
  * ``manual`` — use ``inputs_peaks/<id>.dat`` (our remapped manual peak list)
                 directly as the d-I list; intensities renormalized to max 100.

Reports 1-sample (candidate #1 matches) and best-of-N (any candidate) match
rates. Run inside the PXRDGen venv; point ``--repo`` at the extracted
``RealPXRD-Solver`` tree and ``--weights`` at the ``pxrd-atom25`` checkpoint dir.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")


def load_model(repo: Path, weights: Path, label: int = -1):
    sys.path.insert(0, str(repo))
    from hydra import compose, initialize_config_dir
    from app.model.flow import CSPFlow

    with initialize_config_dir(str(weights / ".hydra")):
        cfg = compose(config_name="config")
    ckpt = sorted(weights.glob("*.ckpt"))[label]
    print(f"loading {ckpt}")
    model = CSPFlow.load_from_checkpoint(str(ckpt), map_location="cpu", **cfg.model)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model


def peaks_from_raw(dat: Path):
    """RealPXRDSolver's own raw -> sparse peak list (their exact functions)."""
    from scripts.sample_flow import search_peaks, subtract_polynomial_background

    xrd = np.loadtxt(dat)
    cx, cy = xrd[:, 0], xrd[:, 1]
    cy = subtract_polynomial_background(cx, cy, poly_order=6)
    return search_peaks(cx, cy)  # (pxrd_x 2θ, pxrd_y 0-100)


def peaks_from_manual(dat: Path):
    """Our remapped manual peak list, renormalized to max 100 and masked py>5.

    The renormalize-to-100-then-drop-below-5 step mirrors what the model was
    trained on (``CrystDataset_pxrd``: LMDB peaks are 0-100, then ``pxrd_y > 5``),
    so the manual list enters the encoder in the same regime as the raw-mode
    ``search_peaks`` output (whose ``find_peaks(height=5)`` does the same).
    """
    pk = np.loadtxt(dat)
    if pk.ndim == 1:
        pk = pk.reshape(1, -1)
    px, py = pk[:, 0], pk[:, 1]
    if py.max() > 0:
        py = py / py.max() * 100.0
    keep = py > 5
    return px[keep], py[keep]


def build_data(formula: str, px: np.ndarray, py: np.ndarray, num_evals: int):
    from scripts.eval_utils import chemical_symbols
    from torch_geometric.data import Data
    import chemparse

    comp = chemparse.parse_formula(formula)
    chem_list = []
    for elem, cnt in comp.items():
        chem_list.extend([chemical_symbols.index(elem)] * int(cnt))
    px_t = torch.Tensor(px).view(-1, 1)
    py_t = torch.Tensor(py).view(-1, 1)
    return [
        Data(atom_types=torch.LongTensor(chem_list), pxrd_x=px_t, pxrd_y=py_t,
             peak_num=len(px), num_atoms=len(chem_list), num_nodes=len(chem_list))
        for _ in range(num_evals)
    ]


def generate(model, data_list, step_lr=5, infer_timesteps=200):
    from scripts.eval_utils import get_crystals_list, lattices_to_params_shape
    from torch_geometric.data import Batch

    batch = Batch.from_data_list(data_list)
    if torch.cuda.is_available():
        batch = batch.cuda()
    out = model.sample(batch, step_lr=step_lr, infer_timesteps=infer_timesteps)
    lengths, angles = lattices_to_params_shape(out["lattices"].detach().cpu())
    return get_crystals_list(
        out["frac_coords"].detach().cpu(), out["atom_types"].detach().cpu(),
        lengths, angles, out["num_atoms"].detach().cpu())


def to_structure(arr):
    from pymatgen.core.lattice import Lattice
    from pymatgen.core.structure import Structure
    try:
        return Structure(
            lattice=Lattice.from_parameters(*(arr["lengths"].tolist()
                                              + arr["angles"].tolist())),
            species=arr["atom_types"], coords=arr["frac_coords"],
            coords_are_cartesian=False)
    except Exception:  # noqa: BLE001
        return None


MATCHERS = {
    "default": dict(),
    "paper": dict(ltol=0.3, stol=0.5, angle_tol=10),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="extracted RealPXRD-Solver tree")
    ap.add_argument("--weights", required=True, help="pxrd-atom25 checkpoint dir (.hydra + *.ckpt)")
    ap.add_argument("--inputs", required=True, help="prepare.py output dir")
    ap.add_argument("--out", required=True, help="output dir for candidate CIFs + results.csv")
    ap.add_argument("--peak-mode", choices=["raw", "manual"], default="raw",
                    help="raw = their search_peaks on our raw pattern; "
                         "manual = our remapped manual peak list")
    ap.add_argument("--num-evals", type=int, default=20)
    ap.add_argument("--infer-timesteps", type=int, default=200)
    ap.add_argument("--step-lr", type=float, default=5)
    ap.add_argument("--max-atoms", type=int, default=None,
                    help="skip manifest rows whose primitive cell exceeds this")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    repo = Path(args.repo).resolve()
    indir = Path(args.inputs).resolve()
    out = Path(args.out).resolve()
    (out / "cifs").mkdir(parents=True, exist_ok=True)

    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core.structure import Structure

    model = load_model(repo, Path(args.weights).resolve())
    matchers = {k: StructureMatcher(**kw) for k, kw in MATCHERS.items()}
    subdir = "inputs_peaks" if args.peak_mode == "manual" else "inputs_raw"

    rows = list(csv.DictReader(open(indir / "manifest.csv")))
    if args.peak_mode == "manual":
        rows = [r for r in rows if int(r.get("has_peaks", 0))]
    if args.max_atoms is not None:
        rows = [r for r in rows if int(r["n_atoms"]) <= args.max_atoms]
    if args.limit:
        rows = rows[: args.limit]

    results = []
    t0 = time.time()
    for i, r in enumerate(rows):
        sid = r["id"]
        dat = indir / subdir / f"{sid}.dat"
        if not dat.exists():
            print(f"skip {sid}: no {subdir} input")
            continue
        px, py = (peaks_from_manual if args.peak_mode == "manual"
                  else peaks_from_raw)(dat)
        if len(px) == 0:
            print(f"skip {sid}: no peaks found")
            continue
        gt = Structure.from_file(indir / "gt" / f"{sid}.cif")
        data_list = build_data(r["formula"], px, py, args.num_evals)
        arrs = generate(model, data_list, step_lr=args.step_lr,
                        infer_timesteps=args.infer_timesteps)

        rmsd = {k: [] for k in matchers}
        for j, arr in enumerate(arrs):
            s = to_structure(arr)
            if s is not None:
                s.to(filename=str(out / "cifs" / f"{sid}_{j + 1}.cif"), fmt="cif")
            for k, m in matchers.items():
                d = None
                if s is not None:
                    try:
                        rd = m.get_rms_dist(gt, s)
                        d = None if rd is None else float(rd[0])
                    except Exception:  # noqa: BLE001
                        d = None
                rmsd[k].append(d)

        row = {"id": sid, "formula": r["formula"], "n_atoms": int(r["n_atoms"]),
               "n_peaks": len(px), "coverage": float(r["coverage"])}
        for k in matchers:
            vals = rmsd[k]
            matched = [v for v in vals if v is not None]
            row[f"{k}_match1"] = int(vals[0] is not None)
            row[f"{k}_matchN"] = int(len(matched) > 0)
            row[f"{k}_best_rmsd"] = round(min(matched), 4) if matched else ""
        results.append(row)
        print(f"[{i + 1}/{len(rows)}] {sid} {r['formula']} np={len(px)} "
              f"default(1/N)={row['default_match1']}/{row['default_matchN']} "
              f"paper(1/N)={row['paper_match1']}/{row['paper_matchN']} "
              f"({time.time() - t0:.0f}s)")

    fields = (["id", "formula", "n_atoms", "n_peaks", "coverage"]
              + [f"{k}_{s}" for k in matchers
                 for s in ("match1", "matchN", "best_rmsd")])
    with open(out / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)

    n = len(results)
    print(f"\n=== RealPXRDSolver (pxrd-atom25) on {n} patterns, "
          f"{args.peak_mode}-peaks, {args.num_evals}-sample ===")
    for k in matchers:
        m1 = sum(r[f"{k}_match1"] for r in results) / n if n else 0.0
        mN = sum(r[f"{k}_matchN"] for r in results) / n if n else 0.0
        print(f"{k:8s}: 1-sample={m1:.3f}  best-of-{args.num_evals}={mN:.3f}")
    print(f"results -> {out / 'results.csv'}")


if __name__ == "__main__":
    main()
