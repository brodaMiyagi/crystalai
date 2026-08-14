"""Run PXRDGen (flow-CNN, XRD+formula) on prepared lab inputs and score matches.

Loads the trained ``flow_CNN`` generator once, then for each manifest row draws
``--num-evals`` candidate structures from the remapped pattern + full-cell
formula, writes the candidate CIFs, and compares each against the ground-truth
ICSD structure with pymatgen's ``StructureMatcher`` under two settings:

  * ``default``  — StructureMatcher() defaults (ltol=0.2, stol=0.3, angle_tol=5)
  * ``paper``    — the PXRDGen paper's setting (ltol=0.3, stol=0.5, angle_tol=10)

Reports 1-sample (candidate #1 matches) and best-of-N (any candidate matches)
match rates. Depends on the Code Ocean capsule code + trained weights; point
``--capsule`` at the extracted capsule root. Run inside the PXRDGen venv.
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


def load_model(capsule: Path, model_rel: str, label: int = -1):
    xrd2struc = capsule / "code" / "xrd2struc"
    sys.path.insert(0, str(xrd2struc))
    sys.path.insert(0, str(xrd2struc / "scripts"))
    import hydra
    from hydra import initialize_config_dir
    from pxrdgen.model.flow_shift_align import CSPFlow

    model_path = capsule / model_rel
    with initialize_config_dir(str(model_path / ".hydra")):
        cfg = hydra.compose(config_name="config")
    ckpt = sorted(model_path.glob("*.ckpt"))[label]
    model = CSPFlow.load_from_checkpoint(str(ckpt), map_location="cpu", **cfg.model)
    model.eval()
    return model


def build_data(formula: str, xrd: np.ndarray, num_evals: int):
    from eval_utils import chemical_symbols
    from torch_geometric.data import Data
    import chemparse

    comp = chemparse.parse_formula(formula)
    chem_list = []
    for elem, cnt in comp.items():
        chem_list.extend([chemical_symbols.index(elem)] * int(cnt))
    xrd_t = torch.Tensor(xrd).view(1, -1)
    return [
        Data(atom_types=torch.LongTensor(chem_list), xrd_array=xrd_t,
             num_atoms=len(chem_list), num_nodes=len(chem_list))
        for _ in range(num_evals)
    ]


def generate(model, data_list, step_lr=5, infer_timesteps=200):
    from torch_geometric.data import Batch
    from eval_utils import get_crystals_list, lattices_to_params_shape
    batch = Batch.from_data_list(data_list)
    if torch.cuda.is_available():
        batch = batch.cuda()
    out = model.sample(batch, step_lr=step_lr, infer_timesteps=infer_timesteps)
    lengths, angles = lattices_to_params_shape(out["lattices"].detach().cpu())
    return get_crystals_list(
        out["frac_coords"].detach().cpu(), out["atom_types"].detach().cpu(),
        lengths, angles, out["num_atoms"].detach().cpu())


def to_structure(arr):
    from pymatgen.core.structure import Structure
    from pymatgen.core.lattice import Lattice
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
    ap.add_argument("--capsule", required=True, help="extracted Code Ocean capsule root")
    ap.add_argument("--inputs", required=True, help="prepare.py output dir (manifest + inputs/ + gt/)")
    ap.add_argument("--out", required=True, help="output dir for candidate CIFs + results.csv")
    ap.add_argument("--model-rel", default="data/outs/xrd2struc/flow_CNN")
    ap.add_argument("--num-evals", type=int, default=20)
    ap.add_argument("--infer-timesteps", type=int, default=200)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-atoms", type=int, default=None,
                    help="skip manifest rows whose primitive cell exceeds this "
                         "(PXRDGen's MP-20 regime is <=20; large cells are OOD)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    capsule = Path(args.capsule).resolve()
    indir = Path(args.inputs).resolve()
    out = Path(args.out).resolve()
    (out / "cifs").mkdir(parents=True, exist_ok=True)

    from pymatgen.core.structure import Structure
    from pymatgen.analysis.structure_matcher import StructureMatcher

    model = load_model(capsule, args.model_rel)
    matchers = {k: StructureMatcher(**kw) for k, kw in MATCHERS.items()}

    rows = list(csv.DictReader(open(indir / "manifest.csv")))
    if args.max_atoms is not None:
        rows = [r for r in rows if int(r["n_atoms"]) <= args.max_atoms]
    if args.limit:
        rows = rows[: args.limit]

    results = []
    t0 = time.time()
    for i, r in enumerate(rows):
        sid = r["id"]
        xrd = np.loadtxt(indir / "inputs" / f"{sid}.dat")
        gt = Structure.from_file(indir / "gt" / f"{sid}.cif")
        data_list = build_data(r["formula"], xrd, args.num_evals)
        arrs = generate(model, data_list, infer_timesteps=args.infer_timesteps)

        rmsd = {k: [] for k in matchers}  # best rms per candidate, per matcher
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
               "coverage": float(r["coverage"])}
        for k in matchers:
            vals = rmsd[k]
            matched = [v for v in vals if v is not None]
            row[f"{k}_match1"] = int(vals[0] is not None)
            row[f"{k}_matchN"] = int(len(matched) > 0)
            row[f"{k}_best_rmsd"] = round(min(matched), 4) if matched else ""
        results.append(row)
        print(f"[{i + 1}/{len(rows)}] {sid} {r['formula']} "
              f"default(1/N)={row['default_match1']}/{row['default_matchN']} "
              f"paper(1/N)={row['paper_match1']}/{row['paper_matchN']} "
              f"({time.time() - t0:.0f}s)")

    fields = (["id", "formula", "n_atoms", "coverage"]
              + [f"{k}_{s}" for k in matchers
                 for s in ("match1", "matchN", "best_rmsd")])
    with open(out / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)

    n = len(results)
    print(f"\n=== PXRDGen flow-CNN on {n} lab patterns "
          f"({args.num_evals}-sample) ===")
    for k in matchers:
        m1 = sum(r[f"{k}_match1"] for r in results) / n
        mN = sum(r[f"{k}_matchN"] for r in results) / n
        print(f"{k:8s}: 1-sample={m1:.3f}  best-of-{args.num_evals}={mN:.3f}")
    print(f"results -> {out / 'results.csv'}")


if __name__ == "__main__":
    main()
