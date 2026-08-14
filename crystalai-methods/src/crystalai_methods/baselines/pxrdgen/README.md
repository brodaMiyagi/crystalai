# PXRDGen external-baseline eval

Evaluates **PXRDGen** (Li et al., *Nat. Commun.* 16:7428, 2025;
[paper](https://www.nature.com/articles/s41467-025-62708-8),
[Code Ocean capsule 7727770](https://codeocean.com/capsule/7727770/tree/v1)) on our
experimental stores — the internal **lab** patterns and the public **opXRD** set.
Reference harness only — no PXRDGen checkpoint feeds our own model (see
`crystalai-methods/CLAUDE.md`). Mirrors `baselines/alphadiffract/`.

## What PXRDGen is

A flow/diffusion crystal-structure generator conditioned on **a PXRD pattern + the
chemical formula**. We use the `flow_CNN` variant ("XRD + formula", Fig. 2 of the
paper); the `flow_CNN_L` variant additionally takes the lattice `L`. It generates
N candidate structures; a candidate "matches" if pymatgen's `StructureMatcher`
aligns it to the ground truth.

## Input contract (critical)

- **7500 intensities**, equally spaced in 2θ over **[5°, 80°) at 0.01°** (grid
  `5.00…79.99`; the paper says "5°–80°, step 0.01°" — the training loader holds 7501
  points and drops the last, and the shipped `*_XRD.dat` are the 7500 form).
- **Normalized by max peak** (max → 1).
- **No wavelength input.** MP-20 training patterns are Cu Kα, so a non-Cu pattern is
  Bragg-remapped to the **Cu Kα1** (λ=1.540598 Å) 2θ scale before use (`remap.py`).
- **Formula = full-cell composition** of the *primitive* cell (MP-20 convention,
  e.g. `Cr4O6`, not the reduced `Cr2O3`). Ground-truth structure source depends on
  the store: **lab** rows resolve it from the matched ICSD entry via `cif_id`;
  **opXRD** rows carry no ICSD match, so GT comes from the pattern's shipped
  `cif_path`. Disordered (partial-occupancy) structures are skipped — PXRDGen emits
  ordered cells.

## Environment

Needs torch 2.1.0+cu118 / torch_geometric / pymatgen 2023.8.10, which **conflict with
the methods env** — use a dedicated venv (built at `PXRDGen-code/pxrdgen-venv`):

```
uv venv --python 3.10 pxrdgen-venv && source pxrdgen-venv/bin/activate
uv pip install "torch==2.1.0" --index-url https://download.pytorch.org/whl/cu118
uv pip install torch_scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
uv pip install torch_geometric==2.4.0 lightning==2.1.4 hydra-core==1.3.2 \
    pymatgen==2023.8.10 chemparse p_tqdm tqdm SMACT fastdtw "setuptools<81" "numpy<2" pytest
```

Runs on CPU (~1 min/pattern for 20 candidates); the shipped GPU passthrough is not
required. The extracted capsule root (`PXRDGen-code/capsule`) holds the code and the
trained weights `data/outs/xrd2struc/flow_CNN/`. That config's `encoder_xrd_fix` is
set to `'None'` so it doesn't re-load the pretrain encoder (weights come from the full
checkpoint) — required for CPU loading.

## Run

```
export PYTHONPATH=src CUDA_VISIBLE_DEVICES=""   # profile sets CVD='' which hides GPUs
# 1. build inputs (remapped .dat + gt cif + manifest) from a chosen exp_data source
python -m crystalai_methods.baselines.pxrdgen.prepare \
    --data-root ../crystalai-data --source lab --out <WORK>/eval_lab
#   opXRD instead:  --source opxrd --out <WORK>/eval_opxrd
# 2. generate + score
python -m crystalai_methods.baselines.pxrdgen.run_eval \
    --capsule <WORK>/capsule --inputs <WORK>/eval_lab --out <WORK>/eval_lab/run_full \
    --num-evals 20
```

`--source` (default `lab`) picks the exp_data store; `opxrd` selects the public set
(GT via `cif_path`). The scored opXRD run below filtered to ≤20-atom patterns.

`run_eval` scores each candidate under two `StructureMatcher` settings: `default`
(ltol=0.2, stol=0.3, angle_tol=5) and `paper` (ltol=0.3, stol=0.5, angle_tol=10),
reporting 1-sample (candidate #1) and best-of-N match rates to `results.csv`.

## Headline results

**Lab** (80 patterns → 76 scored; 4 disordered skipped):

| set | paper 1-sample | paper best-of-20 | default 1-sample | default best-of-20 |
|---|---|---|---|---|
| all (76) | 0.368 | 0.697 | 0.263 | 0.592 |
| ≤20 atoms (68, in-dist) | 0.397 | 0.735 | 0.279 | 0.647 |
| >20 atoms (8, OOD) | 0.125 | 0.375 | 0.125 | 0.125 |

**opXRD** (≤20-atom subset, 141 patterns scored; 125 CNRS + 16 HKUST-A):

| set | paper 1-sample | paper best-of-20 | default 1-sample | default best-of-20 |
|---|---|---|---|---|
| ≤20 atoms (141) | 0.298 | 0.610 | 0.213 | 0.411 |

opXRD lands in the same ballpark as lab but a notch lower — noisier, more chemically
diverse real-world data. Median best RMSD among matched ≈ 0.001 (tight when it
matches). For reference the paper reports 82% (1-sample) / 96% (20-sample) on
*simulated* MP-20 with paper tolerances — real experimental data is substantially
harder.
