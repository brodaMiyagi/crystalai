# RealPXRDSolver external-baseline eval

Evaluates **RealPXRDSolver** (DP Technology, 2025; `RealPXRD-Solver` repo — peak-set
BERT encoder + DiffCSP-lineage flow decoder) on our experimental stores — the
internal **lab** patterns and the public **opXRD** set. Reference harness only — no
RealPXRDSolver checkpoint feeds our own model (see `crystalai-methods/CLAUDE.md`).
Mirrors `baselines/pxrdgen/` and reuses its `remap.py`.

## What RealPXRDSolver is

A crystal-structure generator conditioned on **a sparse PXRD peak list + the chemical
formula**. A peak-set BERT embeds each peak's **integer 2θ degree** (an
`nn.Embedding(180)` position table; peak angles are floored to whole degrees via
`.long()`) together with its intensity; a flow decoder emits N candidate structures.
We use the **`pxrd-atom25`** checkpoint — the XRD+formula variant trained on ≤25-atom
cells, the analog of PXRDGen's `flow_CNN`. (The `-L` variants additionally take the
lattice `L`; `pxrd-all` trains on all cell sizes.) A candidate "matches" if pymatgen's
`StructureMatcher` aligns it to the ground truth.

## Input contract (critical)

- **Sparse peak list**: `(pxrd_x, pxrd_y)` — peak 2θ positions and intensities. The
  model floors `pxrd_x` to an **integer 2θ degree** index into `nn.Embedding(180)`,
  so effective position resolution is 1° in 2θ. Intensities are on a 0–100 scale.
- **No wavelength input.** Training patterns are Cu Kα, so a non-Cu pattern is
  Bragg-remapped to the **Cu Kα1** (λ=1.540598 Å) 2θ scale before use — we reuse
  `pxrdgen.remap` (identity for genuine Cu data; our lab is Cu Kα, λ≈1.5406).
- **Formula = full-cell composition** of the *primitive* cell (e.g. `Ti4O8`, not the
  reduced `TiO2`) — same convention as the PXRDGen baseline. Ground truth: **lab**
  rows resolve the matched ICSD entry via `cif_id`; **opXRD** rows carry no ICSD
  match, so GT comes from the shipped `cif_path`. Disordered (partial-occupancy)
  cells are skipped — the generator emits ordered cells.

### Two ways to get the peak list (`--peak-mode`)

1. **`manual`** — our **manually specified peak list** (`exp_data/.../peaks.xy`, an
   EXPO d-list sampled to intensities), each peak's 2θ remapped to Cu Kα1, used
   *directly* as the d-I list. Lab only (opXRD ships no manual peaks).
2. **`raw`** — RealPXRDSolver's **own** `subtract_polynomial_background` +
   `search_peaks` (6th-order polynomial background, `scipy.find_peaks`), imported
   verbatim from the upstream `scripts/sample_flow.py`, applied to our raw pattern
   after remapping to the Cu Kα1 2θ grid. Used for both lab and opXRD.

## Environment

Reuses the **PXRDGen venv** (`PXRDGen-code/pxrdgen-venv`) — torch 2.1.0+cu118 /
torch_geometric / torch_scatter / lightning 2.1.4 / hydra / pymatgen — which loads
and runs this model unchanged. Unlike the PXRDGen baseline this runs on **GPU**
(GTX 1660 SUPER, ~0.3 s/candidate for a batch of 20; the unicore fused
softmax/layer-norm CUDA kernels fall back to pure-torch automatically). The upstream
tree lives at `RealPXRDSolver/RealPXRD-Solver`; the extracted checkpoints at
`RealPXRDSolver/weight/2501/<variant>/` (`.hydra/config.yaml` + `last_one.ckpt`).

> Note: the shell profile sets `CUDA_VISIBLE_DEVICES=''` which hides the GPU — export
> it explicitly (`export CUDA_VISIBLE_DEVICES=0`) when running.

## Run

```
export PYTHONPATH=src CUDA_VISIBLE_DEVICES=0
RPS=/…/RealPXRDSolver
# 1. build inputs (raw remapped .dat + manual-peak .dat + gt cif + manifest)
python -m crystalai_methods.baselines.realpxrdsolver.prepare \
    --data-root ../crystalai-data --source lab --out $RPS/eval_lab
#   opXRD instead:  --source opxrd --out $RPS/eval_opxrd
# 2. generate + score (pick --peak-mode; --max-atoms 20 for the in-distribution set)
python -m crystalai_methods.baselines.realpxrdsolver.run_eval \
    --repo $RPS/RealPXRD-Solver --weights $RPS/weight/2501/pxrd-atom25 \
    --inputs $RPS/eval_lab --out $RPS/eval_lab/run_manual \
    --peak-mode manual --num-evals 20 --max-atoms 20
#   lab search_peaks:  --peak-mode raw  --out $RPS/eval_lab/run_raw
#   opXRD search_peaks: --inputs $RPS/eval_opxrd --peak-mode raw --out $RPS/eval_opxrd/run_raw
```

`run_eval` scores each candidate under two `StructureMatcher` settings: `default`
(ltol=0.2, stol=0.3, angle_tol=5) and `paper` (ltol=0.3, stol=0.5, angle_tol=10 — the
upstream `scripts/compute.py` setting), reporting 1-sample (candidate #1) and
best-of-N match rates to `results.csv`.

## Headline results

_(populated after the runs complete — see below)_

<!-- RESULTS -->
