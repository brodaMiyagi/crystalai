# OpenAlphaDiffract — external baseline on CrystalAI experimental data

Evaluation harness that runs the published **OpenAlphaDiffract** model
(arXiv 2603.23367; weights: [`linked-liszt/OpenAlphaDiffract`](https://huggingface.co/linked-liszt/OpenAlphaDiffract))
over our real experimental PXRD store (`crystalai-data/exp_data`).

This is a **downstream reference baseline only** — it reads existing data + labels
and runs a frozen third-party model. It does not touch our encoder, simulator, or
any Track-A/B gate, and it is not itself a gate.

---

## The model's input contract

8.7M-param 1D ConvNeXt, three heads: crystal system (7), space group (230),
lattice parameters (6). Critically, the model has **no wavelength input** —
`forward(x)` takes **8192 intensities on a grid equally spaced in 2θ over
[5°, 20°] at monochromatic 20 keV** (λ = 0.61992 Å), i.e. d ∈ [1.785, 7.106] Å.
Preprocessing (per the model card): floor negatives at zero, then rescale to
[0, 100]. CS head order is Triclinic..Cubic (index 0..6); SG prediction = argmax + 1.

## Input remap (the crux)

Our patterns are mostly Cu Kα, so each measured point is moved onto the model's
20 keV 2θ scale via Bragg's law (wavelength-invariant d-spacing):

```
d       = λ_src / (2 sin θ_src)
sin θ20 = λ_20 / (2 d) = (λ_20 / λ_src) · sin θ_src
```

Intensities are then linearly resampled onto the fixed 8192-point 2θ grid **in the
2θ coordinate** and normalized to [0, 100]. For genuine 20 keV data this is the
identity. Bins outside the measured range are zero-filled; `coverage` records the
fraction of the model's window a pattern actually spans. See [`remap.py`](remap.py).

## Scored set

Filter: `(audit == 'pass' OR source == 'lab') AND wavelength AND space_group AND
bgsub`. Note this is **3570**, not the 4145 `audit=='pass'` rows — 655 pass rows
lack an SG label, and the 80 lab rows (blank audit, but cleanest ground truth) are
added explicitly. Scored as two sets by background provenance:

| set | n | composition |
|---|---|---|
| **manual** (autobg=0) | 1114 | RRUFF native-curated `XY_Processed` (1034) + lab hand-verified bg (80) |
| **auto** (autobg=1) | 2456 | RRUFF (1581) + opXRD (875), our pybaselines auto-background |

> Terminology: "manual" here means *not our autobg*. The 80 lab patterns are
> genuinely hand-verified; the 1034 RRUFF are RRUFF's own curated subtraction.
> Only ~1484 of RRUFF's 3003 powder patterns ship an `XY_Processed` file, which is
> why the other 1773 fall into the auto set.

---

## Results

Single released checkpoint (not the paper's 10-model ensemble), evaluated on
**real** data. Background-subtracted input. **Overall: CS top-1 0.530, SG top-1 0.340.**

### By background set and source

| set / source | n | CS top-1 | SG top-1 | SG top-5 | CS↔SG consistency |
|---|--:|--:|--:|--:|--:|
| **manual** (all) | 1114 | 0.559 | 0.364 | 0.650 | 0.697 |
| · lab | 80 | **0.700** | 0.412 | 0.762 | 0.738 |
| · rruff | 1034 | 0.548 | 0.361 | 0.641 | 0.694 |
| **auto** (all) | 2456 | 0.516 | 0.329 | 0.612 | 0.699 |
| · rruff | 1581 | 0.585 | 0.388 | 0.688 | 0.708 |
| · opxrd | 875 | 0.392 | 0.223 | 0.474 | 0.682 |

*CS↔SG consistency = fraction where the CS head agrees with the crystal system
implied by the predicted SG (via `sg_to_cs`); a label-order-independent sanity signal.*

### Lattice-parameter MAE (Å / °)

The lattice head effectively fails on real data — reported for completeness.

| set / source | a | b | c | α | β | γ |
|---|--:|--:|--:|--:|--:|--:|
| manual (all) | 3.11 | 2.99 | 5.63 | 3.26 | 6.03 | 7.70 |
| · lab | 1.52 | 2.65 | 5.31 | 1.80 | 1.06 | 5.37 |
| auto (all) | 3.84 | 3.48 | 5.38 | 3.34 | 6.14 | 6.51 |

### CS top-1 by true crystal system (coverage ≥ 0.95, both sets)

| system | n | CS top-1 |
|---|--:|--:|
| triclinic | 338 | 0.506 |
| monoclinic | 1140 | 0.662 |
| orthorhombic | 793 | 0.483 |
| tetragonal | 317 | 0.448 |
| trigonal | 397 | 0.370 |
| hexagonal | 226 | 0.473 |
| cubic | 283 | 0.562 |

### Reading of the results

- **Manual bg > auto bg** (0.559 vs 0.516 CS): cleaner backgrounds help.
- **Lab is best** (0.70 CS): reported labels + matched ICSD structure + hand-verified bg.
- **opXRD is worst** (0.39 CS / 0.22 SG): spglib-derived P1-expanded labels + auto bg.
- **RRUFF is consistent** (~0.55–0.59 CS) across both bg types.
- The **lattice head does not transfer** to real data (MAE 3–6 Å on cell edges).

### Caveats

Single model, not the paper's 10-model ensemble (so this is a floor on its
ability). Trained on **MP DFT-relaxed, GSAS-II-simulated 20 keV** patterns and
evaluated on **real** data (domain gap: peak shapes, resolution, residual
background). The model sees only d ∈ [1.785, 7.106] Å; high-Q information is
discarded. opXRD SGs are spglib-derived (our sanctioned experimental carve-out),
not reported ground truth; RRUFF ships no atomic CIF.

---

## Reproduce

```bash
# 1. Fetch weights into models/alphadiffract/ (gitignored, ~35 MB)
python scripts/fetch_alphadiffract.py

# 2. Sanity-check the remap (no download needed)
pytest tests/test_alphadiffract_remap.py

# 3. Smoke test (prints predictions next to ground truth)
python -m crystalai_methods.baselines.alphadiffract.run_eval --limit 10

# 4. Full run -> models/alphadiffract/results.csv + the report above
python -m crystalai_methods.baselines.alphadiffract.run_eval
```

CPU is sufficient (8.7M params × 3570 patterns, minutes). Per-pattern predictions
and labels are written to `models/alphadiffract/results.csv`.

## Files

| file | role |
|---|---|
| `remap.py` | 2θ@λ_src → 2θ@20 keV → 8192-bin model input (NumPy) |
| `infer.py` | load vendored model, batch, decode heads (PyTorch) |
| `metrics.py` | CS/SG accuracy, lattice MAE, per-set/source/system report |
| `run_eval.py` | CLI: select → remap → infer → score |
| `vendor/model.py` | upstream OpenAlphaDiffract model code, kept **verbatim** |
| `../../../../scripts/fetch_alphadiffract.py` | downloads weights from the Hub |
