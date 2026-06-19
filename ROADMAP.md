# CrystalAI — Master Roadmap

**Submission target:** ICLR (September 2026 deadline)
**Compute envelope:** 2 nodes × 4 GPUs (8 GPUs total), 24 GB per GPU (~22.5 GiB usable), ~180 GiB aggregate. Single training runs sized to fit within ~1 week.
---

## Goal

Demonstrate two contributions on real experimental PXRD data, in a paper-ready incremental form:

1. **Multimodal representation learning** — full-pattern + peak-position encoders trained jointly produce representations that classify crystal system (CS) and space group (SG) more robustly on real experimental patterns than a single-input full-pattern baseline does.
2. **Generative-model conditioning** — the same representations, when used to condition a frozen XtalNet or PXRDGen generator, produce measurably better crystal structures than naive embeddings from a single-input encoder.

The paper narrative is the two-step comparison: vanilla single-input baseline → multimodal robust representation → quantified lift on both classification and generation.

---

## Three-repo structure

The project lives in three independent repositories, each with its own roadmap. Decoupling lets each piece be validated in isolation before integration — addressing the central rebuild concern that previous work took large steps without validating intermediate results.

| Repo | Scope | Roadmap |
|------|-------|---------|
| `CrystalAI-data` | Curate ICSD CIFs; ingest and structure RRUFF / opXRD / internal lab patterns; provide visualization apps | `DATA_ROADMAP.md` |
| `CrystalAI-simXRD` | Simulate PXRD patterns from CIFs in 2θ and log(d); on-the-fly augmentation pipeline; interactive simulation dashboard | `SIMXRD_ROADMAP.md` |
| `CrystalAI-methods` | Track A (classification) and Track B (generation); encoders, training stages, evaluation | `METHODS_ROADMAP.md` |

Files flow strictly downstream: `data` produces curated CIFs and experimental pattern indices that `simXRD` and `methods` consume; `simXRD` produces simulated patterns and augmentation transforms that `methods` consumes. `methods` is the only repo that does ML training.

---

## Track summary

### Track A — Prediction (CS + SG classification)

**Phase A1: Single-input baseline.**
Full diffraction pattern (log-d binned, background-removed) → encoder (no-pool log-d CNN primary; bidirectional GRU and lightweight Transformer as logged ablations) → CS classification head + CS-gated SG classification head. Wavelength injected via FiLM (CNN) or equivalent per architecture. Trained on simulated patterns + ~5k labeled experimental.

**Phase A2: Multimodal.**
Add a peak-position encoder (no-pool log-d CNN with dilated convolutions for spacing-relational sensitivity). VICReg alignment between the two encoders' z_lattice projections. Classification firewall via a detached probe with a small metered gradient leak. Importance-aware peak-position augmentation. Trained on the same data as A1; comparison is against A1 on the same evaluation set.

### Track B — Generation (XtalNet, PXRDGen)

**Phase B1: Vanilla conditioning.**
Train a lightweight encoder on ICSD-simulated patterns (no contrastive objectives, no multimodal alignment — just a classifier-trained or simple-MSE-pretrained encoder). Condition a frozen XtalNet / PXRDGen generator on this embedding. Train only the conditioning layers. Establishes a vanilla baseline.

**Phase B2: Robust conditioning.**
Swap the B1 encoder for the A2 multimodal encoder (frozen). Same conditioning-layer training, same frozen generator. Demonstrates the lift attributable to representation quality.

The A2 → B2 encoder transfer is the structural reason Tracks A and B are not fully independent.

---

## Core design commitments

These are settled. Each carries rationale and trade-offs in `DESIGN_DECISIONS.md`.

1. **Encoder input coordinate: log(d-spacing)**, with **wavelength as an explicit input** (FiLM-conditioned for CNNs). Removes wavelength as a coordinate confound; enables data pooling across Cu Kα / Mo Kα / Co Kα / synchrotron sources.
2. **Crystal structure source: ICSD only.** No MP, CrystDB, COD, MP-20-PXRD, or SimXRD-4M comparison. Future structure additions preserved as an explicit option.
3. **Primary encoder architecture: no-pool 1D CNN in log-d** for both the full-profile and peak-position views. Position-preserving reduction (flatten + FC), not global average pooling. Bidirectional GRU and lightweight Transformer evaluated as logged ablations on the real-data metric.
4. **Wavelength injection mechanism: FiLM** (CNN primary), prepended token (Transformer ablation), initial hidden state (GRU ablation).
5. **Experimental data: ~5k labeled patterns only.** RRUFF (~3k) + opXRD-labeled (~1k) + internal lab (<1k). The 91k uncurated opXRD pool is dropped. Supervised fine-tuning with matched-sample contrastive alignment replaces unsupervised Sinkhorn DA.
6. **Track A multimodal alignment: VICReg between profile and peak views; classification firewall via detached probe** with a small metered gradient leak.
7. **Peak-position augmentation: importance-aware** — protect high-2θ / low-d peaks (lattice-pinning), and hard-reject any augmentation that fabricates systematic absences.
8. **Generator backbone: frozen.** Compute envelope dictates this. Only conditioning layers are trained in Track B.

---

## Compute & timeline

| Constraint | Value |
|------------|-------|
| GPUs | 2 nodes × 4 GPUs = 8 GPUs total; 24 GB per GPU (~22.5 GiB usable); ~180 GiB aggregate |
| Single-experiment training time budget | ≤ 1 week |
| Submission deadline | ICLR, September 2026 |
| Estimated major training runs before submission | ~6 (A1 baseline; A1 architecture ablations; A2 multimodal; B1 vanilla; B2 robust; one buffer for re-runs) |

The frozen-generator decision is the load-bearing compute concession. Retraining a diffusion / flow backbone from scratch is out of scope for this submission; the conditioning-first path is the pragmatic one. RWTH HPC / cloud allocation is held in reserve as a fallback if conditioning fundamentally fails.

---

## Validation philosophy

Each piece is validated before stacking the next:

- `CrystalAI-data` produces curated, indexed data — validated by visualization, distribution plots, and spot-checks against source databases.
- `CrystalAI-simXRD` produces simulated patterns — validated against pymatgen's `XRDCalculator` for peak positions and integrated intensities; validated against experimental patterns from `CrystalAI-data` via Rwp on a small held-out set.
- `CrystalAI-methods` Phase A1 — validated by reaching a defensible CS/SG accuracy floor on the 82 lab patterns before Phase A2 begins.
- Phase A2 — validated by demonstrating a measurable lift over A1 on the same held-out set, *before* the encoder is transferred to Track B.
- Phase B1 — validated by reaching a defensible generation baseline (top-k match rate, RMSD, Rwp post-Rietveld) on real experimental patterns before B2 begins.
- Phase B2 — validated by demonstrating a measurable lift over B1 on the same held-out set.

No phase begins until its predecessor's validation is in hand. The point of the rebuild is to never again stack work on an unverified intermediate.

---

## Open decisions

These remain genuinely undecided and should be revisited as the project progresses:

- **Track A architecture ablation depth.** Run all three (CNN / GRU / Transformer) at A1 baseline, or commit to CNN-primary and run GRU/Transformer at smaller scale only as a sanity check? Compute pressure argues for the latter; methodology arguments for the former. Default: CNN-primary at full scale; GRU and Transformer at a single reduced-scale configuration for the ablation table.
- **Generator choice for B1/B2.** XtalNet (diffusion), PXRDGen (flow), or both? PXRDGen's flow backbone is cheaper than XtalNet's diffusion. Both have published checkpoints. Decision deferred until B1 prototyping confirms the conditioning-layer interface for at least one.
- **Frozen-encoder risk in Track B.** If the A2 encoder, frozen and inserted into Track B's conditioning path, fails to provide useful gradients to the conditioning layers, the fallback is to unfreeze the projection heads (not the backbone). Decision contingent on B2 results.
- **Track A1 vs A2 encoder relationship.** Should the A2 encoder reuse A1 weights as initialization, or train from scratch? Reuse saves compute and makes the lift a clean transfer story; from-scratch makes the comparison more methodologically honest. Default: A2 reuses the A1 CNN backbone as initialization and adds the peak-position encoder fresh.

---

## File map

| File | Purpose |
|------|---------|
| `ROADMAP.md` | This document — overall structure, commitments, narrative |
| `DATA_ROADMAP.md` | CrystalAI-data: curation, ingestion, visualization |
| `SIMXRD_ROADMAP.md` | CrystalAI-simXRD: simulation engine, augmentation |
| `METHODS_ROADMAP.md` | CrystalAI-methods: Track A and Track B in detail |
| `DESIGN_DECISIONS.md` | Consolidated rationale for the 8 core commitments above |
