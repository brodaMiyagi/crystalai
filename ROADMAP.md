# CrystalAI — Master Roadmap

**Submission target:** ICLR (September 2026 deadline)
**Compute envelope:** 2 nodes × 4 GPUs (8 GPUs total), 24 GB per GPU (~22.5 GiB usable), ~180 GiB aggregate. Single training runs sized to fit within ~1 week.
---

## Goal

Demonstrate two contributions on real experimental PXRD data, in a paper-ready incremental form:

1. **Multimodal representation learning** — full-pattern + peak-position encoders trained jointly produce representations that classify crystal system (CS) and space group (SG) more robustly on real experimental patterns than a single-input full-pattern baseline does.
2. **Generative-model conditioning** — the same representations, when used to condition a flow generator trained from scratch on this representation's output distribution, produce measurably better crystal structures than naive embeddings from a single-input encoder.

The paper narrative is the two-step comparison: vanilla single-input baseline → multimodal robust representation → quantified lift on both classification and generation.

---

## Three-repo structure

The project lives in three independent repositories, each with its own roadmap. Decoupling lets each piece be validated in isolation before integration — addressing the central rebuild concern that previous work took large steps without validating intermediate results.

| Repo | Scope | Roadmap |
|------|-------|---------|
| `CrystalAI-data` | Curate ICSD CIFs and the MP-20 subset; ingest and structure RRUFF / opXRD / internal lab patterns; provide visualization apps | `DATA_ROADMAP.md` |
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

### Track B — Generation (from-scratch flow generator)

**Phase B1: Vanilla conditioning.**
Train a lightweight encoder on simulated patterns (no contrastive objectives, no multimodal alignment — just a classifier-trained or simple-MSE-pretrained encoder), then **freeze it**. Train a flow generator (FlowMM/DiffCSP-lineage equivariant GNN over lattice + fractional coords + atom types) **from scratch**, conditioned on the frozen encoder's embedding — the generator learns to natively consume that conditioning distribution. Establishes a vanilla baseline. PXRDGen (Code Ocean) and XtalNet (Zenodo) are used as code scaffolding, not as loaded checkpoints.

**Phase B2: Robust conditioning.**
Swap the B1 encoder for the A2 multimodal encoder (frozen). Train a fresh flow generator from scratch on the same recipe, conditioned on the A2 encoder. Demonstrates the lift attributable to representation quality — each representation gets its own best-fit generator, a cleaner comparison than remapping a thin adapter onto a fixed denoiser.

The A2 → B2 encoder transfer is the structural reason Tracks A and B are not fully independent. In both phases the **encoder is frozen and the generator is trained**; encoder and generator are never co-trained (that would break the transfer).

---

## Core design commitments

These are settled. Each carries rationale and trade-offs in `DESIGN_DECISIONS.md`.

1. **Encoder input coordinate: log(d-spacing)**, with **wavelength as an explicit input** (FiLM-conditioned for CNNs). Removes wavelength as a coordinate confound; enables data pooling across Cu Kα / Mo Kα / Co Kα / synchrotron sources.
2. **Crystal structure sources: ICSD (primary) + MP-20 (Track B only).** ICSD is experimentally-verified and the sole source for Track A and for the encoder. MP-20 (DFT-relaxed MP subset, ≤20 atoms/cell, redistributable) is added solely to make Track B's generator publicly releasable — ICSD's terms prohibit redistributing a generative model that emits full crystallographic phases. The released generator is trained on MP-20 only; an internal ICSD+MP-20 (≤20-atom) variant may be reported but not shipped. No COD, no MP-20-PXRD pre-computed-pattern comparison, no SimXRD-4M comparison. The driver is *licensing*, not a change in the structure-quality stance (the DFT-to-real caveat still holds and is why MP-20 is scoped to Track B targets, not Track A inputs).
3. **Primary encoder architecture: no-pool 1D CNN in log-d** for both the full-profile and peak-position views. Position-preserving reduction (flatten + FC), not global average pooling. Bidirectional GRU and lightweight Transformer evaluated as logged ablations on the real-data metric.
4. **Wavelength injection mechanism: FiLM** (CNN primary), prepended token (Transformer ablation), initial hidden state (GRU ablation).
5. **Experimental data: ~5k labeled patterns only.** RRUFF (~3k) + opXRD-labeled (~1k) + internal lab (<1k). The 91k uncurated opXRD pool is dropped. Supervised fine-tuning with matched-sample contrastive alignment replaces unsupervised Sinkhorn DA.
6. **Track A multimodal alignment: VICReg between profile and peak views; classification firewall via detached probe** with a small metered gradient leak.
7. **Peak-position augmentation: importance-aware** — protect high-2θ / low-d peaks (lattice-pinning), and hard-reject any augmentation that fabricates systematic absences.
8. **Generator backbone: trained from scratch; encoder frozen.** Track B trains a flow generator (FlowMM/DiffCSP-class small equivariant GNN) from scratch on the ≤20-atom structure subset, conditioned on the frozen Track-A encoder. Only the generator trains in Track B (encoder frozen, never co-trained). A frozen *pretrained* generator was rejected: it would lock in its own training-set structure prior and its own encoder's conditioning space, neither compatible with our target structures or log-d multi-source encoder. The per-run cost (~1 day on a 24 GB card for this model class) fits the ≤1-week budget, so from-scratch is in scope.

---

## Compute & timeline

| Constraint | Value |
|------------|-------|
| GPUs | 2 nodes × 4 GPUs = 8 GPUs total; 24 GB per GPU (~22.5 GiB usable); ~180 GiB aggregate |
| Single-experiment training time budget | ≤ 1 week |
| Submission deadline | ICLR, September 2026 |
| Estimated major training runs before submission | ~6 (A1 baseline; A1 architecture ablations; A2 multimodal; B1 vanilla generator; B2 robust generator; one buffer for re-runs). Each Track B generator run is ~1 day for this model class; encoder runs ~½–1 day. |

The load-bearing Track B constraint is *iteration count*, not per-run compute. A single from-scratch flow-generator run on the ≤20-atom subset is ~1 day on a 24 GB card (FlowMM/DiffCSP-lineage models are small equivariant GNNs, not image/text-scale generators), comfortably inside the ≤1-week budget — so training the generator from scratch is feasible, and the earlier "frozen generator is the load-bearing concession" framing no longer applies. The real calendar cost is the debug/tune loop on the conditioning interface and flow-matching hyperparameters (many runs → weeks of wall-clock). Memory is the caveat that bites on this hardware: flow-matching is heavier than diffusion, but only the large-cell (MOF-scale) regime is problematic; at ≤20 atoms it fits. RWTH HPC / cloud allocation is held in reserve if the ≤20-atom envelope needs to grow.

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
- **Generator scaffolding for B1/B2.** Which codebase to adapt — PXRDGen (flow, Code Ocean) or XtalNet (diffusion, Zenodo) — as the from-scratch flow generator's scaffolding? PXRDGen's flow backbone is the cheaper, more natural base. Both are used as code references only, not loaded checkpoints. Decision deferred until B1 prototyping confirms the flow-matching training loop and conditioning interface against our frozen encoder.
- **Generator training set (released vs internal).** The released generator trains on MP-20 (≤20 atoms) for redistributability; an internal variant may union ICSD ≤20-atom structures (via the `n_atoms_le_20` filter) for an ICSD+MP-20 reported-but-unshipped result. Whether the internal variant is worth the extra run is contingent on whether MP-20 alone gives competitive match rates.
- **Gap-compression risk in Track B.** A from-scratch generator on a small-cell subset can learn a strong *unconditional* prior (small cells are low-entropy), so both B1 and B2 rise on the prior alone and the conditioning lift shrinks. Watch the unconditional-vs-conditioned gap; if it compresses, the B1→B2 comparison loses signal and conditioning strength needs auditing.
- **Conditioning-failure fallback.** If a from-scratch generator conditioned on the frozen A2 encoder fails to beat B1, the encoder side is the suspect (the generator is already trained natively for the encoder, so an adapter mismatch is no longer the explanation). Fallback is to unfreeze the A2 projection heads (not the trunk) and re-run.
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
