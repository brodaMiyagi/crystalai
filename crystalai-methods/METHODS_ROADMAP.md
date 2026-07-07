# CrystalAI-methods — Roadmap

**Scope:** All ML training for the CrystalAI pipeline. Track A (classification: crystal system + space group). Track B (generation: conditional crystal structure generation via a flow generator trained from scratch, using PXRDGen / XtalNet as code scaffolding). Encoders, loss functions, training loops, evaluation.

**Out of scope:** Data curation (lives in `CrystalAI-data`). Pattern simulation (lives in `CrystalAI-simXRD`).

**Reading order:** This file assumes familiarity with `ROADMAP.md` (the project overview) and `DESIGN_DECISIONS.md` (the 8 core commitments and their rationale).

---

## 1. Track-and-phase structure

The pipeline is organized into two tracks, each with two phases. Each phase is validated against real-data metrics before the next phase begins.

| Phase | Goal | What's new vs the previous phase |
|-------|------|----------------------------------|
| A1 | Single-input baseline classifier | Full pattern only → CS + SG heads |
| A2 | Multimodal robust classifier | + peak-position encoder, VICReg alignment, classification firewall |
| B1 | Vanilla generative conditioning | From-scratch flow generator conditioned on a frozen lightweight encoder |
| B2 | Robust generative conditioning | + frozen A2 encoder in place of B1's encoder; fresh from-scratch generator |

Phase A2's encoder is the artifact that gets transferred (frozen) into Phase B2. This is the only structural coupling between the tracks.

---

## 2. Track A — Prediction (CS + SG)

### 2.1 Phase A1 — Single-input baseline

**Inputs.**
- Log-d binned full diffraction pattern. Fixed window (default 0.7 Å to 8 Å, ~4000 bins). Background-removed before binning (a simple Chebyshev-fit subtraction or scipy `find_peaks` baseline is sufficient at this stage; refinement deferred).
- Wavelength scalar (in Å), injected via FiLM for CNN, prepended token for Transformer ablation, initial hidden state for GRU ablation.
- Validity mask (per-bin 0/1), passed as a second input channel (CNN), an attention mask (Transformer), or concatenated per-timestep feature (GRU). Marks the bins corresponding to the actual measured 2θ range (after conversion to log-d) vs. padding.

**Encoder.** No-pool 1D CNN in log-d (primary). The trunk is ~6 strided convolutional blocks with channel progression 32 → 64 → 128 → 256, total downsampling factor ~16×, followed by flatten + FC → 256-dim representation. Wavelength FiLM modulates per-channel γ, β at intermediate blocks. The validity mask is passed as a second input channel. Parameter count: ~5–10M for the trunk.

**Architecture ablations (logged).**
- *Bidirectional GRU* in log-d. Same fixed-window input. Wavelength injected as initial hidden state. Mask via concat-feature per timestep + output masking before reduction.
- *Lightweight Transformer* in log-d. Patch-based (patches of ~50 bins → ~80 patches), sinusoidal positional encoding based on log-d patch center (not sequence index). Wavelength embedded via a small MLP and prepended as a token alongside the CLS token. Mask via attention masking on padding patches. ~1–1.5M parameters total — small, to be a fair comparison rather than a brute-force one.

Both ablations run at a reduced configuration as a sanity check; if either matches or beats the CNN on the A1 real-data metric, the open question of architecture choice (see `ROADMAP.md` Open decisions) re-opens.

**Heads.**
- **CS head**: MLP on the encoder representation (256 → 64 → 7), producing CS logits over the 7 crystal systems and an intermediate CS feature vector `f_CS` (dim 64).
- **CS-gated SG head**: 7 separate sub-heads, one per crystal system, each predicting only the SGs valid for that CS:
  - Triclinic (SG 1–2): 2 classes
  - Monoclinic (SG 3–15): 13 classes
  - Orthorhombic (SG 16–74): 59 classes
  - Tetragonal (SG 75–142): 68 classes
  - Trigonal (SG 143–167): 25 classes
  - Hexagonal (SG 168–194): 27 classes
  - Cubic (SG 195–230): 36 classes
- Sub-head input: `[encoder_representation; f_CS]`. At training time, the ground-truth CS selects which sub-head receives gradient (teacher forcing). At inference, the predicted CS selects which sub-head's output to use. A soft-gating variant — weighting each sub-head's output by `p_CS` and summing — is available for ablation, providing gradient flow through multiple sub-heads when CS is uncertain.

**Training data.**
- Simulated patterns from `CrystalAI-simXRD` (the `PRODUCTION` augmentation preset: wavelength randomization, Caglioti randomization, mild asymmetry, matched d-range, plus the standard zero shift / intensity noise / background perturbation / impurity peaks).
- ~5k labeled experimental patterns: RRUFF + opXRD-labeled + internal lab. Source tracked as metadata.
- Batch composition: 70% sim, 30% experimental (range to explore: 60:40 to 80:20).

**Losses.**

```
L_A1 = α₄ · (L_CS_sim + λ_exp · L_CS_exp) + α₅ · (L_SG_sim + λ_exp_sg · L_SG_exp)
```

- `L_CS_sim`: cross-entropy with sim-pool inverse-frequency class weights.
- `L_CS_exp`: cross-entropy with exp-pool inverse-frequency class weights (separate scheme, derived from the labeled experimental pool).
- `L_SG_sim`: cross-entropy with sim-pool inverse-frequency weights within each CS-gated sub-head.
- `L_SG_exp`: cross-entropy with no class weighting (experimental SG counts too sparse for inverse-frequency to be stable).

**Hyperparameters (starting points).**

| Parameter | Default | Range to explore |
|-----------|---------|------------------|
| `α₄`, `α₅` | 1.0, 1.0 | 0.5 – 2.0 |
| `λ_exp` | 1.0 | 0.5 – 2.0 |
| `λ_exp_sg` | 0.5 | 0.25 – 1.0 |
| Batch size | 256 | 128 – 512 (memory permitting) |
| Learning rate | 3e-4 | 1e-4 – 1e-3 |
| Optimizer | AdamW | — |
| Weight decay | 0.05 | — |
| LR schedule | Cosine with 10-epoch warmup | — |
| Epochs | 60–80 | — |

**Evaluation.**
- **Primary**: CS and SG accuracy on the 82 in-house held-out lab patterns. Per-CS SG accuracy reported separately (overall SG accuracy can be misleadingly high when CS distribution is skewed).
- **Secondary**: CS and SG accuracy on the 20% held-out split of the labeled experimental pool, with per-source breakdown (RRUFF, opXRD-labeled, lab).
- **No-leakage check**: group-aware split by CIF identity, not just by pattern. No CIF appears in both training and held-out sets.

**Phase A1 validation gate.** A defensible accuracy floor on the 82 lab patterns (current rough targets: ~75% CS, ~45% SG, based on DIFCON's 74.4% / 41.5% as the prior baseline; A1 should at minimum match this, and ideally exceed it via the log-d coordinate + wavelength conditioning). If A1 does not match these numbers, debugging A1 takes priority over moving to A2 — the issue is in the foundation.

---

### 2.2 Phase A2 — Multimodal robust classifier

**What's new vs A1.**
- Peak-position encoder added (second view of the input).
- Three projection heads on the full-profile encoder: MLP_lattice, MLP_atomic, MLP_exp.
- VICReg alignment between the two views' lattice projections.
- InfoNCE on z_atomic across augmentations of the same CIF.
- Cross-covariance disentanglement between z_lattice, z_atomic, z_exp.
- Classification firewall via detached probe.

**Architecture.**

*Full-profile encoder (carried from A1, initialized from A1 weights).* Adds three projection heads:
- MLP_lattice: encoder representation → z_lattice_profile (128-dim)
- MLP_atomic: encoder representation → z_atomic (128-dim)
- MLP_exp: encoder representation → z_exp (64-dim)

Each MLP: 2 linear layers, GELU activation, batch normalization between layers.

*Peak-position encoder.* No-pool 1D CNN in log-d on the sparse peak-position histogram (narrow Gaussians at peak positions, uniform height, same binning as the full profile but sparse). Uses dilated convolutions to give the model a wider effective receptive field — recovering the GRU-like sensitivity to spacing relationships and d-ratios while keeping the homogeneous CNN-both architecture. Channels: 32 → 64 → 128. Output: z_lattice_peaks (128-dim, same as z_lattice_profile). Parameter count: ~1–2M.

*Heads.* CS and CS-gated SG heads from A1, now reading from a `detach`-ed copy of the encoder representation (the classification firewall).

**Inputs.**

- Same log-d binned full pattern, wavelength, validity mask as A1, feeding the full-profile encoder.
- Peak-position histogram, no wavelength (peak positions are wavelength-independent in d-space), no mask (peaks-only-view is sparse and doesn't have the padding issue), feeding the peak-position encoder.

The peak-position augmentations from `CrystalAI-simXRD` (importance-aware drops + manufactured-absence guard + angle-aware jitter) are applied to the peak histogram input.

**Losses.**

```
L_A2 = L_A1_classification_via_firewall                  (W3 firewall)
     + α₁ · L_VICReg(z_lattice_profile, z_lattice_peaks)
     + α₂ · L_InfoNCE(z_atomic)                          (sim only, two augmented views)
     + α₃ · L_disentangle(z_lattice, z_atomic, z_exp)
     + α_match · (L_matched_lattice + L_matched_atomic)  (sim-exp pairs)
```

*Classification firewall (`DESIGN_DECISIONS.md` §5).* The CS / SG heads read `detach(encoder_representation)`. A scalar gradient leak permits a metered fraction back into the trunk:

```
L_cls_full_strength → trains the probe
L_cls_full_strength · λ_cls_rep_profile → leaks into the full-profile encoder trunk
L_cls_full_strength_CS_only · λ_cls_rep_peaks → leaks into the peak-position encoder trunk
```

(CS-only on the peak side — SG cross-entropy does not attach to the peak encoder. See `DESIGN_DECISIONS.md` §5.)

*VICReg (Bardes et al., 2022).* Branch-independent variance and covariance terms:

```
L_VICReg = λ · L_invariance + μ · L_variance + ν · L_covariance
L_invariance = (1/N) Σᵢ ||z_lattice_profile_i - z_lattice_peaks_i||²
L_variance   = (1/d) Σⱼ max(0, γ - std(zⱼ))             (per branch, independently)
L_covariance = (1/d²) Σᵢ≠ⱼ Cov(zᵢ, zⱼ)²                 (per branch, independently)
```

Defaults: λ=25, μ=25, ν=1.

*InfoNCE on z_atomic.* Two differently-augmented profiles of the same CIF are positives; profiles from different CIFs in the batch are negatives:

```
L_InfoNCE = -(1/N) Σᵢ log[exp(sim(z_atomic_i_a, z_atomic_i_b) / τ) /
                          Σⱼ exp(sim(z_atomic_i_a, z_atomic_j_b) / τ)]
```

InfoNCE rather than Barlow Twins because z_atomic needs to be both invariant to augmentations (positive pairs) AND equivariant to structural differences (negative pairs). Barlow Twins only enforces invariance + decorrelation.

*Disentanglement.* Cross-covariance penalties between all three pairs:

```
L_disentangle = ||CrossCov(z_lattice, z_atomic)||²_F
              + ||CrossCov(z_lattice, z_exp)||²_F
              + ||CrossCov(z_atomic, z_exp)||²_F
```

*Matched-sample contrastive alignment.* For each labeled experimental pattern with a CIF in the batch, generate the corresponding simulated pattern on-the-fly. InfoNCE between sim-exp pairs (positives: same CIF, negatives: different CIFs in batch). Replaces the original Sinkhorn DA. z_exp is **not** aligned.

**Hyperparameters (starting points; all flexible).**

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| `α₁` (VICReg) | 1.0 | 0.5 – 2.0 | |
| `α₂` (InfoNCE on z_atomic) | 1.0 | 0.5 – 2.0 | |
| `α₃` (disentanglement) | 0.1 | 0.01 – 1.0 | Interacts with InfoNCE τ |
| `α_match` | 0.5 → 1.0 over 20 epochs | final 0.5 – 2.0 | Warmup matters |
| `λ_cls_rep_profile` | 0.1 | 0.0 – 0.3 | Profile-side firewall leak |
| `λ_cls_rep_peaks` | 0.3 | 0.0 – 1.0 | Peak-side firewall leak (CS only) |
| InfoNCE τ | 0.1 | 0.05 – 0.5 | Interacts with `α₃` |
| Batch sim:exp ratio | 70:30 | 60:40 – 80:20 | |
| Learning rate | 3e-5 | 1e-5 – 5e-5 | Lower than A1 — fine-tuning the A1 weights |

**Open hyperparameter interaction.** InfoNCE τ and the disentanglement weight α₃ compete for control of z_atomic's geometry: τ controls negative-pair sharpness, α₃ pushes cross-covariance to zero. These should be measured together rather than tuned independently. Sweep τ ∈ {0.05, 0.1, 0.2, 0.5} × α₃ ∈ {0.01, 0.1, 0.3, 1.0} on a small held-out set at the start of A2.

**Evaluation.**
- Primary: CS and SG accuracy on the 82 lab patterns. **Must improve over A1** for A2 to be considered successful.
- Cosine similarity matrices: z_lattice should cluster by crystal system; z_atomic should cluster by SG within CS; z_exp should cluster by augmentation type.
- Cross-covariance norms → 0 (disentanglement convergence).
- No impossible CS → SG predictions (CS-gating enforces this by construction; verify).
- Matched-sample alignment quality: cosine similarity between z_lattice_sim and z_lattice_exp on held-out matched pairs.
- t-SNE overlays of sim vs exp distributions in z_lattice and z_atomic space. Sim and exp should overlap in z_lattice and z_atomic; z_exp can and should remain distinct.

**Phase A2 validation gate.** A measurable lift on the 82 lab patterns over A1, plus matched-sample alignment quality reaching a meaningful level (cosine similarity > 0.5 on held-out matched pairs is a rough starting target). If A2 does not lift over A1, the firewall hyperparameters are the first thing to debug — likely `λ_cls_rep_profile` is too low (representation is undersupervised) or `α_match` is too high (over-aligning at the expense of supervised learning).

---

## 3. Track B — Generation

### 3.1 Phase B1 — Vanilla conditioning

**Goal.** Establish a vanilla generative baseline: lightweight frozen encoder, a flow generator trained **from scratch** to consume that encoder's conditioning. Quantify what naive conditioning achieves so Phase B2's improvement is measurable.

**Encoder.** Same no-pool 1D CNN architecture as A1, trained with a simpler objective: either (a) CS + SG classification on simulated patterns only (no multimodal, no firewall, no contrastive losses, no experimental alignment), or (b) MSE pretraining on simulated patterns (regression to peak-list or similar). Default: (a) — closer to the supervision a naive baseline would use, and an honest comparison with A2's contrastively-aligned encoder. Once trained it is **frozen**; only the generator trains. The encoder takes a log-d binned pattern + wavelength (FiLM as before); no peak-position view, no projection heads — a single representation vector conditions the generator.

**Generator.** A flow generator (FlowMM/DiffCSP-lineage equivariant GNN over lattice + fractional coords + atom types), **trained from scratch**. PXRDGen (Code Ocean, flow) is the primary scaffolding; XtalNet (Zenodo 13629658, diffusion) is the alternative. These are **code references, not loaded checkpoints** — no pretrained generator weights enter the pipeline. Rationale: a frozen pretrained generator would lock in its own training-set structure prior and its own encoder's conditioning space, neither compatible with our targets or our log-d encoder (`DESIGN_DECISIONS.md` §8). The per-run cost is ~1 day on a 24 GB card for this model class.

**Conditioning interface.** The frozen encoder's representation is fed to the generator's denoiser as an extra input at **every flow-matching step**. Because the generator is trained from scratch, its conditioning interface is built for our encoder's output distribution from step one — there is no adapter remapping our representation onto a foreign denoiser. Conditioning is learned implicitly through the generation loss; there is no separate conditioning objective.

**Training data.** Simulated patterns from `CrystalAI-simXRD` on the **≤20-atom structure subset** (`n_atoms_le_20`). For B1 development this is the redistributable MP-20 pool (`source='mp-20'`); the internal ICSD+MP-20 ≤20-atom union is an optional reported variant (`DESIGN_DECISIONS.md` §2, §8). The encoder being conditioned was trained on full ICSD — superset-train / subset-deploy, which is benign. No experimental data conditions the generator at this phase.

**Loss.** The generator's native flow-matching loss (denoising score-matching if the XtalNet scaffolding is used), training the full generator from scratch (the encoder is frozen via gradient masking, not the generator).

**Evaluation.**
- **Top-k match rate** on a held-out set of experimental patterns from the labeled pool: k=1, 5, 10, 20 via `pymatgen.analysis.structure_matcher.StructureMatcher`.
- **Atomic RMSD (Å)** between generated and ground-truth structures.
- **Rwp after Rietveld refinement** of the generated structure against the experimental pattern.
- **CS/SG correctness** of generated structures (via the source-read / derived labels on the matched CIF; `crystalai-data` symmetry utilities, not spglib).
- **Unconditional-prior probe** (gap-compression canary): sample the generator with conditioning ablated and measure match rate. A high unconditional match rate means the generator is leaning on a learned small-cell prior rather than the conditioning — both B1 and B2 would then rise on the prior and the B1→B2 lift would shrink (`DESIGN_DECISIONS.md` §8).
- **Case studies**: 5–10 generated CIFs side-by-side with experimental patterns, with overlay of re-simulated patterns.

**Phase B1 validation gate.** A defensible baseline well above random, and **clearly conditioning-driven** — the conditioned match rate must beat the unconditional-prior probe by a meaningful margin. (It will be below published PXRDGen/XtalNet numbers on their own clean-input benchmarks; expected, since we evaluate on real data.) If generated structures bear no relationship to the conditioning input (conditioned ≈ unconditional, or RMSD ~ random), the conditioning interface or the from-scratch training is broken and must be debugged before B2.

---

### 3.2 Phase B2 — Robust conditioning

**What's new vs B1.** The B1 encoder is replaced by the **Phase A2 encoder, frozen end-to-end**. A **fresh flow generator is trained from scratch** on the A2 encoder's conditioning distribution — same architecture, same recipe, same training data, same generation loss, same evaluation as B1. The only variable across B1→B2 is the encoder; each encoder gets its own best-fit generator. This is a cleaner comparison than remapping a thin adapter onto a fixed denoiser (the prior design), which was confounded by adapter capacity.

**Conditioning input.** A2's representation has more structure than B1's — three sub-representations (z_lattice, z_atomic, z_exp) with disentanglement guarantees. The generator conditions on `[z_lattice; z_atomic]` (z_exp excluded — experimental artifacts should not condition generation).

Conditioning strategies to evaluate:
1. **Direct concatenation**: `[z_lattice; z_atomic]` as a single 256-dim conditioning vector fed at every flow step.
2. **Hierarchical**: z_lattice conditions the lattice/early flow steps; `[z_lattice; z_atomic]` conditions atom-position/later flow steps.
3. **Confidence-weighted**: modulate conditioning by `cos_sim(z_lattice_profile, z_lattice_peaks)` — condition more confidently when the two views agree. (Requires combined-mode input with manually-verified peaks at inference; reserved for case studies.)

Default: strategy 1, with strategy 2 as a small ablation. Because the generator is trained from scratch on this conditioning, the strategy is a property of the generator's input head, not an adapter bolted onto a fixed denoiser.

**Training data.** Same ≤20-atom subset as B1. Matched-sample structure is available: for each simulated training pattern we have the structure, and for labeled experimental patterns with a CIF we can present matched sim-exp pairs as a curriculum (sim-only → mixed sim-exp partway through). Note the encoder is frozen, so this curriculum shapes the *generator's* use of the conditioning, not the representation.

**Evaluation.** Same as B1, on the same held-out set, including the unconditional-prior probe. **Must improve over B1** for B2 to be considered successful — the lift is the paper's headline result. The lift must survive the gap-compression check: B2's advantage over B1 has to come from better conditioning, not from both generators sharing a strong unconditional small-cell prior.

**Phase B2 validation gate.** Measurable lift over B1 on top-k match rate, RMSD, and Rwp post-Rietveld, with the conditioned-vs-unconditional gap intact. Ideally the lift is visually obvious in case studies (e.g., a pattern where B1 generates the wrong space group but B2 generates the correct one).

---

## 4. Cross-track and cross-phase concerns

### 4.1 Encoder transfer A2 → B2

The A2 encoder, including its peak-position encoder, projection heads, and trained weights, is taken as-is into B2. B2 does not retrain the encoder. The peak-position encoder is unused in B2 (B2 conditions on `[z_lattice_profile; z_atomic]`, which both come from the full-profile encoder). But the A2 training shaped z_lattice_profile via VICReg alignment with z_lattice_peaks — that's the indirect contribution of the peak view to Phase B2.

If at inference time a crystallographer supplies manually-verified peak positions alongside an experimental pattern, the combined-mode input `z_lattice = (z_lattice_profile + z_lattice_peaks) / 2` can be used. This is the only Track B inference path that touches the peak encoder.

### 4.2 Frozen encoder, from-scratch generator

In both B1 and B2 the **encoder is frozen and the generator is trained from scratch**; they are never co-trained. Co-training would let the generation loss reshape the encoder, collapsing the A2→B2 transfer that is the paper's thesis. So corrective gradients from generation never reach the encoder — the A2 firewall (`DESIGN_DECISIONS.md` §5) must therefore have protected the conditioning representation adequately during A2, because B2 cannot fix a categorically-contaminated encoder.

Because the generator is trained natively for the frozen encoder, an "adapter mismatch" is no longer a failure mode — there is no adapter. If B2 fails to beat B1, the suspect is the encoder representation itself. Fallback: unfreeze the A2 projection heads (the three MLPs producing z_lattice / z_atomic / z_exp), keeping the encoder trunk frozen, and retrain the generator. This lets the generative objective shape the conditioning representation slightly without undoing the firewall on the trunk. If even that doesn't help, the encoder side needs deeper reconsideration; this is the worst case where Track B becomes a future-work section (`DESIGN_DECISIONS.md` §8).

**Gap-compression watch.** A from-scratch generator on the low-entropy ≤20-atom subset can learn a strong unconditional prior. The unconditional-prior probe (§3.1 evaluation) is the canary; if the conditioned-vs-unconditional gap compresses, the B1→B2 comparison loses signal regardless of headline match rates.

### 4.3 Input mode dropout in A2

In A2 training, each sample is randomly assigned one of two input modes:
- **Profile-only**: classification heads receive `[detach(z_lattice_profile); detach(z_atomic); f_CS_profile]`. Standard production inference mode.
- **Combined**: classification heads receive `[detach((z_lattice_profile + z_lattice_peaks)/2); detach(z_atomic); f_CS_combined]`. The inference mode used when manually-verified peaks are available.

Peaks-only inference is **not** trained — the architecture commits to the full profile always being available at inference. See `DESIGN_DECISIONS.md` §5 / §7.

Default dropout ratio: 70% profile-only, 30% combined. The combined mode trains the heads to use the VICReg-aligned z_lattice_peaks productively rather than treating it as redundant.

---

## 5. Package structure

```
crystalai-methods/
├── pyproject.toml
├── README.md
├── ROADMAP.md
│
├── configs/
│   ├── a1_baseline_cnn.yaml
│   ├── a1_baseline_gru.yaml            # ablation
│   ├── a1_baseline_transformer.yaml    # ablation
│   ├── a2_multimodal.yaml
│   ├── b1_vanilla.yaml
│   └── b2_robust.yaml
│
├── src/
│   └── crystalai_methods/
│       ├── __init__.py
│       ├── encoders/
│       │   ├── log_d_cnn.py            # No-pool 1D CNN, primary
│       │   ├── log_d_gru.py            # Bidirectional GRU ablation
│       │   ├── log_d_transformer.py    # Lightweight Transformer ablation
│       │   ├── peak_cnn.py             # No-pool CNN with dilated convs for peak histogram
│       │   ├── projection_heads.py     # MLP_lattice, MLP_atomic, MLP_exp
│       │   └── film.py                 # FiLM modulation utility (wavelength conditioning)
│       │
│       ├── heads/
│       │   ├── cs_head.py              # Crystal system classifier
│       │   └── sg_gated_head.py        # CS-gated space group classifier with 7 sub-heads
│       │
│       ├── losses/
│       │   ├── vicreg.py
│       │   ├── infonce.py
│       │   ├── disentanglement.py
│       │   ├── matched_alignment.py    # InfoNCE between sim-exp matched pairs
│       │   ├── classification.py       # Weighted CE with firewall
│       │   └── firewall.py             # Detached-probe gradient routing utility
│       │
│       ├── data/
│       │   ├── simulated_dataset.py    # Wraps CrystalAI-simXRD on-the-fly simulation
│       │   ├── experimental_dataset.py # Loads from CrystalAI-data unified experimental index; native→log-d conversion + binning in __getitem__
│       │   ├── matched_pairs.py        # Pairs labeled experimental patterns with their CIFs (resolves cif_id then cif_path)
│       │   └── mixed_dataloader.py     # 70:30 sim:exp batch composition
│       │
│       ├── training/
│       │   ├── a1_baseline.py          # Phase A1 training loop
│       │   ├── a2_multimodal.py        # Phase A2 training loop
│       │   ├── b1_vanilla.py           # Phase B1 conditioning training
│       │   └── b2_robust.py            # Phase B2 conditioning training
│       │
│       ├── generators/
│       │   ├── flow_generator.py     # From-scratch flow generator (FlowMM/DiffCSP-lineage equivariant GNN)
│       │   ├── conditioning.py       # Encoder-embedding → per-step conditioning input (concat / hierarchical / confidence-weighted)
│       │   ├── flow_matching.py      # Flow-matching loss + sampler (score-matching variant if XtalNet scaffolding)
│       │   └── scaffolding/          # Vendored PXRDGen / XtalNet code references (CSPNet/FlowMM blocks); NO checkpoints
│       │
│       └── evaluation/
│           ├── classification_eval.py  # CS / SG accuracy, per-CS breakdown, per-source breakdown
│           ├── representation_viz.py   # t-SNE, cosine similarity matrices
│           ├── alignment_eval.py       # Matched-sample alignment quality
│           ├── generation_eval.py      # Top-k match, RMSD, Rwp, CS/SG check, unconditional-prior probe
│           └── case_studies.py         # Generate + overlay case study plots
│
├── tests/
│   ├── test_encoders.py
│   ├── test_losses.py
│   ├── test_firewall.py                # Verify gradient masking works as specified
│   ├── test_data.py
│   └── test_evaluation.py
│
└── scripts/
    ├── train_a1.py
    ├── train_a2.py
    ├── train_b1.py
    ├── train_b2.py
    └── evaluate.py
```

---

## 6. Dependencies

```toml
[project]
name = "crystalai-methods"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.0",
    "torch-geometric",                   # equivariant-GNN flow generator (Track B)
    "numpy>=1.24",
    "scipy>=1.10",
    "einops",
    "pymatgen>=2024.1",
    "scikit-learn",                      # t-SNE
    "tqdm",
    "pyyaml",
    "wandb",                             # experiment tracking
    "crystalai-data",
    "crystalai-simxrd",
]

# Track B generator — trained from scratch; upstream repos used as CODE SCAFFOLDING ONLY (no checkpoints)
# PXRDGen (flow, primary scaffolding):  https://codeocean.com/capsule/7727770/tree/v1
# XtalNet  (diffusion, alternative):    https://github.com/dptech-corp/XtalNet  (Zenodo 13629658)
# torch-geometric is required for the equivariant-GNN generator (CSPNet/FlowMM-lineage blocks)

[project.optional-dependencies]
dev = ["pytest", "ruff"]
```

---

## 7. Implementation order

The implementation order is strictly phase-by-phase, with validation gates between each:

### Sprint 1: Infrastructure + A1 baseline
1. Package skeleton, config system.
2. `log_d_cnn.py` — primary encoder.
3. `film.py` — FiLM modulation utility.
4. `cs_head.py` + `sg_gated_head.py`.
5. `classification.py` loss (weighted CE).
6. `simulated_dataset.py` + `experimental_dataset.py`.
7. `mixed_dataloader.py`.
8. `a1_baseline.py` training loop.
9. `classification_eval.py`.
10. Train A1, evaluate, **validation gate**: defensible CS/SG floor on 82 lab patterns.

### Sprint 2: A1 architecture ablations
11. `log_d_gru.py` + `log_d_transformer.py`.
12. Re-run A1 training for GRU and Transformer at reduced scale.
13. Compile ablation table.

### Sprint 3: A2 multimodal
14. `peak_cnn.py` — dilated-conv peak encoder.
15. `projection_heads.py` — MLP_lattice / MLP_atomic / MLP_exp.
16. `vicreg.py`, `infonce.py`, `disentanglement.py`.
17. `matched_alignment.py`.
18. `firewall.py` — detached-probe gradient routing.
19. `a2_multimodal.py` training loop with input-mode dropout.
20. `representation_viz.py`, `alignment_eval.py`.
21. Train A2 (initialize from A1 checkpoint), evaluate, **validation gate**: lift over A1 + meaningful matched-sample alignment.

### Sprint 4: B1 vanilla generative
22. `generators/scaffolding/` — vendor PXRDGen/XtalNet code blocks (CSPNet/FlowMM); no checkpoints.
23. `flow_generator.py` + `flow_matching.py` — from-scratch flow generator and loss/sampler; sanity-check unconditional training on the ≤20-atom subset reproduces a CrystalFlow-class baseline (~1 day/run).
24. `conditioning.py` — frozen-encoder embedding → per-step conditioning input.
25. `b1_vanilla.py` training loop (frozen lightweight encoder + from-scratch generator).
26. `generation_eval.py` (incl. unconditional-prior probe), `case_studies.py`.
27. Train B1, evaluate, **validation gate**: defensible baseline well above random AND conditioned match rate clearly beats the unconditional-prior probe.

### Sprint 5: B2 robust generative
28. `b2_robust.py` training loop (fresh from-scratch generator on the same recipe; swaps in the frozen A2 encoder).
29. Train B2 with A2 encoder frozen, evaluate, **validation gate**: lift over B1 with the conditioned-vs-unconditional gap intact.

### Sprint 6: Paper experiments
30. Final evaluation runs with the best hyperparameter settings.
31. Ablations for the paper (firewall leak weights, conditioning strategy, encoder choice, ICSD+MP-20 vs MP-20-only generator).
32. Case study generation.
33. Figure preparation.

The compute envelope (~6 major training runs before submission, see `ROADMAP.md`) maps to Sprints 1, 2 (combined into one ablation table), 3, 4, 5, plus one buffer run.

---

## 8. Outstanding decisions to revisit during implementation

These are not yet resolved and should be reconsidered when the empirical evidence is in hand:

- **A1 architecture ablation depth.** Reduced-scale or full-scale? Default: reduced. Revisit if A1 GRU or Transformer surprises on the real-data metric.
- **Generator scaffolding (B1/B2).** PXRDGen (flow) vs XtalNet (diffusion) as the from-scratch generator's code base. Default: PXRDGen flow scaffolding (cheaper, more natural). Both are references, not checkpoints. Revisit if the flow-matching loop is harder to stand up than the diffusion one.
- **Generator training set.** MP-20-only (released) vs ICSD+MP-20 ≤20-atom (internal/reported). Default: develop on MP-20; run the ICSD+MP-20 variant only if MP-20 alone underperforms (`DESIGN_DECISIONS.md` §2, §8).
- **Gap compression.** If the unconditional-prior probe shows the generator leaning on a small-cell prior, the B1→B2 lift is suspect; audit conditioning strength before trusting headline numbers.
- **Conditioning-failure fallback.** If B2 doesn't beat B1, unfreeze the A2 projection heads (not the trunk) and retrain the generator as the first fallback.
- **InfoNCE τ × disentanglement weight α₃ interaction.** Sweep at the start of A2.
- **Mask injection mechanism for GRU and Transformer ablations.** Concat-feature + output-masking (GRU); attention masking (Transformer). The fixed-window setup for GRU makes the bidirectional backward pass pad-aware via the mask but doesn't get the `pack_padded_sequence` benefit; this is one reason the fixed-window GRU is a weaker variant and is itself a possible reason an ablation favors CNN.
