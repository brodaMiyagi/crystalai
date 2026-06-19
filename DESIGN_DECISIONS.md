# CrystalAI — Design Decisions & Rationale

This document consolidates the rationale for the 8 core design commitments in `ROADMAP.md`. Each section states the decision, the alternatives considered, the reasoning chain, and the conditions under which the decision should be revisited.

The decisions are presented in order of *foundational dependency* — choices made earlier in the list constrain or motivate choices made later.

---

## 1. Encoder input coordinate: log(d-spacing)

**Decision.** All training and inference PXRD patterns are binned uniformly in log(d) over a fixed d-window (default 0.7 Å to 8 Å, ~4000 bins). The wavelength is provided as an explicit input alongside the binned pattern.

**Alternatives considered.** Uniform binning in 2θ; uniform binning in linear d.

**Reasoning.**

PXRD patterns come from instruments using different X-ray wavelengths (Cu Kα, Mo Kα, Co Kα, synchrotron sources). The opXRD pool spans multiple sources, and our internal lab patterns use either Cu or Mo Kα on STADI-P / STADI-MP. Committing to 2θ implicitly commits to a fixed wavelength — forcing us to either discard non-Cu data, train separate models per wavelength, or hope wavelength-invariance is learned from a single-wavelength distribution. None are good options.

D-spacing is the wavelength-independent reciprocal of the diffraction vector (`d = λ / (2·sin θ)`) and is the natural coordinate for wavelength pooling. But d-space alone is insufficient: the smooth multiplicative modifiers — Lorentz-polarization (LP), Debye-Waller (DW), absorption — are wavelength-dependent even when intensities are placed correctly on a d-axis, because they were computed at a wavelength-specific 2θ angle before being baked into the integrated intensity. The same physical crystal therefore produces *different intensity envelopes in d-space* at different wavelengths. Without explicit wavelength conditioning, the model can exploit wavelength-specific envelope shapes as a shortcut.

**Why log-d, not linear-d.** Three independent advantages.

1. **Lattice scaling becomes a translation.** Two isostructural compounds (e.g., CeO₂, UO₂, ThO₂ — all fluorite-type Fm-3m) have patterns that, in log-d, are shifted copies of each other. Multiplying all d-spacings by a constant becomes adding a constant to log(d). A translation-equivariant CNN exploits this by construction — convolutional filters generalize across lattice scales automatically. In 2θ or linear-d, the same symmetry appears as a stretching operation, which the CNN cannot exploit through its inductive bias and must learn from data. This advantage compounds with the CNN architecture choice (Section 3).
2. **Uniform peak widths.** Caglioti gives FWHM(2θ)² ≈ U·tan²θ + V·tanθ + W; converting to d via the Jacobian gives FWHM in linear d that varies as ~d². Converting to log-d, the d-dependence of FWHM_d and the d-stretching of log(d) cancel to first order, so peak widths are approximately constant across the pattern. This means uniform information density per bin — both CNN filters and Transformer attention operate on a more even substrate.
3. **Honest resolution allocation.** Constant Δ(log d) gives every peak the same number of bins per FWHM — no undersampling at small d, no oversampling at large d. This is the constant Δd/d property that TOF and synchrotron analysis routinely exploit.

**Cost.** Implementation overhead is small: `np.logspace` instead of `np.linspace` for the binning grid; one extra factor of d in the Jacobian when converting from 2θ; trivial changes to positional encoding if a Transformer is used.

**Conditions to revisit.** If wavelength metadata becomes unreliable enough that the explicit-wavelength conditioning breaks down (significantly mislabeled training data), or if the downstream generator turns out to need linear-d / 2θ input and the conversion at the encoder boundary becomes the bottleneck.

---

## 2. Crystal structure source: ICSD only

**Decision.** ICSD is the sole crystal-structure source for training and evaluation. MP-20-PXRD, CrystDB, the self-built MP-2024 subset, and COD are not used. The database schema retains a `source` column for forward compatibility, populated only with `'icsd'` in this iteration.

**Alternatives considered.** ICSD + CrystDB merge; ICSD + self-built MP-2024 subset; ICSD + COD as backup; MP-20-PXRD as an evaluation benchmark.

**Reasoning.**

Two independent arguments converge on ICSD-only.

**(a) DFT-predicted structures introduce a hidden gap.** CrystDB and the self-built MP-2024 subset both rest on DFT-relaxed structures from Materials Project. These pass automated stability and symmetry checks but are not experimentally verified. A pipeline whose central premise is bridging the sim-to-real gap should not be trained on structures that may themselves be wrong — that adds a "DFT-to-real" gap on top of the simulation-to-real gap, and the resulting failure modes are entangled and harder to debug. ICSD's experimental provenance is the gold standard. ~290k experimentally-verified structures is a sufficient training pool. COD's lower curation standards make it unattractive as a primary source.

**(b) Clean-input benchmarks reward the wrong thing.** MP-20-PXRD's pre-computed patterns are stick patterns convolved to a grid without realistic broadening, backgrounds, or noise. Placing a number next to PXRDGen / PXRDnet on MP-20-PXRD would advertise exactly the wrong evaluation regime — one in which clean-input optimization is rewarded over sim-to-real transfer.

**Cost.**

- No head-to-head numbers against PXRDGen / PXRDnet on their published benchmarks. The honest framing for the paper: published baselines use unphysical conditioning signals that do not reflect the real-data use case.
- No DFT-predicted hypothetical phases in training. The pipeline cannot learn from structures that have been predicted but never synthesized. Acceptable given ICSD's size.
- SimXRD-4M comparison dropped (`simxrd_compat.py`, `Pysimxrd` optional dependency, `ase` core dependency all removed). Validation against `pymatgen.analysis.diffraction.xrd.XRDCalculator` covers the simulation-correctness check against a trusted reference.

**Conditions to revisit.** If a curated experimental database with ICSD-comparable provenance becomes available; or if a manually-validated subset of MP/CrystDB structures (whose DFT geometries have been confirmed against experiment) can be assembled. The `source` column in the schema makes adding such a source a small change.

---

## 3. Primary encoder architecture: no-pool 1D CNN in log-d

**Decision.** A homogeneous no-pool 1D CNN operating in log-d serves as the primary architecture for both the full-profile and peak-position encoders. Bidirectional GRU and a lightweight Transformer are evaluated as logged ablations on the real-data metric (CS/SG accuracy on the 82 lab patterns + matched-sample alignment quality on held-out real pairs), not on simulated retrieval.

"No-pool" means strided convolutions for downsampling, no pooling layers, then flatten + FC to the projection heads. Position is preserved into the reduction (flatten or positional-encoded attention pooling), *not* global average pooling.

**Alternatives considered.** Patch Transformer (the original master-roadmap choice, motivated by PXRDGen's contrastive retrieval results); bidirectional GRU; CNN with global average pooling; CNN/GRU mixed encoders (CNN profile + GRU peaks or vice versa).

**Reasoning.**

Three independent legs support CNN-primary.

**(a) Coordinate fit.** In log-d, lattice scaling is a translation (Section 1). A CNN's translation-equivariance exploits this directly through its inductive bias; a Transformer cannot. This is the argument most specific to our pipeline and is unique among the alternatives — neither GRU nor Transformer gains anything from log-d that a CNN doesn't gain more of.

**(b) Empirical sim-to-real evidence.** SimXRD's published results show no-pool 1D CNN and bidirectional GRU beating Transformer variants on RRUFF real-data test accuracy. PXRDGen's Transformer-favoring result (92.42% vs 33.57% top-10 retrieval) measures in-distribution contrastive retrieval, which is silent on the sim-to-real gap. We are graded on transfer to the 82 lab patterns and on real generation, not on retrieval — PXRDGen measured the wrong thing for our use case.

**(c) Firewall tractability.** The classification firewall (Section 5) requires a clean separation between a detached classification probe and the representation feeding the generator. The no-pool CNN admits this cheaply — the trunk produces a position-preserving feature map, the probe attaches to a detached copy of it, the conditioning path takes its own reduction. A Transformer with a CLS token bakes categorical pressure into the attention pattern itself and would require either early branching or near-decoupling, both more expensive.

**Why no global average pooling.** GAP is translation-invariant — it discards position. In log-d, position *is* lattice scale, which is the within-SG discriminator the generator needs. GAP on the conditioning path would erase the very information generation depends on. (GAP is also banned from the peak-position encoder for the same reason — the peak histogram's *entire* information content is positional.) Both encoders use position-preserving reduction.

**Homogeneity (CNN-both, not a CNN/GRU split).** The VICReg invariance term aligns z_lattice_profile and z_lattice_peaks dimension-wise; that alignment is cheaper between two encoders sharing a coordinate basis than between two architectures with different positional inductive biases. The GRU's appeal — relational modeling of peak spacings and d-ratios — is recovered by **dilated convolutions** in the peak CNN, keeping homogeneity. A CNN/GRU mix forces alignment across the largest architectural gap on the one loss that ties the multi-view design together; that's an unnecessary tax.

**Provisional sizing.** Seeded from SimXRD's `NoPoolCNN`: ~6 conv blocks with strides giving total downsampling of ~16×, channel progression 32 → 64 → 128 → 256, flatten + FC to a 256-dim representation, then per-projection-head MLPs to z_lattice (128), z_atomic (128), z_exp (64). The load-bearing relationship: total stride × bin count sets the flatten width. When the bin count or window changes, strides may need re-tuning. Total parameter count ~5–10M for the trunk.

**Conditions to revisit.** If the GRU ablation matches or beats CNN on the real-data metric at comparable parameter count, the homogeneity argument collapses (and a CNN/GRU split or GRU-both becomes the right call). If the Transformer ablation closes the gap, the firewall tractability argument becomes the deciding factor — likely still favors CNN, but worth revisiting.

---

## 4. Wavelength injection: FiLM (CNN) / prepended token (Transformer) / initial hidden state (GRU)

**Decision.** Wavelength enters the CNN via FiLM (feature-wise linear modulation): a small MLP maps the wavelength scalar to per-channel γ, β vectors that modulate intermediate feature maps. The Transformer ablation prepends a wavelength token (small MLP-embedded) alongside the CLS token. The GRU ablation uses the wavelength either as the initial hidden state (after MLP projection to the right dimension) or concatenated with the final hidden state before the projection heads.

**Alternatives considered.** Wavelength as an extra constant input channel (CNN); wavelength concatenated to projection-head input only (post-encoder); ignoring wavelength entirely with aggressive augmentation as a substitute.

**Reasoning.**

Wavelength is a *global* scalar — it modulates how intensity envelopes (LP, DW, absorption) should be interpreted across the whole pattern, not at any particular position. FiLM is the purpose-built mechanism for global conditioning: a per-channel scalar modulation is much cheaper than treating the wavelength as a per-position feature, and it provides the model with a structured way to apply wavelength-dependent transformations at every layer.

Encoding wavelength as a repeated constant input channel is rejected as wasteful (every position carries the same value, so the convolution is doing redundant work) and indirect (the wavelength affects intermediate feature interpretation, not the raw input).

Encoding wavelength only at the projection-head boundary is rejected as too late — by then the encoder has already produced a representation, and the model has no chance to apply wavelength-conditional transformations inside the encoder. The LP/DW correction is needed throughout the feature hierarchy, not only at the readout.

Ignoring wavelength and relying on augmentation alone (the maximally aggressive variant — also randomize the intensity envelope) is rejected because the resulting z_atomic loses real intensity information that the generator needs to interpret structure-factor content. With wavelength as an explicit input, the model can apply a learned inverse transformation to remove wavelength-specific envelope effects, leaving structure-factor-flavored features behind.

**Augmentation that pairs with this choice.** Wavelength must still be randomized at training time (uniform sample from ~[0.5, 1.8] Å covering Cu Kα, Mo Kα, Co Kα, Cr Kα, Ag Kα, and synchrotron values). Without this, the wavelength input becomes meaningless — the model cannot learn to use it. Caglioti parameters and mild profile asymmetry are also randomized; matched-d-range simulation ensures all training patterns cover the same log-d window after binning.

**Canary.** CS accuracy should be independent of wavelength. Dependence indicates that CS is being read off envelope shape — the wavelength-shortcut failure mode the conditioning is designed to prevent.

**Conditions to revisit.** If GRU or Transformer ablations show that their respective injection mechanisms underperform FiLM-equivalent setups on CNN, the comparison is unfair and the injection mechanism becomes the confound. (This is the main reason to keep the ablations *logged* but secondary.)

---

## 5. Classification firewall: detached probe with metered gradient leak

**Decision.** The classification heads (CS, CS-gated SG) read from a `detach`-ed copy of the encoder trunk and train at full strength against their cross-entropy losses. A small, deliberate gradient leak (scalar `λ_cls_rep_profile` ≈ 0.1, and `λ_cls_rep_peaks` ≈ 0.3) permits a metered fraction of the classification gradient to flow back into the trunk. The conditioning path (z_lattice, z_atomic, z_exp) feeds the generator and is shaped only by contrastive / generative losses plus the small classification leak.

**Alternatives considered.** Full-strength classification gradient into the shared trunk (the original design); fully decoupled encoders (separate classification CNN sharing nothing with the conditioning encoder); zero-leak detached probe (probe-only classification, no representational influence at all).

**Reasoning.**

The CS / SG heads serve two roles: as a reported deliverable (a working symmetry classifier on lab patterns) and as auxiliary supervision shaping the representation. Categorical cross-entropy supervision imposes categorical geometry on the representation — sharp class boundaries, ambiguous structures snapped to the nearest class, flattened within-class fine structure. When the generator conditions on that representation, it inherits the distortion *regardless of whether the classifier's predictions are correct*. A 100%-accurate classifier still imposes categorical geometry; the deliverable and the contamination are produced by the same training pressure.

The two surface symptoms with the same root: **collapse** (within-class fine structure lost — fluorite-trio CeO₂/UO₂/ThO₂ mapped to one point) and **misplacement** (borderline / slightly-distorted structures yanked across a class boundary). For a generator, misplacement is arguably worse — borderline structures are where interesting science often lives.

**Why a metered leak rather than zero or full.** Zero leak (pure probe) means classification provides no representational supervision at all, and experimental classification's role of grounding the representation in real data shifts entirely to matched-sample contrastive alignment (Section 6). That may be sufficient, but losing classification's signal entirely is a real cost. A small leak preserves classification's beneficial regularization (e.g., for FiLM wavelength conditioning — see "Canary" in Section 4) while preventing the categorical geometry from dominating.

**Asymmetry: profile vs peaks.** The profile encoder feeds z_atomic into the generator and carries lattice-scale structure; protect it heavily (small leak, default 0.1). The peak encoder consumes position-only input — pure symmetry information — and classification asks it to be symmetry-discriminative, which aligns with rather than fights its purpose; it reaches the generator only through z_lattice via VICReg-aligned combined mode; tolerate a larger leak (default 0.3). The peak-side leak is bounded not by the peak encoder's own tolerance but by how much categorical geometry VICReg is permitted to transmit *into* the profile encoder. Canary: if the profile encoder's within-SG / lattice-scale structure degrades despite its own firewall, `λ_cls_rep_peaks` is too high.

**Peak-side leak carries CS only, not SG.** SG is partly intensity-dependent; the position-only peak input cannot fully support it, so SG cross-entropy on the peak encoder would ask it to predict something its input can't carry — hallucination pressure with no upside. The SG head does not attach to the peak encoder.

**Hyperparameter starting points.**

| Parameter | Default | Range to explore | Trade-off |
|-----------|---------|------------------|-----------|
| `λ_cls_rep_profile` | 0.1 | 0.0 – 0.3 | Higher → more classification shaping (better-supervised representation, risk of categorical contamination); lower → cleaner conditioning, experimental supervision shifts onto matched-sample alignment |
| `λ_cls_rep_peaks` | 0.3 | 0.0 – 1.0 | Higher → stronger CS shaping of z_lattice_peaks; capped by VICReg back-door contamination of z_lattice_profile |

**Fallback.** If the detached probe (reading from the detached trunk) cannot reach acceptable CS/SG accuracy because it can't shape the lower features, escalate to decoupled encoders: a separate small classification CNN sharing nothing with the conditioning encoder. Cost is small (CNNs are 1–6M params, dwarfed by the generator); the lost beneficial transfer is small, because the symmetry structure the conditioning encoder needs comes from the contrastive losses, not from CS/SG cross-entropy.

**Conditions to revisit.** If the leak weights resist tuning (no setting both protects the conditioning representation and gives a useful classifier), the decoupled-encoder fallback applies. If Track B's generative results show clear evidence of categorical contamination (e.g., generated structures collapse to a few prototypes per SG), reduce both leak weights and consider decoupling.

---

## 6. Experimental data: ~5k labeled patterns; matched-sample contrastive alignment

**Decision.** Experimental training data consists of ~5k labeled patterns: RRUFF (~3k) + opXRD-labeled (~1k) + internal lab (<1k). The 91k uncurated opXRD pool is dropped. Stage 0 MAE pretraining is removed. Stage 3 is restructured from unsupervised Sinkhorn domain adaptation to **supervised experimental fine-tuning with matched-sample contrastive alignment**: for each labeled experimental pattern with a CIF, the corresponding simulated pattern is generated on-the-fly, and an InfoNCE objective is applied between z_lattice and z_atomic of the matched sim-exp pair (positives) versus mismatched pairs in the batch (negatives).

**Alternatives considered.** Sinkhorn divergence on the 91k uncurated pool (the original master-roadmap design); MAE pretraining on the mixed sim+exp pool; manual curation of a smaller subset from the 91k pool.

**Reasoning.**

The Sinkhorn DA design aligned *distributions* between simulated and ~91k unlabeled experimental patterns. Sinkhorn aligns distributions, not curated samples — every pattern in the experimental pool exerts pull on the simulated representation. Manual screening of opXRD revealed many patterns too noisy to be useful, some with clean-looking peaks that can't be indexed, and a fraction with strong backgrounds. With no automated way to filter the 91k pool reliably, the safe move is to use only what can be trusted.

Of the ~5k labeled patterns, ~90% have full CIFs available, enabling matched-sample alignment as a direct replacement for Sinkhorn DA. The matched-sample objective is much stronger per-sample: positive pairs are *the same crystal* simulated vs measured, not just samples drawn from the same distribution.

**Failure modes the original design was vulnerable to.**

- Amorphous / glassy patterns (broad humps, no sharp peaks) pulling the model toward treating broad backgrounds as legitimate symmetry signal.
- Multi-phase patterns violating the "one pattern = one structure" assumption that VICReg and InfoNCE depend on.
- Mislabeled-wavelength patterns silently distorting the simulated-to-experimental mapping.
- The Sinkhorn loss decreasing monotonically and being misinterpreted as success, because alignment to a junk-contaminated distribution is still alignment.

All four are eliminated by switching to labeled-only with matched-sample alignment.

**Loss structure.**

- **Continued from earlier phases (simulated data only):** L_VICReg (full-profile ↔ peak-position alignment), L_InfoNCE on z_atomic (two augmented views of simulated patterns), L_disentangle (cross-covariance penalties between z_lattice, z_atomic, z_exp).
- **CS classification — both sim and exp, with separate weighting:**
  - `L_CS_sim` with sim-pool inverse-frequency class weights.
  - `L_CS_exp` with exp-pool inverse-frequency class weights (separate scheme, derived from the labeled experimental pool).
  - Combined: `α₄ · (L_CS_sim + λ_exp · L_CS_exp)`.
- **SG classification — asymmetric weighting:**
  - `L_SG_sim` with sim-pool inverse-frequency weights within each CS-gated sub-head.
  - `L_SG_exp` with no class weighting (standard cross-entropy within each sub-head) — experimental SG counts are too sparse for inverse-frequency weighting to be stable.
  - Combined: `α₅ · (L_SG_sim + λ_exp_sg · L_SG_exp)`.
- **Matched-sample contrastive alignment** (replaces Sinkhorn DA):
  - For each labeled experimental pattern with a CIF in the batch, the corresponding simulated pattern is generated.
  - `L_matched_lattice`: InfoNCE between z_lattice_sim and z_lattice_exp for matched pairs; positives are sim-exp pairs from the same CIF, negatives are sim-exp pairs from different CIFs in the batch.
  - `L_matched_atomic`: analogous for z_atomic.
  - z_exp is **not** aligned — experimental artifacts should differ between domains.
  - Combined: `α_match · (L_matched_lattice + L_matched_atomic)`.

**Partial labels are usable.**

- CS-only labeled (no SG, no CIF): contributes to `L_CS_exp` only.
- CS + SG labeled (no CIF): contributes to `L_CS_exp` and `L_SG_exp`; not to matched-sample alignment.
- Fully labeled with CIF (~90% of the pool): contributes to all three.

**Hyperparameter starting points.**

| Parameter | Default | Range to explore | Trade-off |
|-----------|---------|------------------|-----------|
| `λ_exp` (CS sim-vs-exp weight) | 1.0 | 0.5 – 2.0 | Higher → experimental CS dominates, may overfit to ~5k pool; lower → simulated CS dominates, may not close sim-to-real gap |
| `λ_exp_sg` (SG sim-vs-exp weight) | 0.5 | 0.25 – 1.0 | Higher → trust sparse experimental SG more, risk memorization of rare SGs; lower → rely on sim supervision + matched-sample alignment |
| `α_match` | 0.5 → 1.0 over 20 epochs | final 0.5 – 2.0 | Higher → stronger sim-exp alignment, risks over-aligning at the expense of supervised learning; lower → weaker alignment, sim-to-real gap may persist |
| Batch sim:exp ratio | 70:30 | 60:40 – 80:20 | More exp → more sim-to-real signal but more noise from limited pool; more sim → broader coverage but weaker exp anchoring |
| Learning rate | 3e-5 | 1e-5 – 5e-5 | Standard fine-tuning rate |

The most impactful knob is `λ_exp_sg` — if experimental SG accuracy is poor, escalate toward 1.0; if training is unstable, reduce. Matched-sample alignment quality (cosine similarity between z_lattice_sim and z_lattice_exp on held-out matched pairs) is the canary for `α_match` tuning.

**Composition with the classification firewall (Section 5).** Under the firewall, `L_CS_exp` and `L_SG_exp` reach the representation only through the small `λ_cls_rep` leak. Experimental classification is largely a probe readout; the experimental-supervision-into-representation role shifts onto matched-sample alignment. The two sets of weights compose: this section's `λ_exp` / `λ_exp_sg` shape the *internal* sim/exp balance of the classification gradient; Section 5's `λ_cls_rep` meters *how much* of that gradient reaches the trunk.

**Future re-introduction of unlabeled data.** Deferred. Possible paths if the supervised pipeline succeeds: manual curation of a few hundred to few thousand additional opXRD patterns; a learned quality classifier trained on manually-screened data, used to filter the larger pool. Both require the supervised baseline to be working first.

**Conditions to revisit.** If matched-sample alignment plateaus far short of the desired sim-to-real transfer; if experimental class coverage proves too sparse for the supervised approach to converge. In either case, a curated unlabeled pool becomes worth the effort.

---

## 7. Peak-position augmentation: importance-aware

**Decision.** Peak-position augmentations respect the physical role of each peak in lattice and space-group determination. The previous "drop with probability proportional to weakness" rule is replaced with a three-component drop model (correlated-failure + high-information protection + baseline random) plus a hard-reject manufactured-absence guard, and the position-jitter magnitude is angle-aware (gentler at high 2θ where small δ does more damage).

**Alternatives considered.** Uniform random drops; weakness-proportional drops (the original design); full Fisher-information weighting of every peak's marginal contribution.

**Reasoning.**

The roles for peak-position augmentation in the new design are (a) feeding the VICReg invariance term so z_lattice_peaks meaningfully aligns with z_lattice_profile, and (b) producing combined-mode robustness when a crystallographer supplies manually-verified peaks alongside a full profile. Both roles are degraded — not served — by indiscriminate weakness-proportional dropping.

**Physics framing.**

- **Peak positions at high 2θ / low d constrain lattice parameters most tightly.** The fractional error in extracted lattice constants from a peak scales as cot(θ) · δ(2θ), so a high-angle peak pins the cell far more precisely than a low-angle one — the basis of Nelson-Riley extrapolation and the standard Rietveld preference for high-angle reflections in lattice refinement. High-order peaks are also often weak (form-factor falloff, Debye-Waller damping), so weakness-proportional dropping preferentially deletes the peaks carrying the most lattice information.
- **Space-group determination depends on systematic absences.** The SG signal lives in *which* reflections are present vs absent (extinction conditions for screw axes and glide planes), not their precise positions. A *missed weak peak that is actually present* can be misread as a systematic absence and silently corrupt SG identification — the worst failure mode for SG-relevant signal, and the one uniform weakness-proportional dropping makes most likely.

**Augmentation rules.**

1. **Position jitter is angle-aware.** Per-peak jitter magnitude scales with local peak FWHM in log-d (approximately uniform under W1), with a floor preventing collapse at low d.
2. **Drop probability is a three-component model.**
   - *Correlated-failure*: when one peak is dropped, neighbouring peaks within a configurable log-d window receive elevated drop probability.
   - *High-information protection*: peaks above a 2θ threshold (equivalently, below a d threshold) have drop probability multiplicatively capped.
   - *Baseline random*: small uniform component for true random misses.
3. **Manufactured-absence guard (hard reject).** After every augmentation draw, the resulting peak list is checked against the source CIF's extinction conditions. If any reflection removed is one whose presence is *diagnostic* for the true space group (i.e. its observation rules out a higher-symmetry alternative), the draw is rejected and resampled.
4. **Spurious peaks unchanged.** Existing rule (0–3 random spurious peaks per pattern) retained.

**Hyperparameter starting points.**

| Parameter | Default | Range | Trade-off |
|-----------|---------|-------|-----------|
| High-information 2θ threshold | ~60° at Cu Kα equivalent (d ≲ 1.5 Å) | 50°–80° | Lower → larger protected set, weaker VICReg signal; higher → smaller protected set, more aggressive augmentation possible |
| Drop-cap multiplier in protected region | 0.2× | 0.0×–0.5× | 0× = never drop high-order peaks (strongest protection, unrealistic) |
| Correlated drop cluster radius (in log-d) | 0.05 (≈ one FWHM) | 0.02–0.10 | Larger → more strongly correlated drops (more realistic but more aggressive overall) |
| Baseline random drop rate | 0.05 | 0.02–0.10 | Standard |

**Caveat.** Weighting by 2θ/order is a heuristic for the principled quantity (each peak's marginal Fisher information for the cell parameters and extinction conditions). The principled version is computable from the CIF but adds machinery that may not be worth the cost at the augmentation stage. The heuristic above is the production rule; Fisher-information weighting is the rigorous fallback if the heuristic underperforms — diagnosed by VICReg alignment quality on held-out CIFs and downstream CS / lattice-extraction accuracy.

**Conditions to revisit.** If VICReg alignment between z_lattice_profile and z_lattice_peaks plateaus, or if downstream CS accuracy via the peak encoder degrades, escalate to Fisher-information weighting.

---

## 8. Generator backbone: frozen

**Decision.** XtalNet's diffusion backbone and / or PXRDGen's flow backbone is loaded from the published checkpoint and held frozen throughout Track B. Only the conditioning layers (the encoder → generator interface) are trained. The encoder feeding conditioning is also frozen (it's the Phase A2 encoder in Phase B2; the Phase B1 encoder in Phase B1). If conditioning fails to converge with both encoder and generator frozen, the fallback is to unfreeze the projection heads (not the encoder backbone, not the generator backbone).

**Alternatives considered.** Full fine-tuning of the generator backbone; partial unfreezing (last few layers of the generator + conditioning); from-scratch retraining of a flow/diffusion backbone.

**Reasoning.**

**Compute envelope is the load-bearing constraint.** The hardware — 2 nodes × 4 GPUs (8 × 24 GB, ~22.5 GiB usable per card, ~180 GiB aggregate) — could in principle support distributed from-scratch training of a flow/diffusion backbone, but not within the ≤1-week-per-run budget the ~6-run experiment plan depends on. From-scratch training, or partial unfreezing of large generator segments, pushes individual experiments well past that budget; the modest per-card capacity (~22.5 GiB) tightens this further by limiting how much of the generator fits alongside activations. Conditioning-layer training fits cleanly within the envelope.

**Methodological coherence.** The paper's contribution is *representation quality*, not generator architecture. A frozen generator backbone keeps the comparison clean: Phase B1's encoder vs Phase B2's encoder is the only variable changing. Any backbone unfreezing would entangle "the generator adapted to the encoder" with "the encoder is better," and the contribution claim would weaken correspondingly.

**Inductive-bias risk.** The encoders feeding conditioning may not match the inductive biases the published generators were trained with. Both XtalNet and PXRDGen used their own pretrained encoders (CNN or Transformer flavors); replacing those with our encoder + a thin conditioning adapter is a domain change for the generator. The frozen-generator approach assumes this risk is small enough that adapter training can absorb it. If it isn't, B1 will plateau at a poor baseline — diagnosable, and the fallback (unfreeze projection heads or move to fine-tuning) becomes appropriate.

**Inference-time encoder is also frozen.** This is implicit but worth stating. The Phase A2 encoder, having been trained with the W3 firewall, is taken as-is into Phase B2. No corrective gradients from the generative objective flow into the encoder. The classification firewall during A2 must therefore have protected the conditioning representation adequately — Phase B2 cannot fix a categorically-contaminated encoder.

**Conditions to revisit.**

- If Phase B1 fails to reach a defensible baseline with both encoder and generator frozen — diagnosed by top-k match rate plateauing far below the levels published for fully-trained PXRDGen / XtalNet on their benchmarks — unfreeze the projection heads first. If that doesn't help, the inductive-bias mismatch is severe and the encoder side needs reconsideration.
- If RWTH HPC or cloud allocation becomes available with substantially more compute, partial fine-tuning of the generator's late layers becomes feasible and may be worth running as a final experiment for the paper.
- If neither B1 nor B2 reaches a useful generation baseline, the paper's contribution falls back to Track A alone (classification + representation learning); Track B becomes a "future work" section. This is the worst-case scenario but is consistent with the modular-rebuild philosophy: each phase is validated before the next is committed to.

---

## Decisions explicitly **not** carried into the new project

For clarity, the following were considered, in some cases extensively, and are out of scope for the ICLR submission:

- **Stage 0 MAE pretraining.** Dropped with the 91k uncurated opXRD pool (Section 6). Without a large unlabeled corpus to exploit, MAE on simulated-only data is just an inefficient version of Phase A1.
- **Peaks-only inference mode.** Dropped. Peak positions enter only via the multimodal combined mode (full profile + manually-verified peaks) or as a training-time view for VICReg alignment. There is no inference path that consumes peaks alone.
- **MP-20-PXRD, CrystDB, COD, self-built MP-2024 subset, SimXRD-4M comparison.** All dropped (Section 2).
- **Sinkhorn divergence domain adaptation.** Replaced by matched-sample contrastive alignment (Section 6).
- **`Pysimxrd` and `ase` dependencies.** Dropped.
