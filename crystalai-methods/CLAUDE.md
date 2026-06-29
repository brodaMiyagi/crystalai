# crystalai-methods

All ML training. Track A (classification: crystal system + space group), Track B (generation). Imports `crystalai_data` and `crystalai_simxrd`; nothing imports this package.

Before non-trivial work here, read `METHODS_ROADMAP.md` and the relevant section of `../DESIGN_DECISIONS.md` (§3 architecture, §5 firewall, §8 from-scratch generator / frozen encoder).

## Package-specific rules

- The generator is a flow model (FlowMM/DiffCSP-lineage equivariant GNN) **trained from scratch** on the frozen encoder's conditioning distribution. PXRDGen / XtalNet are **code scaffolding only — never loaded checkpoints**. The conditioning encoder is frozen (the lightweight B1 encoder in B1; the A2 encoder in B2); encoder and generator are **never co-trained**. Conditioning is the encoder embedding fed to the denoiser at every flow-matching step — there is no adapter remapping onto a foreign denoiser. If B2 fails to beat B1, the only fallback is to unfreeze the A2 projection heads (never the encoder trunk) and retrain the generator.
- Generator training set: the **released** generator trains on **MP-20 (≤20 atoms) only** — ICSD's license bars redistributing a generator that emits full phases. An **internal** ICSD+MP-20 ≤20-atom variant (`n_atoms_le_20` filter) may be trained and *reported* but never shipped. The paper discloses the full ICSD+MP-20 training mix; the released weights are MP-20-tuned. The ICSD-trained encoder feeding the generator is disclosed and is not a redistribution problem (it encodes patterns, not structures).
- Watch gap compression: a from-scratch generator on the low-entropy ≤20-atom subset can learn a strong unconditional prior. The unconditional-prior probe is the canary — if the conditioned-vs-unconditional gap compresses, the B1→B2 comparison loses signal regardless of headline match rates.
- The frozen A2 encoder transferred into B2 is the single structural coupling between Tracks A and B. B2 does not retrain it.
- Classification firewall: CS/SG heads read a `detach`-ed copy of the trunk and train at full strength; only a metered gradient leak reaches the trunk (`λ_cls_rep_profile` ≈ 0.1, `λ_cls_rep_peaks` ≈ 0.3). The peak-side leak carries CS only — the SG head never attaches to the peak encoder.
- Encoders use position-preserving reduction (flatten + FC). No global average pooling: in log-d, position is lattice scale, which is exactly the within-SG signal the generator needs.
- Input is log(d)-binned pattern + wavelength via FiLM (CNN primary). Bidirectional GRU and lightweight Transformer are logged ablations judged on the real-data metric, not simulated retrieval.

## Phase gates — do not cross without evidence in hand

- A1 → A2: defensible CS/SG floor on the 82 lab patterns (at minimum match DIFCON's ~74.4% CS / ~41.5% SG).
- A2 → B2: measurable lift over A1, plus matched-sample alignment reaching a meaningful level (cos sim > ~0.5 on held-out matched pairs) before the encoder transfers to Track B.
- B1 → B2: defensible generation baseline (top-k match / RMSD / Rwp post-Rietveld) well above random AND clearly conditioning-driven — the conditioned match rate must beat the unconditional-prior probe by a meaningful margin (conditioned ≈ unconditional means the interface or the from-scratch training is broken).
- B2: measurable lift over B1 with the conditioned-vs-unconditional gap intact — this is the paper's headline result, and the lift must come from better conditioning, not a shared small-cell prior.

## Commands (run from this directory)

- Install editable: `pip install -e .`
- Tests: `pytest` (include `tests/test_firewall.py` — it verifies the gradient masking works as specified)
- Lint: `ruff check .`
- Train: `python scripts/train_a1.py` (a2 / b1 / b2 analogous); evaluate: `python scripts/evaluate.py`
