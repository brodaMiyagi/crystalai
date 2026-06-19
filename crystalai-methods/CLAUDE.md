# crystalai-methods

All ML training. Track A (classification: crystal system + space group), Track B (generation). Imports `crystalai_data` and `crystalai_simxrd`; nothing imports this package.

Before non-trivial work here, read `METHODS_ROADMAP.md` and the relevant section of `../DESIGN_DECISIONS.md` (§3 architecture, §5 firewall, §8 frozen backbone).

## Package-specific rules

- The generator backbone (XtalNet / PXRDGen) is loaded from the published checkpoint and frozen end-to-end. The conditioning encoder is also frozen (the B1 encoder in B1; the A2 encoder in B2). Only the adapter / conditioning layers train. If conditioning fails to converge, the only fallback is to unfreeze the projection heads — never the encoder trunk, never the generator backbone.
- The frozen A2 encoder transferred into B2 is the single structural coupling between Tracks A and B. B2 does not retrain it.
- Classification firewall: CS/SG heads read a `detach`-ed copy of the trunk and train at full strength; only a metered gradient leak reaches the trunk (`λ_cls_rep_profile` ≈ 0.1, `λ_cls_rep_peaks` ≈ 0.3). The peak-side leak carries CS only — the SG head never attaches to the peak encoder.
- Encoders use position-preserving reduction (flatten + FC). No global average pooling: in log-d, position is lattice scale, which is exactly the within-SG signal the generator needs.
- Input is log(d)-binned pattern + wavelength via FiLM (CNN primary). Bidirectional GRU and lightweight Transformer are logged ablations judged on the real-data metric, not simulated retrieval.

## Phase gates — do not cross without evidence in hand

- A1 → A2: defensible CS/SG floor on the 82 lab patterns (at minimum match DIFCON's ~74.4% CS / ~41.5% SG).
- A2 → B2: measurable lift over A1, plus matched-sample alignment reaching a meaningful level (cos sim > ~0.5 on held-out matched pairs) before the encoder transfers to Track B.
- B1 → B2: defensible generation baseline (top-k match / RMSD / Rwp post-Rietveld) well above random.
- B2: measurable lift over B1 — this is the paper's headline result.

## Commands (run from this directory)

- Install editable: `pip install -e .`
- Tests: `pytest` (include `tests/test_firewall.py` — it verifies the gradient masking works as specified)
- Lint: `ruff check .`
- Train: `python scripts/train_a1.py` (a2 / b1 / b2 analogous); evaluate: `python scripts/evaluate.py`
