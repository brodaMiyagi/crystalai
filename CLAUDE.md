# CrystalAI — monorepo root

Multimodal ML for crystal-structure determination from real experimental PXRD. Target: ICLR, September 2026.

Three independently-installable, independently-tested packages. Each keeps its own `pyproject.toml`, `src/`, and `tests/`. Run commands from the package directory, not from this root.

- `crystalai-data/`    — curate ICSD CIFs and the MP-20 subset; ingest RRUFF / opXRD-labeled / internal-lab patterns; visualization apps
- `crystalai-simxrd/`  — simulate PXRD from CIFs (log-d production, 2θ validation); on-the-fly augmentation pipeline
- `crystalai-methods/` — all ML training. Track A (classification: CS + SG), Track B (generation)

## Dependency flow — strictly downstream

`data → simxrd → methods`. `simxrd` and `methods` import `crystalai_data`; `methods` imports `crystalai_simxrd`. Never the reverse. Nothing in `data` imports `simxrd` or `methods`; nothing in `simxrd` imports `methods`. If a task seems to need an upstream import, the design is wrong — stop and flag it.

## Validate before stacking

Every phase has an explicit validation gate (see the per-package roadmaps). Do not scaffold, train, or build on a phase until its predecessor's gate is met and recorded. In particular: no A2 work before the A1 gate; no B1 before A2; no B2 before B1. If asked to skip or assume a gate, say so and stop rather than proceeding.

## Hard constraints (do not reintroduce)

- No `ase`, no `Pysimxrd`, no `mp-api`. pymatgen is the primary structure library. For **crystal structures** (ICSD / MP-20) space groups are read from the source (CIF / MP-20 CSV) and crystal systems derived, never re-analyzed — **no `spglib` on the structure path**. The single carve-out: **experimental opXRD patterns ship no reported space group** (P1-expanded for generative modeling), so their SG is *derived* via pymatgen's `SpacegroupAnalyzer` (spglib) under a conservative symprec-consensus policy, confined to `crystalai-data`'s `xrddata` (experimental) path and left null when unresolved. This never touches the ICSD/MP-20 crystals DB. See `DESIGN_DECISIONS.md` §6.
- ICSD is the primary, experimentally-verified source and the **sole** source for Track A and the encoder. MP-20 (DFT-relaxed MP subset, ≤20 atoms/cell, redistributable) is a **Track-B-only generation source**, added solely for licensing — a publicly releasable generator cannot be ICSD-trained (ICSD bars redistributing a model that emits full phases). Still no general MP, no CrystDB, no COD, no MP-20-PXRD pre-computed-pattern benchmark, no SimXRD-4M. The DFT-to-real caveat is why MP-20 is confined to Track B targets and kept out of Track A inputs.
- The simulation core is NumPy throughout. PyTorch appears only at the augmentor / DataLoader boundary (`crystalai-simxrd` §5.2), never in the simulation math.
- log(d) is the settled encoder/production coordinate; wavelength is an explicit FiLM input. The **from-scratch generator + frozen encoder** strategy is settled: Track B trains a flow generator (FlowMM/DiffCSP-lineage equivariant GNN) from scratch on the frozen encoder; the two are never co-trained, and PXRDGen/XtalNet are code scaffolding, not loaded checkpoints. Do not relitigate these settled commitments unless explicitly asked.

## Sources of truth

- `ROADMAP.md` — structure, narrative, the 8 core commitments.
- `DESIGN_DECISIONS.md` — rationale for those commitments, in order of foundational dependency. Consult the relevant section before changing any settled decision. Justifications must stay defensible against the actual compute spec: 2 nodes × 4 GPUs (8 total), ~22.5 GiB usable per card, ~180 GiB aggregate, ≤ 1 week per run.

## Migration

Parts of the code exist in older repos. Migrated code must be adapted to the constraints above (old code likely uses `ase` / `Pysimxrd`, 2θ-native *output/binning*, supercell disorder expansion, spglib symmetry re-derivation, or a frozen pretrained generator + conditioning adapter — all now disallowed) and re-validated against the gates. Note: the simulator *does* build per-peak profiles in 2θ and resample to **log-d** (the disallowed pattern is 2θ *output/binning*, not 2θ intermediate construction; log-d remains the production/training coordinate — see `crystalai-simxrd/SIMXRD_ROADMAP.md` §1). Never trust a ported file because it worked before.
