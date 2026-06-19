# CrystalAI — monorepo root

Multimodal ML for crystal-structure determination from real experimental PXRD. Target: ICLR, September 2026.

Three independently-installable, independently-tested packages. Each keeps its own `pyproject.toml`, `src/`, and `tests/`. Run commands from the package directory, not from this root.

- `crystalai-data/`    — curate ICSD CIFs; ingest RRUFF / opXRD-labeled / internal-lab patterns; visualization apps
- `crystalai-simxrd/`  — simulate PXRD from CIFs (log-d production, 2θ validation); on-the-fly augmentation pipeline
- `crystalai-methods/` — all ML training. Track A (classification: CS + SG), Track B (generation)

## Dependency flow — strictly downstream

`data → simxrd → methods`. `simxrd` and `methods` import `crystalai_data`; `methods` imports `crystalai_simxrd`. Never the reverse. Nothing in `data` imports `simxrd` or `methods`; nothing in `simxrd` imports `methods`. If a task seems to need an upstream import, the design is wrong — stop and flag it.

## Validate before stacking

Every phase has an explicit validation gate (see the per-package roadmaps). Do not scaffold, train, or build on a phase until its predecessor's gate is met and recorded. In particular: no A2 work before the A1 gate; no B1 before A2; no B2 before B1. If asked to skip or assume a gate, say so and stop rather than proceeding.

## Hard constraints (do not reintroduce)

- No `ase`, no `Pysimxrd`, no `mp-api`. pymatgen is the primary structure library.
- ICSD is the only crystal-structure source. No MP, CrystDB, COD, MP-20-PXRD, or SimXRD-4M.
- The simulation core is NumPy throughout. PyTorch appears only at the augmentor / DataLoader boundary (`crystalai-simxrd` §5.2), never in the simulation math.
- log(d) is the settled encoder/production coordinate; wavelength is an explicit FiLM input. The frozen-generator + frozen-encoder strategy is settled. Do not relitigate these settled commitments unless explicitly asked.

## Sources of truth

- `ROADMAP.md` — structure, narrative, the 8 core commitments.
- `DESIGN_DECISIONS.md` — rationale for those commitments, in order of foundational dependency. Consult the relevant section before changing any settled decision. Justifications must stay defensible against the actual compute spec: 2 nodes × 4 GPUs (8 total), ~22.5 GiB usable per card, ~180 GiB aggregate, ≤ 1 week per run.

## Migration

Parts of the code exist in older repos. Migrated code must be adapted to the constraints above (old code likely uses `ase` / `Pysimxrd`, 2θ-native simulation, or supercell disorder expansion — all now disallowed) and re-validated against the gates. Never trust a ported file because it worked before.
