# crystalai-data

Curate ICSD CIFs and the MP-20 dataset into a single SQLite database; ingest experimental patterns (RRUFF, opXRD-labeled, internal lab) per-source. No simulation, no training in this package.

Before non-trivial work here, read `DATA_ROADMAP.md` and the relevant section of `../DESIGN_DECISIONS.md` (§2 structure sources / licensing, §6 experimental data, §8 the Track B size bound the `n_atoms_le_20` flag serves).

## Package-specific rules

- Structures are pymatgen `Structure`, stored as gzip-compressed compact dicts in `crystals.sqlite`. Never ASE `Atoms`.
- The cell is stored **as parsed** — no Niggli or other reduction at ingest. `StructureMatcher` canonicalizes internally at compare-time, and the stored `a,b,c,α,β,γ` query columns must describe the same cell a user reasons about.
- Disorder: preserve fractional site occupancies exactly. Never majority-species-replace or supercell-expand disordered structures — the downstream Bragg calculator uses occupancy-weighted structure factors, the physically correct treatment. (Relevant to ICSD; MP-20 structures are ordered.)
- **No spglib on the crystals (structure) path.** Space groups are read from the source, never re-derived: ICSD from the CIF `_space_group_IT_number` (fall back to the H-M symbol via `hm_symbol_to_sg_number` only when the IT number is absent); MP-20 from the CSV field. Crystal system is derived from the space group by the range partition in `sg_to_cs`, making SG/CS disagreement impossible by construction. No symmetry re-analysis is performed on ICSD/MP-20.
- **Experimental SG carve-out (`xrddata` only).** opXRD patterns ship structures but *no* reported space group (P1-expanded for generative modeling), so `ingest_opxrd.py` derives the SG via pymatgen's `SpacegroupAnalyzer` (spglib) on an ordered geometric proxy of the structure. Policy: accept only when two symprec settings (0.01 & 0.1) agree; leave `space_group`/`crystal_system` null otherwise; record the accepted symprec in `notes`. This is the *only* place spglib is used, and it never runs on the crystals DB. RRUFF reads its SG from the DIF H-M symbol (no derivation). See `DATA_ROADMAP.md` §2 and `DESIGN_DECISIONS.md` §6.
- `space_group` is stored as the integer 1–230; `crystal_system` as the integer 1–7. Names/symbols come from the `symmetry.py` utilities (`sg_number_to_symbol`, `cs_number_to_name`), which are the single source of truth.
- Two sources share one schema and one table, distinguished by `source` (`'icsd'` | `'mp-20'`). `UNIQUE(source, source_id)` is the dedup key (the pair — numeric ids can collide across sources). `source_file_id` holds the ICSD `icsd_id` for on-disk CIF lookup (CIFs are named by `icsd_id`; `source_id` holds the ICSD collection code, bridged by the prior-paper mapping CSV); null for MP-20.
- A `crystals` row and its `crystal_elements` rows are written **in one transaction**. The junction table (`crystal_elements(crystal_id, element, amount)`, indexed on `element`) backs the relational composition queries — contains-all / exact-set / exclude — that a flat `chemsys` string can't do safely (substring matching is buggy: `O` matches `Os`, `C` matches `Cs`/`Co`). `chemsys` (sentinel-delimited, e.g. `-O-Ti-`) and `anonymized_formula` (e.g. `AB2`) stay as denormalized convenience columns.
- `n_atoms_le_20` is the indexed key for Track B's size-bounded subset (≤20 atoms; `DESIGN_DECISIONS.md` §8). Derivable from `n_atoms` for ICSD but stored so the common filter is one indexed column; always 1 for MP-20.
- `split` (`'train'`/`'val'`/`'test'`/null) is populated **after** ingest. MP-20 inherits its native split from the source CSVs at ingest — honoring it is a leakage-safety requirement, since any generative-model code/scaffolding was developed against that exact split. ICSD rows are null at ingest, assigned later by a crystal-system-stratified split.
- This is not a filtering/curation step. ICSD entries are experimentally verified and all kept; MP-20 is kept as published. Labels come from the sources, not from re-analysis.
- Experimental data is NOT centralized into one database. Each source keeps its own ingest path and per-source `index.csv` pointing at original files in place.
- `wavelength_A` is required in every experimental index row. Patterns with missing or suspicious wavelength metadata are flagged and excluded — the W1 wavelength-conditioning strategy depends on accurate wavelengths.
- The 91k uncurated opXRD pool is not used. ~5k labeled patterns only.

## Hard bans (do not reintroduce)

`ase`, `Pysimxrd`, `mp-api`. MP-20 ingests from plain CSVs of CIF strings, so it adds no dependency. `spglib` stays banned **as a label source on the crystals (ICSD/MP-20) path** — space groups there are read from the source, never re-derived. The sole permitted use is deriving space groups for **experimental opXRD patterns** (which carry none), via pymatgen's `SpacegroupAnalyzer` in `ingest_opxrd.py` under the symprec-consensus policy above; do not extend it to the crystals path.

## Commands (run from this directory)

- Install editable: `pip install -e .`
- Tests: `pytest`
- Lint: `ruff check .`

Steps 1–6 in `DATA_ROADMAP.md` §5 (skeleton → `database.py` → `symmetry.py` → `convert_icsd.py` → `convert_mp20.py` → `CrystalDatabase` API) must be fully tested before `simxrd` or `methods` consume the API.
