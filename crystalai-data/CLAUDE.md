# crystalai-data

Curate ICSD CIFs into a single SQLite database; ingest experimental patterns (RRUFF, opXRD-labeled, internal lab) per-source. No simulation, no training in this package.

Before non-trivial work here, read `DATA_ROADMAP.md` and the relevant section of `../DESIGN_DECISIONS.md` (§2 structure source, §6 experimental data).

## Package-specific rules

- Structures are pymatgen `Structure`, stored as gzip-compressed compact dicts in `crystals.sqlite`. Never ASE `Atoms`.
- Disorder: preserve fractional site occupancies exactly. Never majority-species-replace or supercell-expand disordered structures — the downstream Bragg calculator uses occupancy-weighted structure factors, which is the physically correct treatment.
- spglib is for validation, not filtering. If spglib disagrees with ICSD's reported space group, log a warning and still insert the row (with the spglib SG). Keep every ICSD entry — this is not a curation step.
- Experimental data is NOT centralized into one database. Each source keeps its own ingest path and per-source `index.csv` pointing at original files in place.
- `wavelength_A` is required in every experimental index row. Patterns with missing or suspicious wavelength metadata are flagged and excluded — the W1 wavelength-conditioning strategy depends on accurate wavelengths.
- The `source` column stays for forward compatibility but is populated with `'icsd'` only this iteration.
- The 91k uncurated opXRD pool is not used. ~5k labeled patterns only.

## Commands (run from this directory)

- Install editable: `pip install -e .`
- Tests: `pytest`
- Lint: `ruff check .`

Steps 1–4 (skeleton → `database.py` → `convert_icsd.py` → `CrystalDatabase` API) must be fully tested before `simxrd` or `methods` consume the API.
