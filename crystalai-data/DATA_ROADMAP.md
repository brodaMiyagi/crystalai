# CrystalAI-data — Roadmap

**Scope:** Curate and structure all input data for the CrystalAI pipeline. ICSD CIFs and the MP-20 dataset on the structural side; RRUFF, opXRD, and internal lab patterns on the experimental side. Provide visualization apps for both.

**On the two structure sources.** ICSD (experimentally-verified) is the primary source and the sole source for Track A. MP-20 (DFT-relaxed Materials Project subset, ≤20 atoms/cell, redistributable) is added to support Track B's public generator release — ICSD's licensing terms prohibit redistributing a generative model that emits full crystallographic phases. Both sources share one schema and one database; a `source` column distinguishes them so any per-source filter (including "ICSD-only") is a one-line query. See `DESIGN_DECISIONS.md` §2 for the licensing rationale and the Track-A/Track-B split.

**Out of scope:** Simulation (lives in `CrystalAI-simXRD`). ML training (lives in `CrystalAI-methods`).

---

## 1. Crystal structures (ICSD + MP-20)

### Storage

A single SQLite database (`crystals.sqlite`) holds every structure as a gzip-compressed compact pymatgen dict, alongside indexed metadata columns. Both ICSD and MP-20 structures live in this one table, distinguished by the `source` column. SQLite (over a flat CSV/parquet table) is the backend because the dominant access pattern is the one that determines training throughput:

- **DataLoader random access.** Training simulates patterns on-the-fly: each `__getitem__` is a point lookup of one structure by `id`, across multiple worker *processes*. SQLite serves this from a B-tree index, one blob at a time, with each worker holding its own read-only connection and near-zero resident memory. A CSV would force `read_csv` to load the entire ~290k-row frame eagerly into every worker (Python's refcounting defeats copy-on-write under `fork`), and the bulk of each row is the structure blob — the memory blowup can dominate the RAM budget before training starts. Parquet is columnar and excellent for analytical scans but awkward for single-row point lookups. This is the load-bearing reason; it sits above the three below.
- **Disorder fidelity.** Many ICSD entries have fractional site occupancies (e.g., Mn₀.₆₇Cr₀.₃₃ mixed sites). ASE's `Atoms` requires exactly one element per site, forcing lossy majority-species replacement or supercell expansion. Pymatgen `Structure` handles fractional occupancies natively, and the simulation pipeline uses occupancy-weighted structure factors — the physically correct treatment. (MP-20 structures are ordered, so this matters only for ICSD.)
- **Speed.** `Structure.from_dict()` is 5–20× faster than `Structure.from_str(cif_text)`, which matters in DataLoader workers. (Format property, preserved regardless of backend.)
- **Size.** Gzipped compact dicts land at ~3–4× smaller than `Structure.as_dict()`, comparable to gzipped CIF but deserializing much faster.

**Pandas ergonomics are not sacrificed.** `pd.read_sql_query("SELECT … WHERE …", conn)` returns a DataFrame from any SQL query in one line, and DB Browser for SQLite gives a GUI for eyeballing rows. If a purely analytical artifact is ever wanted, dump a *metadata-only* parquet (no blobs) — small and inspectable — while structures stay in SQLite. The structure blob is unreadable in a CSV cell regardless (raw gzip bytes don't fit text cells; base64 is opaque; un-gzipped JSON balloons the file), so CSV would pay all the performance costs to make only the ~dozen metadata columns human-friendly — which `read_sql` already provides for free.

### Schema

```sql
CREATE TABLE crystals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source          TEXT    NOT NULL,       -- 'icsd' | 'mp-20'
    source_id       TEXT    NOT NULL,       -- ICSD collection code | MP id (e.g. 'mp-1234')
    source_file_id  TEXT,                   -- ICSD: icsd_id (CIF filename on disk); MP-20: null
    structure_blob  BLOB    NOT NULL,       -- gzipped JSON of compact Structure dict (cell as-parsed)

    -- Labels (read from source, not spglib; see Conversion pipeline)
    space_group     INTEGER NOT NULL,       -- 1..230 (1 = lowest symmetry → 230 = highest)
    crystal_system  INTEGER NOT NULL,       -- 1..7 (1 = triclinic → 7 = cubic), derived from space_group

    -- Composition query keys
    formula             TEXT    NOT NULL,    -- reduced cell composition, e.g. 'TiO2'
    chemsys             TEXT    NOT NULL,    -- sentinel-delimited element set, e.g. '-O-Ti-'
    anonymized_formula  TEXT    NOT NULL,    -- stoichiometry prototype, e.g. 'AB2'
    n_elements          INTEGER NOT NULL,
    n_atoms             INTEGER NOT NULL,    -- atoms in the (as-parsed) unit cell
    n_sites             INTEGER NOT NULL,    -- distinct sites (≠ n_atoms under disorder)

    -- Cell query keys (as-parsed cell — no Niggli reduction; StructureMatcher canonicalizes at compare-time)
    volume          REAL    NOT NULL,        -- Å³
    density         REAL    NOT NULL,        -- g/cm³
    a               REAL    NOT NULL,
    b               REAL    NOT NULL,
    c               REAL    NOT NULL,
    alpha           REAL    NOT NULL,        -- degrees
    beta            REAL    NOT NULL,
    gamma           REAL    NOT NULL,

    -- Flags
    is_disordered   INTEGER NOT NULL,        -- 0 | 1
    n_atoms_le_20   INTEGER NOT NULL,        -- 1 if n_atoms <= 20 (Track B subset key); always 1 for mp-20

    -- Split (populated later; see Conversion pipeline)
    split           TEXT,                    -- 'train' | 'val' | 'test' | null

    UNIQUE(source, source_id)
);

CREATE INDEX idx_source            ON crystals(source);
CREATE INDEX idx_space_group       ON crystals(space_group);
CREATE INDEX idx_crystal_system    ON crystals(crystal_system);
CREATE INDEX idx_formula           ON crystals(formula);
CREATE INDEX idx_chemsys           ON crystals(chemsys);
CREATE INDEX idx_anon_formula      ON crystals(anonymized_formula);
CREATE INDEX idx_n_atoms_le_20     ON crystals(n_atoms_le_20);
CREATE INDEX idx_split             ON crystals(split);

-- Element-membership queries (contains-all, exact-set, exclude) via indexed join.
-- Written in the same transaction as the parent crystals row; cascades on delete.
CREATE TABLE crystal_elements (
    crystal_id  INTEGER NOT NULL,
    element     TEXT    NOT NULL,            -- e.g. 'Fe'
    amount      REAL    NOT NULL,            -- per-formula-unit count; fractional under disorder
    FOREIGN KEY (crystal_id) REFERENCES crystals(id) ON DELETE CASCADE
);

CREATE INDEX idx_ce_element  ON crystal_elements(element);
CREATE INDEX idx_ce_crystal  ON crystal_elements(crystal_id);
```

**Column roles.** `structure_blob` carries the full structure for simulation reconstruction; you never query on the lattice/coordinate matrices, so they stay inside the blob and the *scalar projections* (`a,b,c,α,β,γ`, `volume`, `density`) are exposed as query keys. `formula` answers "this exact chemistry"; `anonymized_formula` answers "this stoichiometry pattern, any elements" (`TiO2`, `SiO2`, `ZrO2` all → `AB2`); `chemsys` is the fast denormalized single-element membership path. The `crystal_elements` junction table handles the relational membership queries `chemsys` can't do cleanly — substring matching on a flat element string is buggy (`O` matches `Os`, `C` matches `Cs`/`Co`), so contains-all / exact-set / exclude go through the indexed join instead:

```sql
-- contains-all {Fe, O}
SELECT crystal_id FROM crystal_elements WHERE element IN ('Fe','O')
GROUP BY crystal_id HAVING COUNT(DISTINCT element) = 2;

-- exact element set {Ti, O}
SELECT crystal_id FROM crystal_elements GROUP BY crystal_id
HAVING COUNT(DISTINCT element) = 2 AND SUM(element IN ('Ti','O')) = COUNT(*);

-- exclude carbon
SELECT id FROM crystals WHERE id NOT IN
  (SELECT crystal_id FROM crystal_elements WHERE element = 'C');
```

`n_atoms_le_20` is the indexed key for Track B's size-bounded training subset (≤20 atoms is the DiffCSP/FlowMM-validated regime; see `METHODS_ROADMAP.md` §3 and `DESIGN_DECISIONS.md` §8). It is derivable from `n_atoms` for ICSD but stored so the common filter is a single indexed column and so MP-20 rows assert it by construction. `split` is nullable and populated *after* ingest: MP-20 inherits its native train/val/test split from the source CSVs (see below); ICSD rows are assigned later by a crystal-system-stratified split.

The `source` column distinguishes `'icsd'` from `'mp-20'`; `UNIQUE(source, source_id)` is the dedup key (the pair, since numeric ids can collide across sources). `source_file_id` holds the ICSD `icsd_id` purely for on-disk CIF lookup, because CIF files are named by `icsd_id` while `source_id` holds the ICSD *collection code* (the two are bridged by a mapping CSV from the prior paper). See `DESIGN_DECISIONS.md` §2.

### Compact Structure serialization

```python
import gzip, json
from pymatgen.core import Structure, Lattice

def structure_to_compact_dict(s: Structure) -> dict:
    return {
        "lattice": s.lattice.matrix.tolist(),
        "sites": [
            {
                "species": [{"el": sp.symbol, "occ": occ}
                            for sp, occ in site.species.items()],
                "abc": list(site.frac_coords),
            }
            for site in s
        ],
    }

def compact_dict_to_structure(d: dict) -> Structure:
    return Structure(
        lattice=Lattice(d["lattice"]),
        species=[{sp["el"]: sp["occ"] for sp in site["species"]}
                 for site in d["sites"]],
        coords=[site["abc"] for site in d["sites"]],
    )

def serialize(s: Structure) -> bytes:
    return gzip.compress(json.dumps(structure_to_compact_dict(s)).encode())

def deserialize(blob: bytes) -> Structure:
    return compact_dict_to_structure(json.loads(gzip.decompress(blob).decode()))
```

Round-trip preserves fractional occupancies exactly. The cell is stored **as parsed** — no Niggli or other reduction at ingest. `StructureMatcher` reduces internally at compare-time (used for the Track B generated-vs-known evaluation), so pre-reducing would buy nothing and would desynchronize the stored cell from the `a,b,c,α,β,γ` query columns a user reasons about.

### Conversion pipeline

Two ingest scripts populate the shared `crystals` table.

**`convert_icsd.py`** walks the ICSD CIF directory. For each CIF:

1. Load with `Structure.from_file()` (pymatgen).
2. **Space group is read from the CIF text, not re-derived.** Prefer the integer `_space_group_IT_number` (unambiguous, 1–230); fall back to parsing `_symmetry_space_group_name_H-M` through the `hm_symbol_to_sg_number` utility only when the IT number is absent (H-M symbols have notational variants — `P2_1/c` vs `P21/c`, short/full, origin/axis settings — so naive string matching is brittle).
3. **Crystal system is derived from the space group** by the standard range partition (`sg_to_cs`), making SG/CS disagreement impossible by construction.
4. Map the ICSD collection code → `icsd_id` via the prior-paper CSV; store the collection code as `source_id` and `icsd_id` as `source_file_id`. If the mapping CSV lacks an entry, log a warning and store a null `source_file_id` (the row is still inserted — the CIF was loadable).
5. Compute composition keys (`formula`, `chemsys`, `anonymized_formula`, `n_elements`) from `Composition`, cell scalars (`volume`, `density`, `a..gamma`, `n_atoms`, `n_sites`) from the `Structure`, and `is_disordered` / `n_atoms_le_20` flags.
6. Serialize via `serialize()`; insert the `crystals` row and its `crystal_elements` rows **in one transaction**.

**`convert_mp20.py`** reads the MP-20 `train.csv` / `val.csv` / `test.csv` (CIF-per-row plus metadata). For each row:

1. Parse the structure from the row's CIF string.
2. **Space group is read from the CSV field** (MP-20 ships it); `crystal_system` derived via `sg_to_cs` as above.
3. Store the MP id as `source_id`, null `source_file_id`, `source='mp-20'`.
4. Same composition/cell/flag computation as ICSD. `n_atoms_le_20` is 1 by construction (MP-20 is capped at 20 atoms).
5. **`split` is set from the source file** the row came from (`train`/`val`/`test`) — honoring MP-20's native split is a leakage-safety requirement, since any generative-model checkpoint used as code scaffolding was developed against that exact split (`DESIGN_DECISIONS.md` §8).
6. Serialize and insert with `crystal_elements` in one transaction.

**No spglib.** Both sources carry an authoritative reported space group, so spglib is not used to populate (or validate) symmetry labels. This drops the `spglib>=2.0` dependency. The cost — spglib's old role of catching CIF data-entry errors by disagreement — is accepted: the depositor's reported SG is the canonical training label either way. (If a cheap audit is later wanted, it can be re-added as warn-only logging that never writes the column.)

**Symmetry utilities** (`crystals/symmetry.py`, pure lookup, no spglib):

- `sg_to_cs(sg: int) -> int` — space group number → crystal system number (1–7) by range partition: triclinic 1–2, monoclinic 3–15, orthorhombic 16–74, tetragonal 75–142, trigonal 143–167, hexagonal 168–194, cubic 195–230. (This is exactly the partition `METHODS_ROADMAP.md` §2.1's CS-gated SG head assumes, so the data layer and model agree by construction.)
- `sg_number_to_symbol(sg: int) -> str` — space group number → Hermann-Mauguin symbol.
- `cs_number_to_name(cs: int) -> str` — crystal system number → name.
- `hm_symbol_to_sg_number(symbol: str) -> int` — internal, for the CIF H-M fallback only.

`crystal_system` is stored as the integer (1–7); the name comes from the utility. This keeps a single source of truth and matches the SG storage convention.

This is not a filtering step. ICSD entries are experimentally verified and all kept; MP-20 is kept as published. Symmetry labels come from the sources, not from re-analysis.

### Query API

A thin `CrystalDatabase` class exposes:

- `get(id) -> Structure` — deserialize a single row
- `iter_by_space_group(sg: int) -> Iterator[Structure]`
- `iter_by_crystal_system(cs: int) -> Iterator[Structure]`
- `iter_all(source: str | None = None, max_atoms: int | None = None, split: str | None = None, exclude_disordered: bool = False) -> Iterator[Structure]` — main training iterator; `source='icsd'` gives the ICSD-only pool (Track A), `max_atoms=20` gives the Track B size-bounded subset, and the two compose (e.g. `source='mp-20'` for the redistributable Track B generator set)
- `query_by_elements(contains_all=None, exact=None, exclude=None) -> Iterator[int]` — composition membership via the `crystal_elements` join
- `get_cif(id) -> str` — on-demand CIF reconstruction for manual inspection
- `count(**filters) -> int`
- `to_frame(**filters) -> pd.DataFrame` — metadata-only DataFrame (no blobs) via `pd.read_sql_query`, for exploratory analysis

All structure iterators yield pymatgen `Structure` objects (disorder preserved), ready to be passed to `CrystalAI-simXRD`'s Bragg calculator. The `source` / `max_atoms` / `split` filters are how Track A (full ICSD), Track B internal (ICSD+MP-20 ≤20 atoms), and Track B released (MP-20 ≤20 atoms) select their pools from the one database — no separate tables.

---

## 2. Experimental PXRD patterns

Experimental data **is** centralized, into a common store inside `crystalai-data` (the `xrddata` subpackage). This reverses the earlier per-source-index design: at ~5k patterns the fork-memory argument that drove §1's SQLite-blob choice does not apply, and a single normalized store is what a common DataLoader and the matched-sample CIF lookup (`DESIGN_DECISIONS.md` §6) both need. The store keeps every pattern as **canonical text files** (`.xy` for patterns, `.cif` for structures) as the source of truth — nothing is resampled or re-binned at ingest, so the normalization is lossless and every file stays openable for manual investigation. A single unified `index.csv` carries the metadata and the file pointers (CSV, not SQLite: at ~5k rows the index loads once into a DataFrame and is served from memory, so there is no random-access-across-processes problem to solve and CSV stays the lighter, greppable choice).

**Native format is preserved; all preprocessing is deferred to the consumer's `__getitem__`.** Patterns are stored in the coordinate they arrive in (2θ for angle-dispersive lab/synchrotron data); since the wavelength is in the index, the conversion to d / log-d and the binning happen on-the-fly in `CrystalAI-methods`' `experimental_dataset.py`. Background subtraction is likewise done downstream when `bgsub` is absent.

> **Deferred optimization (note, not built).** If on-the-fly conversion in `__getitem__` proves too slow, add a second pair of preprocessed files per pattern — the binned profile view and the binned peak view, both as `.xy` — to the store, and have the Dataset prefer them when present. Not needed at ~5k patterns; revisit only if profiling shows the conversion dominating DataLoader time.

### Sources

| Source | Size | Labels | Wavelength | Notes |
|--------|------|--------|------------|-------|
| RRUFF | ~3,000 | Full CIF for most | Cu Kα (mostly) | Separate raw and processed (background-subtracted) xy folders; some entries carry refinement folders with phase/structure and peak positions. Files are uniquely named and serve as identifiers |
| opXRD-labeled | ~1,000 | Full structure labels | Mixed (Cu, Mo, Co, synchrotron) | Zenodo opXRD release (CNRS & HKUST). One JSON per pattern; a `phases` key carries the CIF/structure information |
| Internal lab | <1,000 | Variable (some CS-only, some full CIF) | Cu Kα, Mo Kα | STADI-P / STADI-MP instruments |

**Total: ~5k labeled experimental patterns.** ~90% have full CIFs, enabling matched-sample contrastive alignment (see `DESIGN_DECISIONS.md` §6).

The 91k uncurated opXRD pool is **not used** in this iteration — see `DESIGN_DECISIONS.md` §6.

### Store layout

```
src/crystalai_data/
├── __init__.py
├── crystals/                    # ICSD + MP-20 path (see §1)
│   ├── database.py              # SQLite + Structure serialization
│   ├── convert_icsd.py          # ICSD CIF → SQLite
│   ├── convert_mp20.py          # MP-20 train/val/test CSVs → SQLite
│   ├── symmetry.py              # SG↔CS↔symbol lookup utilities (no spglib)
│   └── api.py                   # CrystalDatabase query class
├── xrddata/
│   ├── __init__.py
│   ├── database.py              # XRDDatabase: unified index (CSV) + pattern-file resolver
│   ├── ingest_rruff.py          # RRUFF folders → canonical files + index rows
│   ├── ingest_opxrd.py          # opXRD JSON    → canonical files + index rows
│   ├── ingest_lab.py            # STADI raw     → canonical files + index rows
│   ├── audit.py                 # wavelength + quality audit
│   └── store/
│       ├── index.csv            # unified index (one row per pattern)
│       └── patterns/
│           └── <source>/<source_id>/
│               ├── raw.xy       # as-measured, native coordinate (required)
│               ├── bgsub.xy     # background-subtracted (absent ⇒ null)
│               ├── peaks.xy     # peak list in the SAME coordinate as raw (absent ⇒ null)
│               └── structure.cif# phase(s); multi-block CIF if >1 phase (absent ⇒ null, or cif_id → crystals.sqlite)
└── visualization/
    ├── crystal_browser.py       # Gradio app for CIF browsing
    └── pattern_browser.py       # Gradio app for experimental pattern viewing
```

Each source's `ingest_*.py` normalizes into this one layout: it extracts the pattern (from opXRD's JSON, RRUFF's xy folders, the STADI raw) into `raw.xy` / `bgsub.xy`, the structure (opXRD's `phases` entry, RRUFF's refinement CIF) into `structure.cif`, and the peak list into `peaks.xy` — writing text files verbatim in their native coordinate, never resampling. The canonical files are the source of truth; the index is rebuildable from them.

### Unified index schema

One row per pattern in `store/index.csv`. File paths are relative to `store/`.

| Column | Type | Description |
|--------|------|-------------|
| `id` | int | Our stable internal id (default, autoincrement) — the primary handle downstream |
| `source` | str | `'rruff'` \| `'opxrd'` \| `'lab'` |
| `source_id` | str | RRUFF: `<rruff_id>`; opXRD: `opXRD_<opxrd_source>_<json_filename>`; lab: instrument+run id. `(source, source_id)` is unique |
| `raw_path` | str | As-measured `.xy` (required) |
| `bgsub_path` | str | Background-subtracted `.xy`; null ⇒ absent, subtraction done in `__getitem__` |
| `peaks_path` | str | Peak list `.xy`, positions in the **same coordinate as `raw`**; null ⇒ source ships no peaks |
| `x_coord` | str | Native x-axis: `'two_theta'` \| `'d'` \| `'tof'` — drives the `__getitem__` conversion |
| `wavelength_A` | float | Å; **nullable** — null for TOF (none yet). Required for angle-dispersive data |
| `format` | str | Raw-loader tag (`xy`, `xrdml`, `raw_stadi`, ...) |
| `crystal_system` | int | 1–7 if labeled (else null) |
| `space_group` | int | 1–230 if labeled (else null) |
| `a`, `b`, `c` | float | Cell edge lengths (Å) if labeled (else null) |
| `alpha`, `beta`, `gamma` | float | Cell angles (degrees) if labeled (else null) |
| `formula` | str | Reduced formula — atom types + stoichiometry (e.g. `TiO2`); full atom positions live in the CIF |
| `n_phases` | int | Phase count in `structure.cif` (default 1). `> 1` marks a multi-phase pattern |
| `cif_id` | int | Reference into `crystals.sqlite` when the structure coincides with an ICSD/MP entry (else null) |
| `cif_path` | str | Standalone CIF in the store (RRUFF/opXRD phase), for structures not in `crystals.sqlite` (else null) |
| `quality_flag` | str | `clean` \| `noisy` \| `multi_phase` \| `amorphous` |
| `notes` | str | Free-text notes from ingest |

Atom-level detail (types, stoichiometry, **positions**) is reached through the CIF (`cif_id` first, then `cif_path`); `formula` is the queryable projection of types + stoichiometry, mirroring the crystals-side split where positions stay in the structure and scalars are exposed as columns. A row with both CIF pointers null is labels-only (CS/SG/cell/formula, no positions) or unlabelled. The six cell scalars are stored as columns because a labelled cell can exist without a refined structure, and because they are filterable without opening a CIF.

**Multi-phase handling.** When a pattern resolves to more than one phase (notably opXRD's plural `phases`), all phases are written into a single multi-block `structure.cif` and `n_phases` records the count. The scalar label columns (`crystal_system`, `space_group`, cell, `formula`) describe the **major/first** phase only, so `n_phases > 1` is the signal that those labels are ambiguous. Multi-phase patterns are **excluded from matched-sample training by default** — they violate the one-pattern-one-structure assumption that VICReg / InfoNCE / matched-sample alignment depend on (`DESIGN_DECISIONS.md` §6) — but are retained in the store for inspection and interesting-case studies. `quality_flag='multi_phase'` is set for convenience filtering; `n_phases` is the authoritative count.

Source-specific columns (instrument ID, 2θ range, step size, etc.) can be added per source.

### Wavelength audit

Wavelength metadata accuracy is required for the wavelength-conditioning strategy (`DESIGN_DECISIONS.md` §1). `audit.py` runs over the store:

- For **angle-dispersive** patterns (`x_coord` ∈ {`two_theta`, `d`}): missing wavelength is disqualifying — flagged in `notes` and excluded from training. Suspicious wavelength (values outside [0.4, 2.5] Å, or inconsistent with the recorded instrument) is flagged for manual review.
- For **TOF** patterns (`x_coord = 'tof'`): null wavelength is expected, not an error, and does not exclude the pattern. (TOF also has no FiLM wavelength input — handling deferred; none in the pool yet.)
- Internal lab patterns are already annotated with wavelength (Cu Kα or Mo Kα per STADI-P / STADI-MP) — these need no audit beyond format validation.

---

## 3. Visualization apps

Two Gradio-based apps live in `src/crystalai_data/visualization/`. They are intended for exploratory use during data preparation and for consortium demos, not for training-time inspection.

### Crystal browser (`crystal_browser.py`)

Loads `crystals.sqlite` and provides:

- Filter / search by formula, space group, crystal system, number of atoms.
- Per-structure view: unit cell visualization (via py3Dmol), CIF text dump, simulated PXRD preview (lightweight call into `CrystalAI-simXRD`).
- Distribution plots: CS histogram, SG histogram (with a 230-bar option), atom count distribution, volume distribution.

### Pattern browser (`pattern_browser.py`)

Loads the unified experimental index (`store/index.csv`) and provides:

- Source / label filtering.
- Per-pattern view: 2θ-I and log-d-I overlay (converting from the stored native coordinate via `wavelength_A`), raw vs background-subtracted toggle, peak-position overlay where available, metadata panel.
- Side-by-side comparison: experimental vs simulated (the latter via `CrystalAI-simXRD` if `cif_id` or `cif_path` is non-null).
- Distribution plots over the experimental pool: wavelength distribution per source, CS distribution among labeled, instrument distribution.

---

## 4. Outputs consumed by other repos

| Consumer | What it reads |
|----------|---------------|
| `CrystalAI-simXRD` | `crystals.sqlite` via the `CrystalDatabase` API (for simulating training patterns) |
| `CrystalAI-methods` | `crystals.sqlite` (Track A: `source='icsd'` pool; Track B: `n_atoms_le_20` subset, by `source`; plus matched-sample alignment — finding the CIF behind an experimental pattern, resolving `cif_id` then `cif_path`) and the unified experimental index `store/index.csv` (for loading training/evaluation patterns) |

There is no other coupling. Each downstream repo imports `crystalai_data` and uses the API. The Track-A-vs-Track-B and ICSD-vs-MP-20 distinctions are query filters on the one database, not separate stores.

---

## 5. Implementation order

| Step | Task |
|------|------|
| 1 | Package skeleton, `pyproject.toml`, base directory structure |
| 2 | `database.py` — schema (incl. `crystal_elements`), serialization, round-trip tests on example CIFs including a disordered structure |
| 3 | `symmetry.py` — `sg_to_cs`, `sg_number_to_symbol`, `cs_number_to_name`, `hm_symbol_to_sg_number`; unit tests against the 7 range boundaries |
| 4 | `convert_icsd.py` — full ICSD walk (~290k structures); SG from CIF text, CS derived, collection-code→icsd_id mapping, junction-table population |
| 5 | `convert_mp20.py` — MP-20 train/val/test CSV ingest; SG from CSV, native split honored, `n_atoms_le_20=1` |
| 6 | `CrystalDatabase` API + tests (source/subset/split filters, element-membership queries) |
| 7 | `xrddata/` unified store: `database.py` (index schema + `XRDDatabase` resolver) and per-source normalizers into canonical `.xy`/`.cif` + one `index.csv` (RRUFF first, opXRD second, lab third) |
| 8 | `audit.py` wavelength + quality pass over the store (incl. the TOF wavelength carve-out and `n_phases` multi-phase flagging) |
| 9 | `crystal_browser.py` Gradio app |
| 10 | `pattern_browser.py` Gradio app |

Steps 1–6 should be fully tested before downstream repos start consuming the API.

---

## 6. Dependencies

```toml
[project]
name = "crystalai-data"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.24",
    "pandas>=2.0",             # read_sql_query ergonomics, metadata frames
    "pymatgen>=2024.1",
    "gradio>=4.0",
    "plotly>=5.15",
    "py3dmol",                 # optional, for unit cell visualization
    "tqdm",
    "pyyaml",
]

[project.optional-dependencies]
dev = ["pytest", "ruff"]
```

No `ase` dependency. No `Pysimxrd`. No `mp-api`. No `spglib` — space groups are read from the sources (CIF text / MP-20 CSV) and crystal systems derived from them, so no symmetry re-analysis is performed.
