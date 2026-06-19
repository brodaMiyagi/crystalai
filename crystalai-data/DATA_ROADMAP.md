# CrystalAI-data — Roadmap

**Scope:** Curate and structure all input data for the CrystalAI pipeline. ICSD CIFs on the structural side; RRUFF, opXRD, and internal lab patterns on the experimental side. Provide visualization apps for both.

**Out of scope:** Simulation (lives in `CrystalAI-simXRD`). ML training (lives in `CrystalAI-methods`).

---

## 1. Crystal structures (ICSD only)

### Storage

A single SQLite database (`crystals.sqlite`) holds every structure as a gzip-compressed compact pymatgen dict, alongside indexed metadata columns. This format was chosen over ASE `.db` because:

- **Disorder fidelity.** Many ICSD entries have fractional site occupancies (e.g., Mn₀.₆₇Cr₀.₃₃ mixed sites). ASE's `Atoms` requires exactly one element per site, forcing lossy majority-species replacement or supercell expansion. Pymatgen `Structure` handles fractional occupancies natively, and the simulation pipeline uses occupancy-weighted structure factors — the physically correct treatment.
- **Speed.** `Structure.from_dict()` is 5–20× faster than `Structure.from_str(cif_text)`, which matters in DataLoader workers.
- **Size.** Gzipped compact dicts land at ~3–4× smaller than `Structure.as_dict()`, comparable to gzipped CIF but deserializing much faster.

### Schema

```sql
CREATE TABLE crystals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    structure_blob  BLOB    NOT NULL,      -- gzipped JSON of compact Structure dict
    source          TEXT    NOT NULL DEFAULT 'icsd',  -- forward compatibility
    icsd_id         INTEGER,
    space_group     INTEGER NOT NULL,      -- 1..230
    crystal_system  TEXT    NOT NULL,      -- triclinic | monoclinic | ... | cubic
    formula         TEXT    NOT NULL,      -- reduced formula
    n_atoms         INTEGER NOT NULL,
    n_elements      INTEGER NOT NULL,
    volume          REAL    NOT NULL,      -- Å³
    is_disordered   INTEGER NOT NULL       -- 0 | 1
);

CREATE INDEX idx_space_group    ON crystals(space_group);
CREATE INDEX idx_crystal_system ON crystals(crystal_system);
CREATE INDEX idx_formula        ON crystals(formula);
CREATE INDEX idx_icsd_id        ON crystals(icsd_id);
```

The `source` column is populated only with `'icsd'` in this iteration but retained for forward compatibility: if a future iteration introduces an additional curated experimental database, it slots in without schema changes. See `DESIGN_DECISIONS.md` §2.

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

Round-trip preserves fractional occupancies exactly.

### Conversion pipeline

`convert_icsd.py` walks the ICSD CIF directory and populates `crystals.sqlite`. For each CIF:

1. Load with `Structure.from_file()` (pymatgen).
2. Run `SpacegroupAnalyzer.get_space_group_number()` and `.get_crystal_system()` — used to populate the `space_group` and `crystal_system` columns.
3. **Validation, not filtering.** If Spglib's analysis disagrees with ICSD's reported space group, log a warning and the row is still inserted (with the Spglib-reported SG). This catches CIF data-entry errors without dropping structures.
4. Serialize via `serialize()`, insert into the database.

This is not a filtering step. ICSD entries are experimentally verified; we keep all of them. Spglib's role here is validation/cleanup, not curation.

### Query API

A thin `CrystalDatabase` class exposes:

- `get(id) -> Structure` — deserialize a single row
- `iter_by_space_group(sg: int) -> Iterator[Structure]`
- `iter_by_crystal_system(cs: str) -> Iterator[Structure]`
- `iter_all(exclude_disordered: bool = False) -> Iterator[Structure]` — main training iterator
- `get_cif(id) -> str` — on-demand CIF reconstruction for manual inspection
- `count(**filters) -> int`

All iterators yield pymatgen `Structure` objects (disorder preserved), ready to be passed to `CrystalAI-simXRD`'s Bragg calculator.

---

## 2. Experimental PXRD patterns

Experimental data is **not** centralized into a single database. Each source has its own ingest path, its own per-source CSV index, and points at original files in place. This avoids lossy schema conversions for sources with heterogeneous metadata.

### Sources

| Source | Size | Labels | Wavelength | Notes |
|--------|------|--------|------------|-------|
| RRUFF | ~3,000 | Full CIF for most | Cu Kα (mostly) | Well-characterized minerals; high-quality experimental references |
| opXRD-labeled | ~1,000 | Full structure labels | Mixed (Cu, Mo, Co, synchrotron) | The labeled subset of Zenodo's opXRD release |
| Internal lab | <1,000 | Variable (some CS-only, some full CIF) | Cu Kα, Mo Kα | STADI-P / STADI-MP instruments |

**Total: ~5k labeled experimental patterns.** ~90% have full CIFs, enabling matched-sample contrastive alignment (see `DESIGN_DECISIONS.md` §6).

The 91k uncurated opXRD pool is **not used** in this iteration — see `DESIGN_DECISIONS.md` §6.

### Per-source ingest structure

```
src/crystalai_data/
├── __init__.py
├── crystals/                    # ICSD path (see §1)
│   ├── database.py              # SQLite + Structure serialization
│   ├── convert_icsd.py          # ICSD CIF → SQLite
│   └── api.py                   # CrystalDatabase query class
├── experimental/
│   ├── __init__.py
│   ├── rruff/
│   │   ├── ingest.py            # raw RRUFF → CSV index + canonicalized pattern files
│   │   ├── index.csv            # output: per-pattern metadata
│   │   └── README.md            # source-specific notes
│   ├── opxrd/
│   │   ├── ingest.py
│   │   ├── index.csv
│   │   └── README.md
│   └── lab/
│       ├── ingest.py
│       ├── index.csv
│       └── README.md
└── visualization/
    ├── crystal_browser.py       # Gradio app for CIF browsing
    └── pattern_browser.py       # Gradio app for experimental pattern viewing
```

### CSV index format

Each `index.csv` carries one row per pattern with at minimum:

| Column | Description |
|--------|-------------|
| `pattern_id` | Source-prefixed unique identifier (e.g., `rruff_R040118`) |
| `file_path` | Path to the original pattern file (relative to source root) |
| `wavelength_A` | Wavelength in Å — **required**. Patterns with missing or suspicious wavelength metadata are flagged and excluded |
| `format` | File format tag (`xy`, `xrdml`, `raw_stadi`, ...) — drives the loader |
| `crystal_system` | If labeled (else null) |
| `space_group` | If labeled (else null) |
| `cif_id` | Reference into `crystals.sqlite` if a CIF is available (else null) |
| `quality_flag` | Source-specific quality tag (e.g., `clean`, `noisy`, `multi_phase`, `amorphous`) |
| `notes` | Free-text notes from ingest |

Source-specific columns (instrument ID, 2θ range, step size, etc.) can be added per source.

### Wavelength audit

Wavelength metadata accuracy is required for the W1 wavelength-conditioning strategy (`DESIGN_DECISIONS.md` §1). The ingest pipeline includes an audit step:

- Patterns with no wavelength metadata are flagged in the `notes` column and excluded from training.
- Patterns with suspicious wavelength metadata (values outside [0.4, 2.5] Å, or wavelengths inconsistent with the recorded instrument) are flagged for manual review.
- Internal lab patterns are already annotated with wavelength (Cu Kα or Mo Kα per STADI-P / STADI-MP) — these need no audit beyond format validation.

---

## 3. Visualization apps

Two Gradio-based apps live in `src/crystalai_data/visualization/`. They are intended for exploratory use during data preparation and for consortium demos, not for training-time inspection.

### Crystal browser (`crystal_browser.py`)

Loads `crystals.sqlite` and provides:

- Filter / search by formula, space group, crystal system, number of atoms.
- Per-structure view: unit cell visualization (via py3Dmol or ase.visualize), CIF text dump, simulated PXRD preview (lightweight call into `CrystalAI-simXRD`).
- Distribution plots: CS histogram, SG histogram (with a 230-bar option), atom count distribution, volume distribution.

### Pattern browser (`pattern_browser.py`)

Loads the three experimental indices (RRUFF, opXRD, lab) and provides:

- Source / label filtering.
- Per-pattern view: 2θ-I and log-d-I overlay, raw vs background-subtracted toggle, peak-position overlay where available, metadata panel.
- Side-by-side comparison: experimental vs simulated (the latter via `CrystalAI-simXRD` if `cif_id` is non-null).
- Distribution plots over the experimental pool: wavelength distribution per source, CS distribution among labeled, instrument distribution.

---

## 4. Outputs consumed by other repos

| Consumer | What it reads |
|----------|---------------|
| `CrystalAI-simXRD` | `crystals.sqlite` via the `CrystalDatabase` API (for simulating training patterns) |
| `CrystalAI-methods` | `crystals.sqlite` (for matched-sample alignment: finding the CIF behind an experimental pattern) and the three experimental indices (for loading training/evaluation patterns) |

There is no other coupling. Each downstream repo imports `crystalai_data` and uses the API.

---

## 5. Implementation order

| Step | Task |
|------|------|
| 1 | Package skeleton, `pyproject.toml`, base directory structure |
| 2 | `database.py` — schema, serialization, round-trip tests on example CIFs including a disordered structure |
| 3 | `convert_icsd.py` — full ICSD walk; ~290k structures expected |
| 4 | `CrystalDatabase` API + tests |
| 5 | Per-source experimental ingest scripts (RRUFF first, opXRD second, lab third) |
| 6 | Wavelength audit pass over all experimental sources |
| 7 | `crystal_browser.py` Gradio app |
| 8 | `pattern_browser.py` Gradio app |

Steps 1–4 should be fully tested before downstream repos start consuming the API.

---

## 6. Dependencies

```toml
[project]
name = "crystalai-data"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.24",
    "pymatgen>=2024.1",
    "spglib>=2.0",
    "gradio>=4.0",
    "plotly>=5.15",
    "py3dmol",                 # optional, for unit cell visualization
    "tqdm",
    "pyyaml",
]

[project.optional-dependencies]
dev = ["pytest", "ruff"]
```

No `ase` dependency. No `Pysimxrd`. No `mp-api`.
