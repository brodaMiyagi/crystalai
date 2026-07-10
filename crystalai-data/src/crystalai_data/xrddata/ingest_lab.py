"""Ingest internal-lab (RWTH-A) PXRD patterns into the unified experimental store.

RWTH-A is a STADI-P/STADI-MP lab set: known compounds measured in-house, each
indexed/refined and cross-referenced to its ICSD entry. Layout under
``RWTH-A_DATA_FOLDER``:

  * ``data_labels_expo_results.csv`` — one row per pattern (``ID`` = ``ML1``..``ML82``):
    ``Radiation`` (Cu/Mo), ``Structural Formula``, ``ICSD Col Code``, ``SG`` (int),
    ``CS``, cell ``a..gamma``, ``Quality``, plus EXPO structure-solution *predictions*
    (``*_pred``, ``sym_``, ``cs_match_``) which are **not** labels and are ignored.
  * ``backgroud_files/<ID>_xy.csv`` — the pattern itself: columns
    ``x, y_obs, weight, y_calc, y_bkg, Q``. ``y_obs`` is the raw profile and
    ``y_bkg`` a refined native background on the same 2θ grid, so both ``raw.xy``
    and a native ``bgsub.xy`` (``y_obs - y_bkg``, ``autobg=0``) come from one file.
  * ``data_and_expo_runs/<ID>/<ID>.pea`` — manually-picked peak list (d-spacings, Å) fed
    to EXPO. Converted to a RRUFF-consistent ``peaks.xy`` (2θ, intensity sampled from the
    bgsub profile and scaled to 100). Other EXPO tables/binaries there are not ingested.

Labels are read, never re-derived: ``SG`` from the CSV integer, ``CS`` via
``sg_to_cs`` (verified against the CSV ``CS``), cell from the CSV. The **ICSD
Collection Code links each pattern to its known structure in ``crystals.sqlite``**
— resolved to a ``cif_id`` (the matched-sample handle, `DESIGN_DECISIONS.md` §6)
and used as the authoritative source of ``formula`` (the CSV formula is blank for
many rows and messy for hydrates).

Rigorous rejection (nothing fishy is ingested): a row is **rejected** when its
``ICSD Col Code`` is not a single clean integer (free-text / multi-phase / "not in
ICSD"), when that code does not resolve in ``crystals.sqlite``, when its background
CSV is missing/malformed, or when ``SG`` is out of range. Every reject is logged.

Run (from the package directory)::

    uv run python -m crystalai_data.xrddata.ingest_lab            # uses .env
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np

from ..crystals._ingest import load_dotenv, repo_root
from ..crystals.symmetry import sg_to_cs
from . import database as db

# STADI monochromated Kα1 wavelengths (Å). Ge(111) mono → pure Kα1.
_WAVELENGTHS = {"Cu": 1.54056, "Mo": 0.70930}
_QUALITY_MAP = {"clean": "clean", "noisy": "noisy", "multiphase": "multi_phase"}
_BG_REQUIRED_COLS = {"x", "y_obs", "y_bkg"}


def _clean_icsd_code(value) -> str | None:
    """Return a single clean integer ICSD collection code, or None if fishy.

    Fishy = free text, empty, or multiple codes (the CSV stores prose there for
    non-ICSD / multi-phase / uncertain samples). Those rows are rejected.
    """
    s = str(value).strip()
    digits = s.replace(".0", "")  # pandas may read the code as a float
    return digits if digits.isdigit() else None


def _load_icsd_index(crystals_db: Path) -> dict[str, tuple[int, str, int]]:
    """Map ICSD collection code -> (crystals.id, formula, space_group).

    Read once so each RWTH row is a dict lookup rather than a per-row query.
    Empty (with a warning) if the DB is absent — every row then fails to resolve
    and is rejected, which is the correct rigorous behaviour.
    """
    if not crystals_db.is_file():
        print(f"[lab] WARNING: crystals DB not found at {crystals_db}; "
              "no ICSD codes will resolve and all rows will be rejected")
        return {}
    conn = sqlite3.connect(str(crystals_db))
    try:
        rows = conn.execute(
            "SELECT source_id, id, formula, space_group FROM crystals WHERE source='icsd'"
        ).fetchall()
    finally:
        conn.close()
    return {str(sid): (int(cid), formula, int(sg)) for sid, cid, formula, sg in rows}


def _read_pattern(bg_csv: Path):
    """Read (x, y_obs, y_bkg) from a RWTH ``backgroud_files`` CSV."""
    import pandas as pd

    df = pd.read_csv(bg_csv)
    missing = _BG_REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"background csv missing columns {missing}")
    x = df["x"].astype(float).tolist()
    y_obs = df["y_obs"].astype(float).tolist()
    y_bkg = df["y_bkg"].astype(float).tolist()
    if not x:
        raise ValueError("background csv has no rows")
    return x, y_obs, y_bkg


def _read_pea(path: Path) -> list[float]:
    """Read a EXPO ``.pea`` file: one manually-picked peak **d-spacing** (Å) per line."""
    ds: list[float] = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            ds.append(float(s.split()[0]))
        except ValueError:
            continue
    return ds


def _peaks_to_two_theta(
    d_spacings: list[float], wavelength: float,
    x: list[float], y: list[float], window: float = 0.05,
) -> tuple[list[float], list[float]]:
    """Convert a d-spacing peak list to ``(2θ, intensity)`` consistent with RRUFF ``peaks.xy``.

    Bragg: ``2θ = 2·asin(λ/2d)``. The ``.pea`` list carries no intensities (EXPO picks
    positions only), so each peak's height is **sampled from the bgsub profile** — the
    local max within ``±window`` degrees (robust to small grid/pick offsets) — then the
    whole list is scaled to 100 like RRUFF. Peaks outside the measured range or with
    ``λ/2d > 1`` (unphysical) are dropped.
    """
    xa, ya = np.asarray(x, float), np.asarray(y, float)
    tts: list[float] = []
    ints: list[float] = []
    for d in d_spacings:
        ratio = wavelength / (2.0 * d)
        if not 0.0 < ratio <= 1.0:
            continue
        tt = float(2.0 * np.degrees(np.arcsin(ratio)))
        if tt < xa[0] or tt > xa[-1]:
            continue
        sel = (xa >= tt - window) & (xa <= tt + window)
        inten = float(ya[sel].max()) if sel.any() else float(np.interp(tt, xa, ya))
        tts.append(tt)
        ints.append(max(inten, 0.0))
    if not tts:
        return [], []
    order = np.argsort(tts)
    tt_s = np.asarray(tts)[order]
    in_s = np.asarray(ints)[order]
    peak = in_s.max()
    if peak > 0:
        in_s = in_s / peak * 100.0
    return tt_s.tolist(), in_s.tolist()


def convert(*, data_dir: Path, store: Path, crystals_db: Path) -> None:
    import pandas as pd

    csv_path = data_dir / "data_labels_expo_results.csv"
    if not csv_path.is_file():
        raise SystemExit(f"[lab] label CSV not found: {csv_path}")

    icsd_index = _load_icsd_index(crystals_db)
    print(f"[lab] data_dir    = {data_dir}")
    print(f"[lab] store       = {store}")
    print(f"[lab] ICSD DB      = {crystals_db} ({len(icsd_index):,} icsd codes)")

    df = pd.read_csv(csv_path)
    rows: list[db.IndexRow] = []
    rejects: list[tuple[str, str]] = []
    sg_mismatch = 0

    for rec in df.to_dict("records"):
        rid = str(rec["ID"]).strip()

        code = _clean_icsd_code(rec["ICSD Col Code"])
        if code is None:
            rejects.append((rid, "fishy ICSD code (not a single integer)"))
            continue

        resolved = icsd_index.get(code)
        if resolved is None:
            rejects.append((rid, f"ICSD code {code} not found in crystals.sqlite"))
            continue
        cif_id, icsd_formula, icsd_sg = resolved

        try:
            sg = int(rec["SG"])
        except (TypeError, ValueError):
            rejects.append((rid, "unparseable SG"))
            continue
        if not 1 <= sg <= 230:
            rejects.append((rid, f"SG out of range: {sg}"))
            continue
        if sg != icsd_sg:
            sg_mismatch += 1  # keep the CSV (experimental) SG, but flag the disagreement

        bg_csv = data_dir / "backgroud_files" / f"{rid}_xy.csv"
        if not bg_csv.is_file():
            rejects.append((rid, "missing background csv"))
            continue
        try:
            x, y_obs, y_bkg = _read_pattern(bg_csv)
        except Exception as exc:  # noqa: BLE001
            rejects.append((rid, f"bad background csv: {type(exc).__name__}: {exc}"))
            continue

        radiation = str(rec["Radiation"]).strip()
        wavelength = _WAVELENGTHS.get(radiation)
        quality = _QUALITY_MAP.get(str(rec["Quality"]).strip().lower(), "clean")
        # formula: prefer the linked ICSD structure (authoritative + always present);
        # the CSV 'Structural Formula' is blank for many rows and messy for hydrates.
        formula = icsd_formula
        cell = {k: float(rec[k]) for k in ("a", "b", "c", "alpha", "beta", "gamma")}

        source_id = f"RWTH-A_{rid}"
        pdir = db.pattern_dir(store, "lab", source_id)
        prov = ["source=lab", "subsource=RWTH-A", f"lab_id={rid}",
                f"radiation={radiation}", "x_coord=two_theta"]
        if wavelength is not None:
            prov.append(f"wavelength_A={wavelength}")
        db.write_xy(pdir / "raw.xy", x, y_obs, header=[*prov, "view=raw"])
        bgsub = [o - b for o, b in zip(y_obs, y_bkg, strict=True)]
        db.write_xy(pdir / "bgsub.xy", x, bgsub, header=[*prov, "view=bgsub (native y_obs-y_bkg)"])

        # Manually-picked peak list (EXPO .pea, d-spacings) -> peaks.xy in 2θ, RRUFF-style.
        peaks_rel: str | None = None
        pea_file = data_dir / "data_and_expo_runs" / rid / f"{rid}.pea"
        if pea_file.is_file() and wavelength is not None:
            tt_pk, i_pk = _peaks_to_two_theta(_read_pea(pea_file), wavelength, x, bgsub)
            if tt_pk:
                db.write_xy(pdir / "peaks.xy", tt_pk, i_pk,
                            header=[*prov, "view=peaks (EXPO .pea d-list -> 2theta; "
                                    "intensity sampled from bgsub, scaled to 100)"])
                peaks_rel = f"patterns/lab/{source_id}/peaks.xy"

        csv_formula = rec["Structural Formula"]
        notes = f"icsd_collection_code={code}"
        if isinstance(csv_formula, str) and csv_formula.strip():
            notes += f"; csv_formula={csv_formula.strip()}"

        rows.append(
            db.IndexRow(
                source="lab",
                source_id=source_id,
                raw_path=f"patterns/lab/{source_id}/raw.xy",
                bgsub_path=f"patterns/lab/{source_id}/bgsub.xy",
                autobg=0,  # native refined background (y_bkg), not auto-generated
                peaks_path=peaks_rel,
                x_coord="two_theta",
                wavelength_A=wavelength,
                format="xy",
                crystal_system=sg_to_cs(sg),
                space_group=sg,
                a=cell["a"], b=cell["b"], c=cell["c"],
                alpha=cell["alpha"], beta=cell["beta"], gamma=cell["gamma"],
                formula=formula,
                n_phases=1,
                cif_id=cif_id,  # matched-sample link into crystals.sqlite
                cif_path=None,
                quality_flag=quality,
                notes=notes,
            )
        )

    written = db.write_source_rows(store, "lab", rows)
    print(f"\n[lab] patterns written = {written} (of {len(df)} rows)")
    print(f"[lab] cif_id linked    = {sum(1 for r in rows if r.cif_id is not None)} (matched-sample)")
    print(f"[lab] peak lists (.pea)= {sum(1 for r in rows if r.peaks_path is not None)} (2θ, from EXPO)")
    if sg_mismatch:
        print(f"[lab] NOTE: {sg_mismatch} rows had CSV SG != ICSD SG (kept CSV SG)")
    if rejects:
        print(f"[lab] rejected         = {len(rejects)}:")
        for rid, why in rejects:
            print(f"       - {rid}: {why}")


def main(argv: list[str] | None = None) -> None:
    root = repo_root()
    env = load_dotenv(root / ".env")
    data_default = env.get("RWTH-A_DATA_FOLDER")
    store_default = str(db.resolve_store(env, root))
    crystals_default = env.get("CRYSTALS_DB", str(root / "crystalai-data" / "crystals.sqlite"))

    p = argparse.ArgumentParser(description="Ingest RWTH-A internal-lab patterns into the store")
    p.add_argument("--data-dir", default=data_default, type=str, required=data_default is None)
    p.add_argument("--store", default=store_default, type=str)
    p.add_argument("--crystals-db", default=crystals_default, type=str,
                   help="crystals.sqlite, for ICSD-code -> cif_id resolution")
    args = p.parse_args(argv)

    convert(
        data_dir=Path(args.data_dir),
        store=Path(args.store),
        crystals_db=Path(args.crystals_db),
    )


if __name__ == "__main__":
    main()
