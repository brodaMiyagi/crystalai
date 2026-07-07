"""Ingest the RRUFF powder-XRD dump into the unified experimental store.

RRUFF ships one file per view in parallel folders, all named
``<Mineral>__<RRUFFID>__Powder__<view>__<hash>.txt``:

  * ``XY_RAW/``          — as-measured profile (``##KEY=VALUE`` header + ``x, y`` rows)
  * ``XY_Processed/``    — background-subtracted profile (same layout)
  * ``DIF/``             — computed pattern: X-ray wavelength, space-group *symbol*,
                           refined cell, and a ``2-THETA INTENSITY D-SPACING H K L``
                           peak table
  * ``Refinement_*/``    — cell-refinement worksheets (not ingested: cell/SG already
                           come from the DIF + XY headers)

One ``XY_RAW`` file == one pattern (the store key is its RRUFF id, e.g.
``R080016-1``). We normalize into ``patterns/rruff/<rruff_id>/``:

  raw.xy   <- XY_RAW profile          (required)
  bgsub.xy <- XY_Processed profile    (when present)
  peaks.xy <- DIF (2-theta, intensity) in the SAME coordinate as raw

Labels are read from the source text, never re-derived:

  * wavelength   <- DIF ``X-RAY WAVELENGTH`` (RRUFF's XY headers carry none, so a
                    pattern without a DIF has null wavelength -> excluded by the
                    wavelength audit for angle-dispersive data, DATA_ROADMAP.md §2)
  * space_group  <- DIF ``SPACE GROUP`` H-M symbol, via ``hm_symbol_to_sg_number``
  * crystal_sys  <- derived from the space group (``sg_to_cs``); falls back to the
                    XY header's ``crystal system:`` name when no SG is available
  * cell a..gamma<- XY header ``CELL PARAMETERS`` (fallback: DIF cell line)
  * formula      <- XY header ``IDEAL CHEMISTRY`` (e.g. ``Ag_2_S`` -> ``Ag2S``)

RRUFF provides no atomic-coordinate CIF anywhere in the dump (the DIF references a
published structure but ships only cell + peak indices), so rows are labels-only:
``cif_path``/``cif_id`` stay null. ``x_coord='two_theta'``, single phase.

Run (from the package directory)::

    uv run python -m crystalai_data.xrddata.ingest_rruff            # uses .env
    uv run python -m crystalai_data.xrddata.ingest_rruff --limit 50
"""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

from ..crystals._ingest import load_dotenv, repo_root
from ..crystals.symmetry import cs_number_to_name, hm_symbol_to_sg_number, sg_to_cs
from . import database as db

# RRUFF id embedded in a filename's second ``__``-field, e.g. R080016 or R080016-1.
# A few filenames mangle the field (``R210008PowderX-raydiffractiondata``), so we
# match the id token rather than taking the whole field.
_RRUFF_ID_RE = re.compile(r"(R\d{6}(?:-\d+)?)")
_WAVELENGTH_RE = re.compile(r"X-RAY WAVELENGTH:\s*([0-9.]+)", re.IGNORECASE)
_SPACEGROUP_RE = re.compile(r"SPACE GROUP:\s*(\S+)", re.IGNORECASE)
_CS_NAME_TO_NUM = {cs_number_to_name(i): i for i in range(1, 8)}


def _rruff_id(path: Path) -> str | None:
    parts = path.name.split("__")
    field = parts[1] if len(parts) > 1 else path.name
    m = _RRUFF_ID_RE.search(field)
    return m.group(1) if m else None


def _index_by_id(folder: Path) -> dict[str, Path]:
    """Map RRUFF id -> file for a view folder (last write wins on the rare dup)."""
    out: dict[str, Path] = {}
    if not folder.is_dir():
        return out
    for f in folder.glob("*.txt"):
        rid = _rruff_id(f)
        if rid:
            out[rid] = f
    return out


def _parse_xy(path: Path) -> tuple[dict[str, str], list[float], list[float]]:
    """Split an RRUFF XY file into its ``##`` header dict and (xs, ys) columns."""
    header: dict[str, str] = {}
    xs: list[float] = []
    ys: list[float] = []
    for line in path.read_text(errors="replace").splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("##"):
            key, _, val = s[2:].partition("=")
            header[key.strip().upper()] = val.strip()
            continue
        parts = s.replace(",", " ").split()
        if len(parts) < 2:
            continue
        try:  # append as a pair so a malformed row never desyncs the columns
            x, y = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        xs.append(x)
        ys.append(y)
    return header, xs, ys


def _parse_cell(header: dict[str, str]) -> tuple[dict[str, float | None], int | None]:
    """Pull a,b,c,alpha,beta,gamma and a crystal-system number from the XY header.

    The ``CELL PARAMETERS`` header looks like::
        a: 4.2244 b: 6.9297 c: 7.8653 alpha: 90 beta: 99.65 gamma: 90 ... crystal system: monoclinic
    """
    cell: dict[str, float | None] = dict.fromkeys(
        ("a", "b", "c", "alpha", "beta", "gamma"), None
    )
    cs: int | None = None
    text = header.get("CELL PARAMETERS", "")
    if not text:
        return cell, cs
    for key in cell:
        m = re.search(rf"\b{key}:\s*([0-9.]+)", text)
        if m:
            cell[key] = float(m.group(1))
    m = re.search(r"crystal system:\s*([a-zA-Z]+)", text)
    if m:
        cs = _CS_NAME_TO_NUM.get(m.group(1).strip().lower())
    return cell, cs


def _parse_formula(header: dict[str, str]) -> str | None:
    """RRUFF ``IDEAL CHEMISTRY`` -> reduced formula (best effort).

    RRUFF markup uses ``_n_`` for subscripts and ``^n+^`` / ``^n-^`` for oxidation
    states, e.g. ``Ca_3_Fe^3+^_2_(SiO_4_)_3_``. Strip the charge annotations, then
    the subscript underscores, then let ``Composition`` reduce it. On a parse
    failure fall back to the cleaned string so the row still carries a formula.
    """
    raw = header.get("IDEAL CHEMISTRY", "").strip()
    if not raw:
        return None
    cleaned = re.sub(r"\^[^^]*\^", "", raw)  # drop '^3+^' charge markers
    cleaned = cleaned.replace("_", "")  # 'Ca_5_(PO_4_)_3_' -> 'Ca5(PO4)3'
    try:
        from pymatgen.core import Composition

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return Composition(cleaned).reduced_formula
    except Exception:
        return cleaned or None


def _parse_dif(path: Path) -> tuple[float | None, int | None, list[float], list[float], str | None]:
    """Return (wavelength, space_group_number, peak_2theta, peak_intensity, note).

    Peaks are the DIF's ``2-THETA`` / ``INTENSITY`` columns — same coordinate as
    the raw profile. ``note`` records an unresolved space-group symbol.
    """
    text = path.read_text(errors="replace")
    wavelength: float | None = None
    m = _WAVELENGTH_RE.search(text)
    if m:
        wavelength = float(m.group(1))

    space_group: int | None = None
    note: str | None = None
    m = _SPACEGROUP_RE.search(text)
    if m and m.group(1).lower() not in {"unknown", "none", "n/a"}:
        try:
            space_group = hm_symbol_to_sg_number(m.group(1))
        except ValueError:
            note = f"unresolved DIF space-group symbol {m.group(1)!r}"

    two_theta: list[float] = []
    intensity: list[float] = []
    in_table = False
    for line in text.splitlines():
        s = line.strip()
        if not in_table:
            # The peak table starts right after its column header.
            if s.upper().startswith("2-THETA") and "INTENSITY" in s.upper():
                in_table = True
            continue
        parts = s.split()
        if len(parts) < 2:
            continue
        try:  # append as a pair so a malformed row never desyncs the columns
            tt, inten = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        two_theta.append(tt)
        intensity.append(inten)
    return wavelength, space_group, two_theta, intensity, note


def convert(*, rruff_dir: Path, store: Path, limit: int | None) -> None:
    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover
        def tqdm(x, **k):  # type: ignore
            return x

    raw_dir = rruff_dir / "XY_RAW"
    if not raw_dir.is_dir():
        raise SystemExit(f"[rruff] XY_RAW not found under {rruff_dir}")

    proc_by_id = _index_by_id(rruff_dir / "XY_Processed")
    dif_by_id = _index_by_id(rruff_dir / "DIF")
    print(f"[rruff] rruff_dir = {rruff_dir}")
    print(f"[rruff] store     = {store}")
    print(f"[rruff] XY_Processed matched-ready: {len(proc_by_id):,}  DIF: {len(dif_by_id):,}")

    raw_files = sorted(raw_dir.glob("*.txt"))
    if limit is not None:
        raw_files = raw_files[:limit]

    rows: list[db.IndexRow] = []
    stats = {"no_id": 0, "no_wavelength": 0, "no_sg": 0, "with_bgsub": 0, "with_peaks": 0}

    for raw_file in tqdm(raw_files, desc="[rruff] ingest"):
        rid = _rruff_id(raw_file)
        if rid is None:
            stats["no_id"] += 1
            continue

        header, xs, ys = _parse_xy(raw_file)
        if not xs:
            continue

        cell, cs_from_header = _parse_cell(header)
        formula = _parse_formula(header)

        wavelength = space_group = None
        peak_tt: list[float] = []
        peak_i: list[float] = []
        dif_note: str | None = None
        dif_file = dif_by_id.get(rid)
        if dif_file is not None:
            wavelength, space_group, peak_tt, peak_i, dif_note = _parse_dif(dif_file)

        crystal_system = sg_to_cs(space_group) if space_group is not None else cs_from_header
        if wavelength is None:
            stats["no_wavelength"] += 1
        if space_group is None:
            stats["no_sg"] += 1

        pdir = db.pattern_dir(store, "rruff", rid)
        prov = ["source=rruff", f"rruff_id={rid}", "x_coord=two_theta"]
        if wavelength is not None:
            prov.append(f"wavelength_A={wavelength}")

        db.write_xy(pdir / "raw.xy", xs, ys, header=[*prov, "view=raw"])
        raw_rel = f"patterns/rruff/{rid}/raw.xy"

        bgsub_rel: str | None = None
        autobg: int | None = None  # 0 = native curated bgsub; auto pass fills the rest
        proc_file = proc_by_id.get(rid)
        if proc_file is not None:
            _, pxs, pys = _parse_xy(proc_file)
            if pxs:
                db.write_xy(pdir / "bgsub.xy", pxs, pys, header=[*prov, "view=bgsub"])
                bgsub_rel = f"patterns/rruff/{rid}/bgsub.xy"
                autobg = 0
                stats["with_bgsub"] += 1

        peaks_rel: str | None = None
        if peak_tt:
            db.write_xy(pdir / "peaks.xy", peak_tt, peak_i, header=[*prov, "view=peaks"])
            peaks_rel = f"patterns/rruff/{rid}/peaks.xy"
            stats["with_peaks"] += 1

        rows.append(
            db.IndexRow(
                source="rruff",
                source_id=rid,
                raw_path=raw_rel,
                bgsub_path=bgsub_rel,
                autobg=autobg,
                peaks_path=peaks_rel,
                x_coord="two_theta",
                wavelength_A=wavelength,
                format="xy",
                crystal_system=crystal_system,
                space_group=space_group,
                a=cell["a"],
                b=cell["b"],
                c=cell["c"],
                alpha=cell["alpha"],
                beta=cell["beta"],
                gamma=cell["gamma"],
                formula=formula,
                n_phases=1,
                cif_id=None,
                cif_path=None,
                quality_flag="clean",
                notes=dif_note,
            )
        )

    written = db.write_source_rows(store, "rruff", rows)
    print(f"\n[rruff] patterns written = {written:,}")
    print(f"[rruff] with bgsub={stats['with_bgsub']:,}  with peaks={stats['with_peaks']:,}")
    print(f"[rruff] missing wavelength={stats['no_wavelength']:,}  missing SG={stats['no_sg']:,}  unkeyed files={stats['no_id']:,}")


def main(argv: list[str] | None = None) -> None:
    root = repo_root()
    env = load_dotenv(root / ".env")
    rruff_default = env.get("RRUFF_DATA_FOLDER")
    store_default = env.get("EXP_DATA_FOLDER", str(db.default_store()))

    p = argparse.ArgumentParser(description="Ingest RRUFF into the experimental store")
    p.add_argument("--rruff-dir", default=rruff_default, type=str, required=rruff_default is None)
    p.add_argument("--store", default=store_default, type=str)
    p.add_argument("--limit", default=None, type=int, help="process at most N raw patterns")
    args = p.parse_args(argv)

    convert(rruff_dir=Path(args.rruff_dir), store=Path(args.store), limit=args.limit)


if __name__ == "__main__":
    main()
