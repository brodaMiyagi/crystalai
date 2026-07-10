"""Ingest the Zenodo opXRD-labeled release into the unified experimental store.

opXRD ships one JSON per pattern. Two contributors are present, in three folders
(``.env``: ``OPXRD_CNRS_DATA_FOLDER`` -> CNRS; ``OPXRD_HKUST_DATA_FOLDER`` ->
``HKUST-A`` and ``HKUST-B`` subfolders). Each JSON has::

    two_theta_values : [float]      # the profile x-axis (2-theta)
    intensities      : [float]
    label            : "<json>"     # nested JSON string, see below
    metadata         : "<json>"     # institution, contributor, date, ...

``label`` decodes to ``{phases, xray_info, is_simulated, ...}`` where:

  * ``xray_info.primary_wavelength`` is the wavelength (Angstrom).
  * ``phases`` is a list of nested-JSON phase strings, each carrying a
    ``lattice`` "(a,b,c,al,be,ga)" string and a ``basis`` list of atoms
    ``{x,y,z,occupancy,symbol,...}``. The phase's ``spacegroup`` field is
    always null in this release, so we build a full P1 ``structure.cif`` from
    lattice+basis but leave ``space_group``/``crystal_system`` null (no spglib —
    symmetry is never re-derived, DATA_ROADMAP.md §1). Occupancies are preserved.

We normalize into ``patterns/opxrd/<source_id>/`` with
``source_id = opXRD_<subsource>_<json_stem>``:

  raw.xy        <- two_theta_values / intensities   (required; x_coord='two_theta')
  structure.cif <- phase(s); multi-block if >1 phase (absent when no phases)

Multi-phase patterns keep all phases in one ``structure.cif`` with ``n_phases``
recording the count and ``quality_flag='multi_phase'``; the scalar cell/formula
columns describe the first phase only. Patterns with **no phase** (the HKUST-B
subset) are ingested as unlabelled: pattern + wavelength only, null structure and
labels, ``notes='unlabelled: no phase'``.

Run (from the package directory)::

    uv run python -m crystalai_data.xrddata.ingest_opxrd            # uses .env
    uv run python -m crystalai_data.xrddata.ingest_opxrd --limit 50
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path

from ..crystals._ingest import load_dotenv, repo_root
from ..crystals.symmetry import sg_to_cs
from . import database as db

_ELEMENT_RE = re.compile(r"^([A-Z][a-z]?)")

# Space-group derivation for opXRD (DATA_ROADMAP.md §2, spglib carve-out): opXRD
# ships P1-expanded structures with *no* reported space group, so — unlike the
# ICSD/MP-20 crystals path, which always reads a reported SG — symmetry must be
# derived here. We accept a derived SG only when two symprec settings agree, and
# leave the label null otherwise (thresholds chosen deliberately; see the
# symprec-sensitivity note in the ingestion write-up).
_SG_SYMPRECS = (0.01, 0.1)  # consensus pair; the looser value is the recorded one
_SG_ANGLE_TOL = 5.0


def _clean_wavelength(xray_info) -> float | None:
    if not xray_info:
        return None
    info = json.loads(xray_info) if isinstance(xray_info, str) else xray_info
    val = info.get("primary_wavelength")
    if val in (None, "None", ""):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _parse_lattice(lattice_str: str) -> tuple[float, ...]:
    """'(6.0948, 6.0953, 12.937, 90.0, 90.004, 90.0)' -> 6 floats (a,b,c,al,be,ga)."""
    nums = [float(t) for t in lattice_str.strip().strip("()").split(",")]
    if len(nums) != 6:
        raise ValueError(f"expected 6 lattice params, got {len(nums)}: {lattice_str!r}")
    return tuple(nums)


def _frac_close(u, v, tol: float) -> bool:
    """True if two fractional coords coincide within ``tol`` (periodic wrap)."""
    for a, b in zip(u, v):
        d = abs(a - b) % 1.0
        if min(d, 1.0 - d) > tol:
            return False
    return True


def _group_coincident(atoms: list[tuple[str, float, list[float]]], tol: float):
    """Group atoms sharing a position into one site each.

    opXRD P1-expanded phases list substitutional disorder as several atoms at the
    *same* fractional coordinate, each at occupancy 1.0 (opXRD drops the true
    fractions). Returns one ``(coord, {element: summed_occ})`` group per distinct
    position, summing occupancies of coincident atoms per element.
    """
    groups: list[dict] = []
    for el, occ, co in atoms:
        for g in groups:
            if _frac_close(g["coord"], co, tol):
                g["occ"][el] = g["occ"].get(el, 0.0) + occ
                break
        else:
            groups.append({"coord": co, "occ": {el: occ}})
    return groups


def _build_structures(phase: dict, cell: tuple[float, ...], *, tol: float = 1e-2):
    """Build the CIF structure and the SG-derivation proxy from an opXRD phase.

    ``cell`` is the already-parsed ``(a,b,c,al,be,ga)``. Oxidation-state decoration
    on the symbol (``Ba0+``, ``O2-``) is stripped to the bare element, matching the
    crystals-side convention. Returns ``(structure_cif, structure_sg, disordered)``:

    * ``structure_cif`` — coincident atoms merged into disordered sites, per-site
      occupancies renormalized to sum ≤ 1 (opXRD lists them at 1.0 each, so a
      two-species site becomes 0.5/0.5). Composition-faithful and round-trippable.
    * ``structure_sg`` — an *ordered geometric proxy*: one representative species
      (the majority) per distinct position. This is what ``SpacegroupAnalyzer``
      needs — a solid solution's space group is the symmetry of the averaged site
      lattice, which distinct species at one point would otherwise defeat.
    * ``disordered`` — True if any site carried >1 species (occupancy assumption
      applied). Raises if the phase carries no usable ``basis``.
    """
    from pymatgen.core import Lattice, Structure

    lattice = Lattice.from_parameters(*cell)
    basis = phase.get("basis")
    if not basis:
        raise ValueError("phase has no basis (lattice only)")

    atoms: list[tuple[str, float, list[float]]] = []
    for atom_str in json.loads(basis):
        atom = json.loads(atom_str) if isinstance(atom_str, str) else atom_str
        m = _ELEMENT_RE.match(str(atom["symbol"]))
        if not m:
            raise ValueError(f"unparseable atom symbol {atom.get('symbol')!r}")
        occ = atom.get("occupancy")
        occ = 1.0 if occ in (None, "", "None") else float(occ)
        atoms.append((m.group(1), occ, [float(atom["x"]), float(atom["y"]), float(atom["z"])]))
    if not atoms:
        raise ValueError("phase has empty basis")

    groups = _group_coincident(atoms, tol)
    disordered = any(len(g["occ"]) > 1 for g in groups)

    cif_species: list[dict[str, float]] = []
    sg_species: list[str] = []
    coords: list[list[float]] = []
    for g in groups:
        total = sum(g["occ"].values())
        scale = 1.0 / total if total > 1.0 else 1.0  # renormalize a >1 site to full
        cif_species.append({el: occ * scale for el, occ in g["occ"].items()})
        sg_species.append(max(g["occ"], key=g["occ"].get))  # majority species proxy
        coords.append(g["coord"])

    structure_cif = Structure(lattice, cif_species, coords)
    structure_sg = Structure(lattice, sg_species, coords)
    return structure_cif, structure_sg, disordered


def _derive_space_group(structure_sg) -> tuple[int | None, float | None]:
    """Derive a space group from the ordered proxy, or (None, None) if unresolved.

    Runs ``SpacegroupAnalyzer`` (spglib) at the ``_SG_SYMPRECS`` consensus pair and
    accepts the result only when both settings agree — the deliberate,
    conservative policy for a *derived* (not reported) label. Returns the SG number
    and the looser symprec it was confirmed at, so the threshold is recorded.
    """
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    results: dict[float, int | None] = {}
    for sp in _SG_SYMPRECS:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results[sp] = SpacegroupAnalyzer(
                    structure_sg, symprec=sp, angle_tolerance=_SG_ANGLE_TOL
                ).get_space_group_number()
        except Exception:  # noqa: BLE001 — an SGA failure just means no label
            results[sp] = None

    tight, loose = _SG_SYMPRECS
    if results[tight] is not None and results[tight] == results[loose]:
        return results[loose], loose
    return None, None


def _discover(env: dict[str, str]) -> list[tuple[str, Path]]:
    """Return (subsource, folder) pairs from the .env opXRD paths that exist."""
    out: list[tuple[str, Path]] = []
    cnrs = env.get("OPXRD_CNRS_DATA_FOLDER")
    if cnrs and Path(cnrs).is_dir():
        out.append(("CNRS", Path(cnrs)))
    hkust = env.get("OPXRD_HKUST_DATA_FOLDER")
    if hkust:
        for sub in ("HKUST-A", "HKUST-B"):
            d = Path(hkust) / sub
            if d.is_dir():
                out.append((sub, d))
    return out


def convert(*, subsources: list[tuple[str, Path]], store: Path, limit: int | None) -> None:
    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover
        def tqdm(x, **k):  # type: ignore
            return x

    print(f"[opxrd] store      = {store}")
    for name, folder in subsources:
        print(f"[opxrd] subsource  = {name}: {folder}")

    rows: list[db.IndexRow] = []
    stats = {
        "unlabelled": 0,
        "multi_phase": 0,
        "structure_ok": 0,
        "structure_fail": 0,
        "sg_derived": 0,
        "sg_unresolved": 0,
        "no_wavelength": 0,
        "skipped": 0,
    }

    jobs: list[tuple[str, Path]] = []
    for name, folder in subsources:
        for f in sorted(folder.glob("*.json")):
            jobs.append((name, f))
    if limit is not None:
        jobs = jobs[:limit]

    for subsource, jpath in tqdm(jobs, desc="[opxrd] ingest"):
        try:
            doc = json.loads(jpath.read_text())
            label = json.loads(doc["label"])
            xs = [float(v) for v in doc["two_theta_values"]]
            ys = [float(v) for v in doc["intensities"]]
        except Exception as exc:  # noqa: BLE001 — one bad file must not kill the run
            stats["skipped"] += 1
            print(f"[opxrd] skip {jpath.name}: {type(exc).__name__}: {exc}")
            continue
        if not xs:
            stats["skipped"] += 1
            continue

        source_id = f"opXRD_{subsource}_{jpath.stem}"
        wavelength = _clean_wavelength(label.get("xray_info"))
        if wavelength is None:
            stats["no_wavelength"] += 1

        pdir = db.pattern_dir(store, "opxrd", source_id)
        prov = ["source=opxrd", f"subsource={subsource}", f"opxrd_file={jpath.name}", "x_coord=two_theta"]
        if wavelength is not None:
            prov.append(f"wavelength_A={wavelength}")
        db.write_xy(pdir / "raw.xy", xs, ys, header=[*prov, "view=raw"])
        raw_rel = f"patterns/opxrd/{source_id}/raw.xy"

        phases = label.get("phases") or []
        n_phases = len(phases)

        cell = dict.fromkeys(("a", "b", "c", "alpha", "beta", "gamma"), None)
        formula: str | None = None
        cif_rel: str | None = None
        space_group: int | None = None
        crystal_system: int | None = None
        quality_flag = "clean"
        note_parts: list[str] = []

        if n_phases == 0:
            stats["unlabelled"] += 1
            note_parts.append("unlabelled: no phase")
        else:
            cif_blocks: list[str] = []
            first_cell = None
            first_sg_proxy = None
            first_disordered = False
            for i, ph_str in enumerate(phases):
                phase = json.loads(ph_str) if isinstance(ph_str, str) else ph_str
                # The cell is recorded from the lattice string even when the phase
                # ships no atomic basis (structure build below then fails cleanly).
                params = None
                try:
                    params = _parse_lattice(phase["lattice"])
                    if i == 0:
                        first_cell = params
                except Exception:  # noqa: BLE001 — a malformed lattice just means null cell
                    pass
                try:
                    if params is None:
                        raise ValueError("phase has no parseable lattice")
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        st_cif, st_sg, disordered = _build_structures(phase, params)
                        cif_blocks.append(st_cif.to(fmt="cif"))
                    if i == 0:
                        formula = st_cif.composition.reduced_formula
                        first_sg_proxy = st_sg
                        first_disordered = disordered
                except Exception as exc:  # noqa: BLE001
                    note_parts.append(f"phase {i} structure build failed: {type(exc).__name__}: {exc}")
                    stats["structure_fail"] += 1
                    continue

            if cif_blocks:
                db.write_cif(pdir / "structure.cif", "\n".join(cif_blocks))
                cif_rel = f"patterns/opxrd/{source_id}/structure.cif"
                stats["structure_ok"] += 1
            if first_cell is not None:
                a, b, c, al, be, ga = first_cell
                cell.update(a=a, b=b, c=c, alpha=al, beta=be, gamma=ga)
            if first_disordered:
                note_parts.append("disordered: coincident-site occupancies renormalized")

            # Derive the space group (major/first phase) — opXRD ships none.
            if first_sg_proxy is not None:
                sg, symprec = _derive_space_group(first_sg_proxy)
                if sg is not None:
                    space_group = sg
                    crystal_system = sg_to_cs(sg)
                    note_parts.append(f"sg derived (spglib, symprec={symprec})")
                    stats["sg_derived"] += 1
                else:
                    stats["sg_unresolved"] += 1

            if n_phases > 1:
                quality_flag = "multi_phase"
                stats["multi_phase"] += 1

        rows.append(
            db.IndexRow(
                source="opxrd",
                source_id=source_id,
                raw_path=raw_rel,
                bgsub_path=None,
                autobg=None,  # filled by the background pass
                peaks_path=None,
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
                n_phases=n_phases,
                cif_id=None,
                cif_path=cif_rel,
                quality_flag=quality_flag,
                notes="; ".join(note_parts) or None,
            )
        )

    written = db.write_source_rows(store, "opxrd", rows)
    print(f"\n[opxrd] patterns written = {written:,}")
    print(f"[opxrd] structures ok={stats['structure_ok']:,}  build-fail={stats['structure_fail']:,}  multi-phase={stats['multi_phase']:,}")
    print(f"[opxrd] SG derived={stats['sg_derived']:,}  SG unresolved={stats['sg_unresolved']:,} (consensus symprec {_SG_SYMPRECS})")
    print(f"[opxrd] unlabelled={stats['unlabelled']:,}  missing wavelength={stats['no_wavelength']:,}  skipped={stats['skipped']:,}")


def main(argv: list[str] | None = None) -> None:
    root = repo_root()
    env = load_dotenv(root / ".env")
    store_default = str(db.resolve_store(env, root))

    p = argparse.ArgumentParser(description="Ingest opXRD into the experimental store")
    p.add_argument("--store", default=store_default, type=str)
    p.add_argument("--limit", default=None, type=int, help="process at most N patterns")
    args = p.parse_args(argv)

    subsources = _discover(env)
    if not subsources:
        raise SystemExit("[opxrd] no opXRD folders found in .env (OPXRD_*_DATA_FOLDER)")

    convert(subsources=subsources, store=Path(args.store), limit=args.limit)


if __name__ == "__main__":
    main()
