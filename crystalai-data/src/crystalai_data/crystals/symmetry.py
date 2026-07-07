"""Space-group / crystal-system lookup utilities — pure lookup, no spglib.

These are the single source of truth for the SG<->CS<->symbol relationships the
rest of the package relies on. Crystal system is derived from the space group by
the standard IT-number range partition (``sg_to_cs``), the same partition the
methods package's CS-gated SG head assumes — so the data layer and the model
agree by construction (DATA_ROADMAP.md §1, METHODS_ROADMAP.md §2.1).

Hermann-Mauguin symbols come from pymatgen's internal space-group tables
(``pymatgen.symmetry.groups``), which are plain data lookups and do **not** pull
in spglib.
"""

from __future__ import annotations

import re

from pymatgen.symmetry.groups import SpaceGroup, sg_symbol_from_int_number

__all__ = [
    "sg_to_cs",
    "cs_number_to_name",
    "sg_number_to_symbol",
    "hm_symbol_to_sg_number",
]

# Crystal-system integer -> name (1 = triclinic ... 7 = cubic).
_CS_NAMES = {
    1: "triclinic",
    2: "monoclinic",
    3: "orthorhombic",
    4: "tetragonal",
    5: "trigonal",
    6: "hexagonal",
    7: "cubic",
}

# Inclusive SG-number upper bound -> crystal-system integer, in ascending order.
# triclinic 1-2, monoclinic 3-15, orthorhombic 16-74, tetragonal 75-142,
# trigonal 143-167, hexagonal 168-194, cubic 195-230.
_CS_RANGES = (
    (2, 1),
    (15, 2),
    (74, 3),
    (142, 4),
    (167, 5),
    (194, 6),
    (230, 7),
)


def sg_to_cs(sg: int) -> int:
    """Space-group number (1-230) -> crystal-system number (1-7).

    Raises ``ValueError`` for out-of-range input so a bad label fails loudly at
    ingest rather than silently mislabelling a structure.
    """
    if not isinstance(sg, int) or not 1 <= sg <= 230:
        raise ValueError(f"space group must be an int in 1..230, got {sg!r}")
    for upper, cs in _CS_RANGES:
        if sg <= upper:
            return cs
    raise AssertionError("unreachable")  # pragma: no cover


def cs_number_to_name(cs: int) -> str:
    """Crystal-system number (1-7) -> name."""
    try:
        return _CS_NAMES[cs]
    except KeyError:
        raise ValueError(f"crystal system must be an int in 1..7, got {cs!r}") from None


def sg_number_to_symbol(sg: int) -> str:
    """Space-group number (1-230) -> Hermann-Mauguin symbol (e.g. 'Fd-3m')."""
    if not isinstance(sg, int) or not 1 <= sg <= 230:
        raise ValueError(f"space group must be an int in 1..230, got {sg!r}")
    return sg_symbol_from_int_number(sg)


def _normalize_hm(symbol: str) -> str:
    """Collapse a Hermann-Mauguin symbol to a settings-agnostic comparison key.

    Drops whitespace, underscores, quotes and the ``:1``/``:2`` origin-choice
    suffix so notational variants (``P2_1/c`` vs ``P21/c``, ``F d -3 m`` vs
    ``Fd-3m``) compare equal.
    """
    s = symbol.strip().strip("'\"")
    s = s.split(":", 1)[0]  # drop origin-choice / setting suffix
    s = re.sub(r"\s+", "", s)
    s = s.replace("_", "")
    return s


def _build_hm_norm_map() -> dict[str, int]:
    """Normalized H-M symbol -> SG number, covering short/full/universal settings.

    pymatgen's ``SpaceGroup.SYMM_OPS`` lists every tabulated setting with its
    short, full and universal H-M strings; normalizing all of them lets variants
    like ``P2_1/c`` (short) and ``P12_1/c1`` (full) both resolve. The canonical
    int-number symbols are overlaid last so they win any normalization clash.
    """
    out: dict[str, int] = {}
    for entry in SpaceGroup.SYMM_OPS:
        num = int(entry["number"])
        for key in ("short_h_m", "hermann_mauguin", "universal_h_m", "hermann_mauguin_u"):
            sym = entry.get(key)
            if sym:
                out.setdefault(_normalize_hm(sym), num)
    for n in range(1, 231):
        out[_normalize_hm(sg_symbol_from_int_number(n))] = n
    return out


_HM_NORM_TO_SG = _build_hm_norm_map()


def hm_symbol_to_sg_number(symbol: str) -> int:
    """Hermann-Mauguin symbol -> space-group number (1-230).

    Internal helper, used only as the CIF fallback when ``_space_group_IT_number``
    is absent (DATA_ROADMAP.md §1, conversion pipeline). H-M symbols have
    notational variants, so matching goes through a normalized key rather than
    raw string comparison. Raises ``ValueError`` when the symbol can't be
    resolved, so the caller can fall back further or skip rather than guess.
    """
    if not symbol or not symbol.strip():
        raise ValueError("empty Hermann-Mauguin symbol")

    norm = _normalize_hm(symbol)
    if norm in _HM_NORM_TO_SG:
        return _HM_NORM_TO_SG[norm]

    # Trailing single-letter origin markers ICSD sometimes appends (e.g. the
    # 'S'/'Z' in 'F d -3 m S'): retry once with it stripped.
    if norm[-1:] in {"S", "Z", "O"} and norm[:-1] in _HM_NORM_TO_SG:
        return _HM_NORM_TO_SG[norm[:-1]]

    # Last resort: let pymatgen's own (more permissive) parser try.
    try:
        return SpaceGroup(_normalize_hm(symbol)).int_number
    except Exception:
        raise ValueError(f"unrecognized Hermann-Mauguin symbol: {symbol!r}") from None
