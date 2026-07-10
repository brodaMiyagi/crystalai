"""CIF / structure helpers — thin wrappers around pymatgen ``Structure``.

The pymatgen ``Structure`` is the primary data type throughout the pipeline.

**No spglib here.** Space groups are read from the source (crystals.sqlite / CIF),
never re-derived on the simulation path — the ``SpacegroupAnalyzer`` helper the old
difsim ``crystal.py`` carried is intentionally dropped (root CLAUDE.md hard
constraint; the one sanctioned spglib use is opXRD SG derivation in crystalai-data).
"""

from __future__ import annotations

import warnings
from pathlib import Path

from pymatgen.core import Structure


def load_cif(path: str | Path, *, primitive: bool = False) -> Structure:
    """Load a CIF into a pymatgen ``Structure`` (parse warnings suppressed).

    ``primitive=False`` by default: the conventional cell is kept as-parsed, matching
    the crystals-DB "cell as parsed" convention (DATA_ROADMAP §1) and pymatgen's
    ``XRDCalculator`` reference, so validation comparisons line up.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Structure.from_file(str(path), primitive=primitive)


def is_disordered(structure: Structure) -> bool:
    """True if any site has fractional occupancy or mixed species."""
    return not structure.is_ordered
