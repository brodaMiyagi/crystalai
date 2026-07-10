"""Standard X-ray wavelengths for powder diffraction.

Values from International Tables for Crystallography, Vol. C (2004) and Hölzer
et al., Phys. Rev. A 56, 4554 (1997). All in Ångströms. The full table (per-anode
Kα1/Kα2/Kα-avg/Kβ1 + synchrotron beamlines) is loaded lazily from
``data/reference/wavelengths.yaml``; the most-used lines are exposed as constants.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import yaml

_REFERENCE_DIR = Path(__file__).resolve().parents[3] / "data" / "reference"

_FULL_TABLE: dict | None = None


def _load_table() -> dict:
    global _FULL_TABLE
    if _FULL_TABLE is None:
        with open(_REFERENCE_DIR / "wavelengths.yaml") as f:
            _FULL_TABLE = yaml.safe_load(f)
    return _FULL_TABLE


def get_wavelength(source: str, line: str = "Kalpha1") -> float:
    """Wavelength (Å) for an anode ``source`` (e.g. ``'Cu'``) and emission ``line``
    (``'Kalpha1'`` | ``'Kalpha2'`` | ``'Kalpha_avg'`` | ``'Kbeta1'``)."""
    table = _load_table()
    try:
        return float(table["sources"][source][line])
    except KeyError:
        available = list(table["sources"].keys())
        raise KeyError(
            f"Unknown source/line: {source}/{line}. Available sources: {available}"
        ) from None


def get_synchrotron_wavelength(beamline: str) -> float:
    """Wavelength (Å) for a named synchrotron beamline."""
    table = _load_table()
    try:
        return float(table["synchrotron"][beamline])
    except KeyError:
        available = list(table["synchrotron"].keys())
        raise KeyError(f"Unknown beamline: {beamline}. Available: {available}") from None


# Commonly used constants (Cu Kα is the lab default).
CU_KA1: Final[float] = 1.5405980
CU_KA2: Final[float] = 1.5444260
CU_KA_AVG: Final[float] = 1.5418740
MO_KA1: Final[float] = 0.7093187
CO_KA1: Final[float] = 1.7889960
