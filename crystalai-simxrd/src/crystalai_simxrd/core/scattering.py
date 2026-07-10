"""Cromer-Mann atomic scattering factors.

The 9-parameter analytical approximation
    f(s) = Σ_{i=1..4} a_i · exp(-b_i · s²) + c ,   s = sinθ/λ  (Å⁻¹)
from International Tables for Crystallography, Vol. C, Table 6.1.1.4 (Prince, 2004).
``s`` (= 1/2d) is wavelength-independent, so f(s) — and hence |F(hkl)|² — is
wavelength-independent, the fact the precompute boundary rests on (DESIGN_DECISIONS §9).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

_REFERENCE_DIR = Path(__file__).resolve().parents[3] / "data" / "reference"

_COEFFICIENTS: dict[str, list[float]] | None = None


def _load_coefficients() -> dict[str, list[float]]:
    global _COEFFICIENTS
    if _COEFFICIENTS is None:
        with open(_REFERENCE_DIR / "cromer_mann.json") as f:
            data = json.load(f)
        _COEFFICIENTS = {k: v for k, v in data.items() if not k.startswith("_")}
    return _COEFFICIENTS


def get_coefficients(element: str) -> tuple[NDArray, NDArray, float]:
    """Return ``(a[4], b[4], c)`` Cromer-Mann coefficients for ``element``."""
    table = _load_coefficients()
    try:
        coeffs = table[element]
    except KeyError:
        raise KeyError(f"No Cromer-Mann coefficients for element: {element!r}") from None
    a = np.array([coeffs[0], coeffs[2], coeffs[4], coeffs[6]])
    b = np.array([coeffs[1], coeffs[3], coeffs[5], coeffs[7]])
    c = coeffs[8]
    return a, b, c


def scattering_factor(element: str, s: NDArray) -> NDArray:
    """Atomic scattering factor ``f(s)`` for one element over an array of ``s`` (Å⁻¹)."""
    a, b, c = get_coefficients(element)
    s2 = np.asarray(s, dtype=np.float64) ** 2
    return np.sum(a[:, None] * np.exp(-b[:, None] * s2[None, :]), axis=0) + c
