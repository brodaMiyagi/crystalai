"""Bragg reflections with disorder-aware structure factors.

Two-stage design matching the precompute boundary (DESIGN_DECISIONS §9):

* :func:`compute_peak_list` — the **wavelength-independent** d-space reflection list
  ``{d, |F|²}`` (occupancy-weighted structure factors, merged by d-spacing). This is
  what gets precomputed once per structure.
* :func:`to_two_theta` — maps that list to 2θ at a working wavelength and applies the
  Lorentz-polarization factor, giving the integrated intensities that feed the profile
  build. Wavelength-dependent, run on-the-fly.

Disorder: ``F(hkl) = Σ_j [Σ_s c_{j,s} f_s(1/2d)] · exp(2πi·hkl·r_j)`` with fractional
site occupancies ``c_{j,s}`` — the kinematic-diffraction-correct treatment; disordered
structures are **never** supercell-ordered before computing peaks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from pymatgen.core import Structure

from ..effects.lorentz_polarization import lorentz_polarization_factor
from .scattering import scattering_factor


@dataclass(slots=True)
class BraggPeaks:
    """Wavelength-independent d-space reflection list (merged by d-spacing).

    ``f_squared[i]`` is the sum of ``|F(hkl)|²`` over all reflections at ``d_spacings[i]``
    (so symmetry multiplicity and accidental d-overlaps are folded in — the powder sum).
    ``hkls[i]`` is a representative reflection for that d-group (its dominant ``hkl``,
    useful for preferred orientation).
    """

    d_spacings: NDArray            # Å, ascending
    f_squared: NDArray            # |F|² summed per d-group
    hkls: list[tuple[int, int, int]]
    metric_tensor: NDArray | None = None  # direct-lattice metric (for preferred orientation)

    def __len__(self) -> int:
        return len(self.d_spacings)


# --------------------------------------------------------------------------- #
# hkl generation
# --------------------------------------------------------------------------- #
def _hkl_candidates(reciprocal_lattice: NDArray, d_min: float) -> tuple[NDArray, NDArray]:
    """All ``(h,k,l) ≠ 0`` with d ≥ ``d_min``; returns (hkls, d_spacings)."""
    g_norms = np.linalg.norm(reciprocal_lattice, axis=1)
    max_idx = np.ceil((1.0 / d_min) / g_norms).astype(int) + 1

    ranges = [np.arange(-m, m + 1) for m in max_idx]
    hh, kk, ll = np.meshgrid(*ranges, indexing="ij")
    hkls = np.stack([hh.ravel(), kk.ravel(), ll.ravel()], axis=1)
    hkls = hkls[np.any(hkls != 0, axis=1)]  # drop (0,0,0)

    q = hkls @ reciprocal_lattice          # (N, 3) reciprocal vectors
    q_len = np.linalg.norm(q, axis=1)
    d = 1.0 / q_len
    keep = d >= d_min
    return hkls[keep], d[keep]


# --------------------------------------------------------------------------- #
# Structure factors (element-grouped: f(s) evaluated once per element, not per site)
# --------------------------------------------------------------------------- #
def _structure_factors(structure: Structure, hkls: NDArray, s_values: NDArray) -> NDArray:
    """|F(hkl)|² for all reflections, occupancy-weighted over (possibly mixed) sites."""
    frac = np.array([site.frac_coords for site in structure])        # (n_sites, 3)
    phases = np.exp(2j * np.pi * (hkls @ frac.T))                    # (N_ref, n_sites)

    # Accumulate occupancy per element per site (disorder: a site may hold several).
    n_sites = len(structure)
    elem_occ: dict[str, NDArray] = {}
    for j, site in enumerate(structure):
        for sp, occ in site.species.items():
            elem_occ.setdefault(sp.symbol, np.zeros(n_sites))[j] += occ

    # f(s) once per element; F = Σ_element f_el(s) · (Σ_sites occ · phase).
    f_hkl = np.zeros(len(hkls), dtype=np.complex128)
    for el, occ_vec in elem_occ.items():
        f_el = scattering_factor(el, s_values)          # (N_ref,)
        f_hkl += f_el * (phases @ occ_vec)              # (N_ref,)
    return np.abs(f_hkl) ** 2


# --------------------------------------------------------------------------- #
# Merge reflections by d-spacing (powder overlap)
# --------------------------------------------------------------------------- #
def _merge_by_d(
    hkls: NDArray, d: NDArray, f2: NDArray, tol: float = 1e-5
) -> tuple[list[tuple[int, int, int]], NDArray, NDArray]:
    """Sum |F|² over reflections sharing a d-spacing; keep the strongest hkl as label."""
    order = np.argsort(d)
    d_s, f2_s, hkl_s = d[order], f2[order], hkls[order]

    hkl_rep: list[tuple[int, int, int]] = []
    d_out: list[float] = []
    f2_out: list[float] = []

    i, n = 0, len(d_s)
    while i < n:
        j = i + 1
        while j < n and (d_s[j] - d_s[i]) / d_s[i] < tol:
            j += 1
        group_f2 = f2_s[i:j]
        d_out.append(float(np.mean(d_s[i:j])))
        f2_out.append(float(np.sum(group_f2)))
        hkl_rep.append(tuple(int(x) for x in hkl_s[i + int(np.argmax(group_f2))]))
        i = j
    return hkl_rep, np.array(d_out), np.array(f2_out)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def compute_peak_list(
    structure: Structure, d_min: float = 0.7, f2_rel_cutoff: float = 1e-6
) -> BraggPeaks:
    """Wavelength-independent d-space reflection list down to ``d_min`` (Å).

    ``f2_rel_cutoff`` drops reflections below that fraction of the strongest |F|².
    """
    recip = structure.lattice.reciprocal_lattice_crystallographic.matrix
    metric = np.asarray(structure.lattice.metric_tensor, dtype=np.float64)
    hkls, d = _hkl_candidates(recip, d_min * 0.999)
    if len(hkls) == 0:
        return BraggPeaks(np.array([]), np.array([]), [], metric)

    s_values = 1.0 / (2.0 * d)                       # sinθ/λ = 1/2d (wavelength-free)
    f2 = _structure_factors(structure, hkls, s_values)
    hkl_rep, d_m, f2_m = _merge_by_d(hkls, d, f2)

    if len(f2_m):
        keep = f2_m > f2_m.max() * f2_rel_cutoff
        d_m, f2_m = d_m[keep], f2_m[keep]
        hkl_rep = [h for h, k in zip(hkl_rep, keep) if k]

    order = np.argsort(d_m)                            # ascending d
    return BraggPeaks(d_m[order], f2_m[order], [hkl_rep[i] for i in order], metric)


def to_two_theta(
    peaks: BraggPeaks,
    wavelength: float,
    two_theta_range: tuple[float, float] | None = None,
    *,
    apply_lp: bool = True,
) -> tuple[NDArray, NDArray, NDArray, list[tuple[int, int, int]]]:
    """Map a d-space peak list to 2θ at ``wavelength`` → ``(two_theta, d, intensity, hkls)``.

    Only reflections with ``d ≥ λ/2`` are physically accessible (``sinθ ≤ 1``); those and
    anything outside ``two_theta_range`` (degrees) are dropped. ``intensity = |F|²·LP(2θ)``
    when ``apply_lp`` (the powder-integrated intensity, matching pymatgen's convention).
    """
    d = peaks.d_spacings
    if len(d) == 0:
        return (np.array([]), np.array([]), np.array([]), [])

    sin_theta = wavelength / (2.0 * d)
    ok = sin_theta <= 1.0
    two_theta = np.full_like(d, np.nan)
    two_theta[ok] = np.rad2deg(2.0 * np.arcsin(sin_theta[ok]))

    mask = ok
    if two_theta_range is not None:
        lo, hi = two_theta_range
        mask = mask & (two_theta >= lo) & (two_theta <= hi)

    tt, dd, f2 = two_theta[mask], d[mask], peaks.f_squared[mask]
    hkls = [h for h, m in zip(peaks.hkls, mask) if m]
    intensity = f2 * lorentz_polarization_factor(tt) if apply_lp else f2
    return tt, dd, intensity, hkls
