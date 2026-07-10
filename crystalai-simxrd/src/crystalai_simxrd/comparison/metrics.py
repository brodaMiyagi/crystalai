"""Pattern-agreement metrics (simulated vs experimental, same grid)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def best_scale(obs: NDArray, calc: NDArray) -> float:
    """Least-squares scale ``s`` minimizing ‖obs − s·calc‖² (0 if calc ≡ 0)."""
    o, c = np.asarray(obs, float), np.asarray(calc, float)
    denom = float(np.dot(c, c))
    return float(np.dot(o, c) / denom) if denom > 0 else 0.0


def rwp(obs: NDArray, calc: NDArray, weights: NDArray | None = None) -> float:
    """Weighted profile R-factor ``√(Σw(obs−calc)² / Σw·obs²)`` (%).

    Default weight is counting-statistics ``1/obs``, but **floored at 1% of max(obs)**
    so it does not blow up in the near-zero valleys of a max-normalized pattern
    (true Rietveld Rwp uses count data that never reaches zero).
    """
    o, c = np.asarray(obs, float), np.asarray(calc, float)
    if weights is None:
        floor = 0.01 * float(np.max(o)) if o.size else 1.0
        w = 1.0 / np.clip(o, max(floor, 1e-12), None)
    else:
        w = np.asarray(weights, float)
    num = float(np.sum(w * (o - c) ** 2))
    den = float(np.sum(w * o**2))
    return 100.0 * np.sqrt(num / den) if den > 0 else np.inf


def rwp_counts(obs: NDArray, calc: NDArray) -> tuple[float, float]:
    """Rietveld weighted-profile R (%) on the **count scale**, with weighted-LS scaling.

    ``obs`` is in counts (background-subtracted or raw), weight ``w = 1/max(obs, 1)``
    (Poisson). Returns ``(Rwp%, scale)`` where scale minimizes ``Σw(obs − s·calc)²``.
    Note the identity ``Rwp = √(1 − cos²_w)``: this metric is dominated by the *weak*
    points the ``1/obs`` weight amplifies, which is why a forward simulation with good
    peak agreement (high unweighted cosine) can still show a large Rwp.
    """
    o = np.clip(np.asarray(obs, float), 0, None)
    c = np.asarray(calc, float)
    w = 1.0 / np.clip(o, 1.0, None)
    denom_c = float(np.sum(w * c * c))
    s = float(np.sum(w * o * c) / denom_c) if denom_c > 0 else 0.0
    num = float(np.sum(w * (o - s * c) ** 2))
    den = float(np.sum(w * o * o))
    return (100.0 * np.sqrt(num / den) if den > 0 else np.inf), s


def rp(obs: NDArray, calc: NDArray) -> float:
    """Unweighted profile R-factor ``Σ|obs−calc| / Σ|obs|`` (%)."""
    o, c = np.asarray(obs, float), np.asarray(calc, float)
    den = float(np.sum(np.abs(o)))
    return 100.0 * float(np.sum(np.abs(o - c))) / den if den > 0 else np.inf


def rietveld_r_bragg(
    two_theta: NDArray, obs_net: NDArray, peak_two_theta: NDArray, peak_intensity: NDArray,
    peak_fwhm_g: NDArray, peak_fwhm_l: NDArray, *, window_fwhm: float = 8.0,
) -> float:
    """Rietveld-partition Bragg R-factor (%): ``Σₖ|Iₖ,obs − Iₖ,calc| / Σₖ Iₖ,calc``.

    The crystallographic R_Bragg. Each observed point is apportioned among reflections by each
    reflection's *fractional calculated contribution* (Rietveld, J. Appl. Cryst. 2, 65 (1969)):

        ``Iₖ,obs = Σᵢ [ Sₖ(2θᵢ) / Σₖ' Sₖ'(2θᵢ) ] · yₒᵦₛ,ᵢ``

    where ``Sₖ`` is reflection *k*'s area-normalized (TCH) pseudo-Voigt scaled by its calculated
    integrated intensity ``Iₖ,calc``, and ``yₒᵦₛ`` is the **background-subtracted** observed
    profile. ``obs_net`` and ``peak_intensity`` must be on the same count scale (i.e. pass
    ``scale·sim.peak_intensities``). Overlaps are handled by the partition; because obs is
    apportioned by the calc shape, this is far less inflated by peak width / small position
    offsets / background residual than a fixed-window integral. A high value then genuinely
    means ``|F|²`` disagreement (structure-factor bug, or real texture/absorption).
    """
    from ..profiles.peak_shapes import pseudo_voigt, tch_mix

    tt = np.asarray(two_theta, float)
    y = np.asarray(obs_net, float)
    tt_k = np.asarray(peak_two_theta, float)
    ical = np.asarray(peak_intensity, float)
    in_range = (tt_k >= tt[0]) & (tt_k <= tt[-1]) & (ical > 0)
    if not in_range.any():
        return np.inf

    s_total = np.zeros_like(tt)
    windows: list[tuple[int, int, NDArray]] = []
    for k in np.flatnonzero(in_range):
        fwhm, eta = tch_mix(float(peak_fwhm_g[k]), float(peak_fwhm_l[k]))
        if fwhm <= 0:
            windows.append((0, 0, np.empty(0)))
            continue
        lo = int(np.searchsorted(tt, tt_k[k] - window_fwhm * fwhm))
        hi = int(np.searchsorted(tt, tt_k[k] + window_fwhm * fwhm))
        s = ical[k] * pseudo_voigt(tt[lo:hi], float(tt_k[k]), fwhm, eta)
        s_total[lo:hi] += s
        windows.append((lo, hi, s))

    num = den = 0.0
    for idx in range(len(windows)):
        lo, hi, s = windows[idx]
        if hi <= lo or s.size == 0:
            continue
        # I_calc,k as the calc-profile sum over points (same basis as the obs partition sum,
        # so the pseudo-Voigt per-degree normalization / grid step cancels)
        icalc = float(s.sum())
        frac = s / (s_total[lo:hi] + 1e-30)
        iobs = float(np.sum(frac * y[lo:hi]))
        num += abs(iobs - icalc)
        den += icalc
    return 100.0 * num / den if den > 0 else np.inf


def rwp_noise_floor(
    y_calc: NDArray, weights: NDArray, rng: np.random.Generator | None = None,
    n_rep: int = 8, sigma_rel: float = 0.0,
) -> float:
    """Expected Rwp of a **perfect** model under the pattern's own counting noise (%).

    Realizes Poisson counting noise (and optional relative Gaussian ``σrel``) on the clean
    count-scale model and returns ``mean Rwp(y_calc, noisy)`` over ``n_rep`` draws — the
    ``R_expected`` floor. If the observed Rwp is close to this, the fit is noise-limited
    (a good model measured on a noisy pattern), not defective.
    """
    rng = rng or np.random.default_rng(0)
    yc = np.clip(np.asarray(y_calc, float), 0, None)
    vals = []
    for _ in range(max(n_rep, 1)):
        noisy = rng.poisson(yc).astype(float)
        if sigma_rel > 0:
            noisy = noisy + rng.normal(0.0, sigma_rel * yc)
        vals.append(rwp(yc, noisy, weights=weights))
    return float(np.mean(vals))


def cosine_similarity(a: NDArray, b: NDArray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0


def peak_position_rmsd(obs_positions: NDArray, calc_positions: NDArray, tol: float = np.inf) -> float:
    """RMSD between each observed peak position and its nearest calculated one (within ``tol``)."""
    o, c = np.asarray(obs_positions, float), np.asarray(calc_positions, float)
    if o.size == 0 or c.size == 0:
        return np.nan
    errs = [abs(c - x).min() for x in o]
    errs = [e for e in errs if e <= tol]
    return float(np.sqrt(np.mean(np.square(errs)))) if errs else np.nan
