"""Simulator — CIF/Structure (or cached peak list) → convolved pattern.

Phase 1: Bragg peaks → 2θ assembly (TCH-PV, Caglioti, LP) → single log-d conversion.
Phase 2: optional physical effects via `EffectConfig`, applied in the SIMXRD_ROADMAP
§5 order — Debye-Waller + preferred orientation on the integrated intensities;
size/strain broadening on the widths; zero-shift on the centres; axial-divergence and
slit on the assembled 2θ pattern; residual background in 2θ; Poisson counting noise
(2θ, on total counts) then full-profile Gaussian noise (2θ, on the max-normalized
pattern — SIMXRD_ROADMAP §5a) — all **before** the single conversion to the output
domain, so d/log-d outputs inherit noise added in 2θ rather than adding it natively
post-conversion. Training-time augmentation (`ProfileAugmentor`, Phase 3) samples these
same `EffectConfig` fields — including `gaussian_noise_std` for the σrel noise-floor
conditioning pair — and simulates through this same 2θ-first path; it adds nothing
noise-related natively in log-d.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from pymatgen.core import Structure

from ..core.bragg import BraggPeaks, compute_peak_list, to_two_theta
from ..core.domain import Domain
from ..effects.axial_divergence import apply_axial_divergence
from ..effects.background import physical_background, sample_residual_background
from ..effects.broadening import size_strain_fwhm
from ..effects.noise import poisson_counting, relative_gaussian
from ..effects.preferred_orientation import preferred_orientation_correction
from ..effects.slit import apply_slit
from ..effects.thermal import debye_waller_envelope
from ..effects.zero_shift import apply_zero_shift
from ..profiles.caglioti import SIMXRD_DEFAULT, InstrumentParameters, instrument_fwhm
from ..profiles.convolver import convolve_two_theta, resample_two_theta_to_log_d
from ..utils.binning import make_log_d_grid, make_two_theta_grid
from ..utils.normalization import normalize


@dataclass(slots=True)
class EffectConfig:
    """Toggleable physical effects (Phase 2). Defaults = identity (no effect)."""

    b_iso: float = 0.0                                 # Debye-Waller (Å²); 0 = off
    crystallite_size_nm: float = np.inf                # Scherrer size; ∞ = off
    microstrain: float = 0.0                           # Williamson-Hall strain
    march_r: float = 1.0                               # preferred orientation; 1 = random
    march_axis: tuple[int, int, int] = (0, 0, 1)
    zero_shift_deg: float = 0.0                        # global 2θ offset
    axial_divergence: bool = False
    axial_L: float = 200.0                             # detector distance (mm)
    axial_H: float = 2.0                               # source/sample half-height (mm)
    axial_S: float = 2.0                               # receiving-slit half-height (mm)
    slit_width_deg: float = 0.0                        # explicit slit; 0 = off
    background: str | None = None                      # None | 'residual' | 'physical'
    background_rel_amplitude: float = 0.03
    poisson_lambda_max: float | None = None            # counting noise (2θ); None = off
    gaussian_noise_mean: float = 0.0                    # full-profile noise (2θ, [0,1]); §5a
    gaussian_noise_std: float = 0.0                     # 0 = off
    rng_seed: int | None = None


@dataclass(slots=True)
class SimulatedPattern:
    """A simulated pattern plus the peak list and metadata behind it."""

    x_axis: NDArray                       # 2θ (deg) or log₁₀(d)
    intensity: NDArray
    domain: Domain
    wavelength: float
    peak_positions_d: NDArray
    peak_intensities: NDArray             # integrated |F|²·LP·(DW·PO) per reflection
    peak_two_theta: NDArray | None = None  # reflection centres (deg, incl. zero-shift)
    peak_fwhm_g: NDArray | None = None     # per-reflection Gaussian FWHM (deg)
    peak_fwhm_l: NDArray | None = None     # per-reflection Lorentzian FWHM (deg)
    metadata: dict = field(default_factory=dict)


def simulate(
    source: Structure | BraggPeaks,
    wavelength: float,
    *,
    domain: Domain = Domain.LOG_D,
    instrument: InstrumentParameters = SIMXRD_DEFAULT,
    effects: EffectConfig | None = None,
    two_theta_range: tuple[float, float] = (5.0, 150.0),
    d_window: tuple[float, float] = (0.7, 18.0),
    n_log_d_bins: int = 12000,
    internal_step: float = 0.01,
    min_fwhm_2theta: float = 0.02,
    d_min: float = 0.7,
) -> SimulatedPattern:
    """Simulate a pattern in ``domain`` (LOG_D production / TWO_THETA validation).

    ``source`` is a pymatgen ``Structure`` (Bragg computed now) or a cached
    ``BraggPeaks`` (skip to profile build — the training path). ``effects=None``
    gives the clean Phase-1 pattern.
    """
    cfg = effects or EffectConfig()
    rng = np.random.default_rng(cfg.rng_seed)
    peaks = source if isinstance(source, BraggPeaks) else compute_peak_list(source, d_min=d_min)
    tt, d, inten, hkls = to_two_theta(peaks, wavelength, two_theta_range)

    # --- intensity-domain effects (on integrated |F|²·LP) ---
    if cfg.b_iso > 0:
        inten = inten * debye_waller_envelope(d, cfg.b_iso)
    if abs(cfg.march_r - 1.0) > 1e-9 and peaks.metric_tensor is not None:
        inten = inten * preferred_orientation_correction(
            hkls, peaks.metric_tensor, cfg.march_r, cfg.march_axis
        )

    # --- 2θ assembly ---
    tt_centres = apply_zero_shift(tt, cfg.zero_shift_deg)
    fwhm_g, fwhm_l = instrument_fwhm(tt_centres, instrument)
    extra_g, extra_l = size_strain_fwhm(
        tt_centres, wavelength, cfg.crystallite_size_nm, cfg.microstrain
    )
    fwhm_g = np.sqrt(fwhm_g**2 + extra_g**2)
    fwhm_l = fwhm_l + extra_l
    fwhm_g = np.maximum(fwhm_g, min_fwhm_2theta)

    tt_grid = make_two_theta_grid(two_theta_range[0], two_theta_range[1], internal_step)
    pattern_tt = convolve_two_theta(tt_centres, inten, fwhm_g, fwhm_l, tt_grid.values)

    # --- 2θ instrument effects on the assembled pattern ---
    if cfg.axial_divergence:
        pattern_tt = apply_axial_divergence(
            tt_grid.values, pattern_tt, tt_centres,
            L=cfg.axial_L, H=cfg.axial_H, S=cfg.axial_S,
        )
    if cfg.slit_width_deg > 0:
        pattern_tt = apply_slit(pattern_tt, internal_step, cfg.slit_width_deg)
    if cfg.background == "residual":
        pattern_tt = pattern_tt + sample_residual_background(
            tt_grid.values, float(pattern_tt.max()),
            rel_amplitude=cfg.background_rel_amplitude, rng=rng,
        )
    elif cfg.background == "physical":
        pattern_tt = pattern_tt + physical_background(
            tt_grid.values, wavelength, peak_max=float(pattern_tt.max()),
        )
    # counting (Poisson) noise: count-domain, on total 2θ counts, before conversion
    if cfg.poisson_lambda_max is not None:
        pattern_tt = poisson_counting(pattern_tt, cfg.poisson_lambda_max, rng)

    # full-profile Gaussian noise: imitates the ripple left behind by *manual*
    # background subtraction at inference (SIMXRD_ROADMAP §5a). Max-normalize to
    # [0, 1], add N(mean, std), then re-max-normalize to [0, 1] — always in 2θ, even
    # for d/log-d output, so the single conversion below resamples an already-noisy
    # 2θ pattern rather than adding noise natively post-conversion.
    if cfg.gaussian_noise_std > 0 or cfg.gaussian_noise_mean != 0.0:
        pattern_tt = normalize(pattern_tt, method="max")
        pattern_tt = relative_gaussian(
            pattern_tt, cfg.gaussian_noise_std, rng, mean=cfg.gaussian_noise_mean
        )
        pattern_tt = normalize(pattern_tt, method="max")

    # --- single conversion to the output domain ---
    if domain == Domain.TWO_THETA:
        x_axis, intensity = tt_grid.values, pattern_tt
    elif domain == Domain.LOG_D:
        log_d_grid = make_log_d_grid(d_window[0], d_window[1], n_log_d_bins)
        intensity = resample_two_theta_to_log_d(
            tt_grid.values, pattern_tt, log_d_grid.values, wavelength
        )
        x_axis = log_d_grid.values
    else:
        raise ValueError(f"unsupported production domain: {domain}")

    return SimulatedPattern(
        x_axis=np.asarray(x_axis),
        intensity=np.asarray(intensity),
        domain=domain,
        wavelength=wavelength,
        peak_positions_d=d,
        peak_intensities=inten,
        peak_two_theta=np.asarray(tt_centres),
        peak_fwhm_g=np.asarray(fwhm_g),
        peak_fwhm_l=np.asarray(fwhm_l),
        metadata={"instrument": instrument.name, "n_peaks": int(len(tt)),
                  "effects": cfg},
    )


class Simulator:
    """Thin, reusable wrapper over :func:`simulate` with fixed instrument + grid.

    ``sim = Simulator(); sim(structure_or_peaks, wavelength, effects=...)``.
    """

    def __init__(
        self,
        instrument: InstrumentParameters = SIMXRD_DEFAULT,
        *,
        d_window: tuple[float, float] = (0.7, 18.0),
        n_log_d_bins: int = 12000,
    ):
        self.instrument = instrument
        self.d_window = d_window
        self.n_log_d_bins = n_log_d_bins

    def __call__(
        self, source: Structure | BraggPeaks, wavelength: float,
        *, domain: Domain = Domain.LOG_D, effects: EffectConfig | None = None, **kw,
    ) -> SimulatedPattern:
        return simulate(
            source, wavelength, domain=domain, instrument=self.instrument,
            effects=effects, d_window=self.d_window, n_log_d_bins=self.n_log_d_bins, **kw,
        )
