"""On-the-fly profile augmentation (SIMXRD_ROADMAP §5).

Augmentation here means: **sample** the simulation parameters from config-bounded
ranges (`θ ~ U[lo, hi]`), then call `simulate()` over the matched d-window. All
noise — Poisson counting noise *and* the full-profile Gaussian noise — is applied
inside `simulate()`, in 2θ, before the single conversion to log-d (SIMXRD_ROADMAP
§5a: there is exactly one Gaussian-noise mechanism, and it is 2θ-native like every
other physical effect). What follows the conversion here is only the final log-d
normalization. The `(λmax, σrel)` noise floor is still emitted as conditioning
(DESIGN_DECISIONS §4a) — `σrel` now names the std fed to
`EffectConfig.gaussian_noise_std` rather than a separate log-d-native effect.
Physics is regenerated each call — nothing is pre-computed.

NumPy throughout; the torch-`nn.Module` wrapper is the Phase-5.2 public-API concern.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from pymatgen.core import Structure

from ..core.bragg import BraggPeaks
from ..core.domain import Domain, d_to_two_theta
from ..profiles.caglioti import InstrumentParameters
from ..simulation.simulator import EffectConfig, simulate
from ..utils.normalization import normalize


@dataclass
class AugmentConfig:
    """Sampling ranges for the profile augmentation (uniform unless noted)."""

    wavelength: tuple[float, float] = (0.5, 1.8)          # Å
    caglioti_U: tuple[float, float] = (0.002, 0.02)
    caglioti_V: tuple[float, float] = (-0.02, 0.0)
    caglioti_W: tuple[float, float] = (0.002, 0.02)
    crystallite_nm: tuple[float, float] | None = (30.0, 500.0)   # log-uniform; None = ∞
    microstrain: tuple[float, float] = (0.0, 0.003)
    b_iso: tuple[float, float] = (0.0, 1.5)
    march_r: tuple[float, float] = (0.85, 1.18)
    march_axes: tuple = ((0, 0, 1), (1, 0, 0), (1, 1, 0), (1, 1, 1))
    zero_shift_deg: tuple[float, float] = (-0.02, 0.02)
    axial_divergence: bool = True
    axial_L: tuple[float, float] = (150.0, 300.0)
    axial_H: tuple[float, float] = (1.0, 5.0)
    axial_S: tuple[float, float] = (1.0, 5.0)
    slit_width_deg: float = 0.0                            # off by default (Caglioti covers it)
    background_choices: tuple = ("residual", None)
    background_rel_amplitude: tuple[float, float] = (0.0, 0.05)
    lambda_max: tuple[float, float] = (1.0, 100.0)         # log-uniform (AlphaDiffract)
    sigma_rel: tuple[float, float] = (1e-3, 1e-1)          # log-uniform; -> gaussian_noise_std (2θ, §5a)
    normalization: str = "max"
    d_window: tuple[float, float] = (0.7, 18.0)
    n_log_d_bins: int = 12000


@dataclass(slots=True)
class AugmentedPattern:
    x_axis: NDArray                       # log₁₀(d)
    intensity: NDArray
    wavelength: float
    noise_floor: tuple[float, float]      # (λmax, σrel) — encoder conditioning
    domain: Domain = Domain.LOG_D
    metadata: dict = field(default_factory=dict)


def _u(rng, r):
    return float(rng.uniform(r[0], r[1]))


def _logu(rng, r):
    return float(np.exp(rng.uniform(np.log(r[0]), np.log(r[1]))))


class ProfileAugmentor:
    """Callable: ``(structure_or_peaks, rng) -> AugmentedPattern``."""

    def __init__(self, config: AugmentConfig | None = None):
        self.config = config or AugmentConfig()

    def _matched_two_theta_range(self, wavelength: float) -> tuple[float, float]:
        d_min, d_max = self.config.d_window
        tt_lo = max(float(d_to_two_theta(d_max, wavelength)), 1.0)
        tt_hi = min(float(d_to_two_theta(d_min, wavelength)), 179.0)
        return (tt_lo, tt_hi)

    def sample_effects(self, rng) -> tuple[EffectConfig, InstrumentParameters, float]:
        c = self.config
        wl = _u(rng, c.wavelength)
        inst = InstrumentParameters(_u(rng, c.caglioti_U), _u(rng, c.caglioti_V),
                                    _u(rng, c.caglioti_W), name="sampled")
        eff = EffectConfig(
            b_iso=_u(rng, c.b_iso),
            crystallite_size_nm=(_logu(rng, c.crystallite_nm) if c.crystallite_nm else np.inf),
            microstrain=_u(rng, c.microstrain),
            march_r=_u(rng, c.march_r),
            march_axis=tuple(c.march_axes[rng.integers(len(c.march_axes))]),
            zero_shift_deg=_u(rng, c.zero_shift_deg),
            axial_divergence=c.axial_divergence,
            axial_L=_u(rng, c.axial_L), axial_H=_u(rng, c.axial_H), axial_S=_u(rng, c.axial_S),
            slit_width_deg=c.slit_width_deg,
            background=c.background_choices[rng.integers(len(c.background_choices))],
            background_rel_amplitude=_u(rng, c.background_rel_amplitude),
            poisson_lambda_max=_logu(rng, c.lambda_max),
            gaussian_noise_mean=0.0,
            gaussian_noise_std=_logu(rng, c.sigma_rel),
            # ties simulate()'s internal noise draws (background/Poisson/Gaussian) to this
            # augmentor's own rng, so a seeded `rng` reproduces the whole draw end-to-end.
            rng_seed=int(rng.integers(0, 2**31 - 1)),
        )
        return eff, inst, wl

    def __call__(
        self, source: Structure | BraggPeaks, rng: np.random.Generator | None = None
    ) -> AugmentedPattern:
        rng = rng or np.random.default_rng()
        c = self.config
        eff, inst, wl = self.sample_effects(rng)
        lambda_max = eff.poisson_lambda_max
        sigma_rel = eff.gaussian_noise_std

        sim = simulate(
            source, wl, domain=Domain.LOG_D, instrument=inst, effects=eff,
            two_theta_range=self._matched_two_theta_range(wl),
            d_window=c.d_window, n_log_d_bins=c.n_log_d_bins,
        )
        # Poisson counting noise and the full-profile Gaussian noise are both already
        # applied inside simulate() (2θ, before the log-d conversion — SIMXRD_ROADMAP
        # §5a); this final normalize is the only step left in log-d.
        intensity = normalize(sim.intensity, c.normalization)

        return AugmentedPattern(
            x_axis=sim.x_axis, intensity=intensity, wavelength=wl,
            noise_floor=(float(lambda_max), float(sigma_rel)),
            metadata={"instrument": inst.name, "n_peaks": sim.metadata["n_peaks"],
                      "march_r": eff.march_r, "b_iso": eff.b_iso,
                      "crystallite_nm": eff.crystallite_size_nm},
        )
