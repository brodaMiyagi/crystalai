"""Write a simulated pattern to a headered, self-describing ``.xy`` text file.

The header records provenance (crystalai-simxrd) and every simulation effect that
was actually applied to produce the pattern, with its parameter values, so a saved
``.xy`` can be understood without the dashboard state that produced it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..core.domain import Domain
from ..profiles.caglioti import InstrumentParameters
from ..simulation.simulator import EffectConfig

__all__ = ["effect_summary_lines", "write_xy"]


def effect_summary_lines(cfg: EffectConfig, instrument: InstrumentParameters) -> list[str]:
    """Human-readable ``"name: params"`` lines for the instrument profile and every
    *active* (non-default/non-identity) effect in ``cfg``."""
    lines = [
        f"instrument (Caglioti): name={instrument.name!r} "
        f"U={instrument.U} V={instrument.V} W={instrument.W} X={instrument.X} Y={instrument.Y}"
    ]
    if cfg.b_iso > 0:
        lines.append(f"Debye-Waller thermal factor: B_iso={cfg.b_iso} A^2")
    if np.isfinite(cfg.crystallite_size_nm):
        lines.append(f"crystallite-size (Scherrer) broadening: size={cfg.crystallite_size_nm} nm")
    if cfg.microstrain > 0:
        lines.append(f"microstrain (Williamson-Hall) broadening: microstrain={cfg.microstrain}")
    if abs(cfg.march_r - 1.0) > 1e-9:
        lines.append(
            f"preferred orientation (March-Dollase): r={cfg.march_r} axis={cfg.march_axis}"
        )
    if abs(cfg.zero_shift_deg) > 1e-12:
        lines.append(f"zero-shift: {cfg.zero_shift_deg} deg")
    if cfg.axial_divergence:
        lines.append(
            f"axial divergence: L={cfg.axial_L} mm H={cfg.axial_H} mm S={cfg.axial_S} mm"
        )
    if cfg.slit_width_deg > 0:
        lines.append(f"receiving-slit broadening: width={cfg.slit_width_deg} deg")
    if cfg.background is not None:
        extra = (
            f" rel_amplitude={cfg.background_rel_amplitude}"
            if cfg.background == "residual"
            else ""
        )
        lines.append(f"background: {cfg.background}{extra}")
    if cfg.poisson_lambda_max is not None:
        lines.append(f"Poisson counting noise: lambda_max={cfg.poisson_lambda_max}")
    if cfg.gaussian_noise_std > 0 or cfg.gaussian_noise_mean != 0.0:
        lines.append(
            f"full-profile Gaussian noise (2theta, [0,1]-normalized before/after): "
            f"mean={cfg.gaussian_noise_mean} std={cfg.gaussian_noise_std}"
        )
    if cfg.rng_seed is not None:
        lines.append(f"rng seed: {cfg.rng_seed}")
    if len(lines) == 1:
        lines.append("(no additional effects — pristine Bragg profile)")
    return lines


def write_xy(
    path: str | Path,
    *,
    x: NDArray,
    y: NDArray,
    domain: Domain,
    wavelength: float,
    structure_name: str,
    n_peaks: int,
    effects: EffectConfig,
    instrument: InstrumentParameters,
    crop: tuple[float, float] | None = None,
    scale_to_100: bool = False,
) -> Path:
    """Write a two-column ``x  intensity`` ``.xy`` file with a ``#``-commented header.

    ``crop`` restricts the written range to ``[min(crop), max(crop)]`` on ``x``
    (2θ in degrees, or log10(d/Å), per ``domain``); ``None`` writes the full range.

    ``scale_to_100``, applied *after* cropping, rescales intensity so the strongest
    line in the written range is exactly 100 — the standard PXRD relative-intensity
    convention (ICDD/PDF-style "100 = strongest reflection"), for saved patterns
    meant to be read/compared outside this simulator. Off by default: the raw
    simulator scale is otherwise preserved.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if crop is not None:
        lo, hi = min(crop), max(crop)
        m = (x >= lo) & (x <= hi)
        x, y = x[m], y[m]
    if scale_to_100:
        peak = np.max(y)
        if peak > 0:
            y = y / peak * 100.0

    x_label = "2theta_deg" if domain == Domain.TWO_THETA else "log10_d_over_A"
    header = [
        "# simulated with crystalai-simxrd",
        f"# structure: {structure_name}",
        f"# wavelength: {wavelength} A",
        f"# domain: {domain.value}",
        f"# n_peaks: {n_peaks}",
        "# effects used:",
    ]
    header += [f"#   - {line}" for line in effect_summary_lines(effects, instrument)]
    if scale_to_100:
        header.append("# intensity: scaled to 0-100 (strongest line in this range = 100)")
    header.append(f"# columns: {x_label}  intensity")

    path = Path(path)
    with open(path, "w") as fh:
        fh.write("\n".join(header) + "\n")
        for xi, yi in zip(x, y):
            fh.write(f"{xi:.6f}  {yi:.6f}\n")
    return path
