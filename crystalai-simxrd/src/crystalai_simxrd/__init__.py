"""crystalai-simxrd: physics-based PXRD simulation + on-the-fly augmentation.

Public API consumed by ``crystalai-methods`` (SIMXRD_ROADMAP §8). The simulator
takes a pymatgen ``Structure`` or a cached ``BraggPeaks`` (from the precomputed
``bragg_peaks`` store); the augmentors regenerate patterns on the fly and emit the
``(λmax, σrel)`` noise-floor conditioning.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .augmentations.peak_augmentations import PeakAugmentConfig, PeakAugmentor
from .augmentations.presets import PRESETS, get_preset
from .augmentations.profile_augmentations import (
    AugmentConfig,
    AugmentedPattern,
    ProfileAugmentor,
)
from .comparison.compare import (
    compare_structure_to_experimental,
    compare_structure_to_raw,
)
from .core.bragg import BraggPeaks, compute_peak_list, to_two_theta
from .core.domain import Domain
from .io.bragg_store import BraggStore
from .simulation.simulator import EffectConfig, SimulatedPattern, Simulator, simulate

__all__ = [
    "Domain",
    "Simulator",
    "simulate",
    "SimulatedPattern",
    "EffectConfig",
    "BraggPeaks",
    "compute_peak_list",
    "to_two_theta",
    "BraggStore",
    "ProfileAugmentor",
    "AugmentConfig",
    "AugmentedPattern",
    "PeakAugmentor",
    "PeakAugmentConfig",
    "get_preset",
    "PRESETS",
    "compare_structure_to_raw",
    "compare_structure_to_experimental",
]
