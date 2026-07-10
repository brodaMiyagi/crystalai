"""Shared fixtures: paths to the bundled example CIFs."""

from pathlib import Path

import pytest

_CIF_DIR = Path(__file__).resolve().parents[1] / "data" / "example_cifs"

ORDERED = ["Si", "NaCl", "CeO2", "LaB6", "alpha_quartz", "rutile"]


@pytest.fixture(scope="session")
def cif_dir() -> Path:
    return _CIF_DIR


@pytest.fixture(params=ORDERED)
def ordered_cif(request, cif_dir) -> Path:
    return cif_dir / f"{request.param}.cif"


@pytest.fixture(params=["icsdid_10", "icsdid_1000"])
def disordered_cif(request, cif_dir) -> Path:
    return cif_dir / "disordered" / f"{request.param}.cif"
