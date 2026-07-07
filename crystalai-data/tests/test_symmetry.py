"""Tests for the SG<->CS<->symbol lookup utilities (no spglib)."""

import pytest

from crystalai_data.crystals.symmetry import (
    cs_number_to_name,
    hm_symbol_to_sg_number,
    sg_number_to_symbol,
    sg_to_cs,
)

# (space group, expected crystal system) — both sides of each range boundary.
_BOUNDARIES = [
    (1, 1), (2, 1),                      # triclinic
    (3, 2), (15, 2),                     # monoclinic
    (16, 3), (74, 3),                    # orthorhombic
    (75, 4), (142, 4),                   # tetragonal
    (143, 5), (167, 5),                  # trigonal
    (168, 6), (194, 6),                  # hexagonal
    (195, 7), (230, 7),                  # cubic
]


@pytest.mark.parametrize("sg,cs", _BOUNDARIES)
def test_sg_to_cs_boundaries(sg, cs):
    assert sg_to_cs(sg) == cs


def test_sg_to_cs_covers_full_range_monotonically():
    systems = [sg_to_cs(sg) for sg in range(1, 231)]
    # Non-decreasing across 1..230 and spans exactly the 7 systems.
    assert systems == sorted(systems)
    assert set(systems) == {1, 2, 3, 4, 5, 6, 7}


@pytest.mark.parametrize("bad", [0, 231, -1, 1.5, "5", None])
def test_sg_to_cs_rejects_out_of_range(bad):
    with pytest.raises(ValueError):
        sg_to_cs(bad)


def test_cs_number_to_name():
    assert cs_number_to_name(1) == "triclinic"
    assert cs_number_to_name(7) == "cubic"
    with pytest.raises(ValueError):
        cs_number_to_name(8)


def test_sg_number_to_symbol_known_values():
    assert sg_number_to_symbol(227) == "Fd-3m"
    assert sg_number_to_symbol(1) == "P1"
    with pytest.raises(ValueError):
        sg_number_to_symbol(0)


def test_hm_symbol_to_sg_number_roundtrip_all():
    # Canonical symbol for every group must resolve back to its number.
    for sg in range(1, 231):
        assert hm_symbol_to_sg_number(sg_number_to_symbol(sg)) == sg


def test_hm_symbol_to_sg_number_notational_variants():
    assert hm_symbol_to_sg_number("F d -3 m") == 227   # spaces
    assert hm_symbol_to_sg_number("F d -3 m S") == 227  # trailing origin marker
    assert hm_symbol_to_sg_number("P21/c") == 14        # underscore-free variant
    assert hm_symbol_to_sg_number("P2_1/c") == 14       # IT canonical
    assert hm_symbol_to_sg_number("'Fd-3m'") == 227     # quoted


def test_hm_symbol_to_sg_number_rejects_garbage():
    with pytest.raises(ValueError):
        hm_symbol_to_sg_number("not a space group")
    with pytest.raises(ValueError):
        hm_symbol_to_sg_number("")
