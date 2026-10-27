import pytest

from overtone import estimate_cost


@pytest.fixture(scope="module")
def water_cost(optimised_emt):
    return estimate_cost(optimised_emt("H2O"), derivative_source="energy", stencil=1, coupling=3)


def test_water_has_three_modes(water_cost):
    assert water_cost["n_modes"] == 3


def test_water_harmonic_point_count(water_cost):
    assert water_cost["harmonic"] == 19  # 1 + 6 * 3 atoms


def test_linear_molecule_has_4_modes(optimised_emt):
    assert estimate_cost(optimised_emt("CO2"), stencil=0)["n_modes"] == 4
