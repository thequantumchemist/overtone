"""Detection and treatment of linear molecules."""
import pytest

from helpers.harmonics import run_harmonics
from overtone import Harmonics
from overtone.harmonics.harmonics import is_linear_molecule


@pytest.fixture(scope="module")
def co2(optimised_emt):
    return optimised_emt("CO2")


@pytest.fixture(scope="module")
def co2_harmonics(co2):
    return run_harmonics(co2)


def test_co2_is_linear(co2):
    assert is_linear_molecule(co2)


def test_water_is_not_linear(optimised_emt):
    assert not is_linear_molecule(optimised_emt("H2O"))


def test_result_is_marked_linear(co2_harmonics):
    assert co2_harmonics.is_linear


def test_five_rigid_body_modes_removed(co2_harmonics):
    assert co2_harmonics.n_tr == 5


def test_four_vibrational_modes(co2_harmonics):
    assert len(co2_harmonics.vibrational_modes) == 4


def test_override_wins(co2):
    assert not Harmonics(co2, linear=False).linear
