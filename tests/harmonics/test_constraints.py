"""Constrained atoms switch the rigid-body projection off."""
import pytest

from helpers.harmonics import run_harmonics
from overtone import Harmonics


@pytest.fixture(scope="module")
def water(optimised_emt):
    return optimised_emt("H2O")


@pytest.fixture(scope="module")
def constrained(water):
    with pytest.warns(UserWarning, match="projection"):
        return run_harmonics(water, constrain_index=[0])


def test_constraint_warns_about_projection(water):
    with pytest.warns(UserWarning, match="projection"):
        Harmonics(water, constrain_index=[0])


@pytest.mark.filterwarnings("ignore:Constrained atoms:UserWarning")
def test_constraint_switches_projection_off(water):
    assert not Harmonics(water, constrain_index=[0]).project


def test_no_rigid_body_modes_removed(constrained):
    assert constrained.n_tr == 0


def test_constrained_atom_has_no_modes(constrained):
    assert len(constrained.freq) == 6


def test_free_rotations_fail_active_mode_check(constrained):
    # one fixed atom leaves free rotations -> must stop before the anharmonic part
    with pytest.raises(ValueError, match="very low frequency"):
        constrained.check_active_modes(constrained.vibrational_modes)
