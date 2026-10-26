"""Constrained atoms switch the rigid-body projection off."""
import pytest

from helpers.harmonics import run_harmonics
from overtone import Harmonics


def test_constraints_switch_projection_off_and_fail_early(optimised_emt):
    atoms = optimised_emt("H2O")
    with pytest.warns(UserWarning, match="projection"):
        h = Harmonics(atoms, constrain_index=[0])
    assert not h.project
    r = run_harmonics(atoms, constrain_index=[0])
    assert r.n_tr == 0 and len(r.freq) == 6
    # one fixed atom leaves free rotations -> must stop before the anharmonic part
    with pytest.raises(ValueError, match="very low frequency"):
        r.check_active_modes(r.vibrational_modes)
