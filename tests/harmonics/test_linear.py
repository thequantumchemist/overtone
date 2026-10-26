"""Detection and treatment of linear molecules."""
from helpers.harmonics import run_harmonics
from overtone import Harmonics
from overtone.harmonics.harmonics import is_linear_molecule


def test_linear_molecule_detection(optimised_emt):
    co2 = optimised_emt("CO2")
    assert is_linear_molecule(co2)
    assert not is_linear_molecule(optimised_emt("H2O"))
    r = run_harmonics(co2)
    assert r.is_linear and r.n_tr == 5 and len(r.vibrational_modes) == 4
    # the override wins
    assert not Harmonics(co2, linear=False).linear
