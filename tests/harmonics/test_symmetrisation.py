"""Symmetrisation of the finite-difference Hessian."""
import numpy as np
import pytest

from helpers.harmonics import run_harmonics


@pytest.fixture(scope="module")
def hessians(optimised_emt):
    atoms = optimised_emt("CH3NO2")
    return (run_harmonics(atoms, symmetrize_hessian=True),
            run_harmonics(atoms, symmetrize_hessian=False))


def test_flag_is_recorded(hessians):
    sym, raw = hessians
    assert sym.symmetrized and not raw.symmetrized


def test_asymmetry_diagnostic_is_small_but_nonzero(hessians):
    sym, _ = hessians
    assert 0 < sym.hessian_asymmetry < 1e-3


def test_frequencies_barely_change(hessians):
    sym, raw = hessians
    assert np.allclose(sym.freq, raw.freq, atol=1.0)
