"""Symmetrisation of the finite-difference Hessian."""
import numpy as np

from helpers.harmonics import run_harmonics


def test_symmetrisation_flag_and_diagnostic(optimised_emt):
    atoms = optimised_emt("CH3NO2")
    sym = run_harmonics(atoms, symmetrize_hessian=True)
    raw = run_harmonics(atoms, symmetrize_hessian=False)
    assert sym.symmetrized and not raw.symmetrized
    assert 0 < sym.hessian_asymmetry < 1e-3
    assert np.allclose(sym.freq, raw.freq, atol=1.0)
