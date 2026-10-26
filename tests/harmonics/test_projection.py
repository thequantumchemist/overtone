"""Projection of translations and rotations out of the Hessian."""
import numpy as np
import pytest

from helpers.harmonics import run_harmonics
from overtone.harmonics.harmonics import rigid_body_basis


@pytest.mark.parametrize("name", ["H2O", "CH3NO2"])
def test_projection_gives_pure_vibrations(name, optimised_emt):
    atoms = optimised_emt(name)
    on = run_harmonics(atoms)
    off = run_harmonics(atoms, project_translations_rotations=False)
    assert on.projected and on.n_tr == 6
    assert np.all(on.freq[:6] == 0) and np.all(on.ifreq[:6] == 0)
    T = rigid_body_basis(atoms, linear=False)
    assert np.abs(T.T @ on.modes[:, 6:]).max() < 1e-10
    # modes are orthonormal
    assert np.allclose(on.modes.T @ on.modes, np.eye(len(on.freq)), atol=1e-10)
    # at a tight minimum the vibrational frequencies barely change
    assert np.allclose(on.freq[6:], off.freq[6:], atol=0.5)
