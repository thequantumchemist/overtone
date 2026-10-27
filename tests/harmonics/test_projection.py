"""Projection of translations and rotations out of the Hessian."""
from types import SimpleNamespace

import numpy as np
import pytest

from helpers.harmonics import run_harmonics
from overtone.harmonics.harmonics import rigid_body_basis


@pytest.fixture(scope="module", params=["H2O", "CH3NO2"])
def hessians(request, optimised_emt):
    atoms = optimised_emt(request.param)
    return SimpleNamespace(
        atoms=atoms,
        on=run_harmonics(atoms),
        off=run_harmonics(atoms, project_translations_rotations=False),
    )


def test_projection_is_reported(hessians):
    assert hessians.on.projected and hessians.on.n_tr == 6


def test_rigid_body_frequencies_are_zero(hessians):
    assert np.all(hessians.on.freq[:6] == 0) and np.all(hessians.on.ifreq[:6] == 0)


def test_vibrations_orthogonal_to_rigid_body_motion(hessians):
    T = rigid_body_basis(hessians.atoms, linear=False)
    assert np.abs(T.T @ hessians.on.modes[:, 6:]).max() < 1e-10


def test_modes_are_orthonormal(hessians):
    modes = hessians.on.modes
    assert np.allclose(modes.T @ modes, np.eye(len(hessians.on.freq)), atol=1e-10)


def test_vibrational_frequencies_unchanged_at_minimum(hessians):
    # at a tight minimum the vibrational frequencies barely change
    assert np.allclose(hessians.on.freq[6:], hessians.off.freq[6:], atol=0.5)
