"""The Coriolis contribution must not depend on molecular orientation/position."""
from types import SimpleNamespace

import numpy as np
import pytest
from ase.calculators.emt import EMT

from helpers.harmonics import run_harmonics
from helpers.units import rotational_constants
from overtone import VPT2Config, VPT2Solver


def _coriolis(atoms):
    atoms = atoms.copy()
    atoms.calc = EMT()
    hr = run_harmonics(atoms)
    n = 3 * len(atoms)
    cfg = VPT2Config.from_harmonics_result(hr, active_modes=[6, 7, 8], atoms=atoms)
    s = VPT2Solver(cfg, (np.zeros((n,) * 3), np.zeros((n,) * 4)))
    return s.calculate_rotational_constants(), s.calculate_anharmonic_coriolis()


@pytest.fixture(scope="module")
def frames(optimised_emt):
    ref = optimised_emt("H2O", fmax=1e-6)  # VPT2 needs a minimum
    moved = ref.copy()
    moved.rotate(37, (1, 2, 3), center="COM")
    moved.translate((3.0, -1.0, 2.0))
    (B1, X1), (B2, X2) = _coriolis(ref), _coriolis(moved)
    return SimpleNamespace(ref=ref, B=(B1, B2), X=(X1, X2))


def test_rotational_constants_invariant(frames):
    assert np.allclose(*frames.B, rtol=1e-6)


def test_coriolis_constants_invariant(frames):
    # frame bug gave factor ~60 errors; remaining noise is finite-difference noise
    assert np.allclose(*frames.X, rtol=1e-2, atol=1e-2)


def test_rotational_constants_match_moments_of_inertia(frames):
    I = frames.ref.get_moments_of_inertia()  # ASE: principal moments about COM  # noqa: E741
    assert np.allclose(np.sort(frames.B[0]), np.sort(rotational_constants(I)), rtol=1e-6)
