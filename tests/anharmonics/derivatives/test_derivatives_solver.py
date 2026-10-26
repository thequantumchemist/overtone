"""AnharmonicDerivativesSolver: misuse is reported clearly, trajectories can be saved."""
import numpy as np
import pytest
from ase.calculators.emt import EMT
from ase.io import read

from helpers.harmonics import run_harmonics
from overtone import AnharmonicDerivativesConfig, AnharmonicDerivativesSolver


@pytest.fixture(scope="module")
def water_config(optimised_emt):
    atoms = optimised_emt("H2O")
    harm = run_harmonics(atoms)
    return AnharmonicDerivativesConfig.from_harmonics_result(
        harm, atoms, harm.vibrational_modes, stencil=0)


@pytest.fixture
def solver(water_config):
    return AnharmonicDerivativesSolver(water_config)


def test_energy_derivatives_need_keys(solver):
    with pytest.raises(ValueError, match="generate_displacement_trajectory"):
        solver.calculate_derivatives([])


def test_force_derivatives_need_keys(solver):
    with pytest.raises(ValueError, match="generate_displacement_trajectory_for_forces"):
        solver.calculate_derivatives_from_forces([])


def test_saved_trajectory_round_trips(solver, tmp_path):
    traj, _ = solver.generate_displacement_trajectory()
    solver.save_trajectory(traj, tmp_path / "displacements.traj")
    saved = read(tmp_path / "displacements.traj", index=":")
    assert np.allclose([a.positions for a in saved], [a.positions for a in traj])


def test_missing_dipoles_are_reported(solver):
    keys = solver.get_dipole_displacement_keys()
    traj = solver.create_displaced_trajectory(keys)
    for atoms in traj:
        atoms.calc = EMT()  # no dipole moments
    with pytest.raises(RuntimeError, match="IR intensities need dipole moments"):
        solver.calculate_dipole_derivatives(traj, keys)
