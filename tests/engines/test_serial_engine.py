"""SerialEngine reuses a saved structure only if it is the same structure with all results."""
import pytest
from ase.build import molecule
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

from helpers.calculators import counting_emt
from overtone.engines import SerialEngine


def saved_job(directory, atoms, **results):
    """Write atoms with results as the restart file of structure 0."""
    path = directory / "calculations" / "job_000000.traj"
    path.parent.mkdir(parents=True)
    atoms = atoms.copy()
    atoms.calc = SinglePointCalculator(atoms, **results)
    write(path, atoms)
    return path


def n_calculations(directory, atoms, properties=("energy",)):
    counter = {"n": 0}
    SerialEngine().calculate_trajectory([atoms.copy()], counting_emt(counter),
                                        properties=properties, output_directory=directory)
    return counter["n"]


@pytest.fixture
def water():
    return molecule("H2O")


def test_matching_file_is_reused(tmp_path, water):
    saved_job(tmp_path, water, energy=1.0)
    assert n_calculations(tmp_path, water) == 0


def test_truncated_file_is_recalculated(tmp_path, water):
    path = saved_job(tmp_path, water, energy=1.0)
    path.write_bytes(path.read_bytes()[:50])  # e.g. a crash while writing
    assert n_calculations(tmp_path, water) == 1


def test_other_elements_are_recalculated(tmp_path, water):
    other = water.copy()
    other.symbols[0] = "S"
    saved_job(tmp_path, other, energy=1.0)
    assert n_calculations(tmp_path, water) == 1


def test_missing_property_is_recalculated(tmp_path, water):
    saved_job(tmp_path, water, energy=1.0)
    assert n_calculations(tmp_path, water, properties=("energy", "forces")) == 1
