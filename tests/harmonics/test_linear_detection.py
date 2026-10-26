from ase import Atoms
from ase.build import molecule

from overtone.harmonics.harmonics import is_linear_molecule


def test_single_atom_is_not_linear():
    assert not is_linear_molecule(Atoms("Ar"))


def test_co2_is_linear():
    assert is_linear_molecule(molecule("CO2"))
