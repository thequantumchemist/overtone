import matplotlib
import matplotlib.pyplot as plt
import pytest
from ase.build import molecule
from ase.calculators.emt import EMT
from ase.optimize import BFGS

from helpers.calculators import WATER_CHARGES, ChargedEMT

matplotlib.use("Agg")  # before pyplot is imported anywhere else


@pytest.fixture(scope="session")
def optimised_emt():
    """optimised_emt(name, fmax=1e-7): a copy of the EMT minimum of an ase.build molecule."""
    cache = {}

    def get(name, fmax=1e-7):
        if (name, fmax) not in cache:
            atoms = molecule(name)
            atoms.calc = EMT()
            BFGS(atoms, logfile=None).run(fmax=fmax)
            cache[name, fmax] = atoms
        atoms = cache[name, fmax].copy()
        atoms.calc = EMT()
        return atoms

    return get


@pytest.fixture(scope="session")
def charged_water():
    """Water at the ChargedEMT minimum (EMT forces, point-charge dipole)."""
    atoms = molecule("H2O")
    atoms.calc = ChargedEMT(WATER_CHARGES)
    BFGS(atoms, logfile=None).run(fmax=1e-6)
    return atoms


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")
