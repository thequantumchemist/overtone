"""
The tutorial scripts run without errors and write their output files.

The documentation includes its code from these scripts, so this keeps the
docs in sync with the package (skipped if tblite is not installed).
Tutorials 3 and 6 use PySCF and take minutes; they run with GFN2-xTB in
place of the PySCF calculator, which checks every OVERTONE call in them.

Deselect with `pytest -m "not tutorial"`.
"""
import runpy
import sys
import types
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

tblite_ase = pytest.importorskip("tblite.ase")

TUTORIALS = Path(__file__).parents[2] / "tutorials"

pytestmark = pytest.mark.tutorial


def run(script, workdir):
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(workdir)
        try:
            runpy.run_path(str(TUTORIALS / script), run_name="__main__")
        finally:
            plt.close("all")


@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    # shared, because tutorial 5 analyses the output of tutorial 1
    return tmp_path_factory.mktemp("tutorials")


@pytest.fixture
def xtb_instead_of_pyscf(monkeypatch):
    def fake(**kwargs):
        return tblite_ase.TBLite(method="GFN2-xTB", verbosity=0, accuracy=0.01)

    module = types.ModuleType("pyscf_calculator")
    module.PySCFCalculator = fake
    monkeypatch.setitem(sys.modules, "pyscf_calculator", module)


@pytest.fixture(scope="module")
def quickstart(workdir):
    run("01_quickstart.py", workdir)
    return workdir


@pytest.fixture(scope="module")
def analysis(quickstart):
    run("05_analysis.py", quickstart)
    return quickstart


@pytest.mark.parametrize("output", ["water_quickstart.png",
                                    "water_quickstart/anharmonics/vpt2_result.json"])
def test_01_quickstart_writes(quickstart, output):
    assert (quickstart / output).exists()


def test_02_step_by_step(workdir):
    run("02_step_by_step.py", workdir)


def test_03_water_vs_experiment(workdir, xtb_instead_of_pyscf):
    run("03_water_vs_experiment.py", workdir)
    assert (workdir / "water_vs_experiment.png").exists()


def test_04_fermi_resonances(workdir):
    run("04_fermi_resonances.py", workdir)
    assert (workdir / "formaldehyde_resonance.png").exists()


@pytest.mark.parametrize("output", ["water_spectrum.dat", "water_vs_reference.png"])
def test_05_analysis_writes(analysis, output):
    assert (analysis / output).exists()


def test_06_hybrid(workdir, xtb_instead_of_pyscf):
    run("06_hybrid.py", workdir)


@pytest.fixture(scope="module")
def h2_with_pyscf():
    """H2 with the real PySCF calculator of tutorials 3 and 6 (skipped without PySCF)."""
    pytest.importorskip("pyscf")
    from ase.build import molecule

    with pytest.MonkeyPatch.context() as mp:
        mp.syspath_prepend(str(TUTORIALS))
        from pyscf_calculator import PySCFCalculator

    atoms = molecule("H2")
    atoms.calc = PySCFCalculator(xc="hf", basis="sto-3g")
    return atoms


def test_pyscf_calculator_gives_forces(h2_with_pyscf):
    assert h2_with_pyscf.get_forces().shape == (2, 3)


def test_pyscf_calculator_gives_dipole(h2_with_pyscf):
    assert h2_with_pyscf.get_dipole_moment().shape == (3,)
