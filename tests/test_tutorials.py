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

import matplotlib
import pytest

tblite_ase = pytest.importorskip("tblite.ase")

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TUTORIALS = Path(__file__).parents[1] / "tutorials"

pytestmark = pytest.mark.tutorial


@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    # shared, because tutorial 5 analyses the output of tutorial 1
    return tmp_path_factory.mktemp("tutorials")


def run(script, workdir, monkeypatch):
    monkeypatch.chdir(workdir)
    try:
        runpy.run_path(str(TUTORIALS / script), run_name="__main__")
    finally:
        plt.close("all")


@pytest.fixture
def xtb_instead_of_pyscf(monkeypatch):
    def fake(**kwargs):
        return tblite_ase.TBLite(method="GFN2-xTB", verbosity=0, accuracy=0.01)

    module = types.ModuleType("pyscf_calculator")
    module.PySCFCalculator = fake
    monkeypatch.setitem(sys.modules, "pyscf_calculator", module)


def test_01_quickstart(workdir, monkeypatch):
    run("01_quickstart.py", workdir, monkeypatch)
    assert (workdir / "water_quickstart.png").exists()
    assert (workdir / "water_quickstart" / "anharmonics" / "vpt2_result.json").exists()


def test_02_step_by_step(workdir, monkeypatch):
    run("02_step_by_step.py", workdir, monkeypatch)


def test_03_water_vs_experiment(workdir, monkeypatch, xtb_instead_of_pyscf):
    run("03_water_vs_experiment.py", workdir, monkeypatch)
    assert (workdir / "water_vs_experiment.png").exists()


def test_04_fermi_resonances(workdir, monkeypatch):
    run("04_fermi_resonances.py", workdir, monkeypatch)
    assert (workdir / "formaldehyde_resonance.png").exists()


def test_05_analysis(workdir, monkeypatch):
    if not (workdir / "water_quickstart").exists():
        run("01_quickstart.py", workdir, monkeypatch)
    run("05_analysis.py", workdir, monkeypatch)
    assert (workdir / "water_spectrum.dat").exists()
    assert (workdir / "water_vs_reference.png").exists()


def test_06_hybrid(workdir, monkeypatch, xtb_instead_of_pyscf):
    run("06_hybrid.py", workdir, monkeypatch)


def test_pyscf_calculator(monkeypatch):
    """The real PySCF calculator of tutorials 3 and 6 (skipped without PySCF)."""
    pytest.importorskip("pyscf")
    from ase.build import molecule

    monkeypatch.syspath_prepend(str(TUTORIALS))
    from pyscf_calculator import PySCFCalculator

    atoms = molecule("H2")
    atoms.calc = PySCFCalculator(xc="hf", basis="sto-3g")
    assert atoms.get_forces().shape == (2, 3)
    assert atoms.get_dipole_moment().shape == (3,)
