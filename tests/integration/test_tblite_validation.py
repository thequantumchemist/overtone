"""
Validation of the harmonic options on two molecule sizes with GFN2-xTB
(skipped if tblite is not installed).

Water (3 modes) and methanol (12 modes):
* harmonic frequencies agree with ase.vibrations.Vibrations
* all four symmetrize/project combinations give the same VPT2 fundamentals
* the full VPT2 result (incl. Coriolis) is invariant to rotating and
  translating the molecule
"""
from pathlib import Path

import numpy as np
import pytest

tblite_ase = pytest.importorskip("tblite.ase")

from ase.build import molecule
from ase.optimize import BFGS
from ase.vibrations import Vibrations

from overtone import Harmonics, calculate_anharmonics

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def setup():
    # tight SCF: finite differences amplify SCF noise
    return tblite_ase.TBLite(method="GFN2-xTB", verbosity=0, accuracy=0.01)


@pytest.fixture(scope="module", params=["H2O", pytest.param("CH3OH", marks=pytest.mark.slow)])
def optimised(request):
    atoms = molecule(request.param)
    atoms.calc = setup()
    BFGS(atoms, logfile=None).run(fmax=1e-5)
    return atoms


def _run(atoms, directory, **kw):
    return calculate_anharmonics(atoms, setup, Path(directory),
                                 derivative_source="forces", **kw)


@pytest.fixture(scope="module")
def reference(optimised, tmp_path_factory):
    """Default settings: symmetrised Hessian, translations/rotations projected out."""
    return _run(optimised.copy(), tmp_path_factory.mktemp("reference"))


def test_harmonics_agree_with_ase(optimised, tmp_path):
    h = Harmonics(optimised.copy(), delta=0.005, project_translations_rotations=False)
    traj = h.calculate_displacements()
    shared = setup()  # same calculator re-use as ase.vibrations
    for a in traj:
        a.calc = shared
        shared.reset()
        a.get_forces()
    ours = h.calculate_hessian(traj).freq[6:]
    b = optimised.copy()
    b.calc = setup()
    vib = Vibrations(b, name=str(tmp_path / "vib"), delta=0.005)
    vib.run()
    ref = np.sort(np.real(vib.get_frequencies()))[6:]
    assert np.allclose(ours, ref, atol=0.05)


@pytest.mark.parametrize("sym, proj", [(True, False), (False, True), (False, False)])
def test_switches_give_consistent_vpt2(optimised, reference, sym, proj, tmp_path):
    r = _run(optimised.copy(), tmp_path, symmetrize_hessian=sym,
             project_translations_rotations=proj)
    assert np.abs(r.anharm_freq[6:] - reference.anharm_freq[6:]).max() < 1.0


@pytest.fixture(scope="module")
def moved(optimised, tmp_path_factory):
    atoms = optimised.copy()
    atoms.rotate(37, (1, 2, 3), center="COM")
    atoms.translate((3.0, -1.0, 2.0))
    return _run(atoms, tmp_path_factory.mktemp("moved"))


def test_fundamentals_invariant_to_rotation_and_translation(reference, moved):
    assert np.abs(reference.anharm_freq[6:] - moved.anharm_freq[6:]).max() < 1.0


def test_coriolis_invariant_to_rotation_and_translation(reference, moved):
    assert np.isclose(np.abs(reference.X_coriolis).max(), np.abs(moved.X_coriolis).max(),
                      rtol=0.01)
