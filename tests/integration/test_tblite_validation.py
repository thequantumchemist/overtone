"""
Validation of the harmonic options on two molecule sizes with GFN2-xTB
(skipped if tblite is not installed).

Water (3 modes) and methanol (12 modes):
* harmonic frequencies agree with ase.vibrations.Vibrations
* all four symmetrize/project combinations give the same VPT2 fundamentals
* the full VPT2 result (incl. Coriolis) is invariant to rotating and
  translating the molecule
"""
import itertools
from pathlib import Path

import numpy as np
import pytest

tblite_ase = pytest.importorskip("tblite.ase")

from ase.build import molecule  # noqa: E402
from ase.optimize import BFGS  # noqa: E402
from ase.vibrations import Vibrations  # noqa: E402

from overtone import Harmonics, calculate_anharmonics  # noqa: E402


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


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_switches_give_consistent_vpt2(optimised, tmp_path):
    results = {}
    for sym, proj in itertools.product((True, False), repeat=2):
        r = _run(optimised.copy(), tmp_path / f"{sym}{proj}",
                 symmetrize_hessian=sym, project_translations_rotations=proj)
        results[(sym, proj)] = r.anharm_freq[6:]
    ref = results[(True, True)]
    for v in results.values():
        assert np.abs(v - ref).max() < 1.0


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_vpt2_invariant_to_rotation_and_translation(optimised, tmp_path):
    moved = optimised.copy()
    moved.rotate(37, (1, 2, 3), center="COM")
    moved.translate((3.0, -1.0, 2.0))
    a = _run(optimised.copy(), tmp_path / "a")
    b = _run(moved, tmp_path / "b")
    assert np.abs(a.anharm_freq[6:] - b.anharm_freq[6:]).max() < 1.0
    assert np.isclose(np.abs(a.X_coriolis).max(), np.abs(b.X_coriolis).max(), rtol=0.01)
