"""Without resonances GVPT2 must reduce to standard VPT2."""
import numpy as np
import pytest

from helpers.models import MU1, MU2, U_MODEL, by_quanta, solve, sym3
from overtone.anharmonics.dipole_derivatives import DipoleDerivativesResult


@pytest.fixture(scope="module")
def results():
    w = [3050.0, 1650.0, 1180.0]
    T = sym3({(0, 0, 0): -300, (1, 1, 1): -120, (2, 2, 2): -90, (0, 0, 1): 40,
              (1, 1, 2): 55, (0, 1, 2): 30}, 3)
    dip = DipoleDerivativesResult(mu1=MU1, mu2=MU2)
    return solve(w, T, U_MODEL, "gvpt2", dip), solve(w, T, U_MODEL, "standard", dip)


def _paired(results, attr):
    g, s = (by_quanta(r) for r in results)
    return [getattr(g[k], attr) for k in s], [getattr(s[k], attr) for k in s]


def test_no_fermi_resonance_detected(results):
    gvpt2, _ = results
    assert gvpt2.fermi_type1 == [] and gvpt2.fermi_type2 == []


def test_energies_equal_standard(results):
    assert np.allclose(*_paired(results, "energy"))


def test_intensities_equal_standard(results):
    assert np.allclose(*_paired(results, "intensity"))
