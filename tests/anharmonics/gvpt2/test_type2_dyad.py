"""GVPT2 for a type-2 Fermi dyad (w0 ~ w1 + w2) against exact levels."""
from types import SimpleNamespace

import numpy as np
import pytest

from helpers.models import MU1, MU2, U_MODEL, by_quanta, exact_levels, solve, sym3
from overtone.anharmonics.dipole_derivatives import DipoleDerivativesResult

BANDS = {"fundamental": (((0, 1),), (1, 0, 0)), "combination": (((1, 1), (2, 1)), (0, 1, 1))}


@pytest.fixture(scope="module")
def dyad():
    w = [2850.0, 1650.0, 1190.0]  # w0 ~ w1 + w2
    T = sym3({(0, 0, 0): -300, (1, 1, 1): -120, (2, 2, 2): -90, (0, 1, 2): -140,
              (0, 0, 1): 30, (1, 1, 2): 40}, 3)
    gvpt2 = solve(w, T, U_MODEL, "gvpt2", DipoleDerivativesResult(mu1=MU1, mu2=MU2))
    ex = exact_levels(np.array(w), T, U_MODEL, MU1, MU2)
    return SimpleNamespace(
        result=gvpt2,
        gvpt2=by_quanta(gvpt2),
        exact={band: ex(f)[0] for band, (_, f) in BANDS.items()},
    )


def test_fermi_type2_detected(dyad):
    assert dyad.result.fermi_type2 == [(0, 1, 2)]


@pytest.mark.parametrize("band", BANDS)
def test_energy_matches_exact(dyad, band):
    assert abs(dyad.gvpt2[BANDS[band][0]].energy - dyad.exact[band]) < 3


def test_anharm_freq_holds_variational_fundamental(dyad):
    assert np.isclose(dyad.result.anharm_freq[0], dyad.gvpt2[((0, 1),)].energy)
