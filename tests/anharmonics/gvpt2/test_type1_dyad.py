"""
GVPT2 for a type-1 Fermi dyad (w0 ~ 2 w1): standard VPT2 fails, while GVPT2
must reproduce both exact levels and how the intensity is shared between them.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from helpers.models import MU1, MU2, U_MODEL, by_quanta, exact_levels, solve, sym3
from overtone.anharmonics.dipole_derivatives import DipoleDerivativesResult

# |1_0> and |2_1> mix; only mode 0 carries intensity directly
BANDS = {"fundamental": (((0, 1),), (1, 0, 0)), "overtone": (((1, 2),), (0, 2, 0))}


@pytest.fixture(scope="module")
def dyad():
    w = [3200.0, 1608.0, 1180.0]  # w0 ~ 2 w1
    T = sym3({(0, 0, 0): -300, (1, 1, 1): -120, (2, 2, 2): -90, (0, 1, 1): -150,
              (1, 1, 2): 55, (0, 0, 2): 30}, 3)
    dip = DipoleDerivativesResult(mu1=MU1, mu2=MU2)
    gvpt2 = solve(w, T, U_MODEL, "gvpt2", dip)
    ex = exact_levels(np.array(w), T, U_MODEL, MU1, MU2)
    return SimpleNamespace(
        result=gvpt2,
        gvpt2=by_quanta(gvpt2),
        standard=by_quanta(solve(w, T, U_MODEL, "standard", dip)),
        exact={band: ex(f) for band, (_, f) in BANDS.items()},
    )


def test_fermi_type1_detected(dyad):
    assert dyad.result.fermi_type1 == [(0, 1)]


def test_standard_vpt2_misses_fundamental(dyad):
    e_exact, _ = dyad.exact["fundamental"]
    assert abs(dyad.standard[((0, 1),)].energy - e_exact) > 30


@pytest.mark.parametrize("band", BANDS)
def test_energy_matches_exact(dyad, band):
    e_exact, _ = dyad.exact[band]
    assert abs(dyad.gvpt2[BANDS[band][0]].energy - e_exact) < 3


@pytest.mark.parametrize("band", BANDS)
def test_intensity_share_matches_exact(dyad, band):
    # intensity borrowing: the dark overtone gains a large share
    total = sum(i for _, i in dyad.exact.values())
    _, i_exact = dyad.exact[band]
    assert abs(dyad.gvpt2[BANDS[band][0]].intensity / total - i_exact / total) < 0.05


@pytest.mark.parametrize("band", BANDS)
def test_band_is_mixed(dyad, band):
    assert dyad.gvpt2[BANDS[band][0]].mixing


def test_fundamental_is_resonance_affected(dyad):
    assert dyad.gvpt2[((0, 1),)].resonance_affected
