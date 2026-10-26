"""
GVPT2 for a type-1 Fermi dyad (w0 ~ 2 w1): standard VPT2 fails, while GVPT2
must reproduce both exact levels and how the intensity is shared between them.
"""
import numpy as np

from helpers.models import MU1, MU2, U_MODEL, by_quanta, exact_levels, solve, sym3
from overtone.anharmonics.dipole_derivatives import DipoleDerivativesResult


def test_gvpt2_type1_dyad_energies_and_intensities():
    # w0 ~ 2 w1: |1_0> and |2_1> mix; only mode 0 carries intensity directly
    w = [3200.0, 1608.0, 1180.0]
    T = sym3({(0, 0, 0): -300, (1, 1, 1): -120, (2, 2, 2): -90, (0, 1, 1): -150,
              (1, 1, 2): 55, (0, 0, 2): 30}, 3)
    dip = DipoleDerivativesResult(mu1=MU1, mu2=MU2)
    res = solve(w, T, U_MODEL, "gvpt2", dip)
    g = by_quanta(res)
    std = by_quanta(solve(w, T, U_MODEL, "standard", dip))
    ex = exact_levels(np.array(w), T, U_MODEL, MU1, MU2)
    assert res.fermi_type1 == [(0, 1)]
    fund, over = g[((0, 1),)], g[((1, 2),)]
    e_f, i_f = ex((1, 0, 0))
    e_o, i_o = ex((0, 2, 0))
    # standard VPT2 is far off, GVPT2 matches the exact levels
    assert abs(std[((0, 1),)].energy - e_f) > 30
    assert abs(fund.energy - e_f) < 3 and abs(over.energy - e_o) < 3
    # intensity borrowing: the dark overtone gains a large share
    total = i_f + i_o
    assert abs(fund.intensity / total - i_f / total) < 0.05
    assert abs(over.intensity / total - i_o / total) < 0.05
    assert fund.mixing and over.mixing and fund.resonance_affected
