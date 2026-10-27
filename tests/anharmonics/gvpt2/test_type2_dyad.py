"""GVPT2 for a type-2 Fermi dyad (w0 ~ w1 + w2) against exact levels."""
import numpy as np

from helpers.models import MU1, MU2, U_MODEL, by_quanta, exact_levels, solve, sym3
from overtone.anharmonics.dipole_derivatives import DipoleDerivativesResult


def test_gvpt2_type2_dyad_energies():
    # w0 ~ w1 + w2
    w = [2850.0, 1650.0, 1190.0]
    T = sym3({(0, 0, 0): -300, (1, 1, 1): -120, (2, 2, 2): -90, (0, 1, 2): -140,
              (0, 0, 1): 30, (1, 1, 2): 40}, 3)
    res = solve(w, T, U_MODEL, "gvpt2", DipoleDerivativesResult(mu1=MU1, mu2=MU2))
    g = by_quanta(res)
    ex = exact_levels(np.array(w), T, U_MODEL, MU1, MU2)
    assert res.fermi_type2 == [(0, 1, 2)]
    assert abs(g[((0, 1),)].energy - ex((1, 0, 0))[0]) < 3
    assert abs(g[((1, 1), (2, 1))].energy - ex((0, 1, 1))[0]) < 3
    # the fundamentals in anharm_freq are the variational ones
    assert np.isclose(res.anharm_freq[0], g[((0, 1),)].energy)
