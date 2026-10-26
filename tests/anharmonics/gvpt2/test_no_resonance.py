"""Without resonances GVPT2 must reduce to standard VPT2."""
import numpy as np

from helpers.models import MU1, MU2, U_MODEL, by_quanta, solve, sym3
from overtone.anharmonics.dipole_derivatives import DipoleDerivativesResult


def test_gvpt2_equals_standard_without_resonance():
    w = [3050.0, 1650.0, 1180.0]
    T = sym3({(0, 0, 0): -300, (1, 1, 1): -120, (2, 2, 2): -90, (0, 0, 1): 40,
              (1, 1, 2): 55, (0, 1, 2): 30}, 3)
    dip = DipoleDerivativesResult(mu1=MU1, mu2=MU2)
    rg = solve(w, T, U_MODEL, "gvpt2", dip)
    rs = solve(w, T, U_MODEL, "standard", dip)
    g, s = by_quanta(rg), by_quanta(rs)
    assert rg.fermi_type1 == [] and rg.fermi_type2 == []
    for k in s:
        assert np.isclose(g[k].energy, s[k].energy)
        assert np.isclose(g[k].intensity, s[k].intensity)
