"""
Without resonances, VPT2 fundamentals of a model quartic force field must
match the exact variational ones.
"""
import numpy as np

from helpers.models import U_MODEL, exact_fundamentals, solve, sym3


def test_standard_vpt2_matches_exact_without_resonance():
    w = np.array([3050.0, 1650.0, 1180.0])
    T = sym3({(0, 0, 0): -300, (1, 1, 1): -120, (2, 2, 2): -90, (0, 0, 1): 40,
              (0, 1, 1): -60, (1, 1, 2): 55, (0, 0, 2): 30, (1, 2, 2): -45,
              (0, 2, 2): 25, (0, 1, 2): 70}, 3)
    exact = exact_fundamentals(w, T, U_MODEL)
    for method in ("standard", "deperturbed", "dcpt2"):
        nu = solve(w, T, U_MODEL, method).anharm_freq
        assert np.allclose(nu, exact, atol=1.0), method
