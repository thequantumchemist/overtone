"""
Deperturbed VPT2 + 2x2 diagonalisation of a Fermi dyad must match the exact
variational fundamental (Martin et al. 1995, Eqs. A10/A11).
"""
import numpy as np

from helpers.models import U_MODEL, exact_fundamentals, solve, sym3, term_value


def test_deperturbed_type1_plus_diagonalisation_matches_exact():
    w = np.array([3200.0, 1608.0, 1180.0])  # 2 w1 ~ w0
    T = sym3({(0, 0, 0): -300, (1, 1, 1): -120, (2, 2, 2): -90, (0, 1, 1): -150,
              (1, 1, 2): 55, (0, 0, 2): 30, (0, 1, 2): 20}, 3)
    r = solve(w, T, U_MODEL, "deperturbed")
    assert r.fermi_type1 == [(0, 1)]
    assert np.allclose(r.X, r.X.T)
    H = np.array([[term_value(w, r.X, [1, 0, 0]), T[1, 1, 0] / 4],
                  [T[1, 1, 0] / 4, term_value(w, r.X, [0, 2, 0])]])
    exact = exact_fundamentals(w, T, U_MODEL)[0]
    assert np.min(np.abs(np.linalg.eigvalsh(H) - exact)) < 1.0


def test_deperturbed_type2_plus_diagonalisation_matches_exact():
    w = np.array([2850.0, 1650.0, 1190.0])  # w1 + w2 ~ w0
    T = sym3({(0, 0, 0): -300, (1, 1, 1): -120, (2, 2, 2): -90, (0, 1, 2): -140,
              (0, 0, 1): 30, (1, 1, 2): 40}, 3)
    r = solve(w, T, U_MODEL, "deperturbed")
    assert r.fermi_type2 == [(0, 1, 2)]
    assert np.allclose(r.X, r.X.T)
    H = np.array([[term_value(w, r.X, [1, 0, 0]), T[0, 1, 2] / np.sqrt(8)],
                  [T[0, 1, 2] / np.sqrt(8), term_value(w, r.X, [0, 1, 1])]])
    exact = exact_fundamentals(w, T, U_MODEL)[0]
    assert np.min(np.abs(np.linalg.eigvalsh(H) - exact)) < 1.0
