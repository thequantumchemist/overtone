"""
Deperturbed VPT2 + 2x2 diagonalisation of a Fermi dyad must match the exact
variational fundamental (Martin et al. 1995, Eqs. A10/A11).
"""
from types import SimpleNamespace

import numpy as np
import pytest

from helpers.models import U_MODEL, exact_fundamentals, solve, sym3, term_value

TYPE1_T = sym3({(0, 0, 0): -300, (1, 1, 1): -120, (2, 2, 2): -90, (0, 1, 1): -150,
                (1, 1, 2): 55, (0, 0, 2): 30, (0, 1, 2): 20}, 3)
TYPE2_T = sym3({(0, 0, 0): -300, (1, 1, 1): -120, (2, 2, 2): -90, (0, 1, 2): -140,
                (0, 0, 1): 30, (1, 1, 2): 40}, 3)
CASES = {
    # 2 w1 ~ w0: |1_0> couples to |2_1>
    "type1": dict(w=np.array([3200.0, 1608.0, 1180.0]), T=TYPE1_T,
                  resonances=("fermi_type1", [(0, 1)]),
                  partner=[0, 2, 0], coupling=TYPE1_T[1, 1, 0] / 4),
    # w1 + w2 ~ w0: |1_0> couples to |1_1 1_2>
    "type2": dict(w=np.array([2850.0, 1650.0, 1190.0]), T=TYPE2_T,
                  resonances=("fermi_type2", [(0, 1, 2)]),
                  partner=[0, 1, 1], coupling=TYPE2_T[0, 1, 2] / np.sqrt(8)),
}


@pytest.fixture(scope="module", params=CASES)
def dyad(request):
    case = SimpleNamespace(**CASES[request.param])
    case.result = solve(case.w, case.T, U_MODEL, "deperturbed")
    return case


def test_resonance_detected(dyad):
    attr, expected = dyad.resonances
    assert getattr(dyad.result, attr) == expected


def test_anharmonic_constants_symmetric(dyad):
    assert np.allclose(dyad.result.X, dyad.result.X.T)


def test_dyad_diagonalisation_matches_exact(dyad):
    X, w = dyad.result.X, dyad.w
    H = np.array([[term_value(w, X, [1, 0, 0]), dyad.coupling],
                  [dyad.coupling, term_value(w, X, dyad.partner)]])
    exact = exact_fundamentals(w, dyad.T, U_MODEL)[0]
    assert np.min(np.abs(np.linalg.eigvalsh(H) - exact)) < 1.0
