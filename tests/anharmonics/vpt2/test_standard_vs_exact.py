"""
Without resonances, VPT2 fundamentals of a model quartic force field must
match the exact variational ones.
"""
import numpy as np
import pytest

from helpers.models import U_MODEL, exact_fundamentals, solve, sym3

W = np.array([3050.0, 1650.0, 1180.0])
T = sym3({(0, 0, 0): -300, (1, 1, 1): -120, (2, 2, 2): -90, (0, 0, 1): 40,
          (0, 1, 1): -60, (1, 1, 2): 55, (0, 0, 2): 30, (1, 2, 2): -45,
          (0, 2, 2): 25, (0, 1, 2): 70}, 3)


@pytest.fixture(scope="module")
def exact():
    return exact_fundamentals(W, T, U_MODEL)


@pytest.mark.parametrize("method", ["standard", "deperturbed", "dcpt2"])
def test_fundamentals_match_exact_without_resonance(method, exact):
    assert np.allclose(solve(W, T, U_MODEL, method).anharm_freq, exact, atol=1.0)
