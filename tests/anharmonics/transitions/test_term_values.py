"""Term values: Martin et al. 1995, Eqs. A5-A7."""
import itertools

import numpy as np
import pytest

from overtone.anharmonics.transitions import term_value

W = np.array([3000.0, 1600.0, 1200.0, 700.0])
ACTIVE = list(range(len(W)))
_X = np.random.default_rng(0).normal(0, 10, (len(W), len(W)))
X = (_X + _X.T) / 2


def _others(*modes):
    return [r for r in ACTIVE if r not in modes]


@pytest.mark.parametrize("i", ACTIVE)
def test_fundamental_matches_A5(i):
    nu_i = W[i] + 2 * X[i, i] + 0.5 * sum(X[i, r] for r in _others(i))
    assert np.isclose(term_value(W, X, {i: 1}, ACTIVE), nu_i)


@pytest.mark.parametrize("i", ACTIVE)
def test_first_overtone_matches_A6(i):
    two = 2 * W[i] + 6 * X[i, i] + sum(X[i, r] for r in _others(i))
    assert np.isclose(term_value(W, X, {i: 2}, ACTIVE), two)


@pytest.mark.parametrize("i, j", list(itertools.combinations(ACTIVE, 2)))
def test_binary_combination_matches_A7(i, j):
    comb = (W[i] + W[j] + 2 * X[i, i] + 2 * X[j, j] + 2 * X[i, j]
            + 0.5 * sum(X[i, r] + X[j, r] for r in _others(i, j)))
    assert np.isclose(term_value(W, X, {i: 1, j: 1}, ACTIVE), comb)
