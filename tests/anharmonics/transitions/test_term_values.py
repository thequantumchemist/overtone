"""Term values: Martin et al. 1995, Eqs. A5-A7."""
import itertools

import numpy as np

from overtone.anharmonics.transitions import term_value


def test_term_values_match_martin_A5_A7():
    rng = np.random.default_rng(0)
    n = 4
    w = np.array([3000.0, 1600.0, 1200.0, 700.0])
    X = rng.normal(0, 10, (n, n))
    X = (X + X.T) / 2
    act = list(range(n))
    for i in act:
        others = [r for r in act if r != i]
        nu_i = w[i] + 2 * X[i, i] + 0.5 * sum(X[i, r] for r in others)
        assert np.isclose(term_value(w, X, {i: 1}, act), nu_i)
        two = 2 * w[i] + 6 * X[i, i] + sum(X[i, r] for r in others)
        assert np.isclose(term_value(w, X, {i: 2}, act), two)
    for i, j in itertools.combinations(act, 2):
        rest = [r for r in act if r not in (i, j)]
        comb = (w[i] + w[j] + 2 * X[i, i] + 2 * X[j, j] + 2 * X[i, j]
                + 0.5 * sum(X[i, r] + X[j, r] for r in rest))
        assert np.isclose(term_value(w, X, {i: 1, j: 1}, act), comb)
