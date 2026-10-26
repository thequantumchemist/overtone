"""
Two-quanta transition moments vs exact variational ones
(leading order => error ~ lambda^2).
"""
import itertools

import numpy as np
import scipy.sparse.linalg as spl

from helpers.models import ho_operators
from overtone.anharmonics.transitions import TransitionCalculator


def test_two_quanta_transition_moments_converge_to_exact():
    rng = np.random.default_rng(7)
    n, nmax = 3, 12
    w = np.array([3000.0, 1450.0, 1100.0])
    T0 = np.zeros((n,) * 3)
    for idx in itertools.combinations_with_replacement(range(n), 3):
        v = rng.normal(0, 150)
        for p in set(itertools.permutations(idx)):
            T0[p] = v
    mu1 = rng.normal(0, 0.1, (n, 3))
    mu2 = rng.normal(0, 0.03, (n, n, 3))
    mu2 = (mu2 + mu2.transpose(1, 0, 2)) / 2

    Q, N = ho_operators(n, nmax)
    H0 = sum(w[i] * N[i] for i in range(n))

    def max_error(lam):
        T, m2 = lam * T0, lam * mu2
        H = H0 + sum(T[i, j, k] / 6 * (Q[i] @ Q[j] @ Q[k])
                     for i, j, k in itertools.product(range(n), repeat=3))
        # confining quartic: enters the wavefunction only at second order
        H = H + sum(lam**2 * 60 / 24 * (Q[i] @ Q[i] @ Q[i] @ Q[i]) for i in range(n))
        E, V = spl.eigsh(H, k=80, which="SA")

        def state(f):
            idx = np.ravel_multi_index(f, (nmax,) * n)
            v = V[:, np.argmax(np.abs(V[idx]))]
            return v * np.sign(v[idx])

        psi0 = state((0, 0, 0))
        tc = TransitionCalculator(w, np.zeros((n, n)), range(n), T=T, mu1=mu1, mu2=m2)
        errs = []
        for f in [(2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0), (1, 0, 1), (0, 1, 1)]:
            pf = state(f)
            Mex = np.array([
                psi0 @ ((sum(mu1[i, c] * Q[i] for i in range(n))
                         + sum(0.5 * m2[i, j, c] * (Q[i] @ Q[j])
                               for i in range(n) for j in range(n))) @ pf)
                for c in range(3)
            ])
            Mpt = tc.transition_moment({m: v for m, v in enumerate(f) if v})
            errs.append(np.linalg.norm(Mex - Mpt) / np.linalg.norm(Mex))
        return max(errs)

    e1, e2 = max_error(0.25), max_error(0.125)
    assert e2 < 0.02
    assert 3.0 < e1 / e2 < 5.0  # error scales as lambda^2
