"""
Model quartic force fields in dimensionless normal coordinates (cm^-1).

H = sum w/2 (p^2 + q^2) + 1/6 sum phi_ijk qqq + 1/24 sum phi_iijj qqqq
mu = sum mu_i q_i + 1/2 sum mu_ij q_i q_j

Includes an exact variational solver in a product harmonic-oscillator basis,
used as reference for VPT2 and GVPT2.
"""
import itertools

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spl

from overtone import VPT2Config, VPT2Solver
from overtone.anharmonics.transitions import INTENSITY_CONSTANT


def sym3(d, n):
    T = np.zeros((n,) * 3)
    for idx, v in d.items():
        for p in set(itertools.permutations(idx)):
            T[p] = v
    return T


def sym4(diag, off, n):
    U = np.zeros((n,) * 4)
    for i, v in diag.items():
        U[i, i, i, i] = v
    for (i, j), v in off.items():
        for p in set(itertools.permutations((i, i, j, j))):
            U[p] = v
    return U


U_MODEL = sym4({0: 180, 1: 40, 2: 25}, {(0, 1): -20, (0, 2): -15, (1, 2): 10}, 3)
MU1 = np.array([[0.00, 0.00, 0.05], [0.00, 0.00, 0.30], [0.00, 0.12, 0.00]])
MU2 = np.zeros((3, 3, 3))
MU2[1, 1, 2] = 0.01
MU2[1, 2, 1] = MU2[2, 1, 1] = 0.0


def model_force_field(w, seed=1):
    """
    Random but physically scaled constants: |phi_ijk| ~ 0.002 sqrt(w_i w_j w_k)
    and |phi_iijj| ~ 1e-5 w_i w_j (about 60 and 10 cm^-1 at 1000 cm^-1), so
    that soft modes are not unrealistically anharmonic.
    """
    rng = np.random.default_rng(seed)
    n = len(w)
    T = np.zeros((n,) * 3)
    for idx in itertools.combinations_with_replacement(range(n), 3):
        v = rng.normal(0, 1) * 0.002 * np.sqrt(np.prod(w[list(idx)]))
        for p in set(itertools.permutations(idx)):
            T[p] = v
    U = np.zeros((n,) * 4)
    for i, j in itertools.combinations_with_replacement(range(n), 2):
        scale = 1e-5 * w[i] * w[j]
        v = abs(rng.normal(0, 1)) * scale if i == j else rng.normal(0, 1) * scale
        for p in set(itertools.permutations((i, i, j, j))):
            U[p] = v
    return T, U


def ho_operators(n, nmax):
    """q_i and (n_i + 1/2) for n modes in a product basis with nmax levels each."""
    eye = sps.identity(nmax, format="csr")
    a = sps.diags(np.sqrt(np.arange(1, nmax)), 1, format="csr")
    q = (a + a.T) / np.sqrt(2)
    num = sps.diags(np.arange(nmax) + 0.5, format="csr")

    def op(single, idx):
        out = None
        for m in range(n):
            mat = single if m == idx else eye
            out = mat if out is None else sps.kron(out, mat, format="csr")
        return out

    return [op(q, i) for i in range(n)], [op(num, i) for i in range(n)]


def exact_levels(w, T, U, mu1=None, mu2=None, nmax=14, nstates=60):
    """
    Diagonalise the model Hamiltonian. Returns level(f) giving the transition
    energy from the ground state to the state with the largest weight on the
    basis function f, and with dipoles also its intensity (km/mol).
    """
    n = len(w)
    Q, N = ho_operators(n, nmax)
    H = sum(w[i] * N[i] for i in range(n))
    for i, j, k in itertools.product(range(n), repeat=3):
        if T[i, j, k]:
            H = H + T[i, j, k] / 6 * (Q[i] @ Q[j] @ Q[k])
    for i, j, k, l in itertools.product(range(n), repeat=4):
        if U[i, j, k, l]:
            H = H + U[i, j, k, l] / 24 * (Q[i] @ Q[j] @ Q[k] @ Q[l])
    E, V = spl.eigsh(H, k=nstates, which="SA")
    o = np.argsort(E)
    E, V = E[o], V[:, o]
    if mu1 is not None:
        MU = [sum(mu1[i, c] * Q[i] for i in range(n))
              + sum(0.5 * mu2[i, j, c] * (Q[i] @ Q[j]) for i in range(n) for j in range(n))
              for c in range(3)]
    psi0 = V[:, 0]

    def level(f):
        idx = np.ravel_multi_index(f, (nmax,) * n)
        s = np.argmax(np.abs(V[idx]) ** 2)
        nu = E[s] - E[0]
        if mu1 is None:
            return nu
        M = np.array([psi0 @ (MU[c] @ V[:, s]) for c in range(3)])
        return nu, INTENSITY_CONSTANT * nu * M @ M

    return level


def exact_fundamentals(w, T, U, nmax=16):
    level = exact_levels(w, T, U, nmax=nmax, nstates=12)
    return np.array([level(tuple(np.eye(len(w), dtype=int)[i])) for i in range(len(w))])


def solve(w, T, U, method, dipole_derivatives=None):
    n = len(w)
    cfg = VPT2Config(freq=np.asarray(w, float), active_modes=list(range(n)),
                     min_mode_idx=0, max_mode_idx=n - 1, len_modes=n, method=method)
    return VPT2Solver(cfg, (T, U), dipole_derivatives=dipole_derivatives).calculate()


def by_quanta(result):
    """Transitions of a VPT2Result keyed by their sorted (mode, quanta) pairs."""
    return {tuple(sorted(t.quanta.items())): t for t in result.transitions}


def term_value(w, X, v):
    """Reference term value relative to v = 0 (Martin et al. 1995, Eq. A4)."""
    v = np.asarray(v, float)
    e = np.dot(w, v)
    for r in range(len(w)):
        e += X[r, r] * (v[r] ** 2 + v[r])
        for s in range(r + 1, len(w)):
            e += X[r, s] * (v[r] * v[s] + 0.5 * (v[r] + v[s]))
    return e
