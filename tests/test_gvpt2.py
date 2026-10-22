"""
GVPT2 against exact variational results on model force fields.

H = sum w/2 (p^2 + q^2) + 1/6 sum phi_ijk qqq + 1/24 sum phi_iijj qqqq
mu = sum mu_i q_i + 1/2 sum mu_ij q_i q_j

For a Fermi dyad, standard VPT2 fails, while GVPT2 must reproduce both exact
levels and how the intensity is shared between them.
"""
import contextlib
import io
import itertools

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spl

from overtone import VPT2Config, VPT2Solver
from overtone.anharmonics.dipole_derivatives import DipoleDerivativesResult
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


def exact(w, T, U, mu1, mu2, nmax=14, nstates=60):
    n = len(w)
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

    Q = [op(q, i) for i in range(n)]
    H = sum(w[i] * op(num, i) for i in range(n))
    for i, j, k in itertools.product(range(n), repeat=3):
        if T[i, j, k]:
            H = H + T[i, j, k] / 6 * (Q[i] @ Q[j] @ Q[k])
    for i, j, k, l in itertools.product(range(n), repeat=4):
        if U[i, j, k, l]:
            H = H + U[i, j, k, l] / 24 * (Q[i] @ Q[j] @ Q[k] @ Q[l])
    E, V = spl.eigsh(H, k=nstates, which="SA")
    o = np.argsort(E)
    E, V = E[o], V[:, o]
    MU = [sum(mu1[i, c] * Q[i] for i in range(n))
          + sum(0.5 * mu2[i, j, c] * (Q[i] @ Q[j]) for i in range(n) for j in range(n))
          for c in range(3)]
    psi0 = V[:, 0]

    def level(f):
        idx = np.ravel_multi_index(f, (nmax,) * n)
        s = np.argmax(np.abs(V[idx]) ** 2)
        M = np.array([psi0 @ (MU[c] @ V[:, s]) for c in range(3)])
        nu = E[s] - E[0]
        return nu, INTENSITY_CONSTANT * nu * M @ M

    return level


def solve(w, T, U, method):
    n = len(w)
    cfg = VPT2Config(freq=np.asarray(w, float), active_modes=list(range(n)),
                     min_mode_idx=0, max_mode_idx=n - 1, len_modes=n, method=method)
    dip = DipoleDerivativesResult(mu1=MU1, mu2=MU2)
    with contextlib.redirect_stdout(io.StringIO()):
        res = VPT2Solver(cfg, (T, U), dipole_derivatives=dip).calculate()
    return {tuple(sorted(t.quanta.items())): t for t in res.transitions}, res


def test_gvpt2_equals_standard_without_resonance():
    w = [3050.0, 1650.0, 1180.0]
    T = sym3({(0, 0, 0): -300, (1, 1, 1): -120, (2, 2, 2): -90, (0, 0, 1): 40,
              (1, 1, 2): 55, (0, 1, 2): 30}, 3)
    g, rg = solve(w, T, U_MODEL, "gvpt2")
    s, rs = solve(w, T, U_MODEL, "standard")
    assert rg.fermi_type1 == [] and rg.fermi_type2 == []
    for k in s:
        assert np.isclose(g[k].energy, s[k].energy)
        assert np.isclose(g[k].intensity, s[k].intensity)


def test_gvpt2_type1_dyad_energies_and_intensities():
    # w0 ~ 2 w1: |1_0> and |2_1> mix; only mode 0 carries intensity directly
    w = [3200.0, 1608.0, 1180.0]
    T = sym3({(0, 0, 0): -300, (1, 1, 1): -120, (2, 2, 2): -90, (0, 1, 1): -150,
              (1, 1, 2): 55, (0, 0, 2): 30}, 3)
    g, res = solve(w, T, U_MODEL, "gvpt2")
    std, _ = solve(w, T, U_MODEL, "standard")
    ex = exact(np.array(w), T, U_MODEL, MU1, MU2)
    assert res.fermi_type1 == [(0, 1)]
    fund, over = g[((0, 1),)], g[((1, 2),)]
    e_f, i_f = ex((1, 0, 0))
    e_o, i_o = ex((0, 2, 0))
    # standard VPT2 is far off, GVPT2 matches the exact levels
    assert abs(std[((0, 1),)].energy - e_f) > 30
    assert abs(fund.energy - e_f) < 3 and abs(over.energy - e_o) < 3
    # intensity borrowing: the dark overtone gains a large share
    total = i_f + i_o
    assert abs(fund.intensity / total - i_f / total) < 0.05
    assert abs(over.intensity / total - i_o / total) < 0.05
    assert fund.mixing and over.mixing and fund.resonance_affected


def test_gvpt2_type2_dyad_energies():
    # w0 ~ w1 + w2
    w = [2850.0, 1650.0, 1190.0]
    T = sym3({(0, 0, 0): -300, (1, 1, 1): -120, (2, 2, 2): -90, (0, 1, 2): -140,
              (0, 0, 1): 30, (1, 1, 2): 40}, 3)
    g, res = solve(w, T, U_MODEL, "gvpt2")
    ex = exact(np.array(w), T, U_MODEL, MU1, MU2)
    assert res.fermi_type2 == [(0, 1, 2)]
    assert abs(g[((0, 1),)].energy - ex((1, 0, 0))[0]) < 3
    assert abs(g[((1, 1), (2, 1))].energy - ex((0, 1, 1))[0]) < 3
    # the fundamentals in anharm_freq are the variational ones
    assert np.isclose(res.anharm_freq[0], g[((0, 1),)].energy)
