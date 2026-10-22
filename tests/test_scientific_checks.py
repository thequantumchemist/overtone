"""
Scientific regression tests for the VPT2 solver.

1. Model quartic force fields (dimensionless normal coordinates, cm^-1) are
   solved exactly in a product harmonic-oscillator basis and compared with
   VPT2 / deperturbed VPT2 + 2x2 diagonalisation (Martin et al. 1995, A10/A11).
2. The Coriolis contribution must not depend on molecular orientation/position.
"""
import contextlib
import io
import itertools

import numpy as np
import pytest
import scipy.sparse as sps
import scipy.sparse.linalg as spl
from ase.build import molecule
from ase.calculators.emt import EMT

from overtone import Harmonics, VPT2Config, VPT2Solver


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def sym3(d, n):
    T = np.zeros((n,) * 3)
    for idx, v in d.items():
        for p in set(itertools.permutations(idx)):
            T[p] = v
    return T


def sym4_iijj(diag, off, n):
    U = np.zeros((n,) * 4)
    for i, v in diag.items():
        U[i, i, i, i] = v
    for (i, j), v in off.items():
        for p in set(itertools.permutations((i, i, j, j))):
            U[p] = v
    return U


def exact_fundamentals(w, T, U, nmax=16):
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
    E, V = spl.eigsh(H, k=12, which="SA")
    order = np.argsort(E)
    E, V = E[order], V[:, order]
    out = []
    for i in range(n):
        v = np.zeros(n, int)
        v[i] = 1
        flat = np.ravel_multi_index(v, (nmax,) * n)
        out.append(E[np.argmax(np.abs(V[flat]) ** 2)] - E[0])
    return np.array(out)


def solve(w, T, U, method):
    n = len(w)
    cfg = VPT2Config(
        freq=np.asarray(w, float), active_modes=list(range(n)),
        min_mode_idx=0, max_mode_idx=n - 1, len_modes=n,
        method=method,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        return VPT2Solver(cfg, (T, U)).calculate()


def term_value(w, X, v):
    v = np.asarray(v, float)
    e = np.dot(w, v)
    for r in range(len(w)):
        e += X[r, r] * (v[r] ** 2 + v[r])
        for s in range(r + 1, len(w)):
            e += X[r, s] * (v[r] * v[s] + 0.5 * (v[r] + v[s]))
    return e  # term value relative to v = 0 (Martin et al. Eq. A4)


U_MODEL = sym4_iijj({0: 180, 1: 40, 2: 25}, {(0, 1): -20, (0, 2): -15, (1, 2): 10}, 3)


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------
def test_standard_vpt2_matches_exact_without_resonance():
    w = np.array([3050.0, 1650.0, 1180.0])
    T = sym3({(0, 0, 0): -300, (1, 1, 1): -120, (2, 2, 2): -90, (0, 0, 1): 40,
              (0, 1, 1): -60, (1, 1, 2): 55, (0, 0, 2): 30, (1, 2, 2): -45,
              (0, 2, 2): 25, (0, 1, 2): 70}, 3)
    exact = exact_fundamentals(w, T, U_MODEL)
    for method in ("standard", "deperturbed", "dcpt2"):
        nu = solve(w, T, U_MODEL, method).anharm_freq
        assert np.allclose(nu, exact, atol=1.0), method


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


def _coriolis(atoms):
    atoms = atoms.copy()
    atoms.calc = EMT()
    h = Harmonics(atoms, delta=0.005)
    traj = h.calculate_displacements()
    for a in traj:
        a.calc = EMT()
        a.get_forces()
    hr = h.calculate_hessian(traj)
    n = 3 * len(atoms)
    cfg = VPT2Config.from_harmonics_result(hr, active_modes=[6, 7, 8], atoms=atoms)
    s = VPT2Solver(cfg, (np.zeros((n,) * 3), np.zeros((n,) * 4)))
    return s.calculate_rotational_constants(), s.calculate_anharmonic_coriolis()


def test_coriolis_is_invariant_to_rotation_and_translation():
    from ase.optimize import BFGS

    ref = molecule("H2O")
    ref.calc = EMT()
    BFGS(ref, logfile=None).run(fmax=1e-6)  # VPT2 needs a minimum
    moved = ref.copy()
    moved.rotate(37, (1, 2, 3), center="COM")
    moved.translate((3.0, -1.0, 2.0))
    B1, X1 = _coriolis(ref)
    B2, X2 = _coriolis(moved)
    assert np.allclose(B1, B2, rtol=1e-6)
    # frame bug gave factor ~60 errors; remaining noise is finite-difference noise
    assert np.allclose(X1, X2, rtol=1e-2, atol=1e-2)
    I = ref.get_moments_of_inertia()  # ASE: principal moments about COM
    B_ref = 6.62607015e-34 / (8 * np.pi**2 * 2.99792458e10 * I * 1.66053906660e-27 * 1e-20)
    assert np.allclose(np.sort(B1), np.sort(B_ref), rtol=1e-6)
