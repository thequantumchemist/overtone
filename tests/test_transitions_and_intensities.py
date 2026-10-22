"""
Tests for overtones/combination bands, IR intensities and spectrum broadening.
"""
import contextlib
import io
import itertools
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sps
import scipy.sparse.linalg as spl
from ase.build import molecule
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.emt import EMT
from ase.optimize import BFGS

from overtone import calculate_anharmonics
from overtone.anharmonics.transitions import (
    INTENSITY_CONSTANT,
    TransitionCalculator,
    term_value,
)
from overtone.spectrum import broaden, broaden_transitions


# ----------------------------------------------------------------------
# term values: Martin et al. 1995, Eqs. A5-A7
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# transition moments vs exact variational ones (leading order => error ~ lambda^2)
# ----------------------------------------------------------------------
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
    H0 = sum(w[i] * op(num, i) for i in range(n))

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


# ----------------------------------------------------------------------
# end-to-end: harmonic intensities vs ASE, and "on request" behaviour
# ----------------------------------------------------------------------
class ChargedEMT(Calculator):
    """EMT energies/forces plus a fixed-point-charge dipole (e*Angstrom)."""

    implemented_properties = ["energy", "forces", "dipole"]

    def __init__(self, charges, with_dipole=True, **kw):
        super().__init__(**kw)
        self.charges = np.asarray(charges, float)
        self.with_dipole = with_dipole
        self._emt = EMT()

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        if "dipole" in properties and not self.with_dipole:
            raise RuntimeError("this calculator cannot provide dipoles")
        a = self.atoms.copy()
        a.calc = self._emt
        self.results["energy"] = a.get_potential_energy()
        self.results["forces"] = a.get_forces()
        if self.with_dipole:
            # nonlinear in the geometry, so second derivatives are non-zero
            r = self.atoms.positions - self.atoms.positions.mean(0)
            self.results["dipole"] = self.charges @ (r + 0.1 * r**2)


@pytest.fixture(scope="module")
def water():
    atoms = molecule("H2O")
    atoms.calc = ChargedEMT([-0.8, 0.4, 0.4])
    BFGS(atoms, logfile=None).run(fmax=1e-6)
    return atoms


def _run(atoms, tmp, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return calculate_anharmonics(atoms.copy(), directory=Path(tmp),
                                     anharmonic_delta=0.1, stencil=1, **kw)


def test_harmonic_intensities_match_ase(water, tmp_path):
    from ase.vibrations import Infrared

    res = _run(water, tmp_path / "vpt2",
               setup_calculator=lambda: ChargedEMT([-0.8, 0.4, 0.4]),
               intensities=True)
    ir_atoms = water.copy()
    ir_atoms.calc = ChargedEMT([-0.8, 0.4, 0.4])
    with contextlib.redirect_stdout(io.StringIO()):
        ir = Infrared(ir_atoms, name=str(tmp_path / "ir"), delta=0.005)
        ir.run()
        ir.read()
    ase_km = np.sort(ir.intensities[6:] * 42.255)  # ASE: (D/A)^2/amu -> km/mol
    fund = [t for t in res.transitions if t.kind == "fundamental"]
    ours = np.sort([INTENSITY_CONSTANT * t.harmonic * np.dot(t.transition_dipole, t.transition_dipole)
                    for t in fund])
    assert np.allclose(ours, ase_km, rtol=0.02, atol=1e-3)
    assert all(t.intensity is not None for t in res.transitions if t.total_quanta <= 2)
    kinds = {t.kind for t in res.transitions}
    assert kinds == {"fundamental", "overtone", "combination"}


def test_intensities_are_only_computed_on_request(water, tmp_path):
    # a calculator without dipoles must work as long as intensities=False
    res = _run(water, tmp_path / "nodip",
               setup_calculator=lambda: ChargedEMT([0, 0, 0], with_dipole=False),
               max_quanta=3)
    assert all(t.intensity is None for t in res.transitions)
    assert len(res.transitions) == 3 + 6 + 10  # 1, 2 and 3 quanta in 3 modes
    with pytest.raises(ValueError, match="intensities=True"):
        broaden_transitions(res.transitions)


def test_broadening_conserves_band_areas():
    pos, inten = [1000.0, 1500.0, 3000.0], [10.0, 1.0, 50.0]
    for shape in ("lorentzian", "gaussian"):
        x, y = broaden(pos, inten, shape=shape, fwhm=10.0,
                       xmin=0, xmax=6000, resolution=0.1)
        tol = 0.01 if shape == "lorentzian" else 1e-6  # Lorentzian tails
        assert np.isclose(np.trapezoid(y, x), sum(inten), rtol=tol)


def test_plot_spectrum_with_reference(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from overtone.anharmonics.transitions import VPT2Transition
    from overtone.spectrum import plot_spectrum

    bands = [
        VPT2Transition(quanta={6: 1}, kind="fundamental", label="v7",
                       harmonic=1650.0, energy=1600.0, intensity=60.0),
        VPT2Transition(quanta={6: 2}, kind="overtone", label="2v7",
                       harmonic=3300.0, energy=3150.0, intensity=0.5),
    ]
    ref = {"positions": [1595.0, 3152.0], "intensities": [70.0, 0.4], "label": "exp."}
    ax = plot_spectrum(bands, reference=ref)                       # mirrored
    assert ax.get_ylim()[0] < 0
    ax = plot_spectrum(bands, reference=ref, reference_mode="overlay", log=True)
    with pytest.raises(ValueError, match="log axis"):
        plot_spectrum(bands, reference=ref, log=True)
    ax.figure.savefig(tmp_path / "spec.png")
    assert (tmp_path / "spec.png").stat().st_size > 0


def test_separate_harmonic_calculator(water, tmp_path):
    """The Hessian may come from a different calculator than the force field."""
    calls = {"harm": 0, "anh": 0}

    def harm_calc():
        calls["harm"] += 1
        return ChargedEMT([-0.8, 0.4, 0.4])

    def anh_calc():
        calls["anh"] += 1
        return ChargedEMT([-0.8, 0.4, 0.4])

    with contextlib.redirect_stdout(io.StringIO()):
        calculate_anharmonics(water.copy(), anh_calc, tmp_path,
                              setup_harmonic_calculator=harm_calc, stencil=0)
    assert calls["harm"] == 19        # 1 + 6 * 3 atoms
    assert calls["anh"] > 0


def test_recompute_vpt2_from_saved_run(water, tmp_path):
    from overtone import recompute_vpt2

    with contextlib.redirect_stdout(io.StringIO()):
        first = calculate_anharmonics(
            water.copy(), directory=Path(tmp_path),
            setup_calculator=lambda: ChargedEMT([-0.8, 0.4, 0.4]),
            intensities=True, save_results=True, stencil=0)
        same = recompute_vpt2(tmp_path, method="standard")
        dep = recompute_vpt2(tmp_path, method="deperturbed", max_quanta=3)
    assert np.allclose(first.anharm_freq, same.anharm_freq)
    assert same.has_intensities
    assert dep.method == "deperturbed"
    assert max(t.total_quanta for t in dep.transitions) == 3
