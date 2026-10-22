"""
Tests for the harmonic analysis options and an exact end-to-end check.

The end-to-end test uses an ASE calculator whose potential IS a quartic force
field in dimensionless normal coordinates. The whole pipeline (Hessian,
projection, displacements, unit conversions, energy- and force-based
derivatives, VPT2) must then reproduce VPT2 evaluated directly from the
model constants.
"""
import contextlib
import io
import itertools
import warnings
from pathlib import Path

import numpy as np
import pytest
from ase.build import molecule
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.emt import EMT
from ase.optimize import BFGS

from overtone import (
    CoordinateTransformer,
    Harmonics,
    VPT2Config,
    VPT2Solver,
    calculate_anharmonics,
    estimate_cost,
)
from overtone.harmonics.harmonics import is_linear_molecule, rigid_body_basis

EV_TO_INV_CM = 8065.543937


def _optimised(name):
    atoms = molecule(name)
    atoms.calc = EMT()
    BFGS(atoms, logfile=None).run(fmax=1e-7)
    return atoms


def _harmonics(atoms, calc_factory=EMT, **kw):
    h = Harmonics(atoms.copy(), delta=0.005, **kw)
    traj = h.calculate_displacements()
    for a in traj:
        a.calc = calc_factory()
        a.get_forces()
    return h.calculate_hessian(traj)


# ----------------------------------------------------------------------
# harmonic analysis
# ----------------------------------------------------------------------
@pytest.mark.parametrize("name", ["H2O", "CH3NO2"])
def test_projection_gives_pure_vibrations(name):
    atoms = _optimised(name)
    on = _harmonics(atoms)
    off = _harmonics(atoms, project_translations_rotations=False)
    assert on.projected and on.n_tr == 6
    assert np.all(on.freq[:6] == 0) and np.all(on.ifreq[:6] == 0)
    T = rigid_body_basis(atoms, linear=False)
    assert np.abs(T.T @ on.modes[:, 6:]).max() < 1e-10
    # modes are orthonormal
    assert np.allclose(on.modes.T @ on.modes, np.eye(len(on.freq)), atol=1e-10)
    # at a tight minimum the vibrational frequencies barely change
    assert np.allclose(on.freq[6:], off.freq[6:], atol=0.5)


def test_symmetrisation_flag_and_diagnostic():
    atoms = _optimised("CH3NO2")
    sym = _harmonics(atoms, symmetrize_hessian=True)
    raw = _harmonics(atoms, symmetrize_hessian=False)
    assert sym.symmetrized and not raw.symmetrized
    assert 0 < sym.hessian_asymmetry < 1e-3
    assert np.allclose(sym.freq, raw.freq, atol=1.0)


def test_linear_molecule_detection():
    co2 = _optimised("CO2")
    assert is_linear_molecule(co2)
    assert not is_linear_molecule(_optimised("H2O"))
    r = _harmonics(co2)
    assert r.is_linear and r.n_tr == 5 and len(r.vibrational_modes) == 4
    # the override wins
    assert not Harmonics(co2, linear=False).linear


def test_constraints_switch_projection_off_and_fail_early():
    atoms = _optimised("H2O")
    with pytest.warns(UserWarning, match="projection"):
        h = Harmonics(atoms, constrain_index=[0])
    assert not h.project
    r = _harmonics(atoms, constrain_index=[0])
    assert r.n_tr == 0 and len(r.freq) == 6
    # one fixed atom leaves free rotations -> must stop before the anharmonic part
    with pytest.raises(ValueError, match="very low frequency"):
        r.check_active_modes(r.vibrational_modes)


def test_estimate_cost_matches_actual_run_size():
    atoms = _optimised("H2O")
    cost = estimate_cost(atoms, derivative_source="energy", stencil=1, coupling=3)
    assert cost["n_modes"] == 3 and cost["harmonic"] == 19
    co2_cost = estimate_cost(_optimised("CO2"), stencil=0)
    assert co2_cost["n_modes"] == 4


# ----------------------------------------------------------------------
# exact end-to-end test with a quartic force field calculator
# ----------------------------------------------------------------------
class QFFCalculator(Calculator):
    """V = 1/2 sum w q^2 + 1/6 sum T qqq + 1/24 sum U qqqq  (cm^-1 -> eV)."""

    implemented_properties = ["energy", "forces", "dipole"]

    def __init__(self, ref, transformer, w, T, U, active, **kw):
        super().__init__(**kw)
        self.ref, self.tr = ref, transformer
        self.w, self.T, self.U, self.active = w, T, U, active
        n = len(ref) * 3
        # q = J (R - R0), J constant
        self.J = transformer.cart_to_reduced_norm(np.eye(n))[active]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        q = self.J @ (self.atoms.positions - self.ref).ravel()
        w, T, U = self.w, self.T, self.U
        V = (0.5 * np.sum(w * q**2) + np.einsum("ijk,i,j,k", T, q, q, q) / 6
             + np.einsum("ijkl,i,j,k,l", U, q, q, q, q) / 24)
        g = (w * q + np.einsum("ijk,j,k->i", T, q, q) / 2
             + np.einsum("ijkl,j,k,l->i", U, q, q, q) / 6)
        self.results["energy"] = V / EV_TO_INV_CM
        self.results["forces"] = -(self.J.T @ g).reshape(-1, 3) / EV_TO_INV_CM
        self.results["dipole"] = np.zeros(3)


def _model_force_field(w, seed=1):
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


@pytest.mark.parametrize("name", ["H2O", "CH3NO2"])
@pytest.mark.parametrize("source", ["energy", "forces"])
def test_pipeline_reproduces_exact_quartic_force_field(name, source, tmp_path):
    ref_atoms = _optimised(name)
    harm = _harmonics(ref_atoms)
    act = harm.vibrational_modes
    n = len(act)
    w = harm.freq[act]
    sqm = np.repeat(ref_atoms.get_masses() ** 0.5, 3)
    tr = CoordinateTransformer(harm.freq, harm.modes, sqm)
    T, U = _model_force_field(w)

    def setup():
        return QFFCalculator(ref_atoms.positions.copy(), tr, w, T, U, act)

    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = calculate_anharmonics(
            ref_atoms.copy(), setup, Path(tmp_path), derivative_source=source,
            stencil=1, anharmonic_delta=0.1, coriolis=False,
            harmonic_delta=0.001,  # noise-free potential: tiny step is fine
        )

    # exact VPT2 from the model constants
    N = len(harm.freq)
    T_full = np.zeros((N,) * 3)
    U_full = np.zeros((N,) * 4)
    T_full[np.ix_(act, act, act)] = T
    U_full[np.ix_(act, act, act, act)] = U
    cfg = VPT2Config(freq=harm.freq, active_modes=act, min_mode_idx=act[0],
                     max_mode_idx=act[-1], len_modes=N, coriolis=False)
    with contextlib.redirect_stdout(io.StringIO()):
        exact = VPT2Solver(cfg, (T_full, U_full)).calculate()
    # stencil 1 is exact for a quartic potential; what remains is the
    # finite-difference error of the Hessian
    assert np.allclose(res.anharm_freq[act], exact.anharm_freq[act], atol=0.1)
    e_res = sorted(t.energy for t in res.transitions)
    e_ex = sorted(t.energy for t in exact.transitions)
    assert np.allclose(e_res, e_ex, atol=0.2)


def test_exclude_below_drops_soft_modes(tmp_path):
    atoms = _optimised("CH3NO2")              # lowest EMT mode ~ 56 cm^-1
    with contextlib.redirect_stdout(io.StringIO()) as out, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = calculate_anharmonics(atoms, EMT, Path(tmp_path), exclude_below=100,
                                    stencil=0, derivative_source="forces")
    assert "Excluded modes below 100" in out.getvalue()
    assert 6 not in [m for t in res.transitions for m in t.quanta]


def test_shared_calculator_is_evaluated_once_per_structure(tmp_path):
    """One calculator instance for all structures (as with FALCON) must not
    be called again when OVERTONE reads the results back."""
    calls = {"n": 0}

    class Counting(EMT):
        def calculate(self, *args, **kwargs):
            calls["n"] += 1
            super().calculate(*args, **kwargs)

    shared = Counting()
    atoms = _optimised("H2O")
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calculate_anharmonics(atoms, lambda: shared, Path(tmp_path), stencil=0,
                              setup_harmonic_calculator=EMT)
    n_points = estimate_cost(atoms, stencil=0)["anharmonic"]
    assert calls["n"] == n_points
