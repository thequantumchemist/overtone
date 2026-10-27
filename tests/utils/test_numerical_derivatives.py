"""
Finite-difference cubic and quartic constants.

Every stencil is exact for a quartic polynomial (nested central differences
only pick up fifth and higher derivatives), so the derivatives of a quartic
force field must come out exactly, both directly from the calculators and
through AnharmonicDerivativesSolver with a QFF calculator.
"""
import itertools
from types import SimpleNamespace

import numpy as np
import pytest

from helpers.calculators import QFFCalculator
from helpers.harmonics import run_harmonics
from helpers.models import model_force_field
from overtone import (
    AnharmonicDerivativesConfig,
    AnharmonicDerivativesSolver,
    CoordinateTransformer,
)
from overtone.engines import SerialEngine
from overtone.utils.numerical_derivatives import (
    CartesianForceDerivativesCalculator,
    NumericalDerivativesCalculator,
)

STENCILS = [0, 1, 2, 3]
N = 3


def two_mode_only(T):
    """Drop the phi_ijk with three distinct indices (what coupling=2 omits)."""
    T = T.copy()
    for idx in itertools.product(range(len(T)), repeat=3):
        if len(set(idx)) == 3:
            T[idx] = 0.0
    return T


def diagonal_only(T):
    out = np.zeros_like(T)
    for i in range(len(T)):
        out[i, i, i] = T[i, i, i]
    return out


def iijj(U):
    return np.array([[U[i, i, j, j] for j in range(len(U))] for i in range(len(U))])


# ----------------------------------------------------------------------
# the calculators on a polynomial in three variables
# ----------------------------------------------------------------------
W_POLY = np.array([3.0, 2.0, 1.5])
T_POLY, U_POLY = model_force_field(np.array([900.0, 700.0, 500.0]), seed=3)


def poly_energy(x):
    return (0.5 * W_POLY @ x**2 + np.einsum("ijk,i,j,k", T_POLY, x, x, x) / 6
            + np.einsum("ijkl,i,j,k,l", U_POLY, x, x, x, x) / 24)


def poly_gradient(x):
    return (W_POLY * x + np.einsum("ijk,j,k->i", T_POLY, x, x) / 2
            + np.einsum("ijkl,j,k,l->i", U_POLY, x, x, x) / 6)


EXPECTED_T = {3: T_POLY, 2: two_mode_only(T_POLY), 1: diagonal_only(T_POLY)}


def from_energies(stencil, coupling):
    calc = NumericalDerivativesCalculator(N, delta=0.1, stencil=stencil)
    calc.setup_derivative_indices(list(range(N)), [3, 4], coupling=coupling)
    values = calc.evaluate_function_at_displacements(poly_energy, np.zeros(N))
    return calc.calculate_derivatives(values)


def from_forces(stencil, coupling):
    calc = CartesianForceDerivativesCalculator(N, delta=0.1, modes=np.eye(N),
                                               masses=np.ones(N), stencil=stencil)
    calc.setup_derivative_indices(list(range(N)), [3, 4], coupling=coupling)
    # flat (3N,) force vectors
    forces = {tuple(k): -poly_gradient(0.1 * k) for k in calc.get_displacement_keys()}
    return calc.calculate_derivatives(forces)


@pytest.fixture(scope="module", params=[
    (src, s, c) for src in ("energy", "forces") for s in STENCILS for c in (1, 2, 3)
], ids=lambda p: f"{p[0]}-stencil{p[1]}-coupling{p[2]}")
def polynomial(request):
    source, stencil, coupling = request.param
    run = from_energies if source == "energy" else from_forces
    return SimpleNamespace(coupling=coupling, derivatives=run(stencil, coupling))


def test_cubic_constants_exact_for_polynomial(polynomial):
    assert np.allclose(polynomial.derivatives["order_3"], EXPECTED_T[polynomial.coupling],
                       atol=1e-6)


def test_quartic_constants_exact_for_polynomial(polynomial):
    assert np.allclose(iijj(polynomial.derivatives["order_4"]), iijj(U_POLY), atol=1e-6)


def test_custom_coordinate_map_is_used():
    calc = NumericalDerivativesCalculator(1, delta=0.1, stencil=0)
    calc.setup_derivative_indices([0], [3], coupling=3)
    values = calc.evaluate_function_at_displacements(
        lambda x: x[0] ** 3, np.zeros(1), lambda key, base: base + 2 * key * calc.delta)
    # f(2x) = 8 x^3 seen as a function of the displacement
    assert np.isclose(calc.calculate_derivatives(values)["order_3"][0, 0, 0], 48.0)


@pytest.mark.parametrize("make", [
    lambda: NumericalDerivativesCalculator(N, 0.1),
    lambda: CartesianForceDerivativesCalculator(N, 0.1, np.eye(N), np.ones(N)),
], ids=["energy", "forces"])
def test_keys_require_setup(make):
    with pytest.raises(ValueError, match="setup_derivative_indices"):
        make().get_displacement_keys()


@pytest.mark.parametrize("make", [
    lambda: NumericalDerivativesCalculator(N, 0.1),
    lambda: CartesianForceDerivativesCalculator(N, 0.1, np.eye(N), np.ones(N)),
], ids=["energy", "forces"])
def test_derivatives_require_setup(make):
    with pytest.raises(ValueError, match="setup_derivative_indices"):
        make().calculate_derivatives({})


# ----------------------------------------------------------------------
# through AnharmonicDerivativesSolver with a QFF calculator (water)
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def water(optimised_emt):
    atoms = optimised_emt("H2O")
    harm = run_harmonics(atoms)
    act = harm.vibrational_modes
    w = harm.freq[act]
    T, U = model_force_field(w)
    T = two_mode_only(T)  # so that coupling=2 is exact as well
    tr = CoordinateTransformer(harm.freq, harm.modes, np.repeat(atoms.get_masses() ** 0.5, 3))
    return SimpleNamespace(
        atoms=atoms, harm=harm, act=act, T=T, U=U,
        setup=lambda: QFFCalculator(atoms.positions.copy(), tr, w, T, U, act),
    )


@pytest.fixture(scope="module", params=[
    ("energy", 2, 3), ("energy", 3, 3), ("energy", 1, 2),
    ("forces", 2, 3), ("forces", 3, 3), ("forces", 1, 2),
], ids=lambda p: f"{p[0]}-stencil{p[1]}-coupling{p[2]}")
def solver_run(request, water):
    source, stencil, coupling = request.param
    config = AnharmonicDerivativesConfig.from_harmonics_result(
        water.harm, water.atoms, water.act, delta=0.1, coupling=coupling, stencil=stencil)
    solver = AnharmonicDerivativesSolver(config)
    if source == "energy":
        traj, _ = solver.generate_displacement_trajectory()
        traj = SerialEngine().calculate_trajectory(traj, water.setup, ["energy"])
        result = solver.calculate_derivatives(traj)  # keys from the solver
    else:
        traj, _ = solver.generate_displacement_trajectory_for_forces()
        traj = SerialEngine().calculate_trajectory(traj, water.setup, ["energy", "forces"])
        result = solver.calculate_derivatives_from_forces(traj)
    act = water.act
    return SimpleNamespace(T=result.T[np.ix_(act, act, act)],
                           U=result.U[np.ix_(act, act, act, act)])


def test_solver_cubic_constants_exact(water, solver_run):
    """With coupling=2 this includes the permutations phi_iji and phi_jii."""
    assert np.allclose(solver_run.T, water.T, atol=1e-3)


def test_solver_quartic_constants_exact(water, solver_run):
    assert np.allclose(iijj(solver_run.U), iijj(water.U), atol=1e-3)
