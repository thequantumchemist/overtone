"""
Exact end-to-end check with an ASE calculator whose potential IS a quartic
force field in dimensionless normal coordinates. The whole pipeline (Hessian,
projection, displacements, unit conversions, energy- and force-based
derivatives, VPT2) must then reproduce VPT2 evaluated directly from the
model constants.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from helpers.calculators import QFFCalculator
from helpers.harmonics import run_harmonics
from helpers.models import model_force_field
from overtone import CoordinateTransformer, VPT2Config, VPT2Solver, calculate_anharmonics

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


@pytest.fixture(scope="module", params=["H2O", pytest.param("CH3NO2", marks=pytest.mark.slow)])
def model(request, optimised_emt):
    """A QFF calculator for the molecule and VPT2 evaluated directly from its constants."""
    ref_atoms = optimised_emt(request.param)
    harm = run_harmonics(ref_atoms)
    act = harm.vibrational_modes
    w = harm.freq[act]
    sqm = np.repeat(ref_atoms.get_masses() ** 0.5, 3)
    tr = CoordinateTransformer(harm.freq, harm.modes, sqm)
    T, U = model_force_field(w)

    N = len(harm.freq)
    T_full = np.zeros((N,) * 3)
    U_full = np.zeros((N,) * 4)
    T_full[np.ix_(act, act, act)] = T
    U_full[np.ix_(act, act, act, act)] = U
    cfg = VPT2Config(freq=harm.freq, active_modes=act, min_mode_idx=act[0],
                     max_mode_idx=act[-1], len_modes=N, coriolis=False)
    return SimpleNamespace(
        atoms=ref_atoms, active=act,
        setup=lambda: QFFCalculator(ref_atoms.positions.copy(), tr, w, T, U, act),
        exact=VPT2Solver(cfg, (T_full, U_full)).calculate(),
    )


@pytest.fixture(scope="module", params=["energy", "forces"])
def pipeline(request, model, tmp_path_factory):
    return calculate_anharmonics(
        model.atoms.copy(), model.setup, tmp_path_factory.mktemp("qff"),
        derivative_source=request.param, stencil=1, anharmonic_delta=0.1, coriolis=False,
        harmonic_delta=0.001,  # noise-free potential: tiny step is fine
    )


# stencil 1 is exact for a quartic potential; what remains is the
# finite-difference error of the Hessian
def test_fundamentals_match_exact(model, pipeline):
    act = model.active
    assert np.allclose(pipeline.anharm_freq[act], model.exact.anharm_freq[act], atol=0.1)


def test_all_transition_energies_match_exact(model, pipeline):
    e_res = sorted(t.energy for t in pipeline.transitions)
    e_ex = sorted(t.energy for t in model.exact.transitions)
    assert np.allclose(e_res, e_ex, atol=0.2)
