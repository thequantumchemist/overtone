"""
Exact end-to-end check with an ASE calculator whose potential IS a quartic
force field in dimensionless normal coordinates. The whole pipeline (Hessian,
projection, displacements, unit conversions, energy- and force-based
derivatives, VPT2) must then reproduce VPT2 evaluated directly from the
model constants.
"""
from pathlib import Path

import numpy as np
import pytest

from helpers.calculators import QFFCalculator
from helpers.harmonics import run_harmonics
from helpers.models import model_force_field
from overtone import CoordinateTransformer, VPT2Config, VPT2Solver, calculate_anharmonics


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("name", ["H2O", pytest.param("CH3NO2", marks=pytest.mark.slow)])
@pytest.mark.parametrize("source", ["energy", "forces"])
def test_pipeline_reproduces_exact_quartic_force_field(name, source, optimised_emt, tmp_path):
    ref_atoms = optimised_emt(name)
    harm = run_harmonics(ref_atoms)
    act = harm.vibrational_modes
    w = harm.freq[act]
    sqm = np.repeat(ref_atoms.get_masses() ** 0.5, 3)
    tr = CoordinateTransformer(harm.freq, harm.modes, sqm)
    T, U = model_force_field(w)

    def setup():
        return QFFCalculator(ref_atoms.positions.copy(), tr, w, T, U, act)

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
    exact = VPT2Solver(cfg, (T_full, U_full)).calculate()
    # stencil 1 is exact for a quartic potential; what remains is the
    # finite-difference error of the Hessian
    assert np.allclose(res.anharm_freq[act], exact.anharm_freq[act], atol=0.1)
    e_res = sorted(t.energy for t in res.transitions)
    e_ex = sorted(t.energy for t in exact.transitions)
    assert np.allclose(e_res, e_ex, atol=0.2)
