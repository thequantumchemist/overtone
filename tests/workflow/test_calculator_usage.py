"""How calculate_anharmonics uses the calculators it is given."""
from pathlib import Path

import pytest
from ase.calculators.emt import EMT

from helpers.calculators import WATER_CHARGES, ChargedEMT, counting_emt
from overtone import calculate_anharmonics, estimate_cost


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_exclude_below_drops_soft_modes(optimised_emt, tmp_path, capsys):
    atoms = optimised_emt("CH3NO2")              # lowest EMT mode ~ 56 cm^-1
    res = calculate_anharmonics(atoms, EMT, Path(tmp_path), exclude_below=100,
                                stencil=0, derivative_source="forces")
    assert "Excluded modes below 100" in capsys.readouterr().out
    assert 6 not in [m for t in res.transitions for m in t.quanta]


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_shared_calculator_is_evaluated_once_per_structure(optimised_emt, tmp_path):
    """One calculator instance for all structures (as with FALCON) must not
    be called again when OVERTONE reads the results back."""
    calls = {"n": 0}
    shared = counting_emt(calls)()
    atoms = optimised_emt("H2O")
    calculate_anharmonics(atoms, lambda: shared, Path(tmp_path), stencil=0,
                          setup_harmonic_calculator=EMT)
    n_points = estimate_cost(atoms, stencil=0)["anharmonic"]
    assert calls["n"] == n_points


def test_separate_harmonic_calculator(charged_water, tmp_path):
    """The Hessian may come from a different calculator than the force field."""
    calls = {"harm": 0, "anh": 0}

    def harm_calc():
        calls["harm"] += 1
        return ChargedEMT(WATER_CHARGES)

    def anh_calc():
        calls["anh"] += 1
        return ChargedEMT(WATER_CHARGES)

    calculate_anharmonics(charged_water.copy(), anh_calc, tmp_path,
                          setup_harmonic_calculator=harm_calc, stencil=0)
    assert calls["harm"] == 19        # 1 + 6 * 3 atoms
    assert calls["anh"] > 0
