"""How calculate_anharmonics uses the calculators it is given."""
import contextlib
import io
from pathlib import Path

import pytest
from ase.calculators.emt import EMT

from helpers.calculators import WATER_CHARGES, ChargedEMT, counting_emt
from overtone import calculate_anharmonics, estimate_cost

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


@pytest.fixture(scope="module")
def excluded(optimised_emt, tmp_path_factory):
    """Run with exclude_below; returns the result and the printed log."""
    atoms = optimised_emt("CH3NO2")              # lowest EMT mode ~ 56 cm^-1
    # module scope cannot use capsys, so capture the log directly
    with contextlib.redirect_stdout(io.StringIO()) as out:
        res = calculate_anharmonics(atoms, EMT, tmp_path_factory.mktemp("exclude"),
                                    exclude_below=100, stencil=0, derivative_source="forces")
    return res, out.getvalue()


def test_exclude_below_is_reported(excluded):
    _, log = excluded
    assert "Excluded modes below 100" in log


def test_exclude_below_drops_soft_modes(excluded):
    res, _ = excluded
    assert 6 not in [m for t in res.transitions for m in t.quanta]


def test_shared_calculator_is_evaluated_once_per_structure(optimised_emt, tmp_path):
    """One calculator instance for all structures (as with FALCON) must not
    be called again when OVERTONE reads the results back."""
    calls = {"n": 0}
    shared = counting_emt(calls)()
    atoms = optimised_emt("H2O")
    calculate_anharmonics(atoms, lambda: shared, Path(tmp_path), stencil=0,
                          setup_harmonic_calculator=EMT)
    assert calls["n"] == estimate_cost(atoms, stencil=0)["anharmonic"]


@pytest.fixture(scope="module")
def separate_calls(charged_water, tmp_path_factory):
    """Number of calculators created for the harmonic and the anharmonic part."""
    calls = {"harm": 0, "anh": 0}

    def harm_calc():
        calls["harm"] += 1
        return ChargedEMT(WATER_CHARGES)

    def anh_calc():
        calls["anh"] += 1
        return ChargedEMT(WATER_CHARGES)

    calculate_anharmonics(charged_water.copy(), anh_calc, tmp_path_factory.mktemp("separate"),
                          setup_harmonic_calculator=harm_calc, stencil=0)
    return calls


def test_hessian_uses_harmonic_calculator(separate_calls):
    """The Hessian may come from a different calculator than the force field."""
    assert separate_calls["harm"] == 19        # 1 + 6 * 3 atoms


def test_force_field_uses_anharmonic_calculator(separate_calls):
    assert separate_calls["anh"] > 0
