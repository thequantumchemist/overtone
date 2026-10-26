"""IR intensities end to end: harmonic limit vs ASE, and "on request" behaviour."""
from pathlib import Path

import numpy as np
import pytest

from helpers.calculators import WATER_CHARGES, ChargedEMT
from helpers.units import ASE_IR_TO_KM_PER_MOL
from overtone import calculate_anharmonics
from overtone.anharmonics.transitions import INTENSITY_CONSTANT
from overtone.spectrum import broaden_transitions


def _run(atoms, tmp, **kw):
    return calculate_anharmonics(atoms.copy(), directory=Path(tmp),
                                 anharmonic_delta=0.1, stencil=1, **kw)


@pytest.fixture(scope="module")
def with_dipoles(charged_water, tmp_path_factory):
    return _run(charged_water, tmp_path_factory.mktemp("vpt2"),
                setup_calculator=lambda: ChargedEMT(WATER_CHARGES), intensities=True)


@pytest.fixture(scope="module")
def without_dipoles(charged_water, tmp_path_factory):
    # a calculator without dipoles must work as long as intensities=False
    return _run(charged_water, tmp_path_factory.mktemp("nodip"),
                setup_calculator=lambda: ChargedEMT([0, 0, 0], with_dipole=False),
                max_quanta=3)


def test_harmonic_intensities_match_ase(charged_water, with_dipoles, tmp_path):
    from ase.vibrations import Infrared

    ir_atoms = charged_water.copy()
    ir_atoms.calc = ChargedEMT(WATER_CHARGES)
    ir = Infrared(ir_atoms, name=str(tmp_path / "ir"), delta=0.005)
    ir.run()
    ir.read()
    ase_km = np.sort(ir.intensities[6:] * ASE_IR_TO_KM_PER_MOL)
    fund = [t for t in with_dipoles.transitions if t.kind == "fundamental"]
    ours = np.sort([INTENSITY_CONSTANT * t.harmonic * np.dot(t.transition_dipole, t.transition_dipole)
                    for t in fund])
    assert np.allclose(ours, ase_km, rtol=0.02, atol=1e-3)


def test_up_to_two_quanta_have_intensities(with_dipoles):
    assert all(t.intensity is not None for t in with_dipoles.transitions if t.total_quanta <= 2)


def test_all_transition_kinds_present(with_dipoles):
    kinds = {t.kind for t in with_dipoles.transitions}
    assert kinds == {"fundamental", "overtone", "combination"}


def test_no_intensities_unless_requested(without_dipoles):
    assert all(t.intensity is None for t in without_dipoles.transitions)


def test_max_quanta_sets_number_of_transitions(without_dipoles):
    assert len(without_dipoles.transitions) == 3 + 6 + 10  # 1, 2 and 3 quanta in 3 modes


def test_broadening_requires_intensities(without_dipoles):
    with pytest.raises(ValueError, match="intensities=True"):
        broaden_transitions(without_dipoles.transitions)
