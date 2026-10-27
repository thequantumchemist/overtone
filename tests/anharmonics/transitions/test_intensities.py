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


def test_harmonic_intensities_match_ase(charged_water, tmp_path):
    from ase.vibrations import Infrared

    res = _run(charged_water, tmp_path / "vpt2",
               setup_calculator=lambda: ChargedEMT(WATER_CHARGES),
               intensities=True)
    ir_atoms = charged_water.copy()
    ir_atoms.calc = ChargedEMT(WATER_CHARGES)
    ir = Infrared(ir_atoms, name=str(tmp_path / "ir"), delta=0.005)
    ir.run()
    ir.read()
    ase_km = np.sort(ir.intensities[6:] * ASE_IR_TO_KM_PER_MOL)
    fund = [t for t in res.transitions if t.kind == "fundamental"]
    ours = np.sort([INTENSITY_CONSTANT * t.harmonic * np.dot(t.transition_dipole, t.transition_dipole)
                    for t in fund])
    assert np.allclose(ours, ase_km, rtol=0.02, atol=1e-3)
    assert all(t.intensity is not None for t in res.transitions if t.total_quanta <= 2)
    kinds = {t.kind for t in res.transitions}
    assert kinds == {"fundamental", "overtone", "combination"}


def test_intensities_are_only_computed_on_request(charged_water, tmp_path):
    # a calculator without dipoles must work as long as intensities=False
    res = _run(charged_water, tmp_path / "nodip",
               setup_calculator=lambda: ChargedEMT([0, 0, 0], with_dipole=False),
               max_quanta=3)
    assert all(t.intensity is None for t in res.transitions)
    assert len(res.transitions) == 3 + 6 + 10  # 1, 2 and 3 quanta in 3 modes
    with pytest.raises(ValueError, match="intensities=True"):
        broaden_transitions(res.transitions)
