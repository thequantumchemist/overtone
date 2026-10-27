from pathlib import Path

import numpy as np

from helpers.calculators import WATER_CHARGES, ChargedEMT
from overtone import calculate_anharmonics, recompute_vpt2


def test_recompute_vpt2_from_saved_run(charged_water, tmp_path):
    first = calculate_anharmonics(
        charged_water.copy(), directory=Path(tmp_path),
        setup_calculator=lambda: ChargedEMT(WATER_CHARGES),
        intensities=True, save_results=True, stencil=0)
    same = recompute_vpt2(tmp_path, method="standard")
    dep = recompute_vpt2(tmp_path, method="deperturbed", max_quanta=3)
    assert np.allclose(first.anharm_freq, same.anharm_freq)
    assert same.has_intensities
    assert dep.method == "deperturbed"
    assert max(t.total_quanta for t in dep.transitions) == 3
