"""recompute_vpt2 re-runs the VPT2 step from a saved calculation."""
from types import SimpleNamespace

import numpy as np
import pytest

from helpers.calculators import WATER_CHARGES, ChargedEMT
from overtone import calculate_anharmonics, recompute_vpt2


@pytest.fixture(scope="module")
def runs(charged_water, tmp_path_factory):
    directory = tmp_path_factory.mktemp("saved")
    first = calculate_anharmonics(
        charged_water.copy(), directory=directory,
        setup_calculator=lambda: ChargedEMT(WATER_CHARGES),
        intensities=True, save_results=True, stencil=0)
    return SimpleNamespace(
        first=first,
        same=recompute_vpt2(directory, method="standard"),
        deperturbed=recompute_vpt2(directory, method="deperturbed", max_quanta=3),
    )


def test_same_settings_reproduce_frequencies(runs):
    assert np.allclose(runs.first.anharm_freq, runs.same.anharm_freq)


def test_saved_dipoles_give_intensities(runs):
    assert runs.same.has_intensities


def test_method_can_be_changed(runs):
    assert runs.deperturbed.method == "deperturbed"


def test_max_quanta_can_be_changed(runs):
    assert max(t.total_quanta for t in runs.deperturbed.transitions) == 3
