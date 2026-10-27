"""
VPT2Config: parameter validation, defaults from a HarmonicsResult, and the
JSON round trip with and without atoms.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from helpers.harmonics import run_harmonics
from overtone import VPT2Config

FREQ = np.array([3200.0, 1608.0, 1180.0])


def _config(**kwargs):
    data = {"freq": FREQ, "active_modes": [0, 1, 2], "min_mode_idx": 0,
            "max_mode_idx": 2, "len_modes": 3, "method": "deperturbed", "max_quanta": 3}
    data.update(kwargs)
    return VPT2Config(**data)


@pytest.fixture(scope="module")
def water(optimised_emt):
    atoms = optimised_emt("H2O", fmax=1e-6)
    return SimpleNamespace(atoms=atoms, hr=run_harmonics(atoms))


@pytest.fixture(scope="module")
def round_trips(water, tmp_path_factory):
    folder = tmp_path_factory.mktemp("vpt2_config")
    bare = _config()
    with_atoms = VPT2Config.from_harmonics_result(water.hr, [6, 7, 8], atoms=water.atoms.copy())
    bare.dump(folder / "bare.json")
    with_atoms.dump(folder / "atoms.json")
    return SimpleNamespace(
        bare=(bare, VPT2Config.load(folder / "bare.json")),
        atoms=(with_atoms, VPT2Config.load(folder / "atoms.json")),
    )


# --- validation -------------------------------------------------------------

@pytest.mark.parametrize("kwargs, message", [
    ({"len_modes": 4}, r"freq array length \(3\) must match len_modes \(4\)"),
    ({"coupling": 1}, "coupling must be 2 or 3, got 1"),
    ({"max_quanta": 0}, "max_quanta must be >= 1, got 0"),
])
def test_rejects_invalid_parameter(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _config(**kwargs)


# --- from_harmonics_result --------------------------------------------------

def test_mode_range_from_active_modes(water):
    cfg = VPT2Config.from_harmonics_result(water.hr, [7, 8])
    assert (cfg.min_mode_idx, cfg.max_mode_idx) == (7, 8)


def test_empty_active_modes_skip_rigid_body_modes(water):
    cfg = VPT2Config.from_harmonics_result(water.hr, [])
    assert (cfg.min_mode_idx, cfg.max_mode_idx) == (6, 8)


def test_masses_only_with_atoms(water):
    assert VPT2Config.from_harmonics_result(water.hr, [6, 7, 8]).masses is None


# --- dump / load ------------------------------------------------------------

@pytest.mark.parametrize("field", ["active_modes", "min_mode_idx", "max_mode_idx",
                                   "len_modes", "method", "max_quanta"])
def test_round_trip_restores_field(round_trips, field):
    original, loaded = round_trips.bare
    assert getattr(loaded, field) == getattr(original, field)


def test_round_trip_restores_freq(round_trips):
    original, loaded = round_trips.bare
    assert np.array_equal(loaded.freq, original.freq)


@pytest.mark.parametrize("field", ["masses", "modes", "atoms"])
def test_round_trip_without_atoms_keeps_none(round_trips, field):
    _, loaded = round_trips.bare
    assert getattr(loaded, field) is None


@pytest.mark.parametrize("field", ["masses", "modes", "freq_array"])
def test_round_trip_with_atoms_restores_array(round_trips, field):
    original, loaded = round_trips.atoms
    assert np.allclose(getattr(loaded, field), getattr(original, field))


def test_round_trip_with_atoms_restores_positions(round_trips):
    original, loaded = round_trips.atoms
    assert np.allclose(loaded.atoms.positions, original.atoms.positions)
