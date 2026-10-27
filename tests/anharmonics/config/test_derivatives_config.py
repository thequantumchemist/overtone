"""
AnharmonicDerivativesConfig: parameter validation, derived masses and indices,
and the JSON round trip (atoms included), for EMT water.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from helpers.harmonics import run_harmonics
from overtone import AnharmonicDerivativesConfig

ACTIVE = [6, 7, 8]


@pytest.fixture(scope="module")
def water(optimised_emt):
    atoms = optimised_emt("H2O", fmax=1e-6)
    return SimpleNamespace(atoms=atoms, hr=run_harmonics(atoms))


def _config(water, **kwargs):
    return AnharmonicDerivativesConfig.from_harmonics_result(
        water.hr, water.atoms.copy(), active_modes=ACTIVE, **kwargs)


@pytest.fixture(scope="module")
def config(water):
    return _config(water, delta=0.25, coupling=2, stencil=2)


@pytest.fixture(scope="module")
def reloaded(config, tmp_path_factory):
    path = tmp_path_factory.mktemp("derivatives_config") / "config.json"
    config.dump(path)
    return AnharmonicDerivativesConfig.load(path)


# --- validation -------------------------------------------------------------

@pytest.mark.parametrize("kwargs, message", [
    ({"coupling": 1}, "coupling must be 2 or 3, got 1"),
    ({"coupling": 4}, "coupling must be 2 or 3, got 4"),
    ({"stencil": 4}, "stencil must be 0, 1, 2, or 3, got 4"),
    ({"stencil": -1}, "stencil must be 0, 1, 2, or 3, got -1"),
    ({"delta": 0.0}, "delta must be positive, got 0.0"),
    ({"delta": -0.2}, "delta must be positive, got -0.2"),
])
def test_rejects_invalid_parameter(water, kwargs, message):
    with pytest.raises(ValueError, match=message):
        _config(water, **kwargs)


# --- derived fields ---------------------------------------------------------

def test_masses_are_sqrt_mass_per_coordinate(config, water):
    expected = np.repeat(np.sqrt(water.atoms.get_masses()), 3).reshape(-1, 1)
    assert np.allclose(config.masses, expected)


def test_masses_restricted_to_force_indices(water):
    # only the oxygen coordinates (H2O from ase.build lists O first)
    cfg = _config(water, force_indices=[0, 1, 2])
    expected = np.full((3, 1), np.sqrt(water.atoms.get_masses()[0]))
    assert np.allclose(cfg.masses, expected)


def test_default_force_indices_cover_all_coordinates(config):
    assert config.force_indices == list(range(9))


def test_default_unconstrained_index_covers_all_atoms(config):
    assert config.unconstrained_index == [0, 1, 2]


def test_constrained_num_atoms_counts_free_atoms(water):
    cfg = _config(water, constrained=True, unconstrained_index=[1, 2])
    assert cfg.num_atoms == 2


def test_freq_array_is_column(config):
    assert config.freq_array.shape == (9, 1)


# --- dump / load ------------------------------------------------------------

@pytest.mark.parametrize("field", ["freq", "modes", "masses", "freq_array"])
def test_round_trip_restores_array(config, reloaded, field):
    assert np.allclose(getattr(reloaded, field), getattr(config, field))


@pytest.mark.parametrize("field", ["active_modes", "delta", "coupling", "stencil",
                                   "force_indices", "unconstrained_index", "len_modes",
                                   "num_atoms", "constrained"])
def test_round_trip_restores_field(config, reloaded, field):
    assert getattr(reloaded, field) == getattr(config, field)


def test_round_trip_restores_atoms(config, reloaded):
    assert reloaded.atoms.get_chemical_symbols() == config.atoms.get_chemical_symbols()


def test_round_trip_restores_positions(config, reloaded):
    assert np.allclose(reloaded.atoms.positions, config.atoms.positions)
