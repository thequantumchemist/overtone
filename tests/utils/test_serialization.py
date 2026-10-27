"""JSON helpers for Atoms and (complex) numpy arrays."""
import json

import numpy as np
import pytest
from ase.build import molecule

from overtone.utils.serialization import (
    deserialize_array,
    deserialize_atoms,
    deserialize_complex,
    serialize_array,
    serialize_atoms,
    serialize_complex,
)


@pytest.fixture(scope="module")
def water():
    return molecule("H2O")


def test_atoms_none_serializes_to_none():
    assert serialize_atoms(None) is None


def test_atoms_none_deserializes_to_none():
    assert deserialize_atoms(None) is None


def test_atoms_round_trip_symbols(water):
    assert deserialize_atoms(serialize_atoms(water)).get_chemical_symbols() == ["O", "H", "H"]


def test_atoms_round_trip_positions(water):
    assert np.allclose(deserialize_atoms(serialize_atoms(water)).positions, water.positions)


@pytest.mark.parametrize("value", [1.5, "text", None, {"a": 1}])
def test_serialize_complex_leaves_other_values(value):
    assert serialize_complex(value) == value


def test_serialize_complex_nested():
    assert serialize_complex([1.0, [2 + 3j]]) == [1.0, [{"real": 2.0, "imag": 3.0}]]


def test_deserialize_complex_nested():
    assert deserialize_complex([1.0, [{"real": 2.0, "imag": 3.0}]]) == [1.0, [2 + 3j]]


def test_deserialize_complex_leaves_other_dicts():
    assert deserialize_complex({"real": 1.0}) == {"real": 1.0}


@pytest.mark.parametrize("array", [
    np.arange(6.0).reshape(2, 3),
    np.array([[1 + 2j, 3 - 4j], [0j, -1j]]),
], ids=["real", "complex"])
def test_array_survives_json(array):
    restored = deserialize_array(json.loads(json.dumps(serialize_array(array))))
    assert restored.dtype == array.dtype and np.array_equal(restored, array)
