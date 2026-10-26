"""
AnharmonicDerivativesResult: tensor statistics, summary, equivalence and the
JSON round trip (tuple keys of the energy dictionary included).
"""
import numpy as np
import pytest

from helpers.models import U_MODEL, sym3
from overtone.anharmonics.derivatives_result import AnharmonicDerivativesResult

T_MODEL = sym3({(0, 0, 0): -300, (0, 1, 1): -150, (1, 1, 2): 55}, 3)


def _result(**update):
    data = {
        "T": T_MODEL.copy(),
        "U": U_MODEL.copy(),
        "V": {(0, 0, 0): 0.0, (1, 0, 0): 812.5, (-1, 0, 0): 790.25, (1, 1, 0): 1603.0},
        "required_calculations": np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [1, 1, 0]]),
        "number_of_calculations": 4,
    }
    data.update(update)
    return AnharmonicDerivativesResult(**data)


@pytest.fixture(scope="module")
def result():
    return _result()


@pytest.fixture(scope="module")
def reloaded(result, tmp_path_factory):
    path = tmp_path_factory.mktemp("derivatives") / "derivatives.json"
    result.dump(path)
    return AnharmonicDerivativesResult.load(path)


# --- statistics -------------------------------------------------------------

def test_nonzero_T_count(result):
    # (0,0,0) once, (0,1,1) and (1,1,2) three permutations each
    assert result.get_nonzero_T_count() == 7


def test_nonzero_U_count(result):
    assert result.get_nonzero_U_count() == np.count_nonzero(U_MODEL)


def test_T_statistics_use_absolute_values(result):
    stats = result.get_T_statistics()
    assert (stats["min"], stats["max"]) == (55.0, 300.0)


def test_U_statistics_max(result):
    assert result.get_U_statistics()["max"] == 180.0


@pytest.mark.parametrize("method", ["get_T_statistics", "get_U_statistics"])
def test_statistics_of_zero_tensor_are_zero(method):
    empty = _result(T=np.zeros((3, 3, 3)), U=np.zeros((3, 3, 3, 3)))
    assert getattr(empty, method)() == {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}


def test_summary_prints_counts(result, capsys):
    result.summary()
    out = capsys.readouterr().out
    assert "Number of configurations calculated: 4" in out


@pytest.mark.parametrize("label", ["Max |T|", "Max |U|"])
def test_summary_prints_tensor_tables(result, capsys, label):
    result.summary()
    assert label in capsys.readouterr().out


# --- equivalence ------------------------------------------------------------

DIFFERENCES = {
    "number_of_calculations": {"number_of_calculations": 5},
    "T": {"T": T_MODEL + 1.0},
    "U": {"U": U_MODEL * 1.01},
    "required_calculations": {"required_calculations": np.zeros((4, 3), dtype=int)},
    "V keys": {"V": {(0, 0, 0): 0.0}},
    "V values": {"V": {(0, 0, 0): 0.0, (1, 0, 0): 812.6, (-1, 0, 0): 790.25, (1, 1, 0): 1603.0}},
}


@pytest.mark.parametrize("field", DIFFERENCES)
def test_is_equivalent_detects_difference(result, field):
    assert not result.is_equivalent(_result(**DIFFERENCES[field]))


def test_is_equivalent_within_tolerance(result):
    assert result.is_equivalent(_result(T=T_MODEL + 1e-3), atol=1e-2)


def test_is_equivalent_to_other_type_is_false(result):
    assert not result.is_equivalent(result.T)


def test_not_equal_to_other_type(result):
    assert result != "derivatives"


def test_equal_to_identical_copy(result):
    assert result == _result()


# --- dump / load ------------------------------------------------------------

def test_round_trip_is_equal(result, reloaded):
    assert reloaded == result


def test_round_trip_restores_tuple_keys(result, reloaded):
    assert set(reloaded.V) == set(result.V)


@pytest.mark.parametrize("field", ["T", "U", "required_calculations"])
def test_round_trip_restores_arrays(reloaded, field):
    assert isinstance(getattr(reloaded, field), np.ndarray)

