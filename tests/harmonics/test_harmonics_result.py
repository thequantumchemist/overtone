"""HarmonicsResult: comparison, JSON round trip, frequency helpers and summary."""
import io

import numpy as np
import pytest

from helpers.harmonics import run_harmonics
from overtone import HarmonicsResult


@pytest.fixture(scope="module")
def water(optimised_emt):
    return run_harmonics(optimised_emt("H2O"))


def test_json_round_trip(water, tmp_path):
    water.dump(tmp_path / "harmonics.json")
    assert HarmonicsResult.load(tmp_path / "harmonics.json") == water


def test_not_equal_to_other_types(water):
    assert water != water.freq


def test_not_equivalent_to_other_types(water):
    assert not water.is_equivalent(water.freq)


def test_other_formula_is_not_equivalent(water):
    assert not water.is_equivalent(water.model_copy(update={"chemical_formula": "D2O"}))


def test_changed_frequencies_are_not_equivalent(water):
    assert water != water.model_copy(update={"freq": water.freq * 1.001})


def test_changes_within_tolerance_are_equivalent(water):
    assert water.is_equivalent(water.model_copy(update={"freq": water.freq * 1.001}), rtol=1e-2)


def test_real_frequencies_are_the_vibrations(water):
    assert np.array_equal(water.get_real_frequencies(), water.freq[6:])


def test_minimum_has_no_imaginary_frequencies(water):
    assert water.get_imaginary_frequencies().size == 0


def test_imaginary_frequencies_are_listed():
    res = HarmonicsResult(hessian=np.zeros((3, 3)), w2=np.zeros(3), modes=np.eye(3),
                          freq=np.array([0.0, 0.0, 1500.0]), ifreq=np.array([0.0, 120.0, 0.0]),
                          freq_eV=np.zeros(3), chemical_formula="X")
    assert np.array_equal(res.get_imaginary_frequencies(), [120.0])


def test_low_frequency_active_modes_warn(water):
    with pytest.warns(UserWarning, match="below 2000"):
        water.check_active_modes(water.vibrational_modes, warn_frequency=2000.0)


def summary_rows(result, **kw):
    out = io.StringIO()
    result.summary(file=out, **kw)
    return [line for line in out.getvalue().splitlines() if line.startswith("│")]


def test_summary_without_zero_modes_skips_rigid_body_rows(water):
    assert len(summary_rows(water)) - len(summary_rows(water, show_all=False)) == 6
