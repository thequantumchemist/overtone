"""method="split": standard VPT2 with cubic and quartic contributions reported separately."""
import contextlib
import io
from types import SimpleNamespace

import numpy as np
import pytest

from helpers.models import U_MODEL, solve, sym3

W = np.array([3100.0, 1650.0, 1150.0])
T = sym3({(0, 0, 0): -300, (1, 1, 1): -120, (2, 2, 2): -90, (0, 1, 1): -60,
          (1, 1, 2): 35, (0, 0, 2): 30, (0, 1, 2): 25}, 3)


@pytest.fixture(scope="module")
def split():
    with contextlib.redirect_stdout(io.StringIO()) as out:
        result = solve(W, T, U_MODEL, "split")
    return SimpleNamespace(
        result=result,
        log=out.getvalue(),
        standard=solve(W, T, U_MODEL, "standard"),
        cubic_only=solve(W, T, np.zeros_like(U_MODEL), "standard"),
        quartic_only=solve(W, np.zeros_like(T), U_MODEL, "standard"),
    )


def test_method_is_reported(split):
    assert split.result.method == "split"


def test_contributions_add_up_to_standard_X(split):
    assert np.allclose(split.result.X_3 + split.result.X_4, split.standard.X)


def test_X_is_the_sum_of_contributions(split):
    assert np.allclose(split.result.X, split.result.X_3 + split.result.X_4)


def test_cubic_part_is_X_without_quartic_constants(split):
    assert np.allclose(split.result.X_3, split.cubic_only.X)


def test_quartic_part_is_X_without_cubic_constants(split):
    assert np.allclose(split.result.X_4, split.quartic_only.X)


def test_fundamentals_match_standard(split):
    assert np.allclose(split.result.anharm_freq, split.standard.anharm_freq)


def test_transitions_match_standard(split):
    energies = [t.energy for t in split.result.transitions]
    assert np.allclose(energies, [t.energy for t in split.standard.transitions])


@pytest.mark.parametrize("header", ["Third order:", "Fourth order:"])
def test_contributions_are_printed(split, header):
    assert header in split.log
