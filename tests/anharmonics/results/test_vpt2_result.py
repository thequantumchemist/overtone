"""
VPT2Result: filtering and printing transitions, spectra, summaries,
equivalence and the JSON round trip, on a type-1 Fermi dyad model.
"""
import io
import re
from types import SimpleNamespace

import numpy as np
import pytest
from matplotlib.axes import Axes

from helpers.models import MU1, MU2, U_MODEL, solve, sym3
from overtone.anharmonics.dipole_derivatives import DipoleDerivativesResult
from overtone.anharmonics.vpt2_result import VPT2Result

W = [3200.0, 1608.0, 1180.0]  # w0 ~ 2 w1: v1 / 2v2 Fermi dyad
T = sym3({(0, 0, 0): -300, (1, 1, 1): -120, (2, 2, 2): -90, (0, 1, 1): -150,
          (1, 1, 2): 55, (0, 0, 2): 30}, 3)


@pytest.fixture(scope="module")
def results():
    dip = DipoleDerivativesResult(mu1=MU1, mu2=MU2)
    return SimpleNamespace(
        gvpt2=solve(W, T, U_MODEL, "gvpt2", dip),
        deperturbed=solve(W, T, U_MODEL, "deperturbed", dip),
        split=solve(W, T, U_MODEL, "split", dip),
        dark=solve(W, T, U_MODEL, "gvpt2"),
    )


@pytest.fixture(scope="module")
def reloaded(results, tmp_path_factory):
    path = tmp_path_factory.mktemp("vpt2") / "result.json"
    results.gvpt2.dump(path)
    return VPT2Result.load(path)


def _table(result, **kwargs):
    out = io.StringIO()
    result.transitions_table(file=out, **kwargs)
    return out.getvalue()


# --- get_transitions --------------------------------------------------------

@pytest.mark.parametrize("kind", ["fundamental", "overtone", "combination"])
def test_get_transitions_filters_by_kind(results, kind):
    got = results.gvpt2.get_transitions(kinds=[kind])
    assert got and all(t.kind == kind for t in got)


def test_get_transitions_filters_by_window(results):
    got = results.gvpt2.get_transitions(xmin=2000, xmax=4000)
    assert [t.label for t in got] == ["2v3", "v2+v3", "v1", "2v2"]


def test_get_transitions_without_filters_returns_all(results):
    assert results.gvpt2.get_transitions() == results.gvpt2.transitions


def test_get_transitions_without_transitions_raises(results):
    empty = results.gvpt2.model_copy(update={"transitions": None})
    with pytest.raises(ValueError, match="No transitions"):
        empty.get_transitions()


# --- transitions_table ------------------------------------------------------

def test_table_footnote_gvpt2_says_variational(results):
    assert "treated variationally" in _table(results.gvpt2)


def test_table_footnote_other_methods_recommend_gvpt2(results):
    assert "Use method='gvpt2'" in _table(results.deperturbed)


def test_table_without_resonant_bands_has_no_footnote(results):
    assert "Fermi resonance" not in _table(results.gvpt2, xmax=3000)


def test_table_has_intensity_column_with_dipoles(results):
    assert "Intensity" in _table(results.gvpt2)


def test_table_has_no_intensity_column_without_dipoles(results):
    assert "Intensity" not in _table(results.dark)


# --- spectra ----------------------------------------------------------------

def test_spectrum_peaks_at_strongest_band(results):
    x, y = results.gvpt2.spectrum(fwhm=10)
    strongest = max(results.gvpt2.transitions, key=lambda t: t.intensity)
    assert abs(x[np.argmax(y)] - strongest.energy) < 1


def test_spectrum_without_intensities_raises(results):
    with pytest.raises(ValueError, match="No transitions with intensities"):
        results.dark.spectrum()


def test_plot_spectrum_returns_axes(results):
    assert isinstance(results.gvpt2.plot_spectrum(), Axes)


# --- summary ----------------------------------------------------------------

def test_summary_restricted_to_active_modes(results):
    text = results.gvpt2.summary(active_modes=[1, 2], file=io.StringIO())
    assert "active modes: [1, 2]" in text and "Number of modes: 2" in text


def test_summary_active_modes_frequency_range(results):
    text = results.gvpt2.summary(active_modes=[1, 2], file=io.StringIO())
    lo, hi = sorted(results.gvpt2.anharm_freq[[1, 2]])
    assert f"Min anharmonic frequency: {lo:.2f}" in text and f"Max anharmonic frequency: {hi:.2f}" in text


def test_summary_split_reports_cubic_and_quartic_parts(results):
    out = io.StringIO()
    results.split.summary(file=out)
    assert "Max |X_3|" in out.getvalue() and "Max |X_4|" in out.getvalue()


def test_summary_reports_coriolis_part(results):
    out = io.StringIO()
    with_coriolis = results.gvpt2.model_copy(update={"X_coriolis": np.full((3, 3), -0.5)})
    with_coriolis.summary(file=out)
    assert "Max |X_coriolis|" in out.getvalue()


def test_summary_reports_fermi_counts(results):
    out = io.StringIO()
    results.gvpt2.summary(file=out)
    assert re.search(r"Type 1\W+1\W", out.getvalue())


# --- equivalence ------------------------------------------------------------

def _shift_transition(result, **update):
    ts = list(result.transitions)
    ts[0] = ts[0].model_copy(update=update)
    return {"transitions": ts}


DIFFERENCES = {
    "method": lambda r: {"method": "standard"},
    "X": lambda r: {"X": r.X + 1.0},
    "anharm_freq": lambda r: {"anharm_freq": r.anharm_freq + 1.0},
    "X_3 missing on one side": lambda r: {"X_3": np.zeros_like(r.X)},
    "fermi_type1": lambda r: {"fermi_type1": []},
    "fermi_type2": lambda r: {"fermi_type2": [(0, 1, 2)]},
    "transitions missing": lambda r: {"transitions": None},
    "number of transitions": lambda r: {"transitions": r.transitions[:-1]},
    "transition quanta": lambda r: _shift_transition(r, quanta={2: 2}),
    "transition energy": lambda r: _shift_transition(r, energy=r.transitions[0].energy + 1),
    "intensity missing": lambda r: _shift_transition(r, intensity=None),
    "intensity value": lambda r: _shift_transition(r, intensity=2 * r.transitions[0].intensity),
}


@pytest.mark.parametrize("field", DIFFERENCES)
def test_is_equivalent_detects_difference(results, field):
    r = results.gvpt2
    assert not r.is_equivalent(r.model_copy(update=DIFFERENCES[field](r)))


def test_is_equivalent_detects_different_optional_array(results):
    r = results.split
    other = r.model_copy(update={"X_3": r.X_3 + 1.0})
    assert not r.is_equivalent(other)


def test_is_equivalent_within_tolerance(results):
    r = results.gvpt2
    assert r.is_equivalent(r.model_copy(update={"X": r.X + 1e-3}), atol=1e-2)


def test_is_equivalent_to_other_type_is_false(results):
    assert not results.gvpt2.is_equivalent("gvpt2")


def test_not_equal_to_other_type(results):
    assert results.gvpt2 != results.gvpt2.X


def test_different_methods_not_equal(results):
    assert results.gvpt2 != results.deperturbed


# --- dump / load ------------------------------------------------------------

def test_round_trip_is_equal(results, reloaded):
    assert reloaded == results.gvpt2


def test_round_trip_restores_arrays(reloaded):
    assert isinstance(reloaded.X, np.ndarray) and isinstance(reloaded.anharm_freq, np.ndarray)


def test_round_trip_restores_fermi_tuples(results, reloaded):
    assert reloaded.fermi_type1 == results.gvpt2.fermi_type1


def test_round_trip_restores_mixing(results, reloaded):
    assert [t.mixing for t in reloaded.transitions] == [t.mixing for t in results.gvpt2.transitions]


def test_round_trip_split_keeps_x3_x4(results, tmp_path):
    results.split.dump(tmp_path / "split.json")
    assert VPT2Result.load(tmp_path / "split.json") == results.split
