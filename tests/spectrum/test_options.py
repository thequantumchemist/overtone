"""plot_spectrum and broaden options: units, harmonic overlay, axes, invalid input."""
from types import SimpleNamespace

import numpy as np
import pytest

from overtone.anharmonics.transitions import VPT2Transition
from overtone.spectrum import DEFAULT_FWHM, broaden, plot_spectrum

BANDS = [
    VPT2Transition(quanta={6: 1}, kind="fundamental", label="v7",
                   harmonic=1650.0, energy=1600.0, intensity=60.0),
    VPT2Transition(quanta={6: 2}, kind="overtone", label="2v7",
                   harmonic=3300.0, energy=3150.0, intensity=0.5),
]


def line(ax, label):
    return next(ln for ln in ax.get_lines() if ln.get_label() == label)


@pytest.mark.parametrize("shape", ["lorentzian", "gaussian"])
def test_default_width_depends_on_shape(shape):
    x = np.linspace(900, 1100, 401)
    _, y_default = broaden([1000.0], [1.0], x=x, shape=shape)
    _, y_explicit = broaden([1000.0], [1.0], x=x, shape=shape, fwhm=DEFAULT_FWHM[shape])
    assert np.array_equal(y_default, y_explicit)


def test_unknown_line_shape_raises():
    with pytest.raises(ValueError, match="shape must be one of"):
        broaden([1000.0], [1.0], shape="voigt")


def test_result_without_transitions_raises():
    with pytest.raises(ValueError, match="no transitions"):
        plot_spectrum(SimpleNamespace(transitions=None))


def test_wavelength_axis():
    ax = plot_spectrum(BANDS, x_unit="nm", by_kind=False)
    peak = line(ax, "VPT2 (lorentzian)")
    x, y = peak.get_data()
    assert np.isclose(x[np.argmax(y)], 1e7 / 1600.0, rtol=1e-3)


def test_unknown_x_unit_raises():
    with pytest.raises(ValueError, match="x_unit"):
        plot_spectrum(BANDS, x_unit="eV")


def test_harmonic_overlay_peaks_at_harmonic_wavenumber():
    ax = plot_spectrum(BANDS, show_harmonic=True)
    x, y = line(ax, "harmonic").get_data()
    assert np.isclose(x[np.argmax(y)], 1650.0)


def test_single_kind_draws_no_curves_per_kind():
    ax = plot_spectrum(BANDS[:1])
    assert [ln.get_label() for ln in ax.get_lines()] == ["VPT2 (lorentzian)"]


def test_curves_per_kind_for_several_kinds():
    ax = plot_spectrum(BANDS)
    labels = {ln.get_label() for ln in ax.get_lines()}
    assert {"fundamentals", "overtones"} <= labels


def test_unknown_reference_mode_raises():
    reference = {"positions": [1600.0], "intensities": [1.0]}
    with pytest.raises(ValueError, match="reference_mode"):
        plot_spectrum(BANDS, reference=reference, reference_mode="sideways")


def test_wavenumbers_decrease_to_the_right_by_default():
    left, right = plot_spectrum(BANDS).get_xlim()
    assert left > right


def test_axis_not_inverted_on_request():
    left, right = plot_spectrum(BANDS, invert_x=False).get_xlim()
    assert left < right
