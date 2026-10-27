import pytest

from overtone.anharmonics.transitions import VPT2Transition
from overtone.spectrum import plot_spectrum

BANDS = [
    VPT2Transition(quanta={6: 1}, kind="fundamental", label="v7",
                   harmonic=1650.0, energy=1600.0, intensity=60.0),
    VPT2Transition(quanta={6: 2}, kind="overtone", label="2v7",
                   harmonic=3300.0, energy=3150.0, intensity=0.5),
]
REFERENCE = {"positions": [1595.0, 3152.0], "intensities": [70.0, 0.4], "label": "exp."}


def test_mirrored_reference_points_down():
    ax = plot_spectrum(BANDS, reference=REFERENCE)
    assert ax.get_ylim()[0] < 0


def test_overlay_on_log_axis_can_be_saved(tmp_path):
    ax = plot_spectrum(BANDS, reference=REFERENCE, reference_mode="overlay", log=True)
    ax.figure.savefig(tmp_path / "spec.png")
    assert (tmp_path / "spec.png").stat().st_size > 0


def test_mirrored_reference_rejects_log_axis():
    with pytest.raises(ValueError, match="log axis"):
        plot_spectrum(BANDS, reference=REFERENCE, log=True)
