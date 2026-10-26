import pytest

from overtone.anharmonics.transitions import VPT2Transition
from overtone.spectrum import plot_spectrum


def test_plot_spectrum_with_reference(tmp_path):
    bands = [
        VPT2Transition(quanta={6: 1}, kind="fundamental", label="v7",
                       harmonic=1650.0, energy=1600.0, intensity=60.0),
        VPT2Transition(quanta={6: 2}, kind="overtone", label="2v7",
                       harmonic=3300.0, energy=3150.0, intensity=0.5),
    ]
    ref = {"positions": [1595.0, 3152.0], "intensities": [70.0, 0.4], "label": "exp."}
    ax = plot_spectrum(bands, reference=ref)                       # mirrored
    assert ax.get_ylim()[0] < 0
    ax = plot_spectrum(bands, reference=ref, reference_mode="overlay", log=True)
    with pytest.raises(ValueError, match="log axis"):
        plot_spectrum(bands, reference=ref, log=True)
    ax.figure.savefig(tmp_path / "spec.png")
    assert (tmp_path / "spec.png").stat().st_size > 0
