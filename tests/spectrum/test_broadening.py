import numpy as np
import pytest

from overtone.spectrum import broaden


@pytest.mark.parametrize("shape, rtol", [
    ("lorentzian", 0.01),  # Lorentzian tails
    ("gaussian", 1e-6),
])
def test_broadening_conserves_band_areas(shape, rtol):
    pos, inten = [1000.0, 1500.0, 3000.0], [10.0, 1.0, 50.0]
    x, y = broaden(pos, inten, shape=shape, fwhm=10.0, xmin=0, xmax=6000, resolution=0.1)
    assert np.isclose(np.trapezoid(y, x), sum(inten), rtol=rtol)
