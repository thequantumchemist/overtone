import numpy as np

from overtone.spectrum import broaden


def test_broadening_conserves_band_areas():
    pos, inten = [1000.0, 1500.0, 3000.0], [10.0, 1.0, 50.0]
    for shape in ("lorentzian", "gaussian"):
        x, y = broaden(pos, inten, shape=shape, fwhm=10.0,
                       xmin=0, xmax=6000, resolution=0.1)
        tol = 0.01 if shape == "lorentzian" else 1e-6  # Lorentzian tails
        assert np.isclose(np.trapezoid(y, x), sum(inten), rtol=tol)
