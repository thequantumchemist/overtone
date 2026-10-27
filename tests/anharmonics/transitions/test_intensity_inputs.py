"""Intensities need cubic constants and both dipole derivatives."""
import numpy as np
import pytest

from overtone import TransitionCalculator

W = np.array([3000.0, 1600.0])
X = np.array([[-40.0, -10.0], [-10.0, -15.0]])
MU1 = np.ones((2, 3))
MU2 = np.zeros((2, 2, 3))
T = np.zeros((2, 2, 2))


def test_intensities_need_cubic_constants():
    with pytest.raises(ValueError, match="Cubic force constants"):
        TransitionCalculator(W, X, [0, 1], mu1=MU1, mu2=MU2)


def test_intensities_need_second_dipole_derivatives():
    with pytest.raises(ValueError, match="Second dipole derivatives"):
        TransitionCalculator(W, X, [0, 1], T=T, mu1=MU1)
