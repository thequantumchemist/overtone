import numpy as np
import pytest

from overtone import DipoleDerivativesCalculator


def test_missing_dipoles_raise():
    calc = DipoleDerivativesCalculator(n_modes=3, active_modes=[0, 1, 2], delta=0.1, stencil=0)
    keys = calc.get_displacement_keys()
    dipoles = {tuple(k): np.zeros(3) for k in keys[1:]}
    with pytest.raises(KeyError, match="1 displacement keys have no dipole"):
        calc.calculate(dipoles)
