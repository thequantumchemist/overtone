"""VPT2Solver reports missing configuration instead of failing obscurely."""
import numpy as np
import pytest

from helpers.models import U_MODEL, sym3
from overtone import VPT2Config, VPT2Solver

W = np.array([3100.0, 1650.0, 1150.0])
T = sym3({(0, 0, 0): -300, (1, 1, 1): -120, (2, 2, 2): -90}, 3)


@pytest.fixture
def solver():
    cfg = VPT2Config(freq=W, active_modes=[0, 1, 2], min_mode_idx=0, max_mode_idx=2,
                     len_modes=3, method="standard")
    return VPT2Solver(cfg, (T, U_MODEL))


def test_unknown_method_raises(solver):
    solver.config.method = "vpt3"  # bypasses the validated constructor
    with pytest.raises(ValueError, match="Unknown calculation type"):
        solver.calculate()


def test_rotational_constants_need_atoms(solver):
    with pytest.raises(ValueError, match="Atoms object not provided"):
        solver.calculate_rotational_constants()


def test_zeta_needs_modes(solver):
    with pytest.raises(ValueError, match="Normal modes not provided"):
        solver.calculate_zeta()
