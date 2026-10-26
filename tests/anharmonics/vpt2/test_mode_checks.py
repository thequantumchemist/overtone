"""VPT2Solver refuses modes it cannot treat and warns about degenerate ones."""
import numpy as np
import pytest

from helpers.models import U_MODEL, solve, sym3
from overtone import VPT2Config, VPT2Solver

T = sym3({(0, 0, 0): -300, (1, 1, 1): -120, (2, 2, 2): -90}, 3)


@pytest.mark.parametrize("bad", [0.0, -500.0], ids=["zero", "imaginary"])
def test_zero_or_imaginary_active_mode_raises(bad):
    with pytest.raises(ValueError, match=r"Active modes \[1\]"):
        solve(np.array([3000.0, bad, 1200.0]), T, U_MODEL, "standard")


def test_inactive_zero_mode_is_accepted():
    """Rigid-body modes have zero frequency but are not active."""
    w = np.array([0.0, 3000.0, 1600.0, 1200.0])
    cfg = VPT2Config(freq=w, active_modes=[1, 2, 3], min_mode_idx=1, max_mode_idx=3,
                     len_modes=4, method="standard")
    Tf, Uf = np.zeros((4,) * 3), np.zeros((4,) * 4)
    Tf[1:, 1:, 1:], Uf[1:, 1:, 1:, 1:] = T, U_MODEL
    result = VPT2Solver(cfg, (Tf, Uf)).calculate()
    assert result.anharm_freq[0] == 0.0


def test_degenerate_modes_warn():
    with pytest.warns(UserWarning, match=r"Degenerate harmonic modes \[\(1, 2\)\]"):
        solve(np.array([3100.0, 1500.0, 1500.5]), T, U_MODEL, "standard")


@pytest.mark.filterwarnings("error::UserWarning")
def test_nearly_degenerate_modes_do_not_warn():
    solve(np.array([3100.0, 1500.0, 1502.0]), T, U_MODEL, "standard")
