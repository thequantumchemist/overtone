"""Central-difference stencils for first derivatives."""
import numpy as np
import pytest

from overtone.utils.stencils import get_stencil

# stencil n uses 2n + 2 points and is exact for polynomials up to degree 2n + 2
EXACT_DEGREE = {0: 2, 1: 4, 2: 6, 3: 8}
X0, H = 0.3, 0.1


def first_derivative(stencil, f):
    disp, coef, dfac = get_stencil(stencil)
    return sum(c * f(X0 + d * H) for d, c in zip(disp, coef)) / (dfac * H)


@pytest.mark.parametrize("stencil, degree", [
    (s, n) for s, top in EXACT_DEGREE.items() for n in range(top + 1)
])
def test_exact_for_polynomials_up_to_its_order(stencil, degree):
    approx = first_derivative(stencil, lambda x: x**degree)
    assert np.isclose(approx, degree * X0 ** (degree - 1) if degree else 0.0,
                      rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("stencil", EXACT_DEGREE)
def test_not_exact_beyond_its_order(stencil):
    degree = EXACT_DEGREE[stencil] + 1
    approx = first_derivative(stencil, lambda x: x**degree)
    assert not np.isclose(approx, degree * X0 ** (degree - 1), rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("stencil, n_points", [(0, 2), (1, 4), (2, 6), (3, 8)])
def test_number_of_points(stencil, n_points):
    disp, coef, _ = get_stencil(stencil)
    assert len(disp) == len(coef) == n_points


@pytest.mark.parametrize("stencil", EXACT_DEGREE)
def test_displacements_are_antisymmetric(stencil):
    disp, coef, _ = get_stencil(stencil)
    assert dict(zip(disp, coef)) == {-d: -c for d, c in zip(disp, coef)}


@pytest.mark.xfail(strict=True, raises=UnboundLocalError,
                   reason="get_stencil has no else branch: an unknown stencil raises "
                          "UnboundLocalError instead of a ValueError")
def test_unknown_stencil_raises_value_error():
    with pytest.raises(ValueError):
        get_stencil(4)
