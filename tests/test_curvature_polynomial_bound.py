"""A cubic curvature bound is over the full closed arc-length interval."""

import numpy as np
import pytest

from roboticstoolbox.mobile.CurvaturePolyPlanner import solvepath, xcurvature


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("length", [0.1, 1.0, 3.0])
def test_interior_curvature_extremum_is_not_an_ode_sampling_estimate(sign, length):
    # k(s) = sign * (1 - (2s/L - 1)^2), so |k| peaks at exactly 1 at L/2.
    coefficients = np.array([0, -4 / length**2, 4 / length, 0]) * sign
    parameters = np.r_[coefficients, length]
    assert xcurvature(parameters) == pytest.approx(1.0, abs=1e-12)
    for max_step in (np.inf, length / 13):
        _, maximum = solvepath(parameters, max_step=max_step)
        assert maximum == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize(
    "coefficients, length, expected",
    [
        ([0, 0, 0, -2], 3, 2),
        ([0, 0, 2, -1], 2, 3),
        ([0, 1, -4, 0], 3, 4),
        ([4, -6, 0, 0], 1, 2),
        ([4, -6, 0, 0], 0, 0),
    ],
)
def test_degenerate_polynomials_and_endpoints(coefficients, length, expected):
    assert xcurvature(np.r_[coefficients, length]) == pytest.approx(expected)


def test_maximum_curvature_is_independent_of_world_pose():
    parameters = [0, -4, 4, 0, 1]
    _, origin_maximum = solvepath(parameters)
    _, moved_maximum = solvepath(parameters, q0=(1e4, -1e4, 2.3))
    assert origin_maximum == pytest.approx(1)
    assert moved_maximum == pytest.approx(1)


@pytest.mark.parametrize("length", [0.1, 1.0, 3.0])
def test_cubic_checks_both_interior_stationary_points(length):
    # In normalized arc length: 6u^3 - 9u^2 + 3u has opposite extrema
    # at (3 +/- sqrt(3))/6, with equal magnitude sqrt(3)/6.
    parameters = [6 / length**3, -9 / length**2, 3 / length, 0, length]
    assert xcurvature(parameters) == pytest.approx(np.sqrt(3) / 6, abs=1e-12)
