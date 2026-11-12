"""Short feasible paths must not be replaced by a full turning circle."""

import numpy as np
import numpy.testing as nt
import pytest
from spatialmath.base import wrap_mpi_pi

from roboticstoolbox import ReedsSheppPlanner


@pytest.mark.parametrize("curvature", [0.25, 1.0, 4.0])
@pytest.mark.parametrize(
    "distance", [-0.009, -0.001, -1e-5, 1e-5, 0.001, 0.009, -0.1, 0.1]
)
def test_short_straight_paths_retain_their_distance(
    curvature: float, distance: float
) -> None:
    path, status = ReedsSheppPlanner(curvature=curvature, stepsize=0.1).query(
        (0, 0, 0), (distance, 0, 0)
    )
    nt.assert_allclose(status.length, abs(distance), atol=1e-12)
    nt.assert_allclose(path[0], [0, 0, 0], atol=1e-12)
    nt.assert_allclose(path[-1], [distance, 0, 0], atol=1e-12)
    assert len(status.direction) == len(path)


@pytest.mark.parametrize("curvature", [0.25, 1.0, 4.0])
@pytest.mark.parametrize("start", [(0, 0, 0), (3, -4, 0.7), (-2, 5, -1.2)])
def test_stationary_path_has_no_motion(
    curvature: float, start: tuple[float, float, float]
) -> None:
    path, status = ReedsSheppPlanner(curvature=curvature).query(start, start)
    nt.assert_allclose(status.length, 0, atol=1e-12)
    nt.assert_allclose(path, np.tile(start, (len(path), 1)), atol=1e-12)
    assert len(path) > 0
    assert len(status.direction) == len(path)


@pytest.mark.parametrize("curvature", [0.25, 1.0, 4.0])
@pytest.mark.parametrize("direction", [-1, 1])
def test_short_arcs_retain_their_length(curvature: float, direction: int) -> None:
    angle = 1e-3
    goal = (
        np.sin(angle) / curvature,
        direction * (1 - np.cos(angle)) / curvature,
        direction * angle,
    )
    path, status = ReedsSheppPlanner(curvature=curvature, stepsize=0.1).query(
        (0, 0, 0), goal
    )
    nt.assert_allclose(status.length, angle / curvature, atol=1e-9)
    nt.assert_allclose(path[-1, :2], goal[:2], atol=1e-12)
    nt.assert_allclose(wrap_mpi_pi(path[-1, 2] - goal[2]), 0, atol=1e-12)
