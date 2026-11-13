"""Coincident turning circles admit an arc without an extra revolution."""

from unittest.mock import patch

import numpy as np
import numpy.testing as nt
import pytest
from spatialmath import SE2, Polygon2
from spatialmath.base import wrap_mpi_pi

from roboticstoolbox import Bicycle, DubinsPlanner, PolygonMap, RRTPlanner


@pytest.mark.parametrize(
    "angle",
    [np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2, 2 * np.pi / 3, np.pi, 3 * np.pi / 2],
)
@pytest.mark.parametrize("direction", [-1, 1])
@pytest.mark.parametrize("start", [(0, 0, 0), (3, -4, 0.7), (-2, 5, -1.2)])
def test_coincident_circles_do_not_add_a_revolution(
    angle: float, direction: int, start: tuple[float, float, float]
) -> None:
    relative_goal = SE2(
        np.sin(angle), direction * (1 - np.cos(angle)), direction * angle
    )
    goal = (SE2(start) * relative_goal).xyt()
    path, status = DubinsPlanner(curvature=1, stepsize=0.1).query(start, goal)

    # A unit-radius arc is a feasible path. For angles up to pi it also
    # attains the lower bound imposed by the heading change and curvature.
    assert status.length <= angle + 1e-12
    if angle <= np.pi:
        nt.assert_allclose(status.length, angle, atol=1e-12)
    nt.assert_allclose(path[0], start, atol=1e-12)
    nt.assert_allclose(path[-1, :2], goal[:2], atol=1e-12)
    nt.assert_allclose(wrap_mpi_pi(path[-1, 2] - goal[2]), 0, atol=1e-12)


@pytest.mark.parametrize("offset", [-1e-4, 1e-4])
@pytest.mark.parametrize("direction", [-1, 1])
def test_distinct_nearby_circles_retain_the_requested_endpoint(
    offset: float, direction: int
) -> None:
    goal = (1 + offset, direction, direction * np.pi / 2)
    path, _ = DubinsPlanner(curvature=1, stepsize=0.1).query((0, 0, 0), goal)
    nt.assert_allclose(path[-1, :2], goal[:2], atol=1e-10, rtol=0)
    nt.assert_allclose(wrap_mpi_pi(path[-1, 2] - goal[2]), 0, atol=1e-12)


def test_rrt_accepts_a_quarter_circle_connection() -> None:
    outline = Polygon2([(-0.1, -0.1), (-0.1, 0.1), (0.1, 0.1), (0.1, -0.1)])
    planner = RRTPlanner(
        map=PolygonMap(workspace=[-2, 2]),
        vehicle=Bicycle(polygon=outline),
        curvature=1,
        npoints=1,
        progress=False,
    )
    # A quarter-circle is below the planner's six-unit connection limit.
    # Bound sampling to expose rejection rather than enter an infinite loop.
    with patch.object(planner, "qrandom_free", side_effect=[np.array([0.0, 0, 0])]):
        planner.plan((1, 1, np.pi / 2), showsamples=False, showvalid=False)

    path, status = planner.query((0, 0, 0))
    nt.assert_allclose(status.length, np.pi / 2, atol=1e-12)
    nt.assert_allclose(path[-1], [1, 1, np.pi / 2], atol=1e-12)
