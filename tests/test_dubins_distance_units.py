"""Dubins lengths and sample intervals use the same distance unit as poses."""

from unittest.mock import patch

import numpy as np
import numpy.testing as nt
import pytest
from spatialmath import Polygon2

from roboticstoolbox import Bicycle, DubinsPlanner, PolygonMap, RRTPlanner


@pytest.mark.parametrize("curvature", [0.25, 0.5, 1.0, 2.0, 4.0])
def test_straight_path_lengths_and_spacing(curvature: float) -> None:
    path, status = DubinsPlanner(curvature=curvature, stepsize=0.2).query(
        (0, 0, 0), (3, 0, 0)
    )

    nt.assert_allclose(status.length, 3.0, atol=1e-12)
    nt.assert_allclose(status.seglengths, [0, 3, 0], atol=1e-12)
    nt.assert_allclose(path[0], [0, 0, 0], atol=1e-12)
    nt.assert_allclose(path[-1], [3, 0, 0], atol=1e-12)
    intervals = np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1)
    assert intervals.max() <= 0.2 + 1e-12
    nt.assert_allclose(np.median(intervals[intervals > 1e-12]), 0.2, atol=1e-12)


@pytest.mark.parametrize("curvature", [0.25, 0.5, 1.0, 2.0, 4.0])
def test_arc_and_straight_lengths_and_spacing(curvature: float) -> None:
    radius = 1 / curvature
    path, status = DubinsPlanner(curvature=curvature, stepsize=0.1).query(
        (0, 0, 0), (radius, 3 * radius, np.pi / 2)
    )

    # A quarter-circle followed by a straight segment of two turning radii.
    nt.assert_allclose(status.length, (np.pi / 2 + 2) * radius, atol=1e-7)
    nt.assert_allclose(sum(status.seglengths), status.length, atol=1e-12)
    nt.assert_allclose(path[-1], [radius, 3 * radius, np.pi / 2], atol=1e-7)
    intervals = np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1)
    assert intervals.max() <= 0.1 + 1e-12


def test_rrt_accepts_a_connection_below_its_world_distance_limit() -> None:
    outline = Polygon2([(-0.1, -0.1), (-0.1, 0.1), (0.1, 0.1), (0.1, -0.1)])
    planner = RRTPlanner(
        map=PolygonMap(workspace=[0, 5]),
        vehicle=Bicycle(polygon=outline),
        curvature=2.0,
        npoints=1,
        progress=False,
    )
    # A four-unit straight connection is below RRT's existing six-unit limit.
    # Bound the sampler so a rejected connection fails instead of looping.
    with patch.object(planner, "qrandom_free", side_effect=[np.array([0.0, 0, 0])]):
        planner.plan((4, 0, 0), showsamples=False, showvalid=False)

    path, status = planner.query((0, 0, 0))
    nt.assert_allclose(status.length, 4.0, atol=1e-12)
    nt.assert_allclose(path[0], [0, 0, 0], atol=1e-12)
    nt.assert_allclose(path[-1], [4, 0, 0], atol=1e-12)
