"""Replanning must replace the tree rooted at the preceding goal."""

import numpy as np
import numpy.testing as nt
import pytest
from spatialmath import Polygon2

from roboticstoolbox import Bicycle, PolygonMap, RRTPlanner


def make_planner() -> RRTPlanner:
    outline = Polygon2([(-0.1, -0.1), (-0.1, 0.1), (0.1, 0.1), (0.1, -0.1)])
    return RRTPlanner(
        map=PolygonMap(workspace=[0, 5]),
        vehicle=Bicycle(polygon=outline),
        npoints=8,
        seed=1,
    )


@pytest.mark.parametrize("goal", [(1, 1, 0), (4, 4, 0)])
def test_replanning_matches_a_fresh_tree(goal: tuple[int, int, int]) -> None:
    planner = make_planner()
    planner.plan((4, 4, 0), showsamples=False, showvalid=False)
    old_vertices = list(planner.g)

    planner.plan(goal, showsamples=False, showvalid=False)
    fresh = make_planner()
    fresh.plan(goal, showsamples=False, showvalid=False)

    assert planner.g.n == fresh.g.n == 9
    assert planner.g.ne == fresh.g.ne == 8
    assert not any(vertex is old for vertex in planner.g for old in old_vertices)
    nt.assert_array_equal(planner.g[0].coord, goal)
    nt.assert_array_equal(
        np.stack([vertex.coord for vertex in planner.g]),
        np.stack([vertex.coord for vertex in fresh.g]),
    )
    path, status = planner.query(fresh.g[3].coord)
    expected, expected_status = fresh.query(fresh.g[3].coord)
    nt.assert_array_equal(path, expected)
    assert status.length == expected_status.length
    nt.assert_allclose(path[-1], goal, atol=1e-12)


def test_query_reaches_the_replanned_goal() -> None:
    planner = make_planner()
    planner.plan((4, 4, 0), showsamples=False, showvalid=False)
    planner.plan((1, 1, 0), showsamples=False, showvalid=False)

    path, status = planner.query((4, 4, 0))

    assert len(path) > 0
    nt.assert_allclose(path[-1], (1, 1, 0), atol=1e-12)
    nt.assert_array_equal(status.vertices[-1].coord, (1, 1, 0))
