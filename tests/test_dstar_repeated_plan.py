"""A new D* goal needs a fresh shortest-path problem, not the old goal's tree."""

import numpy as np
import pytest

from roboticstoolbox import DstarPlanner, OccupancyGrid


@pytest.mark.parametrize("cellsize, origin", [(1.0, (0, 0)), (0.25, (-4, 2))])
@pytest.mark.parametrize("middle_goal", [(4, 4), (4, 0), (0, 4)])
def test_replanning_to_new_goal_matches_a_fresh_planner(cellsize, origin, middle_goal):
    costs = np.ones((5, 5))
    costs[1:4, 2] = 8
    geometry = OccupancyGrid(costs, origin=origin, cellsize=cellsize)
    planner = DstarPlanner(geometry)
    planner.plan(geometry.g2w(middle_goal))
    goal = geometry.g2w((0, 0))
    planner.plan(goal)
    fresh = DstarPlanner(geometry)
    fresh.plan(goal)

    actual = np.array([[state.h for state in row] for row in planner._Map._Map])
    expected = np.array([[state.h for state in row] for row in fresh._Map._Map])
    np.testing.assert_allclose(actual, expected)
    path, status = planner.query(geometry.g2w((4, 4)))
    np.testing.assert_allclose(path[-1], goal)
    assert status.cost == pytest.approx(expected[4, 4])
    assert len(path) <= costs.size


def test_explicit_plan_recomputes_after_costmap_edit():
    costs = np.ones((5, 5))
    planner = DstarPlanner(costs)
    planner.plan((4, 4))
    planner.costmap[2, :] = 10
    planner.plan((4, 4))
    fresh = DstarPlanner(planner.costmap.copy())
    fresh.plan((4, 4))
    path, status = planner.query((0, 0))
    expected_path, expected_status = fresh.query((0, 0))
    np.testing.assert_allclose(path, expected_path)
    assert status.cost == pytest.approx(expected_status.cost)
