"""D* must not return a route through an impenetrable cost barrier."""

from unittest.mock import patch

import numpy as np
import numpy.testing as nt
import pytest

from roboticstoolbox import DstarPlanner, NoPathError, OccupancyGrid


@pytest.mark.parametrize("cellsize", [0.25, 1.0])
@pytest.mark.parametrize("origin", [(0, 0), (-2, 3)])
@pytest.mark.parametrize("block_at", [0, 1, 2])
def test_unreachable_query_raises(
    cellsize: float, origin: tuple[float, float], block_at: int
) -> None:
    cost = np.ones((5, 5))
    if block_at == 0:
        cost[2, :] = np.inf
    grid = OccupancyGrid(cost, cellsize=cellsize, origin=origin)
    start, goal = grid.grid_to_world((0, 0)), grid.grid_to_world((4, 4))
    planner = DstarPlanner(grid, goal=goal)
    planner.plan()
    calls = 0

    def sensor(position: tuple[float, float]) -> list[tuple[float, float, float]]:
        nonlocal calls
        calls += 1
        assert calls < 20, "unreachable query must terminate"
        if calls == block_at:
            return [(*grid.grid_to_world((x, 2)), np.inf) for x in range(5)]
        return []

    with patch.object(planner, "animation_end", wraps=planner.animation_end) as finish:
        with pytest.raises(NoPathError, match="not reachable"):
            planner.query(start, sensor=sensor if block_at else None)
        if block_at:
            finish.assert_called_once()
        else:
            finish.assert_not_called()


@pytest.mark.parametrize("cellsize", [0.25, 1.0])
@pytest.mark.parametrize("origin", [(0, 0), (-2, 3)])
def test_barrier_with_a_gap_remains_reachable(
    cellsize: float, origin: tuple[float, float]
) -> None:
    cost = np.ones((5, 5))
    cost[2, :] = np.inf
    cost[2, 0] = 1
    grid = OccupancyGrid(cost, cellsize=cellsize, origin=origin)
    start, goal = grid.grid_to_world((0, 0)), grid.grid_to_world((4, 4))
    planner = DstarPlanner(grid, goal=goal)
    planner.plan()
    path, status = planner.query(start)

    nt.assert_allclose(path[0], start)
    nt.assert_allclose(path[-1], goal)
    assert np.isfinite(status.cost)
    for point in path:
        x, y = grid.world_to_grid(point)
        assert np.isfinite(cost[y, x])
