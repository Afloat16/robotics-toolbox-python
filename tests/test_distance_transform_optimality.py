"""Extract paths using the same legal moves and costs as the distance field."""

import heapq
import math

import numpy as np
import pytest

from roboticstoolbox import BinaryOccupancyGrid, DistanceTransformPlanner


def shortest_cost(grid, start, goal, diagonal):
    """Independent Dijkstra oracle on the finite cell graph."""
    frontier = [(0.0, start)]
    costs = {start: 0.0}
    while frontier:
        cost, point = heapq.heappop(frontier)
        if point == goal:
            return cost
        if cost > costs[point]:
            continue
        x, y = point
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if (dx == dy == 0) or (not diagonal and dx * dy != 0):
                    continue
                nx, ny = x + dx, y + dy
                if not (0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1]):
                    continue
                if grid[ny, nx]:
                    continue
                candidate = cost + math.hypot(dx, dy)
                if candidate < costs.get((nx, ny), math.inf):
                    costs[nx, ny] = candidate
                    heapq.heappush(frontier, (candidate, (nx, ny)))
    return math.inf


@pytest.mark.parametrize("metric", ["manhattan", "cityblock", "MANHATTAN"])
def test_manhattan_paths_use_only_axis_neighbours(metric):
    planner = DistanceTransformPlanner(np.zeros((5, 5)), metric=metric)
    planner.plan((4, 4))
    path = planner.query((0, 0))
    steps = np.abs(np.diff(path, axis=0))
    np.testing.assert_array_equal(steps.sum(axis=1), 1)
    assert len(path) == 9


@pytest.mark.parametrize("cellsize, origin", [(1.0, (0, 0)), (0.2, (-3, 4))])
@pytest.mark.parametrize("metric", ["euclidean", "manhattan", "cityblock"])
def test_path_cost_matches_independent_grid_shortest_path(cellsize, origin, metric):
    rows = [".......", "..##...", "..#....", "...#.#.", ".###.#.", ".#.....", "...##.."]
    grid = np.array([[c == "#" for c in row] for row in rows])
    occupancy = BinaryOccupancyGrid(grid, origin=origin, cellsize=cellsize)
    planner = DistanceTransformPlanner(occupancy, metric=metric)
    planner.plan(occupancy.g2w((6, 6)))
    path = planner.query(occupancy.g2w((0, 0)))
    expected = shortest_cost(grid, (0, 0), (6, 6), metric == "euclidean") * cellsize
    actual = np.linalg.norm(np.diff(path, axis=0), axis=1).sum()
    assert actual == pytest.approx(expected)
    assert planner.distancemap[0, 0] == pytest.approx(expected)
    assert not any(occupancy.isoccupied(point) for point in path)


def test_start_at_goal_has_one_pose():
    planner = DistanceTransformPlanner(np.zeros((3, 3)), metric="manhattan")
    planner.plan((1, 1))
    np.testing.assert_array_equal(planner.query((1, 1)), [[1, 1]])
